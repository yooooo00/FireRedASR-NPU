# Copyright (c) 2025 JiuTian ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.
import warnings

warnings.filterwarnings("ignore")

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None

try:
    import torchair as tng
    from torchair import CompilerConfig
except ModuleNotFoundError:  # pragma: no cover
    tng = None
    CompilerConfig = None


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr.models.fireredasr import load_fireredasr_aed_model  # noqa: E402
from fireredasr.models.module.transformer_decoder import TransformerDecoder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASR decoder compile benchmark (NPU)")
    parser.add_argument("--device", type=str, default="npu:0", help="e.g. npu:0 / cuda:0 / cpu")
    parser.add_argument("--repeat", type=int, default=32, help="Repeat count for base encoder outputs")

    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    parser.add_argument("--softmax_smoothing", type=float, default=1.25)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--eos_penalty", type=float, default=1.0)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--ckpt", type=str, default=None, help="FireRedASR AED model.pth.tar (optional)")
    parser.add_argument("--enc_outputs", type=str, default=None, help="torch.save tensor, shape [1,T,H] (optional)")
    parser.add_argument("--enc_mask", type=str, default=None, help="torch.save tensor, shape [1,1,T] or [1,T] (optional)")
    parser.add_argument("--T", type=int, default=152, help="Synthetic encoder length when no --enc_outputs")
    parser.add_argument("--H", type=int, default=1280, help="Hidden size when no --enc_outputs")

    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Enable torch.compile (default on)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.set_defaults(compile=True)

    parser.add_argument(
        "--compile_impl",
        type=str,
        default="auto",
        choices=["auto", "decoder_method", "run_kernel", "batch_beam_search"],
        help=(
            "auto: prefer decoder.compile_kernel() / run_kernel compilation if available; "
            "otherwise compile batch_beam_search."
        ),
    )
    parser.add_argument("--dynamic", action="store_true", help="torch.compile(dynamic=True)")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.set_defaults(dynamic=True)
    parser.add_argument("--fullgraph", action="store_true", help="torch.compile(fullgraph=True)")
    parser.add_argument("--compile_mode", type=str, default=None, help="torchair CompilerConfig.mode (optional)")

    parser.add_argument("--out", type=str, default="out/decoder_compile_bench.json")
    return parser.parse_args()


def _dtype_from_arg(dtype: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def _device_from_arg(device: str) -> torch.device:
    if device.isdigit():
        return torch.device(f"npu:{device}")
    return torch.device(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _torch_load_compat(path: str):
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_or_make_inputs(args, device: torch.device, dtype: torch.dtype):
    if args.enc_outputs:
        enc_outputs = _torch_load_compat(args.enc_outputs)
    else:
        torch.manual_seed(args.seed)
        enc_outputs = torch.randn(1, args.T, args.H, dtype=torch.float32)

    if args.enc_mask:
        enc_mask = _torch_load_compat(args.enc_mask)
    else:
        enc_mask = torch.ones(enc_outputs.size(0), 1, enc_outputs.size(1), dtype=torch.uint8)

    if enc_mask.dim() == 2:
        enc_mask = enc_mask.unsqueeze(1)

    enc_outputs = enc_outputs.repeat(args.repeat, 1, 1).to(device=device, dtype=dtype)
    enc_mask = enc_mask.repeat(args.repeat, 1, 1).to(device=device)
    return enc_outputs, enc_mask


def _bench(label: str, device: torch.device, fn, warmup: int, iters: int):
    print(f"\n[{label}] Warm-up ({warmup} iters)...")
    for _ in range(warmup):
        fn()
        _synchronize(device)

    print(f"[{label}] Timed run ({iters} iters)...")
    times_ms = []
    for i in range(iters):
        _synchronize(device)
        start = time.time() * 1000
        fn()
        _synchronize(device)
        end = time.time() * 1000
        dt = end - start
        times_ms.append(dt)
        print(f"[{label}] iter {i + 1}: {dt:.3f} ms")

    stats = {
        "avg_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }
    print(f"[{label}] stats: {stats}")
    return stats


def _get_npu_backend(args):
    if tng is None or CompilerConfig is None:
        return None
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    if args.compile_mode:
        config.mode = args.compile_mode
    return tng.get_npu_backend(compiler_config=config)


def _prepare_decoder_inputs_v2(decoder, enc_outputs, enc_mask, beam_size: int):
    batch_size = enc_outputs.size(0)
    T = enc_outputs.size(1)
    H = enc_outputs.size(2)

    enc_mask = enc_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_size * beam_size, -1, T)
    sos_id = decoder.get_sos_id()
    inf = decoder.get_inf()
    ys = torch.ones(batch_size * beam_size, 1, device=enc_outputs.device).fill_(sos_id).long()
    scores_temp = torch.tensor([0.0] + [-inf] * (beam_size - 1), device=enc_outputs.device).float()
    scores = scores_temp.repeat(batch_size).view(batch_size * beam_size, 1)
    is_finished = torch.zeros_like(scores, device=enc_outputs.device)
    input_tgt_mask = torch.ones(1, T, device=enc_outputs.device).to(torch.uint8)
    mask_score = scores_temp.view(1, beam_size).repeat(batch_size * beam_size, 1)
    stride = (
        beam_size
        * torch.arange(batch_size, device=enc_outputs.device).view(batch_size, 1).repeat(1, beam_size).view(batch_size * beam_size)
    )
    index_add = beam_size * torch.arange(batch_size, device=enc_outputs.device).view(batch_size, 1).long()
    enc_outputs = enc_outputs.unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_size * beam_size, T, H)

    return enc_outputs, enc_mask, ys, scores, is_finished, input_tgt_mask, mask_score, stride, index_add


def main():
    args = parse_args()
    device = _device_from_arg(args.device)
    dtype = _dtype_from_arg(args.dtype)

    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("Requested NPU device but torch_npu is not available")
        torch.npu.set_device(device.index or 0)

    torch.manual_seed(args.seed)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        model = load_fireredasr_aed_model(args.ckpt).to(device).eval()
        decoder = model.decoder
    else:
        d_model = args.H
        if args.enc_outputs:
            probe = _torch_load_compat(args.enc_outputs)
            d_model = int(probe.size(-1))
        try:
            decoder = TransformerDecoder(
                sos_id=3,
                eos_id=4,
                pad_id=2,
                odim=7832,
                n_layers=16,
                n_head=20,
                d_model=d_model,
                residual_dropout=0.1,
                pe_maxlen=5000,
            ).to(device).eval()
        except TypeError:
            decoder = TransformerDecoder(
                3,
                4,
                2,
                7832,
                16,
                20,
                d_model,
                0.1,
                5000,
            ).to(device).eval()

    decoder = decoder.to(device).to(dtype=dtype)

    enc_outputs, enc_mask = _load_or_make_inputs(args, device=device, dtype=dtype)

    sig = inspect.signature(decoder.batch_beam_search)
    uses_prepared_inputs = "ys" in sig.parameters

    if uses_prepared_inputs:
        (
            enc_outputs_v,
            enc_mask_v,
            ys,
            scores,
            is_finished,
            input_tgt_mask,
            mask_score,
            stride,
            index_add,
        ) = _prepare_decoder_inputs_v2(decoder, enc_outputs, enc_mask, args.beam_size)

        def run_decode():
            decoder.batch_beam_search(
                enc_outputs_v,
                enc_mask_v,
                ys,
                scores,
                is_finished,
                input_tgt_mask,
                mask_score,
                stride,
                index_add,
                args.beam_size,
                args.nbest,
                args.decode_max_len,
                args.softmax_smoothing,
                args.length_penalty,
                args.eos_penalty,
            )

    else:

        def run_decode():
            decoder.batch_beam_search(
                enc_outputs,
                enc_mask,
                args.beam_size,
                args.nbest,
                args.decode_max_len,
                args.softmax_smoothing,
                args.length_penalty,
                args.eos_penalty,
            )

    eager_stats = _bench("eager", device, run_decode, args.warmup, args.iters)

    compiled_stats = None
    compile_time_s = None
    compile_error = None
    compile_path = None

    if args.compile:
        backend = _get_npu_backend(args) if device.type == "npu" else None
        compile_choice = args.compile_impl
        if compile_choice == "auto":
            if hasattr(decoder, "compile_kernel"):
                compile_choice = "decoder_method"
            elif hasattr(decoder, "_run_kernel_v1") or hasattr(decoder, "_run_kernel_v2"):
                compile_choice = "run_kernel"
            else:
                compile_choice = "batch_beam_search"

        print(f"\n[compile] impl={compile_choice} dynamic={args.dynamic} fullgraph={args.fullgraph} mode={args.compile_mode}")

        try:
            start = time.time()
            if compile_choice == "decoder_method":
                decoder.compile_kernel()
                compile_path = "decoder.compile_kernel()"
            elif compile_choice == "run_kernel":
                if backend is None:
                    raise RuntimeError("torchair is required for NPU compilation")
                if hasattr(decoder, "_run_kernel_v1"):
                    decoder.run_kernel_v1 = torch.compile(
                        decoder._run_kernel_v1, dynamic=args.dynamic, fullgraph=args.fullgraph, backend=backend
                    )
                if hasattr(decoder, "_run_kernel_v2"):
                    decoder.run_kernel_v2 = torch.compile(
                        decoder._run_kernel_v2, dynamic=args.dynamic, fullgraph=args.fullgraph, backend=backend
                    )
                compile_path = "decoder._run_kernel_v1/_run_kernel_v2"
            else:
                if backend is not None:
                    compiled_fn = torch.compile(
                        run_decode, dynamic=args.dynamic, fullgraph=args.fullgraph, backend=backend
                    )
                else:
                    compiled_fn = torch.compile(run_decode, dynamic=args.dynamic, fullgraph=args.fullgraph)

                def run_decode_compiled():
                    return compiled_fn()

                compile_path = "run_decode()"

                run_decode = run_decode_compiled

            run_decode()
            _synchronize(device)
            compile_time_s = time.time() - start
            print(f"[compile] first-run (compile+execute) time: {compile_time_s:.4f} s")

            compiled_stats = _bench("compiled", device, run_decode, args.warmup, args.iters)
        except Exception as e:  # pragma: no cover
            compile_error = repr(e)
            print(f"[compile] failed: {compile_error}")

    summary = {
        "device": str(device),
        "dtype": args.dtype,
        "repeat": args.repeat,
        "beam_size": args.beam_size,
        "T": int(enc_outputs.size(1)),
        "H": int(enc_outputs.size(2)),
        "uses_prepared_inputs": uses_prepared_inputs,
        "compile": args.compile,
        "compile_impl": args.compile_impl,
        "compile_mode": args.compile_mode,
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
        "compile_path": compile_path,
        "compile_time_s": compile_time_s,
        "compile_error": compile_error,
        "eager": eager_stats,
        "compiled": compiled_stats,
        "env": {
            "TORCH_LOGS": os.environ.get("TORCH_LOGS"),
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "versions": {
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None) if torch_npu is not None else None,
        },
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

