# Copyright (c) 2025 JiuTian ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.
import warnings

warnings.filterwarnings("ignore")

import argparse
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

from fireredasr.data.asr_feat import ASRFeatExtractor  # noqa: E402
from fireredasr.models.fireredasr import load_fireredasr_aed_model  # noqa: E402


TARGET_FBANK_FRAMES = {0: 606, 1: 1214, 2: 2430}


def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASR AED compile benchmark with real audio input (NPU)")
    parser.add_argument("--device", type=str, default="npu:0", help="e.g. npu:0 / cuda:0 / cpu")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing model.pth.tar and cmvn.ark (preferred).",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model.pth.tar (optional)")
    parser.add_argument("--cmvn", type=str, default=None, help="Path to cmvn.ark (optional)")

    # Audio input
    parser.add_argument("--wav_path", type=str)
    parser.add_argument("--wav_paths", type=str, nargs="*")
    parser.add_argument("--wav_dir", type=str)
    parser.add_argument("--wav_scp", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--repeat_to_batch",
        action="store_true",
        help="If provided wavs < batch_size, repeat last wav to fill the batch.",
    )

    # Optional padding to match customer server buckets (post-fbank length, before encoder subsampling).
    parser.add_argument(
        "--pad_level",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Pad fbank frames to 606/1214/2430 to align with server buckets.",
    )

    # Decode options
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    parser.add_argument("--softmax_smoothing", type=float, default=1.25)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--eos_penalty", type=float, default=1.0)

    # Benchmark
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)

    # Compile
    parser.add_argument(
        "--compile_target",
        type=str,
        default="none",
        choices=["none", "encoder", "decoder", "end2end"],
        help="Which part to torch.compile.",
    )
    parser.add_argument("--dynamic", action="store_true", help="torch.compile(dynamic=True)")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.set_defaults(dynamic=True)
    parser.add_argument("--fullgraph", action="store_true", help="torch.compile(fullgraph=True)")
    parser.add_argument("--compile_mode", type=str, default=None, help="torchair CompilerConfig.mode (optional)")
    parser.add_argument(
        "--capture_scalar_outputs",
        action="store_true",
        help="Set torch._dynamo.config.capture_scalar_outputs=True (reduces item()-related graph breaks).",
    )

    # Outputs
    parser.add_argument("--out", type=str, default="out/aed_compile_audio_bench.json")
    parser.add_argument("--save_enc_outputs", type=str, default=None, help="Save encoder outputs (.pt) for reuse.")
    parser.add_argument("--save_enc_mask", type=str, default=None, help="Save encoder mask (.pt) for reuse.")
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


def _timed_ms(device: torch.device, fn):
    if device.type == "npu":
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.npu.synchronize()
        return float(start.elapsed_time(end)), out
    _synchronize(device)
    t0 = time.time() * 1000
    out = fn()
    _synchronize(device)
    t1 = time.time() * 1000
    return float(t1 - t0), out


def _get_npu_backend(args):
    if tng is None or CompilerConfig is None:
        return None
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    if args.compile_mode:
        config.mode = args.compile_mode
    return tng.get_npu_backend(compiler_config=config)


def _get_wav_paths(args):
    if args.wav_path:
        wavs = [args.wav_path]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = list(args.wav_paths)
    elif args.wav_scp:
        wavs = [line.strip().split()[-1] for line in open(args.wav_scp, encoding="utf-8")]
    elif args.wav_dir:
        wavs = []
        for root, _, files in os.walk(args.wav_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(root, f))
        wavs.sort()
    else:
        raise ValueError("Please provide wav input via --wav_path/--wav_paths/--wav_scp/--wav_dir")

    if len(wavs) < args.batch_size:
        if not args.repeat_to_batch:
            raise ValueError(f"Need >= --batch_size wavs (got {len(wavs)}), or pass --repeat_to_batch")
        if len(wavs) == 0:
            raise ValueError("No wavs found")
        wavs = wavs + [wavs[-1]] * (args.batch_size - len(wavs))
    return wavs[: args.batch_size]


def _pad_fbank_to(feats: torch.Tensor, lengths: torch.Tensor, target_frames: int):
    # Pad/truncate each sample to exactly target_frames and set lengths accordingly.
    bsz, cur_frames, dim = feats.shape
    if cur_frames == target_frames:
        return feats, torch.full((bsz,), target_frames, dtype=torch.long)
    if cur_frames > target_frames:
        feats = feats[:, :target_frames, :].contiguous()
        return feats, torch.full((bsz,), target_frames, dtype=torch.long)
    pad = feats.new_zeros((bsz, target_frames - cur_frames, dim))
    feats = torch.cat([feats, pad], dim=1)
    return feats, torch.full((bsz,), target_frames, dtype=torch.long)


def _bench(label: str, device: torch.device, fn, warmup: int, iters: int):
    print(f"\n[{label}] Warm-up ({warmup} iters)...")
    for _ in range(warmup):
        fn()

    print(f"[{label}] Timed run ({iters} iters)...")
    times_ms = []
    for i in range(iters):
        dt, _ = _timed_ms(device, fn)
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


def main():
    args = parse_args()
    device = _device_from_arg(args.device)
    dtype = _dtype_from_arg(args.dtype)

    if args.capture_scalar_outputs:
        torch._dynamo.config.capture_scalar_outputs = True

    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("Requested NPU device but torch_npu is not available")
        torch.npu.set_device(device.index or 0)

    torch.manual_seed(args.seed)

    model_dir = args.model_dir
    if model_dir is None and args.ckpt:
        model_dir = str(Path(args.ckpt).resolve().parent)
    if model_dir is None:
        raise ValueError("Provide --model_dir or --ckpt")

    ckpt_path = args.ckpt or str(Path(model_dir) / "model.pth.tar")
    cmvn_path = args.cmvn or str(Path(model_dir) / "cmvn.ark")

    wavs = _get_wav_paths(args)
    print(f"#wavs(batch)={len(wavs)}")

    feat_extractor = ASRFeatExtractor(cmvn_path)
    feats, lengths, durs = feat_extractor(wavs)
    total_dur = float(sum(durs))

    if args.pad_level is not None:
        feats, lengths = _pad_fbank_to(feats, lengths, TARGET_FBANK_FRAMES[args.pad_level])

    feats = feats.to(device=device, dtype=dtype)
    lengths = lengths.to(device=device)

    model = load_fireredasr_aed_model(ckpt_path).to(device).eval()
    model = model.to(dtype=dtype)

    # Optional compilation.
    backend = _get_npu_backend(args) if device.type == "npu" else None
    compile_error = None
    compiled_targets = []

    encoder_fn = model.encoder
    decoder_fn = model.decoder

    if args.compile_target != "none":
        try:
            if args.compile_target in ("encoder", "end2end"):
                if backend is not None:
                    encoder_fn = torch.compile(
                        model.encoder, dynamic=args.dynamic, fullgraph=args.fullgraph, backend=backend
                    )
                else:
                    encoder_fn = torch.compile(model.encoder, dynamic=args.dynamic, fullgraph=args.fullgraph)
                compiled_targets.append("encoder")

            if args.compile_target in ("decoder", "end2end"):

                def _decoder_call(enc_outputs, enc_mask):
                    return model.decoder.batch_beam_search(
                        enc_outputs,
                        enc_mask,
                        args.beam_size,
                        args.nbest,
                        args.decode_max_len,
                        args.softmax_smoothing,
                        args.length_penalty,
                        args.eos_penalty,
                    )

                if backend is not None:
                    decoder_fn = torch.compile(
                        _decoder_call, dynamic=args.dynamic, fullgraph=args.fullgraph, backend=backend
                    )
                else:
                    decoder_fn = torch.compile(_decoder_call, dynamic=args.dynamic, fullgraph=args.fullgraph)
                compiled_targets.append("decoder(batch_beam_search)")
        except Exception as e:  # pragma: no cover
            compile_error = repr(e)
            print(f"[compile] failed during setup: {compile_error}")

    def run_encoder_only():
        with torch.no_grad():
            return encoder_fn(feats, lengths)

    def run_decoder_only(enc_outputs, enc_mask):
        with torch.no_grad():
            # decoder_fn may be a module (not callable wrapper) if we didn't compile decoder.
            if callable(decoder_fn) and not isinstance(decoder_fn, torch.nn.Module):
                return decoder_fn(enc_outputs, enc_mask)
            return model.decoder.batch_beam_search(
                enc_outputs,
                enc_mask,
                args.beam_size,
                args.nbest,
                args.decode_max_len,
                args.softmax_smoothing,
                args.length_penalty,
                args.eos_penalty,
            )

    def run_end2end():
        with torch.no_grad():
            enc_outputs, _, enc_mask = encoder_fn(feats, lengths)
            return run_decoder_only(enc_outputs, enc_mask)

    # Eager or compiled benchmark.
    encoder_stats = _bench("encoder", device, run_encoder_only, args.warmup, args.iters)

    enc_outputs0, _, enc_mask0 = run_encoder_only()
    if args.save_enc_outputs:
        torch.save(enc_outputs0.detach().cpu(), args.save_enc_outputs)
        print(f"Saved encoder outputs: {args.save_enc_outputs}")
    if args.save_enc_mask:
        torch.save(enc_mask0.detach().cpu(), args.save_enc_mask)
        print(f"Saved encoder mask: {args.save_enc_mask}")

    decoder_stats = _bench(
        "decoder",
        device,
        lambda: run_decoder_only(enc_outputs0, enc_mask0),
        args.warmup,
        args.iters,
    )
    end2end_stats = _bench("end2end", device, run_end2end, args.warmup, args.iters)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "pad_level": args.pad_level,
        "total_audio_dur_s": total_dur,
        "compile_target": args.compile_target,
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
        "compile_mode": args.compile_mode,
        "capture_scalar_outputs": bool(args.capture_scalar_outputs),
        "compiled_targets": compiled_targets,
        "compile_error": compile_error,
        "encoder": encoder_stats,
        "decoder": decoder_stats,
        "end2end": end2end_stats,
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

