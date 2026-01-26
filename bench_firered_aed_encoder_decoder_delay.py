# Copyright (c) 2025 JiuTian ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.
import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr.models.fireredasr import load_fireredasr_aed_model  # noqa: E402


@dataclass(frozen=True)
class BenchCase:
    batch_size: int
    length_t: int


DEFAULT_CASES: List[BenchCase] = [
    BenchCase(8, 152),
    BenchCase(16, 152),
    BenchCase(32, 152),
    BenchCase(64, 152),
    BenchCase(8, 304),
    BenchCase(16, 304),
    BenchCase(32, 304),
    BenchCase(8, 608),
    BenchCase(16, 608),
    BenchCase(32, 608),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FireRedASR-AED encoder/decoder benchmark (eager vs compiled-kernel) in md-table format"
    )
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ckpt", type=str, required=True, help="Path to FireRedASR-AED model.pth.tar")

    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    parser.add_argument("--softmax_smoothing", type=float, default=1.25)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--eos_penalty", type=float, default=1.0)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)

    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Optional override cases, e.g. --cases 8,152 16,152 8,304 (default uses built-in table cases)",
    )
    parser.add_argument("--max_cases", type=int, default=None, help="Optional truncate case list (useful for quick smoke)")

    parser.add_argument("--compile", action="store_true", help="Run compiled-kernel benchmark (default on)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=True)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--dynamic", action="store_true", help="torch.compile(dynamic=True)")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.set_defaults(dynamic=True)
    parser.add_argument("--fullgraph", action="store_true", help="torch.compile(fullgraph=True)")

    parser.add_argument("--out_md", type=str, default="out/firered_aed_encoder_decoder_delay.md")
    return parser.parse_args()


def _parse_cases(args) -> List[BenchCase]:
    if not args.cases:
        cases = list(DEFAULT_CASES)
    else:
        cases = []
        for spec in args.cases:
            parts = [p.strip() for p in spec.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid --cases item: {spec!r}, expected 'B,T'")
            cases.append(BenchCase(int(parts[0]), int(parts[1])))

    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    return cases


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


def _timed_ms(device: torch.device, fn) -> float:
    if device.type == "npu":
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        return float(start.elapsed_time(end))

    _synchronize(device)
    t0 = time.time() * 1000
    fn()
    _synchronize(device)
    t1 = time.time() * 1000
    return float(t1 - t0)


def _bench_ms(device: torch.device, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = [_timed_ms(device, fn) for _ in range(iters)]
    return float(np.mean(times))


def _conv_subsample2_out_len(input_len: int) -> int:
    # Two conv layers with kernel_size=3, stride=2, no padding:
    # L1 = (L0 - 3)//2 + 1, L2 = (L1 - 3)//2 + 1
    l1 = (input_len - 3) // 2 + 1
    l2 = (l1 - 3) // 2 + 1
    return int(l2)


def _find_input_len_for_t(t_out: int) -> int:
    # Prefer the bucket-aligned padded length used by the customer pipeline:
    # for Conv2dSubsampling (k=3,s=2) x2, padded_len = 4*T + 4 gives stable matches
    # (e.g. 152->612, 304->1220, 608->2436).
    candidate = t_out * 4 + 4
    if _conv_subsample2_out_len(candidate) == t_out:
        return candidate

    start = max(8, t_out * 4 - 16)
    end = t_out * 4 + 64
    for l in range(start, end):
        if _conv_subsample2_out_len(l) == t_out:
            return l
    raise ValueError(f"Cannot find input length for T_out={t_out} in range [{start}, {end})")


def _make_md(title: str, rows: List[Tuple[int, int, float, float]]) -> str:
    lines = [
        title,
        "",
        "|Batch Size|Length (T)|Encoder 推理时间 (ms)|Decoder 推理时间 (ms)|",
        "|---|---|---|---|",
    ]
    for b, t, enc_ms, dec_ms in rows:
        lines.append(f"|{b}|{t}|{enc_ms:.4f}|{dec_ms:.4f}|")
    return "\n".join(lines)


def main():
    args = parse_args()
    device = _device_from_arg(args.device)
    dtype = _dtype_from_arg(args.dtype)
    cases = _parse_cases(args)

    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("Requested NPU device but torch_npu is not available")
        torch.npu.set_device(device.index or 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    model = load_fireredasr_aed_model(args.ckpt)
    print(f"[load] model constructed in {(time.time() - t0):.3f}s", flush=True)

    t1 = time.time()
    print(f"[load] moving model to {device} ...", flush=True)
    model = model.to(device).eval()
    print(f"[load] moved to device in {(time.time() - t1):.3f}s", flush=True)

    t2 = time.time()
    print(f"[load] converting dtype to {dtype} ...", flush=True)
    model = model.to(dtype=dtype)
    print(f"[load] converted dtype in {(time.time() - t2):.3f}s", flush=True)

    print(
        f"bench: #cases={len(cases)} warmup={args.warmup} iters={args.iters} "
        f"compile={args.compile} dynamic={args.dynamic} fullgraph={args.fullgraph} mode={args.compile_mode}"
    )

    def run_case(case: BenchCase, phase: str) -> Tuple[float, float]:
        print(f"[{phase}] case: B={case.batch_size} T={case.length_t} ...", flush=True)
        pad_extra = int(model.encoder.input_preprocessor.context - 1)
        padded_len = _find_input_len_for_t(case.length_t)
        input_len = padded_len - pad_extra
        if input_len <= 0:
            raise ValueError(f"Invalid input_len={input_len} for T={case.length_t} (pad_extra={pad_extra})")
        feats = torch.randn(case.batch_size, input_len, 80, device=device, dtype=dtype)
        lengths = torch.full((case.batch_size,), input_len, device=device, dtype=torch.long)

        def enc_step():
            with torch.no_grad():
                model.encoder(feats, lengths)

        enc_ms = _bench_ms(device, enc_step, args.warmup, args.iters)

        with torch.no_grad():
            enc_outputs, _, enc_mask = model.encoder(feats, lengths)

        def dec_step():
            with torch.no_grad():
                model.decoder.batch_beam_search(
                    enc_outputs,
                    enc_mask,
                    args.beam_size,
                    args.nbest,
                    args.decode_max_len,
                    args.softmax_smoothing,
                    args.length_penalty,
                    args.eos_penalty,
                )

        dec_ms = _bench_ms(device, dec_step, args.warmup, args.iters)
        print(f"[{phase}] done: B={case.batch_size} T={case.length_t} enc={enc_ms:.3f}ms dec={dec_ms:.3f}ms", flush=True)
        return enc_ms, dec_ms

    eager_rows = []
    print("\n[phase] eager", flush=True)
    for case in cases:
        enc_ms, dec_ms = run_case(case, "eager")
        eager_rows.append((case.batch_size, case.length_t, enc_ms, dec_ms))

    compiled_rows = []
    if args.compile:
        print("\n[phase] compile-kernel setup", flush=True)
        if hasattr(model.encoder, "reset_kernel"):
            model.encoder.reset_kernel()
        if hasattr(model.decoder, "reset_kernel"):
            model.decoder.reset_kernel()
        if hasattr(model.encoder, "compile_kernel"):
            model.encoder.compile_kernel(dynamic=args.dynamic, fullgraph=args.fullgraph, mode=args.compile_mode)
        if hasattr(model.decoder, "compile_kernel"):
            model.decoder.compile_kernel(dynamic=args.dynamic, fullgraph=args.fullgraph, mode=args.compile_mode)

        print("\n[phase] compiled", flush=True)
        for case in cases:
            enc_ms, dec_ms = run_case(case, "compiled")
            compiled_rows.append((case.batch_size, case.length_t, enc_ms, dec_ms))

    out_path = REPO_ROOT / args.out_md
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_parts = [
        "# 模型推理性能数据汇总表（精简版）",
        "",
        "## 一、原始代码推理性能",
        "",
        _make_md("", eager_rows).lstrip(),
    ]
    if args.compile:
        md_parts.extend(
            [
                "",
                "## 二、torch.compile 优化后推理性能",
                "",
                _make_md("", compiled_rows).lstrip(),
            ]
        )
    md = "\n".join(md_parts).rstrip() + "\n"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
