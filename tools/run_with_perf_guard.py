#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command with a performance guard. "
            "The process is terminated once elapsed time exceeds the allowed threshold."
        )
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        required=True,
        help="Reference runtime in seconds for the old/known-good command.",
    )
    parser.add_argument(
        "--margin-seconds",
        type=float,
        default=5.0,
        help="Allowed slowdown in seconds before the new run is terminated.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="How often to check elapsed time.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Working directory for the command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Prefix with -- to separate from tool flags.",
    )
    args = parser.parse_args()
    if args.baseline_seconds <= 0:
        parser.error("--baseline-seconds must be > 0")
    if args.margin_seconds < 0:
        parser.error("--margin-seconds must be >= 0")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be > 0")
    if not args.command:
        parser.error("No command provided. Use -- <command> [args...]")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("No command provided after -- separator.")
    return args


def main() -> int:
    args = parse_args()
    threshold = args.baseline_seconds + args.margin_seconds
    print(
        f"[perf-guard] baseline={args.baseline_seconds:.2f}s "
        f"margin={args.margin_seconds:.2f}s threshold={threshold:.2f}s"
    )
    print(f"[perf-guard] cwd={args.cwd}")
    print(f"[perf-guard] command={' '.join(args.command)}")
    start = time.perf_counter()
    process = subprocess.Popen(args.command, cwd=args.cwd)
    timed_out = False
    try:
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            if elapsed > threshold:
                timed_out = True
                print(
                    f"[perf-guard] terminating: elapsed {elapsed:.2f}s "
                    f"exceeded threshold {threshold:.2f}s",
                    file=sys.stderr,
                )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    elapsed = time.perf_counter() - start
    if timed_out:
        print(f"[perf-guard] final elapsed={elapsed:.2f}s", file=sys.stderr)
        return 124
    print(f"[perf-guard] completed in {elapsed:.2f}s with exit code {process.returncode}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
