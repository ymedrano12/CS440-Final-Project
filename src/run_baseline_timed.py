"""
Running Competition Baseline
Command: python src/run_baseline_timed.py

Warning! The original implementation uses dense feature vectors, 
so running this baseline takes a LOT of time and memory. 

Therefore this file stops the baseline if it runs longer than 60 min,
uses more than 80% RAM, or if it crashes. What happens is recorded

"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
 
import psutil
 
from db_config import REPO_ROOT, get_db_path
 
RESULTS = REPO_ROOT / "results"
GB = 1024**3
 
 
def tree_rss(proc):
    """Total RAM in bytes used by the process and all its children."""
    total = 0
    try:
        for p in [proc] + proc.children(recursive=True):
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total
 
 
def kill_tree(proc):
    try:
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
    except psutil.Error:
        pass
 
 
def last_line(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return lines[-1][:90] if lines else "(no output yet)"
    except OSError:
        return "(no output yet)"
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-ram-percent", type=float, default=80)
    ap.add_argument("--db", help="optional path to the database")
    ap.add_argument(
        "--script",
        type=Path,
        default=REPO_ROOT / "competition_baseline" / "main.py",
    )
    args = ap.parse_args()
 
    if not args.script.is_file():
        sys.exit(f"Could not find the baseline script at: {args.script}")
 
    db = get_db_path(args.db)
    RESULTS.mkdir(exist_ok=True)
    stdout_path = RESULTS / "baseline_stdout.txt"
    metrics_path = RESULTS / "baseline_metrics.json"
    metrics_path.unlink(missing_ok=True)  # avoid stale results
 
    total_ram = psutil.virtual_memory().total
    ram_cap = total_ram * args.max_ram_percent / 100
    limit = args.minutes * 60
    cmd = [sys.executable, "-u", str(args.script), str(db), "--output", str(metrics_path)]
 
    print(f"Total RAM: {total_ram / GB:.1f} GB | RAM cap: {ram_cap / GB:.1f} GB")
    print(f"Time limit: {args.minutes:.0f} min | Ctrl+C stops it early\n")
 
    started = datetime.now().isoformat(timespec="seconds")
    start = time.time()
    peak = 0
    outcome = None
 
    with open(stdout_path, "w", encoding="utf-8") as out:
        child = subprocess.Popen(
            cmd, cwd=args.script.parent, stdout=out, stderr=subprocess.STDOUT
        )
        proc = psutil.Process(child.pid)
        tick = 0
        try:
            while child.poll() is None:
                time.sleep(5)
                tick += 1
                elapsed = time.time() - start
                rss = tree_rss(proc)
                peak = max(peak, rss)
                if tick % 6 == 0:  # status line every 30 seconds
                    print(
                        f"[{elapsed / 60:5.1f} min] RAM {rss / GB:5.2f} GB | "
                        f"{last_line(stdout_path)}"
                    )
                if elapsed > limit:
                    outcome = "stopped: time limit reached"
                    kill_tree(proc)
                    break
                if rss > ram_cap:
                    outcome = "stopped: RAM cap reached"
                    kill_tree(proc)
                    break
        except KeyboardInterrupt:
            outcome = "stopped: by user (Ctrl+C)"
            kill_tree(proc)
        child.wait()
 
    runtime = time.time() - start
    if outcome is None:
        outcome = "completed" if child.returncode == 0 else "crashed"
 
    record = {
        "outcome": outcome,
        "exit_code": child.returncode,
        "runtime_seconds": round(runtime, 1),
        "runtime_minutes": round(runtime / 60, 2),
        "peak_ram_gb": round(peak / GB, 2),
        "total_ram_gb": round(total_ram / GB, 1),
        "ram_cap_gb": round(ram_cap / GB, 1),
        "time_limit_minutes": args.minutes,
        "started": started,
        "metrics_file_written": metrics_path.is_file(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": " ".join(cmd),
    }
    (RESULTS / "baseline_run_raw.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
 
    print(f"\nOutcome: {outcome}")
    print(f"Runtime: {runtime / 60:.1f} min | Peak RAM: {peak / GB:.2f} GB")
    print(f"Saved to: {RESULTS}")
 
 
if __name__ == "__main__":
    main()
 