#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import time
import os
import signal

def tail_file(filepath, wait_timeout):
    """Generator to continuously read new lines from a file, with initial wait for creation."""
    print(f"[INFO] Waiting for log file to be created: {filepath}")
    
    elapsed = 0
    interval = 2
    while not os.path.exists(filepath):
        if elapsed >= wait_timeout:
            print(f"[ERROR] Timeout ({wait_timeout}s) waiting for log file: {filepath}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval)
        elapsed += interval
        
    print(f"[INFO] File found. Tailing {filepath}...")

    with open(filepath, 'r') as file:
        file.seek(0, os.SEEK_END)
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line

def track_log(args):
    """Tracks a log file for specific metrics or divergence patterns."""
    diverge_pattern = re.compile(args.diverge_regex, re.IGNORECASE) if args.diverge_regex else None
    converge_pattern = re.compile(args.converge_regex, re.IGNORECASE) if args.converge_regex else None
    metric_pattern = re.compile(args.metric_regex, re.IGNORECASE) if args.metric_regex else None

    print(f"[INFO] Started tracking log: {args.log_file}")
    if metric_pattern:
        print(f"       Tracking metric with regex: {args.metric_regex}")
    
    try:
        for line in tail_file(args.log_file, args.wait_timeout):
            line_str = line.strip()
            
            # Print new lines if verbose mode
            if args.verbose:
                print(f"[LOG] {line_str}")

            # Extract metric
            if metric_pattern:
                match = metric_pattern.search(line_str)
                if match:
                    # Assume first group is the metric value
                    val = match.group(1)
                    if not args.verbose:  # Only print if not already printing every line
                        print(f"[METRIC] {val}")

            # Check Divergence (Failure)
            if diverge_pattern and diverge_pattern.search(line_str):
                print(f"\n[ALERT - DIVERGENCE DETECTED]")
                print(f"Trigger Line: {line_str}")
                print(f"Action required: Check KNOWLEDGE_ERRORS.md for root cause and self-healing strategy.")
                
                if args.kill_job_id and args.scheduler:
                    print(f"[ACTION] Killing {args.scheduler.upper()} Job ID {args.kill_job_id}...")
                    if args.scheduler == "slurm":
                        subprocess.run(["scancel", args.kill_job_id], check=True)
                    elif args.scheduler == "pbs":
                        subprocess.run(["qdel", args.kill_job_id], check=True)
                    elif args.scheduler == "lsf":
                        subprocess.run(["bkill", args.kill_job_id], check=True)
                
                # Exit with error code to trigger Agent's self-healing loop
                sys.exit(2)

            # Check Convergence (Success)
            if converge_pattern and converge_pattern.search(line_str):
                print(f"\n[ALERT - CONVERGENCE DETECTED]")
                print(f"Trigger Line: {line_str}")
                print("[SUCCESS] Simulation completed successfully according to track rules.")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n[INFO] Tracking interrupted by user.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="HPC Log Tracker for Real-time Monitoring and Divergence Detection")
    parser.add_argument("log_file", type=str, help="Path to the log file to track")
    parser.add_argument("--diverge_regex", type=str, default="floating point exception|NaN", help="Regex to detect divergence")
    parser.add_argument("--converge_regex", type=str, default="solution converged|loop time of", help="Regex to detect convergence")
    parser.add_argument("--metric_regex", type=str, help="Regex to extract metric")
    parser.add_argument("--wait_timeout", type=int, default=3600, help="Seconds to wait for log file creation")
    parser.add_argument("--kill_job_id", type=str, help="Job ID to kill if divergence is detected")
    parser.add_argument("--scheduler", type=str, choices=["slurm", "pbs", "lsf"], help="Scheduler system used")
    parser.add_argument("--verbose", action="store_true", help="Print every line read from the log")

    args = parser.parse_args()
    track_log(args)

if __name__ == "__main__":
    main()
