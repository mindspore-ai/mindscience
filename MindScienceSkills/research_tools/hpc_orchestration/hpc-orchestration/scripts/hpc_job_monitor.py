#!/usr/bin/env python3
"""
HPC Job Monitor Tool for AI Agents.
Monitors the status of a submitted job across Slurm, PBS/Torque, or LSF schedulers.
Returns standardized states: PENDING, RUNNING, COMPLETED, FAILED, or UNKNOWN.
"""

import argparse
import subprocess
import sys
import time

def get_slurm_status(job_id):
    try:
        # %T is the state format in squeue
        res = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"], capture_output=True, text=True)
        state = res.stdout.strip()
        if not state:
            # If not in queue, check sacct (accounting) if available
            try:
                acct = subprocess.run(["sacct", "-j", str(job_id), "-n", "-X", "-o", "State"], capture_output=True, text=True)
                acct_state = acct.stdout.strip().split()[0] if acct.stdout.strip() else ""
                if acct_state:
                    return acct_state
            except Exception:
                pass
            return "COMPLETED_OR_UNKNOWN"
        
        if state in ["PENDING", "CONFIGURING"]: return "PENDING"
        if state in ["RUNNING", "COMPLETING"]: return "RUNNING"
        return state
    except Exception as e:
        return f"ERROR: {e}"

def get_pbs_status(job_id):
    try:
        # PBS qstat status codes: Q (queued), R (running), C (completed), E (exiting)
        res = subprocess.run(["qstat", "-f", str(job_id)], capture_output=True, text=True)
        if res.returncode != 0:
            return "COMPLETED_OR_UNKNOWN"
        for line in res.stdout.split('\n'):
            if "job_state =" in line:
                state_code = line.split("=")[1].strip()
                if state_code in ["Q", "W"]: return "PENDING"
                if state_code in ["R", "E"]: return "RUNNING"
                if state_code in ["C"]: return "COMPLETED"
                return f"UNKNOWN_CODE_{state_code}"
        return "UNKNOWN"
    except Exception as e:
        return f"ERROR: {e}"

def get_lsf_status(job_id):
    try:
        res = subprocess.run(["bjobs", "-noheader", "-o", "stat", str(job_id)], capture_output=True, text=True)
        if res.returncode != 0:
            return "COMPLETED_OR_UNKNOWN"
        state = res.stdout.strip()
        if state in ["PEND", "PSUSP"]: return "PENDING"
        if state in ["RUN", "USUSP", "SSUSP"]: return "RUNNING"
        if state == "DONE": return "COMPLETED"
        if state == "EXIT": return "FAILED"
        return state
    except Exception as e:
        return f"ERROR: {e}"

def main():
    parser = argparse.ArgumentParser(description="Multi-Scheduler HPC Job Monitor for Agents")
    parser.add_argument("job_id", type=str, help="The Job ID returned by the submission command.")
    parser.add_argument("--scheduler", type=str, choices=["slurm", "pbs", "lsf"], required=True, 
                        help="Target scheduler system (must be explicitly provided).")
    parser.add_argument("--wait_for_running", action="store_true", 
                        help="Block and wait until the job leaves the PENDING state.")
    parser.add_argument("--timeout", type=int, default=3600, 
                        help="Maximum seconds to wait if --wait_for_running is set.")
    
    args = parser.parse_args()
    
    def check_status():
        if args.scheduler == "slurm":
            return get_slurm_status(args.job_id)
        elif args.scheduler == "pbs":
            return get_pbs_status(args.job_id)
        elif args.scheduler == "lsf":
            return get_lsf_status(args.job_id)
        return "UNKNOWN"

    initial_status = check_status()
    print(f"[JOB_ID] {args.job_id}")
    print(f"[STATUS] {initial_status}")

    if args.wait_for_running and initial_status == "PENDING":
        print(f"[INFO] Waiting for job {args.job_id} to start running (Timeout: {args.timeout}s)...")
        elapsed = 0
        interval = 10  # check every 10 seconds
        
        while elapsed < args.timeout:
            time.sleep(interval)
            elapsed += interval
            current_status = check_status()
            
            if current_status != "PENDING":
                print(f"[UPDATE] Job {args.job_id} state changed to: {current_status}")
                sys.exit(0)
                
        print(f"[TIMEOUT] Job {args.job_id} is still PENDING after {args.timeout} seconds.")
        sys.exit(1)

if __name__ == "__main__":
    main()
