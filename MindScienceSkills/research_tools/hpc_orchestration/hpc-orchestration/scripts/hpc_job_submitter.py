#!/usr/bin/env python3
"""
HPC Job Submitter Tool for AI Agents.
Generates batch scripts for Slurm, PBS/Torque, or LSF schedulers,
and optionally submits them to the queue.
"""

import argparse
import os
import subprocess
import sys

def detect_scheduler():
    """Detects the available HPC job scheduler by checking system commands."""
    try:
        subprocess.run(["sbatch", "--version"], capture_output=True, check=True)
        return "slurm"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        subprocess.run(["qsub", "--version"], capture_output=True, check=True)
        return "pbs"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        subprocess.run(["bsub", "-V"], capture_output=True, check=True)
        return "lsf"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "unknown"

def generate_slurm_script(args):
    """Generates a Slurm submission script (#SBATCH)."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --ntasks-per-node={args.ntasks_per_node}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={args.output or 'slurm-%j.out'}",
        f"#SBATCH --error={args.error or 'slurm-%j.err'}"
    ]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.mem:
        lines.append(f"#SBATCH --mem={args.mem}")
    
    return "\n".join(lines)

def generate_pbs_script(args):
    """Generates a PBS/Torque submission script (#PBS)."""
    lines = [
        "#!/bin/bash",
        f"#PBS -N {args.job_name}",
        f"#PBS -l nodes={args.nodes}:ppn={args.ntasks_per_node}",
        f"#PBS -l walltime={args.time}",
        f"#PBS -o {args.output or args.job_name + '.out'}",
        f"#PBS -e {args.error or args.job_name + '.err'}"
    ]
    if args.partition:
        lines.append(f"#PBS -q {args.partition}")
    if args.mem:
        lines.append(f"#PBS -l mem={args.mem.lower()}")
    
    # PBS scripts often need to cd into the submission directory
    lines.append("cd $PBS_O_WORKDIR")
    
    return "\n".join(lines)

def generate_lsf_script(args):
    """Generates an LSF submission script (#BSUB)."""
    total_cores = args.nodes * args.ntasks_per_node
    # Convert HH:MM:SS to HH:MM for LSF
    lsf_time = ":".join(args.time.split(":")[:2])
    
    lines = [
        "#!/bin/bash",
        f"#BSUB -J {args.job_name}",
        f"#BSUB -n {total_cores}",
        f'#BSUB -R "span[ptile={args.ntasks_per_node}]"',
        f"#BSUB -W {lsf_time}",
        f"#BSUB -o {args.output or '%J.out'}",
        f"#BSUB -e {args.error or '%J.err'}"
    ]
    if args.partition:
        lines.append(f"#BSUB -q {args.partition}")
    if args.mem:
        # LSF memory can be per-core or total depending on config. Assume total MB for simplicity or pass explicitly.
        lines.append(f'#BSUB -R "rusage[mem={args.mem}]"')
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Multi-Scheduler HPC Job Generator for Agents")
    
    # Scheduler selection
    parser.add_argument("--scheduler", type=str, choices=["slurm", "pbs", "lsf", "auto"], default="auto", 
                        help="Target scheduler system (auto detects based on available commands).")
    
    # Resource specifications
    parser.add_argument("--job_name", type=str, default="hpc_agent_job", help="Name of the job.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of compute nodes.")
    parser.add_argument("--ntasks_per_node", type=int, default=1, help="Number of MPI tasks (cores) per node.")
    parser.add_argument("--time", type=str, default="01:00:00", help="Wallclock time limit (HH:MM:SS).")
    parser.add_argument("--partition", type=str, help="Queue or partition name (optional).")
    parser.add_argument("--mem", type=str, help="Memory requirement (e.g., '16G', '16gb').")
    
    # IO / Environment
    parser.add_argument("--output", type=str, help="Standard output file name.")
    parser.add_argument("--error", type=str, help="Standard error file name.")
    parser.add_argument("--modules", type=str, help="Environment modules to load (e.g., 'openmpi/4.0').")
    
    # Execution
    parser.add_argument("--command", type=str, required=True, help="The main compute command to execute (e.g., 'mpirun lmp -in in.script').")
    parser.add_argument("--script_name", type=str, default="submit_job.sh", help="Filename for the generated script.")
    parser.add_argument("--submit", action="store_true", help="Automatically submit the job after generating the script.")

    args = parser.parse_args()

    # Determine scheduler
    scheduler = args.scheduler
    if scheduler == "auto":
        scheduler = detect_scheduler()
        if scheduler == "unknown":
            print("[WARNING] Could not auto-detect scheduler. Defaulting to 'slurm'.", file=sys.stderr)
            scheduler = "slurm"
        else:
            print(f"[INFO] Auto-detected scheduler: {scheduler.upper()}")

    # Generate Headers
    if scheduler == "slurm":
        header = generate_slurm_script(args)
        submit_cmd = "sbatch"
    elif scheduler == "pbs":
        header = generate_pbs_script(args)
        submit_cmd = "qsub"
    elif scheduler == "lsf":
        header = generate_lsf_script(args)
        submit_cmd = "bsub <" # LSF often requires redirection
    else:
        print(f"[ERROR] Unsupported scheduler: {scheduler}", file=sys.stderr)
        sys.exit(1)

    # Assemble final script
    script_body = [
        header,
        "",
        "# ----------------- Environment Setup -----------------",
        f"module load {args.modules}" if args.modules else "# No modules specified",
        "",
        "# ----------------- Execution -------------------------",
        "echo \"[AGENT-LOG] Job started at $(date)\"",
        args.command,
        "echo \"[AGENT-LOG] Job finished at $(date)\""
    ]
    
    final_script = "\n".join(script_body)

    # Write to file
    try:
        with open(args.script_name, "w", encoding="utf-8") as f:
            f.write(final_script)
        print(f"[SUCCESS] {scheduler.upper()} script generated: {os.path.abspath(args.script_name)}")
    except Exception as e:
        print(f"[ERROR] Failed to write script: {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-submit
    if args.submit:
        try:
            print(f"[ACTION] Submitting job via: {submit_cmd} {args.script_name}")
            if submit_cmd == "bsub <":
                with open(args.script_name, "r") as f:
                    result = subprocess.run(["bsub"], stdin=f, capture_output=True, text=True, check=True)
            else:
                result = subprocess.run([submit_cmd, args.script_name], capture_output=True, text=True, check=True)
            
            print(f"[SUCCESS] Job submitted successfully! Output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Job submission failed. Return code: {e.returncode}\nStderr: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print(f"[ERROR] Submission command '{submit_cmd.split()[0]}' not found on this system.", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
