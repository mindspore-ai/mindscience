#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def generate_slurm_script(args):
    """Generate the SLURM batch script."""
    script_content = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --ntasks-per-node={args.ntasks_per_node}",
        f"#SBATCH --time={args.time}",
    ]

    if args.partition:
        script_content.append(f"#SBATCH --partition={args.partition}")
    if args.mem:
        script_content.append(f"#SBATCH --mem={args.mem}")
    if args.output:
        script_content.append(f"#SBATCH --output={args.output}")
    if args.error:
        script_content.append(f"#SBATCH --error={args.error}")

    script_content.extend([
        "",
        "# Load necessary modules if required",
        f"{args.modules}" if args.modules else "# No modules specified",
        "",
        "# Run the command",
        "echo \"Starting job at $(date)\"",
        f"srun {args.command}" if not args.raw_command else args.command,
        "echo \"Job finished at $(date)\"",
    ])

    return "\n".join(script_content)

def main():
    parser = argparse.ArgumentParser(description="HPC SLURM Deploy Tool for AI Agents")
    parser.add_argument("--job_name", type=str, default="hpc_job", help="Name of the job")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--ntasks_per_node", type=int, default=1, help="Tasks per node (MPI ranks)")
    parser.add_argument("--time", type=str, default="01:00:00", help="Time limit (HH:MM:SS)")
    parser.add_argument("--partition", type=str, help="Partition/queue name")
    parser.add_argument("--mem", type=str, help="Memory per node (e.g., '16G')")
    parser.add_argument("--output", type=str, default="slurm-%j.out", help="Standard output file")
    parser.add_argument("--error", type=str, default="slurm-%j.err", help="Standard error file")
    parser.add_argument("--modules", type=str, help="Module load commands (e.g., 'module load openmpi/4.1.0')")
    parser.add_argument("--command", type=str, required=True, help="Command to execute (e.g., 'simpleFoam -parallel')")
    parser.add_argument("--raw_command", action="store_true", help="Do not prepend 'srun' to the command")
    parser.add_argument("--submit", action="store_true", help="Auto-submit using sbatch after generation")
    parser.add_argument("--script_name", type=str, default="submit.sh", help="Name of the generated script")

    args = parser.parse_args()

    # Generate script content
    script_text = generate_slurm_script(args)

    # Save to file
    try:
        with open(args.script_name, "w", encoding="utf-8") as f:
            f.write(script_text)
        print(f"[SUCCESS] SLURM script generated: {os.path.abspath(args.script_name)}")
    except Exception as e:
        print(f"[ERROR] Failed to write script: {e}", file=sys.stderr)
        sys.exit(1)

    # Auto-submit if requested
    if args.submit:
        try:
            print(f"Submitting job via sbatch {args.script_name}...")
            result = subprocess.run(["sbatch", args.script_name], capture_output=True, text=True, check=True)
            print(f"[SUCCESS] Job submitted! Output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] sbatch submission failed: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("[ERROR] 'sbatch' command not found. Are you on a SLURM cluster?", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
