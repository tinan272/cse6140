#!/usr/bin/env python3

import os
import subprocess
import time
import glob

# Configuration
DATA_DIR = "data"
ALGORITHM = "LS2"  # or "LS1"
CUTOFF_TIME = 600
SEED = 30
EXECUTABLE = "local_search.py"
DATASETSIZE = "large"

def read_optimal_solution(filepath):
    """Reads optimal quality from .out file."""
    with open(filepath, 'r') as f:
        return int(f.readline().strip())

def extract_computed_quality(sol_file):
    """Reads solution quality from .sol file."""
    if not os.path.exists(sol_file):
        print(f"⚠️ Solution file {sol_file} not found!")
        return None  # Return None if the file doesn't exist
    with open(sol_file, 'r') as f:
        return int(f.readline().strip())

def run_instance(instance_file):
    """Runs the solver for a given instance."""
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    output_prefix = f"solutions/{ALGORITHM}/{DATASETSIZE}/{instance_name}_{ALGORITHM}_{int(CUTOFF_TIME)}_{SEED}"
    sol_file = f"{output_prefix}.sol"

    # Run the algorithm
    cmd = [
        "python3", EXECUTABLE,
        "-inst", instance_file,
        "-alg", ALGORITHM,
        "-time", str(CUTOFF_TIME),
        "-seed", str(SEED),
        "-sol", output_prefix
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return sol_file

def verify_all():
    results = []

    # Change to glob the large dataset files
    instance_files = sorted(glob.glob(f"{DATA_DIR}/{DATASETSIZE}*.in"))
    for inst in instance_files:
        out_file = inst.replace(".in", ".out")

        print(f"Verifying {inst} ...")

        opt_quality = read_optimal_solution(out_file)
        sol_file = run_instance(inst)

        # Check if the solution file was created successfully
        found_quality = extract_computed_quality(sol_file)
        if found_quality is None:
            results.append((os.path.basename(inst), opt_quality, "Solution file not found", "❌"))
            continue

        match = "✅ Match" if found_quality == opt_quality else f"⚠️ Off by {found_quality - opt_quality}"
        results.append((os.path.basename(inst), opt_quality, found_quality, match))

    print("\nSummary:")
    print(f"{'Instance':<15} {'Optimal':<10} {'Found':<10} {'Status'}")
    for r in results:
        print(f"{r[0]:<15} {r[1]:<10} {r[2]:<10} {r[3]}")

if __name__ == "__main__":
    verify_all()