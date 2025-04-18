#!/usr/bin/env python3
import os
import subprocess
import time
import glob
import csv
import statistics
from collections import defaultdict

# Configuration
DATA_DIR = "data"
ALGORITHM = "LS2"  # or "LS1"
CUTOFF_TIME = 600
SEED = 30
EXECUTABLE = "local_search.py"
DATASETSIZE = "large"  # or "small" or "test"
NUM_RUNS = 10  # Number of runs for each instance

def read_optimal_solution(filepath):
    """Reads optimal quality from .out file."""
    with open(filepath, 'r') as f:
        return int(f.readline().strip())

def extract_solution_info(sol_file):
    """Reads solution quality and collection size from .sol file."""
    if not os.path.exists(sol_file):
        print(f"⚠️ Solution file {sol_file} not found!")
        return None, None  # Return None if the file doesn't exist
    
    with open(sol_file, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        print(f"⚠️ Solution file {sol_file} is empty!")
        return None, None
    
    quality = int(lines[0].strip())
    
    # Extract collection size (number of selected subsets)
    # If format is different, this may need adjustment
    collection_size = 0
    if len(lines) > 1:
        # Try to count the number of selected subsets from second line
        # Assuming it's a space-separated list of subset indices
        try:
            selected_subsets = lines[1].strip().split()
            collection_size = len(selected_subsets)
        except:
            # If we can't parse the second line, count '1's in the first line
            # Assuming binary representation where 1 = selected subset
            try:
                collection_size = lines[0].strip().count('1')
            except:
                print(f"⚠️ Could not determine collection size from {sol_file}")
    
    return quality, collection_size

def run_instance(instance_file, seed):
    """Runs the solver for a given instance with specific seed."""
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    dataset_type = "large" if "large" in instance_file else "small" if "small" in instance_file else "test"
    
    # Create the directory structure if it doesn't exist
    output_dir = f"solutions/{ALGORITHM}/{dataset_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = f"{output_dir}/{instance_name}_{ALGORITHM}_{int(CUTOFF_TIME)}_{seed}"
    sol_file = f"{output_prefix}.sol"
    
    # Run the algorithm
    start_time = time.time()
    cmd = [
        "python3", EXECUTABLE,
        "-inst", instance_file,
        "-alg", ALGORITHM,
        "-time", str(CUTOFF_TIME),
        "-seed", str(seed),
        "-sol", output_prefix
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    execution_time = time.time() - start_time
    
    return sol_file, execution_time

def verify_all():
    results = []
    csv_data = []
    
    # Process all dataset types
    for dataset_type in ["large", "small", "test"]:
        # Change pattern to match the specified dataset files
        instance_files = sorted(glob.glob(f"{DATA_DIR}/{dataset_type}*.in"))
        
        for inst in instance_files:
            instance_name = os.path.basename(inst)
            out_file = inst.replace(".in", ".out")
            
            print(f"Verifying {inst} with {NUM_RUNS} runs...")
            
            opt_quality = read_optimal_solution(out_file)
            run_qualities = []
            run_times = []
            run_sizes = []
            run_relerrs = []
            
            # Perform multiple runs with different seeds
            for run in range(1, NUM_RUNS + 1):
                seed = SEED + run
                sol_file, exec_time = run_instance(inst, seed)
                
                # Check if the solution file was created successfully
                found_quality, collection_size = extract_solution_info(sol_file)
                if found_quality is not None:
                    run_qualities.append(found_quality)
                    run_times.append(exec_time)
                    run_sizes.append(collection_size if collection_size is not None else 0)
                    
                    # Calculate relative error
                    rel_error = (found_quality - opt_quality) / opt_quality if opt_quality != 0 else float('inf')
                    run_relerrs.append(rel_error)
                    
                    print(f"  Run {run}: Quality={found_quality}, Size={collection_size}, Time={exec_time:.2f}s, RelErr={rel_error:.4f}")
            
            if run_qualities:
                avg_quality = statistics.mean(run_qualities)
                avg_time = statistics.mean(run_times)
                avg_size = statistics.mean(run_sizes)
                avg_rel_error = statistics.mean(run_relerrs)
                
                status = "✅ Match" if avg_quality == opt_quality else f"⚠️ Off by {avg_quality - opt_quality}"
                results.append((instance_name, opt_quality, avg_quality, avg_size, status, avg_time, avg_rel_error))
                
                # Collect data for CSV
                csv_data.append({
                    'Instance': instance_name,
                    'Optimal': opt_quality,
                    'Algorithm': ALGORITHM,
                    'Avg_Quality': round(avg_quality, 2),
                    'Avg_Size': round(avg_size, 2),
                    'Avg_Time': round(avg_time, 2),
                    'RelErr': round(avg_rel_error, 4),
                    'Num_Runs': len(run_qualities)
                })
            else:
                results.append((instance_name, opt_quality, "No successful runs", "N/A", "❌", 0, float('inf')))
                csv_data.append({
                    'Instance': instance_name,
                    'Optimal': opt_quality,
                    'Algorithm': ALGORITHM,
                    'Avg_Quality': "N/A",
                    'Avg_Size': "N/A",
                    'Avg_Time': "N/A",
                    'RelErr': "N/A",
                    'Num_Runs': 0
                })

    # Print summary of results
    print("\nSummary:")
    print(f"{'Instance':<15} {'Optimal':<10} {'Found':<10} {'Size':<10} {'Status':<15} {'Avg Time':<10} {'RelErr':<10}")
    for r in results:
        if isinstance(r[2], str):  # No successful runs
            print(f"{r[0]:<15} {r[1]:<10} {r[2]:<10} {r[3]:<10} {r[4]:<15} N/A N/A")
        else:
            print(f"{r[0]:<15} {r[1]:<10} {r[2]:<10.2f} {r[3]:<10.2f} {r[4]:<15} {r[5]:.2f}s {r[6]:.4f}")
    
    # Save to CSV
    csv_filename = f"results_{ALGORITHM}_{DATASETSIZE}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Optimal', 'Algorithm', 'Avg_Quality', 'Avg_Size', 'Avg_Time', 'RelErr', 'Num_Runs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"\nResults saved to {csv_filename}")
    return csv_data

def compare_algorithms():
    """Run both algorithms and create a comprehensive comparison CSV."""
    all_results = []
    
    for alg in ["LS1", "LS2"]:
        global ALGORITHM
        ALGORITHM = alg
        print(f"\n===== Running {alg} =====")
        
        # Store current results
        results = verify_all()
        if results:
            all_results.extend(results)
    
    # Create comprehensive comparison CSV
    comp_csv_filename = "comprehensive_comparison.csv"
    
    # Reorganize data for comprehensive table
    instances = defaultdict(dict)
    for result in all_results:
        instance = result['Instance']
        alg = result['Algorithm']
        instances[instance][f"{alg}_Quality"] = result['Avg_Quality']
        instances[instance][f"{alg}_Size"] = result['Avg_Size']
        instances[instance][f"{alg}_Time"] = result['Avg_Time']
        instances[instance][f"{alg}_RelErr"] = result['RelErr']
        instances[instance]['Optimal'] = result['Optimal']
    
    # Write to CSV
    with open(comp_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Optimal', 
                      'LS1_Quality', 'LS1_Size', 'LS1_Time', 'LS1_RelErr',
                      'LS2_Quality', 'LS2_Size', 'LS2_Time', 'LS2_RelErr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for instance, data in sorted(instances.items()):
            row = {'Instance': instance, 'Optimal': data.get('Optimal', 'N/A')}
            for alg in ["LS1", "LS2"]:
                row[f"{alg}_Quality"] = data.get(f"{alg}_Quality", "N/A")
                row[f"{alg}_Size"] = data.get(f"{alg}_Size", "N/A")
                row[f"{alg}_Time"] = data.get(f"{alg}_Time", "N/A")
                row[f"{alg}_RelErr"] = data.get(f"{alg}_RelErr", "N/A")
            writer.writerow(row)
    
    print(f"\nComprehensive comparison saved to {comp_csv_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run and verify local search algorithms')
    parser.add_argument('--alg', choices=['LS1', 'LS2', 'both'], default='both', 
                        help='Algorithm to run (default: both)')
    parser.add_argument('--dataset', choices=['large', 'small', 'test', 'all'], default='all',
                        help='Dataset size to use (default: all)')
    parser.add_argument('--runs', type=int, default=10, 
                        help='Number of runs per instance (default: 10)')
    parser.add_argument('--time', type=float, default=600, 
                        help='Cutoff time in seconds (default: 600)')
    parser.add_argument('--seed', type=int, default=30,
                        help='Base random seed (default: 30)')
    
    args = parser.parse_args()
    
    CUTOFF_TIME = args.time
    SEED = args.seed
    NUM_RUNS = args.runs
    
    if args.alg == 'both':
        compare_algorithms()
    else:
        ALGORITHM = args.alg
        if args.dataset == 'all':
            for dataset in ['large', 'small', 'test']:
                DATASETSIZE = dataset
                print(f"\n===== Processing {dataset} dataset with {ALGORITHM} =====")
                verify_all()
        else:
            DATASETSIZE = args.dataset
            verify_all()