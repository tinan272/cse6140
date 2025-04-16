#!/usr/bin/env python3
import numpy as np
"""
Implementation of two Local Search algorithms for Minimum Set Cover problem:
LS1. Hill Climbing w/ Random Restarts
LS2. Simulated Annealing

run with:
    python minimum_set_cover.py -inst <filename> -alg [LS1|LS2] -time <cutoff in seconds> -seed <random seed>
"""

import argparse
import random
import time
import os
import sys
import math
from typing import List, Set, Dict, Tuple, Optional

class MinimumSetCover:
    def __init__(self, instance_file: str):
        """
        Init Minimum Set Cover problem
        
        Args: instance_file: Path to the instance file
        """
        self.instance_name = os.path.basename(instance_file).split('.')[0]
        self.n = 0  # num of ele
        self.m = 0  # num of subsets
        self.subsets = []  # list sets
        
        self._parse_input_file(instance_file)
    
    def _parse_input_file(self, instance_file: str) -> None:
        """
        Parse input file & extract problem instance.
        
        Args:
            instance_file: Path to the instance file
        """
        try:
            with open(instance_file, 'r') as f:
                lines = f.readlines()
                
                # read 1st line
                self.n, self.m = map(int, lines[0].strip().split())
                
                # read each subset
                self.subsets = []
                for i in range(1, self.m + 1):
                    if i <= len(lines):
                        subset_info = list(map(int, lines[i].strip().split()))
                        subset_size = subset_info[0]
                        subset = set(subset_info[1:])
                        self.subsets.append(subset)
                    else:
                        print(f"Warning: Expected {self.m} subsets but file has fewer lines.")
                        break
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    
    def evaluate_solution(self, solution: List[bool]) -> Tuple[int, bool]:
        """
        Evaluate solution by counting selected subsets & checking coverage.
        
        Args: solution: boolean list w/ selected subsets
            
        Returns: tuple of (number of selected subsets, is fully covering)
        """
        selected_count = sum(1 for s in solution if s) # num selected subsets
        
        # coverage
        covered = set()
        for i, selected in enumerate(solution):
            if selected:
                covered.update(self.subsets[i])
        
        is_fully_covering = covered == set(range(1, self.n + 1))
        
        return selected_count, is_fully_covering
    
    def generate_initial_solution(self) -> List[bool]:
        """
        Construct init solution using greedy.
        
        Returns: boolean list w/ solution
        """
        solution = [False] * self.m
        remaining_elements = set(range(1, self.n + 1))
        
        while remaining_elements:
            best_subset_idx = -1
            max_covered = -1
            
            # find subset covering most remaining elements
            for i in range(self.m): 
                if solution[i]: # skip selected subsets
                    continue  
                
                covered = len(self.subsets[i].intersection(remaining_elements))
                if covered > max_covered:
                    max_covered = covered
                    best_subset_idx = i
            
            if best_subset_idx == -1 or max_covered == 0: # no more elements
                break  
            
            solution[best_subset_idx] = True
            
            
            remaining_elements -= self.subsets[best_subset_idx] # update
        if remaining_elements:
            for i in range(self.m):
                solution[i] = True
        return solution
    
    def hill_climbing(self, cutoff_time: float, random_seed: int, output_prefix: str) -> Tuple[List[bool], int]:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        trace_filename = f"{output_prefix}.trace"
        with open(trace_filename, 'w') as _:
            pass
        
        start_time = time.time()
        
        # Initial solution (greedy + ensure coverage)
        current_solution = self.generate_initial_solution()
        current_solution = self._ensure_coverage(current_solution)
        
        # Precompute element coverage information
        # Use a dictionary instead of a list to handle any element values
        element_to_subsets = {}
        for i, subset in enumerate(self.subsets):
            for element in subset:
                if element not in element_to_subsets:
                    element_to_subsets[element] = []
                element_to_subsets[element].append(i)
        
        # Track current coverage state
        current_covered_elements = set()
        for i, selected in enumerate(current_solution):
            if selected:
                current_covered_elements.update(self.subsets[i])
        
        current_quality = sum(current_solution)
        
        best_solution = current_solution.copy()
        best_quality = current_quality
        best_covered = current_covered_elements.copy()
        
        self._append_to_trace_file(trace_filename, 0.0, best_quality)
        
        max_iterations_without_improvement = 1000
        restart_probability = 0.05  # Reduced from 0.1
        iterations_without_improvement = 0
        
        # Tabu list
        tabu_list = {}  # index -> tabu expiration iteration
        tabu_tenure = 10  # How long an index stays in tabu list
        iteration = 0
        
        # Cache subset sizes for quick lookup
        subset_sizes = [len(subset) for subset in self.subsets]
        
        while time.time() - start_time < cutoff_time:
            iteration += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= cutoff_time:
                break
            
            # --- Random Restart Logic ---
            if iterations_without_improvement >= max_iterations_without_improvement:
                # More intelligent restart: preserve some of the best solution
                if random.random() < 0.7:  # 70% chance of partial preservation
                    # Start with best solution and perturb it
                    current_solution = best_solution.copy()
                    current_covered_elements = best_covered.copy()
                    
                    # Flip some random bits (more aggressively)
                    flip_count = max(1, self.m // 10)
                    indices_to_flip = random.sample(range(self.m), flip_count)
                    
                    for i in indices_to_flip:
                        if current_solution[i]:
                            # Try to remove, check if coverage maintained
                            current_solution[i] = False
                            
                            # Update coverage
                            affected_elements = self.subsets[i]
                            removed_elements = set()
                            
                            for elem in affected_elements:
                                still_covered = False
                                for subset_idx in element_to_subsets.get(elem, []):
                                    if subset_idx != i and current_solution[subset_idx]:
                                        still_covered = True
                                        break
                                if not still_covered:
                                    removed_elements.add(elem)
                            
                            if removed_elements:
                                # Restore if coverage broken
                                current_solution[i] = True
                            else:
                                # Need to update coverage status for affected elements
                                for elem in affected_elements:
                                    is_covered = False
                                    for subset_idx in element_to_subsets.get(elem, []):
                                        if subset_idx != i and current_solution[subset_idx]:
                                            is_covered = True
                                            break
                                    if is_covered:
                                        current_covered_elements.add(elem)
                                    else:
                                        current_covered_elements.discard(elem)
                        else:
                            # Add this subset
                            current_solution[i] = True
                            current_covered_elements.update(self.subsets[i])
                else:
                    # Completely fresh restart with greedy
                    current_solution = self.generate_initial_solution()
                    current_solution = self._ensure_coverage(current_solution)
                    
                    # Recompute coverage
                    current_covered_elements = set()
                    for i, selected in enumerate(current_solution):
                        if selected:
                            current_covered_elements.update(self.subsets[i])
                
                current_quality = sum(current_solution)
                iterations_without_improvement = 0
                tabu_list = {}  # Clear tabu list on restart
            
            # --- Neighbor Selection Strategy ---
            potential_moves = []
            
            # Prioritize removing subsets (reducing solution size)
            removal_candidates = [i for i, selected in enumerate(current_solution) if selected]
            
            # Shuffle to randomize equal-quality moves
            random.shuffle(removal_candidates)
            
            # Evaluate each potential removal
            for i in removal_candidates:
                if i in tabu_list and iteration < tabu_list[i] and current_quality - 1 >= best_quality:
                    # Skip tabu moves unless they'd lead to a new best solution
                    continue
                    
                # Check if removal would break coverage
                can_remove = True
                affected_elements = self.subsets[i]
                
                for elem in affected_elements:
                    still_covered = False
                    for subset_idx in element_to_subsets.get(elem, []):
                        if subset_idx != i and current_solution[subset_idx]:
                            still_covered = True
                            break
                    if not still_covered:
                        can_remove = False
                        break
                
                if can_remove:
                    # This move is legal - can remove without breaking coverage
                    potential_moves.append((i, -1))  # (index, delta)
            
            # If we're stuck, also consider adding subsets for diversification
            if iterations_without_improvement > 100:
                addition_candidates = [i for i, selected in enumerate(current_solution) if not selected]
                random.shuffle(addition_candidates)
                
                # Just take a sample if there are too many candidates
                addition_candidates = addition_candidates[:max(10, self.m // 10)]
                
                for i in addition_candidates:
                    if i in tabu_list and iteration < tabu_list[i]:
                        continue
                    potential_moves.append((i, 1))  # (index, delta)
            
            # --- Make best move ---
            if potential_moves:
                # Prioritize removals (negative delta)
                potential_moves.sort(key=lambda x: x[1])
                best_move_idx, delta = potential_moves[0]
                
                # Apply the move
                current_solution[best_move_idx] = not current_solution[best_move_idx]
                
                # Update coverage
                if delta == -1:  # Removal
                    for elem in self.subsets[best_move_idx]:
                        still_covered = False
                        for subset_idx in element_to_subsets.get(elem, []):
                            if subset_idx != best_move_idx and current_solution[subset_idx]:
                                still_covered = True
                                break
                        if still_covered:
                            current_covered_elements.add(elem)
                        else:
                            current_covered_elements.discard(elem)
                else:  # Addition
                    current_covered_elements.update(self.subsets[best_move_idx])
                
                # Update quality
                current_quality += delta
                
                # Update tabu list
                tabu_list[best_move_idx] = iteration + tabu_tenure
                
                if current_quality < best_quality:
                    best_solution = current_solution.copy()
                    best_quality = current_quality
                    best_covered = current_covered_elements.copy()
                    self._append_to_trace_file(trace_filename, time.time() - start_time, best_quality)
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            # --- Early Exit if Optimal ---
            if best_quality == 1:  # Cannot do better than selecting one subset
                break
        
        # Final redundancy removal
        best_solution = self._remove_redundant_subsets_efficient(best_solution, element_to_subsets)
        best_quality = sum(best_solution)
        
        # Write output
        selected_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        solution_filename = f"{output_prefix}.sol"
        self._write_solution_file(solution_filename, best_quality, selected_indices)
        
        return best_solution, best_quality

    def _remove_redundant_subsets_efficient(self, solution, element_to_subsets):
        """More efficient redundancy removal that uses precomputed element_to_subsets data"""
        modified = True
        result = solution.copy()
        
        while modified:
            modified = False
            # Check each selected subset
            for i in range(self.m):
                if not result[i]:
                    continue
                    
                # Try removing this subset
                result[i] = False
                
                # Check if coverage is maintained
                all_covered = True
                for j in range(self.n):
                    covered = False
                    for subset_idx in element_to_subsets.get(j, []):
                        if result[subset_idx]:
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break
                
                if all_covered:
                    # Can safely remove this subset
                    modified = True
                else:
                    # Need to put it back
                    result[i] = True
        
        return result
    def simulated_annealing(self, cutoff_time: float, random_seed: int, output_prefix: str) -> Tuple[List[bool], int]:
        """
        Simulated Annealing for Minimum Set Cover.
        
        Args:
            cutoff_time: Cutoff time in seconds
            random_seed: Random seed for reproducibility
            output_prefix: Prefix for output files
            
        Returns:
            Tuple of (best solution, best quality)
        """
        # init random num generator
        random.seed(random_seed)
        
        trace_filename = f"{output_prefix}.trace"
        with open(trace_filename, 'w') as _:
            pass 
        
        start_time = time.time()
        
        # init w/ greedy solution
        current_solution = self.generate_initial_solution()
        # Make sure the initial solution covers all elements
        current_solution = self._ensure_coverage(current_solution)
        current_quality, is_covering = self.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_quality = current_quality
        
        # record init sol
        self._append_to_trace_file(trace_filename, 0.0, best_quality)
        
        temperature = 5_000_000.0  # init temp
        cooling_rate = 0.99      # temp cooling rate
        temperature_limit = 0.0  # min temp
        
        # get ALL subset indices for random selection
        all_subset_indices = list(range(self.m))
        
        while time.time() - start_time < cutoff_time and temperature > temperature_limit: # main loop
            elapsed_time = time.time() - start_time 
            if elapsed_time >= cutoff_time: # check time pt.1
                break
            # generate neighbor
            neighbor = current_solution.copy()
            
            # randomly select subset to flip (add or remove)
            flip_idx = random.choice(all_subset_indices)
            in_current_solution = neighbor[flip_idx]
            
            if in_current_solution:
                # If subset currently selected, try to remove it (if it maintains coverage)
                neighbor[flip_idx] = False
                _, is_still_covering = self.evaluate_solution(neighbor)
                
                if not is_still_covering: #reverting if breaks coverage
                    neighbor[flip_idx] = True 
            else:
                neighbor[flip_idx] = True
            
            # calc delta = change in cost
            neighbor_quality, _ = self.evaluate_solution(neighbor)
            delta = neighbor_quality - current_quality
            
            # decision to accept neighbor
            if delta < 0: #(1) always accept better solution,
                current_solution = neighbor
                current_quality = neighbor_quality
                
                if current_quality < best_quality: #(2) update if best solution improves current
                    best_solution = current_solution.copy()
                    best_quality = current_quality
                    elapsed_time = time.time() - start_time
                    self._append_to_trace_file(trace_filename, elapsed_time, best_quality)
            else:  # worse solution, accept with probability
                # calc acceptance probability based on delta & temp
                # mod probability based on subset importance
                subset_size = len(self.subsets[flip_idx])
                total_elements = self.n
                
                # degree calculation
                importance_factor = subset_size / total_elements
                
                if in_current_solution:  # was in solution -> tried removing
                    p = math.exp(-delta * (1 + importance_factor) / temperature)
                else:  # not in solution -> tried adding
                    p = math.exp(-delta * (1 - importance_factor) / temperature)
                
                if random.random() < p:
                    current_solution = neighbor
                    current_quality = neighbor_quality
            
            # update cool down temp
            temperature *= cooling_rate
        
        # remove redundant subsets
        if best_solution:
            self._remove_redundant_subsets(best_solution)
            best_quality, _ = self.evaluate_solution(best_solution)
                
        # Write solution file
        selected_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        if (neighbor_quality == best_quality) and (selected_indices < best_indices):
            best_solution = neighbor.copy()
            best_quality = neighbor_quality
        solution_filename = f"{output_prefix}.sol"
        self._write_solution_file(solution_filename, best_quality, selected_indices)
        
        return best_solution, best_quality

    def _ensure_coverage(self, solution: List[bool]) -> List[bool]:
        """
        Ensure that the solution covers all elements, adding necessary subsets if needed.
        
        Args:
            solution: Current solution (boolean list)
            
        Returns:
            Modified solution that covers all elements
        """
        # Check coverage
        covered = set()
        universal_set = set(range(1, self.n + 1))
        
        for i, selected in enumerate(solution):
            if selected:
                covered.update(self.subsets[i])
        
        # If all elements are covered, return as is
        if covered == universal_set:
            return solution
        
        # Add subsets to cover remaining elements
        remaining = universal_set - covered
        while remaining:
            best_idx = -1
            max_covered = -1
            
            # Find subset that covers most remaining elements
            for i in range(self.m):
                if not solution[i]:
                    covered_count = len(remaining.intersection(self.subsets[i]))
                    if covered_count > max_covered:
                        max_covered = covered_count
                        best_idx = i
            
            if best_idx != -1 and max_covered > 0:
                solution[best_idx] = True
                covered.update(self.subsets[best_idx])
                remaining = universal_set - covered
            else:
                for i in range(self.m):
                    solution[i] = True
                break
        
        return solution
    
    def _write_solution_file(self, filename: str, quality: int, selected_indices: List[int]) -> None:
        """
        Write solution to file.
        
        Args:
            filename: Output filename
            quality: Solution quality
            selected_indices: Indices of selected subsets
        """
        with open(filename, 'w') as f:
            f.write(f"{quality}\n")
            f.write(" ".join(map(str, selected_indices)))
    
    def _append_to_trace_file(self, filename: str, timestamp: float, quality: int) -> None:
        """
        Append improvement to trace file.
        
        Args:
            filename: Trace filename
            timestamp: Time in seconds when improvement was found
            quality: Solution quality
        """
        with open(filename, 'a') as f:
            f.write(f"{timestamp:.2f} {quality}\n")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Minimum Set Cover Solver')
    parser.add_argument('-inst', required=True, help='Instance file path')
    parser.add_argument('-alg', required=True, choices=['LS1', 'LS2'], help='Algorithm to use')
    parser.add_argument('-time', required=True, type=float, help='Cutoff time in seconds')
    parser.add_argument('-seed', required=True, type=int, help='Random seed')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_args()
    problem = MinimumSetCover(args.inst)
    instance_name = os.path.basename(args.inst).split('.')[0]
    
    if args.alg == 'LS1':
        output_prefix = f"solutions/LS1/{instance_name}_LS1_{int(args.time)}_{args.seed}"
    else:  # LS2
        output_prefix = f"solutions/LS2/{instance_name}_LS2_{int(args.time)}_{args.seed}"
    
    if args.alg == 'LS1':
        solution, quality = problem.hill_climbing(args.time, args.seed, output_prefix)
    else:  # LS2
        solution, quality = problem.simulated_annealing(args.time, args.seed, output_prefix)
    
    print(f"Best solution quality: {quality}")
    print(f"Selected subsets: {[i+1 for i, selected in enumerate(solution) if selected]}")

if __name__ == "__main__":
    main()