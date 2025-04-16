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
        
        # init solution (greedy + ensure coverage)
        current_solution = self.generate_initial_solution()
        current_solution = self._ensure_coverage(current_solution)
        
        # precompute ele coverage info
        element_to_subsets = {}
        for i, subset in enumerate(self.subsets):
            for element in subset:
                if element not in element_to_subsets:
                    element_to_subsets[element] = []
                element_to_subsets[element].append(i)
        
        # track curr coverage state
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
        restart_probability = 0.05 #test this value out
        iterations_without_improvement = 0
        
        # our tabu list
        tabu_list = {}  # idx -> tabu expiration iter
        tabu_tenure = 10  # duration idx stays in list
        iteration = 0
        
        # cache subset sizes 
        subset_sizes = [len(subset) for subset in self.subsets]
        
        while time.time() - start_time < cutoff_time:
            iteration += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= cutoff_time:
                break
            
            # RANDOM RESTARTS
            if iterations_without_improvement >= max_iterations_without_improvement:
                if random.random() < 0.95:
                    # start w/ best solution & perturb it
                    current_solution = best_solution.copy()
                    current_covered_elements = best_covered.copy()
                    
                    # flip rando bits
                    flip_count = max(1, self.m // 10)
                    indices_to_flip = random.sample(range(self.m), flip_count)
                    
                    for i in indices_to_flip:
                        if current_solution[i]:
                            # try to remove & check if coverage maintained
                            current_solution[i] = False
                            
                            # update coverage
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
                                # if coverage broken, restore it
                                current_solution[i] = True
                            else: # update coverage status for affected
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
                        else: #adding subset
                            current_solution[i] = True
                            current_covered_elements.update(self.subsets[i])
                else:
                    break
                
                current_quality = sum(current_solution)
                iterations_without_improvement = 0
                tabu_list = {}  # clear list when restarting
            
            # NEIGHBOR SELECTION
            potential_moves = []
            
            # prioritze removing subsets (reducing solution size)
            removal_candidates = [i for i, selected in enumerate(current_solution) if selected]
            
            # shuffle to randomize equal-quality moves
            random.shuffle(removal_candidates)
            
            # eval potential removal
            for i in removal_candidates:
                if i in tabu_list and iteration < tabu_list[i] and current_quality - 1 >= best_quality:
                    # skip tabu moves unless better new best solution
                    continue
                    
                # check if removal breaks coverage
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
                
                if can_remove: # w/o breaking coverage
                    potential_moves.append((i, -1)) # (index, delta)
            
            # if stuck, consider adding subsets
            if iterations_without_improvement > 100:
                addition_candidates = [i for i, selected in enumerate(current_solution) if not selected]
                random.shuffle(addition_candidates)
                
                # Just take a sample if there are too many candidates
                addition_candidates = addition_candidates[:max(10, self.m // 10)]
                
                for i in addition_candidates:
                    if i in tabu_list and iteration < tabu_list[i]:
                        continue
                    potential_moves.append((i, 1))  # (index, delta)
            
            # PERFORM BEST MOVES
            if potential_moves:
                potential_moves.sort(key=lambda x: x[1]) # prioritize removals (negative delta)
                best_move_idx, delta = potential_moves[0]
                
                current_solution[best_move_idx] = not current_solution[best_move_idx] #apply removal
                
                # update coverage
                if delta == -1:  # removing
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
                else:  # adding
                    current_covered_elements.update(self.subsets[best_move_idx])
                
                # updates
                current_quality += delta
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
            
            if best_quality == 1:
                break
        
        # redundancy removal
        best_solution = self._remove_redundant_subsets(best_solution)
        best_quality = sum(best_solution)
        
        # Write output
        selected_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        solution_filename = f"{output_prefix}.sol"
        self._write_solution_file(solution_filename, best_quality, selected_indices)
        
        return best_solution, best_quality

    def _remove_redundant_subsets(self, solution: List[bool]) -> List[bool]:
        """
        Remove redundant subsets from a solution while maintaining full coverage.
        Works for both Hill Climbing and Simulated Annealing.
        
        Args:
            solution: Current solution (list of booleans indicating selected subsets)
        
        Returns:
            A new solution with redundant subsets removed
        """
        new_solution = solution.copy() # convert to list
        improved = True
        
        while improved:
            improved = False
            selected_indices = [i for i, selected in enumerate(new_solution) if selected] # indices of currently selected subsets
            
            for i in selected_indices:
                # temp remove this subset
                new_solution[i] = False
                
                # if coverage is still maintained, else put subset back 
                covered = set()
                for j, selected in enumerate(new_solution):
                    if selected:
                        covered.update(self.subsets[j])
                
                if len(covered) != self.n:
                    new_solution[i] = True
                else:
                    improved = True  # Found a redundant subset
        
        return new_solution
    
    def simulated_annealing(self, cutoff_time: float, random_seed: int, output_prefix: str) -> Tuple[List[bool], int]:
        random.seed(random_seed)
        
        trace_filename = f"{output_prefix}.trace"
        with open(trace_filename, 'w') as _:
            pass 
        
        start_time = time.time()
        
        # init solution (greedy + ensure coverage)
        current_solution = self.generate_initial_solution()
        current_solution = self._ensure_coverage(current_solution)
        current_quality, is_covering = self.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_quality = current_quality
        
        # track selected indices for best solution
        best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        
        self._append_to_trace_file(trace_filename, 0.0, best_quality)
        
        temperature = 5_000_000.0  # init
        cooling_rate = 0.99        # cooling rate
        temperature_limit = 0.001  # min temp
        
        all_subset_indices = list(range(self.m))
        
        while time.time() - start_time < cutoff_time and temperature > temperature_limit:
            elapsed_time = time.time() - start_time 
            if elapsed_time >= cutoff_time:
                break
            
            # Generate neighbor
            neighbor = current_solution.copy()
            flip_idx = random.choice(all_subset_indices)
            in_current_solution = neighbor[flip_idx]
            
            if in_current_solution:
                # Try removing the subset (if coverage is maintained)
                neighbor[flip_idx] = False
                _, is_still_covering = self.evaluate_solution(neighbor)
                
                if not is_still_covering:
                    neighbor[flip_idx] = True  # Revert if coverage breaks
            else:
                neighbor[flip_idx] = True  # add subset
            
            # evaluate neighbor
            neighbor_quality, _ = self.evaluate_solution(neighbor)
            delta = neighbor_quality - current_quality
            
            # decide where to accept neighber
            if delta < 0:  # always accept better solutions
                current_solution = neighbor
                current_quality = neighbor_quality
                
                if current_quality < best_quality:
                    best_solution = current_solution.copy()
                    best_quality = current_quality
                    best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
                    self._append_to_trace_file(trace_filename, elapsed_time, best_quality)
            else:  # accept worse solutions w/ prob
                subset_size = len(self.subsets[flip_idx])
                importance_factor = subset_size / self.n
                
                if in_current_solution:  # Tried removing
                    p = math.exp(-delta * (1 + importance_factor) / temperature)
                else:  # Tried adding
                    p = math.exp(-delta * (1 - importance_factor) / temperature)
                
                if random.random() < p:
                    current_solution = neighbor
                    current_quality = neighbor_quality
            
            temperature *= cooling_rate # cool down temp
        
        # remove redundant subsets
        best_solution = self._remove_redundant_subsets(best_solution)
        best_quality = sum(best_solution)
        best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        
        solution_filename = f"{output_prefix}.sol"
        self._write_solution_file(solution_filename, best_quality, best_indices)
        
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
    
    # for verification files!!
    parser.add_argument('-sol', help='Solution output prefix', default='solutions/LS1/large')

    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_args()
    solution_filename = f"{args.sol}.sol"
    trace_filename = f"{args.sol}.trace"
    problem = MinimumSetCover(args.inst)
    instance_name = os.path.basename(args.inst).split('.')[0]
    
    if args.alg == 'LS1':
        output_prefix = f"solutions/LS1/large/{instance_name}_LS1_{int(args.time)}_{args.seed}"
    else:  # LS2
        output_prefix = f"solutions/LS2/large/{instance_name}_LS2_{int(args.time)}_{args.seed}"
    
    if args.alg == 'LS1':
        solution, quality = problem.hill_climbing(args.time, args.seed, output_prefix)
    else:  # LS2
        solution, quality = problem.simulated_annealing(args.time, args.seed, output_prefix)
    
    print(f"Best solution quality: {quality}")
    print(f"Selected subsets: {[i+1 for i, selected in enumerate(solution) if selected]}")

if __name__ == "__main__":
    main()