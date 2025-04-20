#!/usr/bin/env python3
import numpy as np
"""
Implementation of two Local Search algorithms for Minimum Set Cover problem:
LS1. Hill Climbing w/ Random Restarts
LS2. Simulated Annealing

run with:
    python local_search.py -inst data/<test>.in -alg [LS1|LS2] -time 600 -seed <random seed>
"""

import argparse
import random
import time
import os
import sys
import math
from typing import List, Set, Dict, Tuple, Optional
import itertools

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

    def generate_random_solution(self) -> List[bool]:
        """
        Construct a random initial solution, then enforce coverage.
        """
        # random pick each subset with 50% probability
        solution = [random.random() < 0.5 for _ in range(self.m)]
        # make sure it actually covers everything
        return self._ensure_coverage(solution)

    
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
        
        sol_dir = os.path.dirname(output_prefix)
        if sol_dir:
            os.makedirs(sol_dir, exist_ok=True)

        trace_filename = f"{output_prefix}.trace"
        with open(trace_filename, 'w') as _:
            pass
        
        start_time = time.time()
        
        # init solution (greedy + ensure coverage)
        current_solution = self.generate_initial_solution()
        #comment line below back in if you want to use random start rather than greedy
        #current_solution = self.generate_random_solution()  
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
        
        max_iterations_without_improvement = min(5000, max(1000, self.n * 10))
        restart_probability = 0.2 #test this value out
        iterations_without_improvement = 0
        
        tabu_list = {}  # idx -> tabu expiration iter
        tabu_tenure = min(20, max(5, self.m // 100)) # gets bigger w/ incr. number of subsets
        # tabu_tenure = 10  # duration idx stays in list
        iteration = 0
        
        # cache subset sizes 
        subset_sizes = [len(subset) for subset in self.subsets]
        
        while time.time() - start_time < cutoff_time:
            iteration += 1
            elapsed_time = time.time() - start_time
            elapsed_fraction = (time.time() - start_time) / cutoff_time
            
            if elapsed_time >= cutoff_time:
                break
            current_tabu_tenure = int(tabu_tenure * (1 + elapsed_fraction))  # tabu incr. tenure over time
            current_restart_probability = restart_probability * (1 + elapsed_fraction)

            # RANDOM RESTARTS
            if iterations_without_improvement >= max_iterations_without_improvement:
                if random.random() < current_restart_probability:
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

            # 1-opt single removals
            removal_candidates = [i for i, selected in enumerate(current_solution) if selected]
            random.shuffle(removal_candidates) # shuffle to randomize moves w/ equal quality
            
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
                    potential_moves.append(([i], -1)) # (index, delta)


            #2-opt moves, only if cannot find good 1-opt moves
            if not potential_moves or iterations_without_improvement > 50:
                for idx1, idx2 in itertools.combinations(removal_candidates, 2): #removing two subsets simultaneously
                    # if either is tabu (unless it would be best solution), SKIP!
                    if ((idx1 in tabu_list and iteration < tabu_list[idx1]) or 
                        (idx2 in tabu_list and iteration < tabu_list[idx2])) and current_quality - 2 >= best_quality:
                        continue
                    # checker for simultaneous removal coverage is maintained
                    can_remove_both = True
                    combined_elements = set(self.subsets[idx1]).union(self.subsets[idx2])
                    for elem in combined_elements:
                        still_covered = False
                        for subset_idx in element_to_subsets.get(elem, []):
                            if subset_idx != idx1 and subset_idx != idx2 and current_solution[subset_idx]:
                                still_covered = True
                                break
                        if not still_covered:
                            can_remove_both = False
                            break
                    if can_remove_both:
                        potential_moves.append(([idx1, idx2], -2))  # remove both
                
            # adding subsets or swap
            if iterations_without_improvement > 100:
                addition_candidates = [i for i, selected in enumerate(current_solution) if not selected]
                random.shuffle(addition_candidates)
                addition_candidates = addition_candidates[:max(10, self.m // 10)] # max number of candidates (for better runtime)
                
                # 1-opt
                for i in addition_candidates:
                    if i in tabu_list and iteration < tabu_list[i]:
                        continue
                    potential_moves.append(([i], 1))
                
                # swap 2-opt: remove one & add one
                if iterations_without_improvement > 200:
                    for remove_idx in removal_candidates[:10]:  # max first 10
                        for add_idx in addition_candidates[:10]:
                            if ((remove_idx in tabu_list and iteration < tabu_list[remove_idx]) or 
                                (add_idx in tabu_list and iteration < tabu_list[add_idx])):
                                continue
                            # checker if swap leaves coverage still maintained
                            new_solution = current_solution.copy()
                            new_solution[remove_idx] = False
                            new_solution[add_idx] = True
                            # eval coverage after swap
                            covered = set()
                            for j, selected in enumerate(new_solution):
                                if selected:
                                    covered.update(self.subsets[j])
                            if len(covered) == self.n:
                                potential_moves.append(([remove_idx, add_idx], 0))
            
            # PERFORM BEST MOVES
            if potential_moves:
                potential_moves.sort(key=lambda x: x[1]) # prioritize removals (negative delta)
                best_move_indices, delta = potential_moves[0]
                
                for idx in best_move_indices: #applying 1opt or 2opt
                    new_value = not current_solution[idx]
                    current_solution[idx] = new_value

                    #tracking coverage 
                    if new_value:  # adding subset
                        current_covered_elements.update(self.subsets[idx])
                    else:  # removing subset, rechecking coverage
                        for elem in self.subsets[idx]:
                            still_covered = False
                            for subset_idx in element_to_subsets.get(elem, []):
                                if subset_idx != idx and current_solution[subset_idx]:
                                    still_covered = True
                                    break
                            if still_covered:
                                current_covered_elements.add(elem)
                            else:
                                current_covered_elements.discard(elem)
                    
                    tabu_list[idx] = iteration + current_tabu_tenure
                # updates
                current_quality += delta
                
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
        #comment back in if we need random
        #current_solution = self.generate_random_solution()
        current_solution = self._ensure_coverage(current_solution)
        current_quality, is_covering = self.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_quality = current_quality
        
        # params for RESTARTS, save best 5 & too many iterations w/o improvements
        elite_solutions = [(best_solution.copy(), best_quality)]
        max_elite_size = 3
        iterations_since_improvement = 0
        stagnation_limit = 10

        # params for TABU
        tabu_list = {}  
        tabu_tenure = 15  # longevity of moves in tabu list
        current_iteration = 0  # add this to track current iteration
        tabu_cleanup_frequency = 100 #shorten run time

        # track selected indices for best solution
        best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
        self._append_to_trace_file(trace_filename, 0.0, best_quality)
        
        temperature = 5_000_000.0
        initial_temperature = temperature
        cooling_rate = 0.99
        temperature_limit = 0.001
        
        all_subset_indices = list(range(self.m))
        
        p = 0.3 #probability p
        while time.time() - start_time < cutoff_time and temperature > temperature_limit:
            current_iteration += 1 
            
            elapsed_time = time.time() - start_time 
            if elapsed_time >= cutoff_time:
                break
                
            if current_iteration % tabu_cleanup_frequency == 0:
                tabu_list = {idx: expiry for idx, expiry in tabu_list.items() if expiry > current_iteration}
            #RESTART LOGIC
            if iterations_since_improvement > stagnation_limit:
                # restarting w/ best found solutions
                restart_idx = random.randint(0, len(elite_solutions) - 1)
                current_solution = elite_solutions[restart_idx][0].copy()
                current_quality = elite_solutions[restart_idx][1]
                temperature = initial_temperature * 0.5
                iterations_since_improvement = 0
                tabu_list.clear()
            
            if random.random() < p:
                #random move & tabu check
                non_tabu_indices = [idx for idx in all_subset_indices if current_iteration > tabu_list.get(idx, 0)]
                
                # if all tabu, select randomly
                if not non_tabu_indices:
                    non_tabu_indices = all_subset_indices

                flip_idx = random.choice(non_tabu_indices)

                neighbor_is_selected = not current_solution[flip_idx]
                if not neighbor_is_selected:
                    # Create temporary neighbor only when needed to check coverage
                    neighbor = current_solution.copy()
                    neighbor[flip_idx] = False
                    _, is_still_covering = self.evaluate_solution(neighbor)
                    if not is_still_covering:
                        continue
                
                # evaluate neighbor 
                neighbor = current_solution.copy()
                neighbor[flip_idx] = neighbor_is_selected
                neighbor_quality, _ = self.evaluate_solution(neighbor)
                delta = neighbor_quality - current_quality
                is_tabu = current_iteration <= tabu_list.get(flip_idx, 0)
                is_aspiration = neighbor_quality < best_quality
            
                if (not is_tabu) or is_aspiration:
                    if delta < 0:
                        current_solution = neighbor
                        current_quality = neighbor_quality
                        tabu_list[flip_idx] = current_iteration + tabu_tenure
                        iterations_since_improvement = 0

                        if current_quality < best_quality:
                            best_solution = current_solution.copy()
                            best_quality = current_quality
                            
                            elite_solutions.append((best_solution.copy(), best_quality))
                            elite_solutions.sort(key=lambda x: x[1])
                            if len(elite_solutions) > max_elite_size:
                                elite_solutions = elite_solutions[:max_elite_size]
                            
                            best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
                            self._append_to_trace_file(trace_filename, elapsed_time, best_quality)
                    else:
                        subset_size = len(self.subsets[flip_idx])
                        importance_factor = subset_size / self.n
                        
                        accept_prob = math.exp(-delta * ((1 + importance_factor) if current_solution[flip_idx] else (1 - importance_factor)) / temperature)
                        
                        if random.random() < accept_prob:
                            current_solution = neighbor
                            current_quality = neighbor_quality
                            tabu_list[flip_idx] = current_iteration + tabu_tenure
                        iterations_since_improvement += 1
            else: # best neighbor search
                sampling_size = min(40, self.m)
                sampled_indices = random.sample(all_subset_indices, sampling_size) if sampling_size < self.m else all_subset_indices
                best_neighbor_quality = current_quality
                best_flip_idx = -1
                best_neighbor = None

                for flip_idx in sampled_indices:
                    is_tabu = current_iteration <= tabu_list.get(flip_idx, 0)
                    if is_tabu and flip_idx != -1:  # already have candidate
                        continue
                    temp_neighbor = current_solution.copy()
                    temp_neighbor[flip_idx] = not temp_neighbor[flip_idx]
                    
                    if temp_neighbor[flip_idx] == False:
                        _, is_still_covering = self.evaluate_solution(temp_neighbor)
                        if not is_still_covering:
                            continue
                    
                    temp_quality, _ = self.evaluate_solution(temp_neighbor)
                    if temp_quality < best_neighbor_quality:
                        if best_neighbor is None:
                            best_neighbor = temp_neighbor
                        else:
                            best_neighbor = temp_neighbor
                        best_neighbor_quality = temp_quality
                        best_flip_idx = flip_idx
                        if best_neighbor_quality < current_quality * 0.9:  # 10% improvement
                            break
                
                if best_flip_idx != -1 and best_neighbor_quality < current_quality:
                    current_solution = best_neighbor
                    current_quality = best_neighbor_quality
                    iterations_since_improvement = 0
                    tabu_list[best_flip_idx] = current_iteration + tabu_tenure
                    
                    if current_quality < best_quality:
                        best_solution = current_solution.copy()
                        best_quality = current_quality
                        
                        elite_solutions.append((best_solution.copy(), best_quality))
                        elite_solutions.sort(key=lambda x: x[1])
                        if len(elite_solutions) > max_elite_size:
                            elite_solutions = elite_solutions[:max_elite_size]
                        
                        best_indices = [i + 1 for i, selected in enumerate(best_solution) if selected]
                        self._append_to_trace_file(trace_filename, elapsed_time, best_quality)
                else:
                    if random.random() < 0.1:
                        flip_idx = random.choice(all_subset_indices)
                        neighbor = current_solution.copy()
                        neighbor[flip_idx] = not neighbor[flip_idx]
                        
                        if neighbor[flip_idx] == False:
                            _, is_still_covering = self.evaluate_solution(neighbor)
                            if not is_still_covering:
                                continue
                                
                        current_solution = neighbor
                        current_quality, _ = self.evaluate_solution(current_solution)
            
            temperature *= cooling_rate
        
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
        # check coverage
        covered = set()
        universal_set = set(range(1, self.n + 1))
        
        for i, selected in enumerate(solution):
            if selected:
                covered.update(self.subsets[i])
        
        # if all elements covered, return
        if covered == universal_set:
            return solution
        
        # add subsets to cover remaining elements
        remaining = universal_set - covered
        while remaining:
            best_idx = -1
            max_covered = -1
            
            # find subset that covers most remaining elements
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
            f.write(f"{timestamp:.4f} {quality}\n")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Minimum Set Cover Solver')
    parser.add_argument('-inst', required=True, help='Instance file path')
    parser.add_argument('-alg', required=True, choices=['LS1', 'LS2'], help='Algorithm to use')
    parser.add_argument('-time', required=True, type=float, help='Cutoff time in seconds')
    parser.add_argument('-seed', required=True, type=int, help='Random seed')
    parser.add_argument('-sol', required=True, help='Output solution prefix')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # ensure output directory exists (skip if no directory component)
    sol_dir = os.path.dirname(args.sol)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)

    
    problem = MinimumSetCover(args.inst)
    
    if args.alg == 'LS1':
        solution, quality = problem.hill_climbing(args.time, args.seed, args.sol)
    else:
        solution, quality = problem.simulated_annealing(args.time, args.seed, args.sol)
    
    print(f"Best solution quality: {quality}")
    selected_subsets = [i + 1 for i, selected in enumerate(solution) if selected]
    print(f"Selected subsets: {selected_subsets}")

if __name__ == "__main__":
    main()