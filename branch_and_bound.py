import argparse
import math
import heapq
import time

"""
Branch and Bound Set Cover Solver (BFS-based)

This script implements a Branch and Bound algorithm using Breadth-First Search (BFS) to solve the 
Minimum Set Cover problem. Given a universe of elements {1, ..., n} and a list of m subsets, the goal 
is to find the smallest collection of these subsets that together cover the entire universe.

Key Features:
- Parses the input file in the specified format.
- Uses an initial greedy algorithm to obtain a baseline upper bound on solution size.
- Each state in the BFS search tree keeps track of:
    • The selected subset indices so far
    • The union of covered elements
    • A computed lower bound to prune unpromising branches
- Priority queue (heapq) is used to explore the most promising paths first, with priority calculated 
  as the size of the current partial solution plus the estimated lower bound.
- A cutoff parameter (in seconds) is used to limit the runtime.
"""
def parse_input_file(filename):
    """
    Parse input file in project format.

    Returns:
    - n: Size of the universe
    - m: Number of subsets
    - subsets: List of sets
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    n, m = map(int, lines[0].strip().split())
    subsets = [set(map(int, line.strip().split()[1:])) for line in lines[1:]]
    return n, m, subsets


def branch_and_bound_set_cover_bfs(n, m, subsets_list,cutoff):
    universe = set(range(1, n + 1))
    subsets = [set(s) for s in subsets_list]

    def greedy_solution():
        rem = universe.copy()
        cover = []
        while rem:
            best_idx = None
            best_cov = 0
            for i in range(m):
                cov = len(subsets[i] & rem)
                if cov > best_cov:
                    best_cov = cov
                    best_idx = i
            if best_idx is None or best_cov == 0:
                break
            cover.append(best_idx)
            rem -= subsets[best_idx]
        return cover

    # Compute an initial cover using the greedy algorithm. This is based from the approximation algorithm
    # for the set cover problem. It is not guaranteed to be optimal, but it provides a good starting point.
    init_cover = greedy_solution()
    if universe - set.union(*(subsets[i] for i in init_cover)) == set():
        best_sol_size = len(init_cover)
        best_sol_cover = init_cover.copy()
    else:
        best_sol_size = math.inf
        best_sol_cover = []

    # -- Revised lower bound: 
    def lower_bound(covered):
        remaining = universe - covered
        if not remaining:
            return 0
        max_cov = 0
        for s in subsets:
            cov = len(s & remaining)
            if cov > max_cov:
                max_cov = cov
        if max_cov == 0:
            return math.inf
        return math.ceil(len(remaining) / max_cov)

    # Initialize the frontier as a priority queue.
    # Each node: (priority, current_selection, covered, next_index)
    # Priority = len(current_selection) + lower_bound(covered)
    start_time = time.time()
    frontier = []
    heapq.heappush(frontier, (lower_bound(set()), [], set(), 0))
    
    while frontier:
        elapsed_time = time.time() - start_time
        if elapsed_time >= cutoff:
            #print("Cutoff time reached, returning best solution found so far.")
            break
        priority, current_sel, covered, next_index = heapq.heappop(frontier)
        if priority > best_sol_size:
            continue
        if covered == universe:
            if len(current_sel) < best_sol_size:
                best_sol_size = len(current_sel)
                best_sol_cover = current_sel.copy()
            continue
        # Expand node: consider all subsets with index >= next_index.
        for i in range(next_index, m):
            if not (subsets[i] & (universe - covered)):
                continue
            new_sel = current_sel + [i]
            new_covered = covered | subsets[i]
            new_priority = len(new_sel) + lower_bound(new_covered)
            if new_priority <= best_sol_size:
                heapq.heappush(frontier, (new_priority, new_sel, new_covered, i + 1))
                
    return best_sol_size, best_sol_cover, time.time() - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Branch and Bound Set Cover Solver (BFS)")
    parser.add_argument("--cutoff", type=int, default=600, help="Cutoff time in seconds (default: 600)")
    parser.add_argument("--filename", type=str, default="test1", help="file name (default: test1)")
    args = parser.parse_args()
    n, m, subsets_list = parse_input_file(f'data/{args.filename}.in')
    # for i in range(1, 19):
    #     n, m, subsets_list = parse_input_file(f'data/small{i}.in')
    best_size, best_cover,timeTaken = branch_and_bound_set_cover_bfs(n, m, subsets_list,600)
    print("Best cover size:", best_size)
    print("Indices of subsets in the cover (0-indexed):", best_cover)
    print("Time taken:", timeTaken)
