#!/usr/bin/env python3
import argparse, os, subprocess, sys

def main():
    p = argparse.ArgumentParser(
        description="Unified entry point for BnB, Approx, LS1, LS2 solvers"
    )
    p.add_argument('-inst', required=True,
                   help='instance filename (e.g. small1.in)')
    p.add_argument('-alg',  required=True,
                   choices=['BnB','Approx','LS1','LS2'],
                   help='which algorithm to run')
    p.add_argument('-time', required=True, type=float,
                   help='cutoff time (seconds)')
    p.add_argument('-seed', type=int,
                   help='random seed (required for LS1, LS2)')
    args = p.parse_args()

    inst_name = args.inst
    if not inst_name.endswith('.in'):
        p.error("–inst must end in .in")
    inst_path = os.path.join('data', inst_name)
    if not os.path.exists(inst_path):
        sys.exit(f"Error: cannot find {inst_path}")

    base = os.path.splitext(inst_name)[0]
    cutoff = args.time
    cutoff_str = str(int(cutoff)) if cutoff.is_integer() else str(cutoff)
    seed = args.seed

    if args.alg == 'BnB':
        # Branch & Bound ➔ branch_and_bound.py handles seed‐free
        cmd = [sys.executable, 'branch_and_bound.py',
               '--filename', base,
               '--cutoff',   cutoff_str]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            sys.stderr.write(proc.stderr); sys.exit(proc.returncode)

        out = proc.stdout.strip().splitlines()
        best_size = int(out[0].split(':',1)[1].strip())
        raw_cover = out[1].split(':',1)[1].strip()
        cover0 = eval(raw_cover) if raw_cover.startswith('[') else []
        sol_indices = sorted(i+1 for i in cover0)
        timeTaken = float(out[2].split(':',1)[1].strip())

        prefix = f"{base}_BnB_{cutoff_str}"
        with open(prefix + '.sol','w') as f:
            f.write(f"{best_size}\n")
            f.write(" ".join(map(str, sol_indices))+"\n")
        with open(prefix + '.trace','w') as f:
            f.write(f"{timeTaken:.2f} {best_size}\n")

        print(f"Wrote {prefix}.sol and {prefix}.trace")


    elif args.alg == 'Approx':
        # Approx is fully deterministic — no seed needed
        prefix = f"{base}_Approx_{cutoff_str}"
        # clean old log
        if os.path.exists('times.txt'):
            os.remove('times.txt')

        cmd = [sys.executable, 'approxAlgo.py',
               inst_path,
               prefix + '.sol',
               cutoff_str]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            sys.stderr.write(proc.stderr); sys.exit(proc.returncode)

        with open(prefix + '.sol') as f:
            quality = int(f.readline().strip())

        tval = 0.0
        if os.path.exists('times.txt'):
            for L in open('times.txt'):
                fn, t = L.split()
                if fn == inst_path:
                    tval = float(t)
        with open(prefix + '.trace','w') as f:
            f.write(f"{tval:.2f} {quality}\n")
        os.remove('times.txt')

        print(f"Wrote {prefix}.sol and {prefix}.trace")


    else:  # LS1 or LS2
        if seed is None:
            p.error("–seed required for LS1/LS2")
        prefix = f"{base}_{args.alg}_{cutoff_str}_{seed}"
        cmd = [sys.executable, 'local_search.py',
               '-inst', inst_path,
               '-alg',  args.alg,
               '-time', str(cutoff),
               '-seed', str(seed),
               '-sol',  prefix]
        proc = subprocess.run(cmd)
        if proc.returncode:
            sys.exit(proc.returncode)
        print(f"Wrote {prefix}.sol and {prefix}.trace")


if __name__ == '__main__':
    main()
