"""Runs the whole analysis: data -> geometry -> statistics -> energy -> uncertainty -> models -> figures.
Usage:  python run_all.py      (outputs are written to ./out)"""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['run_geometry.py', 'run_stats.py', 'run_energy.py', 'run_facts.py', 'run_uncertainty.py', 'run_models.py', 'make_figures.py']
for s in STEPS:
    t0 = time.time(); print(f'--> {s}', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
    if r.returncode != 0: sys.exit(f'{s} failed')
    print(f'    done ({time.time()-t0:.0f} s)', flush=True)
