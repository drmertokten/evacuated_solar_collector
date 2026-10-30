"""Tüm analizi baştan sona çalıştırır: veri -> geometri -> istatistik -> enerji -> belirsizlik -> modeller -> şekiller.
Kullanım:  python run_all.py      (çıktılar ../out klasörüne yazılır)"""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['run_geometry.py', 'run_stats.py', 'run_energy.py', 'run_facts.py', 'run_uncertainty.py', 'run_models.py', 'make_figures.py']
for s in STEPS:
    t0 = time.time(); print(f'--> {s}', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
    if r.returncode != 0: sys.exit(f'{s} başarısız')
    print(f'    tamam ({time.time()-t0:.0f} s)', flush=True)
