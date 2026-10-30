"""Adım 4: Monte Carlo belirsizlik yayılımı (K tipi termokupl ±0,75 °C, pyranometre ±1,001 W/m²; düzgün dağılım)."""
import json, numpy as np, pandas as pd
from config import *; from data import *
D = pd.read_pickle(f'{OUT}/D_en.pkl').reset_index(drop=True); rng = np.random.default_rng(42); N = 3000
sensors = ['Tin', 'Tfo', 'Tpo', 'Ta', 'Tt', 'G']; day = D.day.values
Ith = D.I085.values; base = {s: D[s].values.copy() for s in sensors}
def metrics(T):
    Tmf = (T['Tin']+T['Tfo'])/2; Tmp = (T['Tin']+T['Tpo'])/2; dTf = T['Tfo']-T['Tin']; dTp = T['Tpo']-T['Tin']; Gv = T['G']
    QfA = A_F*(Gv*ETA_F-UL*(Tmf-T['Ta'])); QpA = A_P*(Gv*ETA_P-UL*(Tmp-T['Ta'])); QpB = A_P*Gv*ETA_F*Ith-UL*A_F*(Tmp-T['Ta'])
    mf = QfA.mean()/(cp(Tmf).mean()*1000*dTf.mean())*1000; mpA = QpA.mean()/(cp(Tmp).mean()*1000*dTp.mean())*1000; mpB = QpB.mean()/(cp(Tmp).mean()*1000*dTp.mean())*1000
    tot = 0
    for d in (1, 2, 3):
        m = day == d; Tt = T['Tt'][m]; tot += M_TANK*cp(Tt.mean())*1000*(Tt[-1]-Tt[0])/((m.sum()-1)*600.0)
    return dict(dTf=dTf.mean(), dTp=dTp.mean(), ratio=dTp.mean()/dTf.mean(), mf=mf, mpA=mpA, mpB=mpB, clA=(QfA+QpA).mean()/(tot/3), clB=(QfA+QpB).mean()/(tot/3),
                etafA=(QfA/(A_F*Gv)).mean(), etapA=(QpA/(A_P*Gv)).mean(), etapB=(QpB/(A_P*Gv)).mean())
b0 = metrics(base); res = {'base': b0}
for case in ('random', 'systematic', 'both'):
    out = {k: [] for k in b0}
    for _ in range(N):
        T = {}
        for s in sensors:
            b_ = U_PYR if s == 'G' else U_TC; e = np.zeros(len(D))
            if case in ('random', 'both'): e += rng.uniform(-b_, b_, len(D))
            if case in ('systematic', 'both'): e += rng.uniform(-b_, b_)
            T[s] = base[s]+e
        for k, v in metrics(T).items(): out[k].append(v)
    res[case] = {k: [float(np.mean(v)), float(np.std(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
# tek okuma ΔT belirsizliği ve gözlenen dalgalanma
res['u_single'] = float(U_TC/np.sqrt(3)*np.sqrt(2)); res['worst_dT'] = 2*U_TC
res['detr_std_f'] = float(D.groupby('day').dTf.apply(lambda s: (s-s.rolling(7, center=True, min_periods=1).mean()).std()).mean())
res['detr_std_p'] = float(D.groupby('day').dTp.apply(lambda s: (s-s.rolling(7, center=True, min_periods=1).mean()).std()).mean())
json.dump(res, open(f'{OUT}/uncertainty.json', 'w'))
for k in ('base',): print({a: round(b, 3) for a, b in res[k].items()})
for c in ('random', 'systematic', 'both'): print(c, {a: [round(x, 3) for x in (b[2], b[3])] for a, b in res[c].items()})
print(res['u_single'], res['detr_std_f'], res['detr_std_p'])
