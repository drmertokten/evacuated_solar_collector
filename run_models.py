"""Adım 5: LODO (gün-dışarıda-bırakma) ile ΔT ve T_çıkış tahmini: ML, gri-kutu ve hibrit karşılaştırması."""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from config import *; from data import *; from models import *
D = pd.read_pickle(f'{OUT}/D_en.pkl').reset_index(drop=True); geo = json.load(open(f'{OUT}/geometry.json')); angs = np.array(geo['ray']['angs']); I = np.array(geo['ray']['rho085'])
D['thT2'] = D.thT**2
ray_fun = lambda g: np.interp(np.abs(g.thT.values), angs, I)/I[0]
FE = ['G', 'Ta', 'Tin', 'clock']; FEg = FE+['thT', 'thT2']
kinds = ['qs', 'lag', 'time', 'geo', 'geo_lag', 'ray_lag']
res = {}; preds = {}; fin = {}
for col, A, eta0 in (('f', A_F, ETA_F), ('p', A_P, ETA_P)):
    y = D['dT'+col].values; P = {k: np.zeros(len(D)) for k in ['Train mean', 'ML Ridge', 'ML Ridge+thT', 'ML GBR', 'ML RF', 'Phys qs', 'Phys lag', 'Phys time', 'Phys geo', 'Phys geo+lag', 'Phys ray+lag', 'Hyb Ridge', 'Hyb GBR', 'Hyb RF']}
    ths = []
    for hd in (1, 2, 3):
        tr = D[D.day != hd].reset_index(drop=True); te = D[D.day == hd].reset_index(drop=True); m = (D.day == hd).values
        P['Train mean'][m] = tr['dT'+col].mean()
        mm = ml_models()
        for nm in ('Ridge', 'GBR', 'RF'): mm[nm].fit(tr[FE], tr['dT'+col]); P['ML '+nm][m] = mm[nm].predict(te[FE])
        r_ = make_pipeline(StandardScaler(), Ridge(1.0)); r_.fit(tr[FEg], tr['dT'+col]); P['ML Ridge+thT'][m] = r_.predict(te[FEg])
        for kind, nm in (('qs', 'Phys qs'), ('lag', 'Phys lag'), ('time', 'Phys time'), ('geo', 'Phys geo'), ('geo_lag', 'Phys geo+lag'), ('ray_lag', 'Phys ray+lag')):
            f = fit_model(tr, col, A, eta0, UL, kind, ray_fun); P[nm][m] = simulate(te, A, eta0, UL, f['K'], f['tau'], f['mfun'])
            if kind == 'geo_lag':
                ths.append(f['a']); trp = simulate(tr, A, eta0, UL, f['K'], f['tau'], f['mfun']); tep = P[nm][m]
        Xtr = tr[FE].copy(); Xtr['phys'] = trp; Xte = te[FE].copy(); Xte['phys'] = tep
        for nm in ('Ridge', 'GBR', 'RF'):
            mm = ml_models(); mm[nm].fit(Xtr, tr['dT'+col]-trp); P['Hyb '+nm][m] = tep+mm[nm].predict(Xte)
    res[col] = {k: {**metrics(y, v), 'by_day': [float(np.sqrt(np.mean((y[(D.day == d_).values]-v[(D.day == d_).values])**2))) for d_ in (1, 2, 3)]} for k, v in P.items()}
    preds[col] = {k: v.tolist() for k, v in P.items()}; res[col+'_tha'] = ths
    f = fit_model(D, col, A, eta0, UL, 'geo_lag'); fin[col] = dict(K=f['K'], tau=f['tau'], tha=f['a'])
    f2 = fit_model(D, col, A, eta0, UL, 'time'); fin[col+'_time'] = dict(K=f2['K'], tau=f2['tau'], b=f2['a'], t0=f2['b'])
    print(col, 'theta_a folds', np.round(ths, 1))
    for k, v in res[col].items(): print(f"  {k:14s} RMSE {v['rmse']:.2f} MAE {v['mae']:.2f} R2 {v['r2']:.2f} days {[round(x,2) for x in v['by_day']]}")
# --- T_çıkış kıyaslaması (iki kolektör üst üste; yapılandırma değişkeni) ---
S = pd.concat([D.assign(cfg=0, Tout=D.Tfo), D.assign(cfg=1, Tout=D.Tpo)]).reset_index(drop=True)
feat = ['G', 'Ta', 'Tt', 'Tin', 'cfg']; TOUT = {}
yy = S.Tout.values
P = {'Naive (Tout=Tin)': S.Tin.values.copy()}
for nm in ml_models(): P[nm] = np.zeros(len(S))
for hd in (1, 2, 3):
    tr = S[S.day != hd]; m = (S.day == hd).values
    for nm, mdl in ml_models().items(): mdl.fit(tr[feat], tr.Tout); P[nm][m] = mdl.predict(S.loc[m, feat])
P['Gray-box (m(θT)+lag)'] = np.concatenate([D.Tin.values+np.array(preds['f']['Phys geo+lag']), D.Tin.values+np.array(preds['p']['Phys geo+lag'])])
TOUT = {k: metrics(yy, v) for k, v in P.items()}
for k, v in TOUT.items(): print(f"  Tout {k:24s} R2 {v['r2']:.3f} MAE {v['mae']:.2f} RMSE {v['rmse']:.2f}")
json.dump(dict(dT=res, tout=TOUT, fin=fin, preds=preds), open(f'{OUT}/models.json', 'w'))
S.to_pickle(f'{OUT}/S_stack.pkl'); pd.to_pickle(P, f'{OUT}/tout_preds.pkl')
