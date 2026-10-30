"""Şekil açıklamalarında kullanılan günlük gerçekler (tepe değerleri, saatler)."""
import json, numpy as np, pandas as pd
from config import *
D = pd.read_pickle(f'{OUT}/D_en.pkl'); fx = {}
for d, g in D.groupby('day'):
    i = g.dTp.idxmax(); j = g.dTf.idxmax()
    fx[int(d)] = dict(pk_p=float(g.dTp.max()), pk_p_t=float(g.clock[i]), p10=float(g.dTp.iloc[0]), p15=float(g.dTp.iloc[-1]), pk_f=float(g.dTf.max()), pk_f_t=float(g.clock[j]),
        f10=float(g.dTf.iloc[0]), f15=float(g.dTf.iloc[-1]), G10=float(g.G.iloc[0]), G1130=float(g.G[g.clock == 11.5].iloc[0]), G15=float(g.G.iloc[-1]),
        Tt0=float(g.Tt.iloc[0]), Tt1=float(g.Tt.iloc[-1]), tank_gain=float(g.Tt.iloc[-1]-g.Tt.iloc[0]))
fx['rows_pos'] = int(((D.dTp-D.dTf) > 0).sum()); fx['dTf_min'] = float(D.dTf.min()); fx['dTf_max'] = float(D.dTf.max()); fx['dTp_min'] = float(D.dTp.min()); fx['dTp_max'] = float(D.dTp.max())
fx['Tpo_minus_Tfo_mean'] = float((D.Tpo-D.Tfo).mean()); fx['Tfo_minus_Tin_mean'] = float((D.Tfo-D.Tin).mean())
json.dump(fx, open(f'{OUT}/facts.json', 'w'), indent=1); print('facts ok')
