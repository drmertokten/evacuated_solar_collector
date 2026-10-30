"""Adım 1: yönelim çıkarımı, geliş açıları, oluk geometrisi ve ışın izleme IAM eğrisi."""
import json, numpy as np, pandas as pd
from scipy.optimize import lsq_linear
from config import *; from data import load; from geometry import *
D = load(); beta = np.radians(BETA_DEG)
G, clk, day = D.G.values, D.clock.values, D.day.values
def azfit(gam, idmax=200, ibmax=1100):
    tot = 0
    for d in (1, 2, 3):
        m = day == d; dec, h, cz, _ = sun(DOY0-1+d, clk[m]); ci = np.clip(cos_inc(dec, h, beta, np.radians(gam)), 0, None)
        X = np.c_[ci, np.full(m.sum(), (1+np.cos(beta))/2)]; r = lsq_linear(X, G[m], bounds=([0, 0], [ibmax, idmax])); tot += np.sum((G[m]-X@r.x)**2)
    return float(np.sqrt(tot/len(G)))
AZ = [dict(gam=g, bearing=(180+g) % 360, rmse=azfit(g)) for g in (-45, -22.5, 0, 10, 20, 30, 40, 60, 85, 135, 157.5)]
# geliş açıları
thT = np.zeros(len(D)); thL = np.zeros(len(D)); cosT = np.zeros(len(D))
for d in (1, 2, 3):
    m = day == d; a, b, c = angles(DOY0-1+d, clk[m]); thT[m], thL[m], cosT[m] = a, b, c
D['thT'], D['thL'], D['cosT'] = thT, thL, cosT
# oluk geometrisi
zt = tangent_zc(); rim = np.degrees(np.pi-np.arctan((W_AP/2)/(H_TR-F_TR)))   # kenar açısı (derece), f < derinlik
geo = dict(f=F_TR, zt=zt, tube_top=zt+R_TUBE, rim_angle=float(rim), C_geo=W_AP/D_C, acc_ideal=float(np.degrees(np.arcsin(D_C/W_AP))))
# ışın izleme IAM
angs = np.arange(0, 76, 2.5)
ray = {'angs': angs.tolist()}
for nm, rho_ in (('rho085', 0.85), ('rho1', 1.0)):
    ray[nm] = [trace(a, zt, N=2001, rho=rho_) for a in angs]
# diyagram için örnek ışınlar
paths = {}
for a in (0, 30, 50):
    xs = np.linspace(-W_AP/2*0.97, W_AP/2*0.97, 13)
    f_, p_ = trace(a, zt, rho=1.0, return_paths=True, xs_sel=xs); paths[str(a)] = [p.tolist() for p in p_]
json.dump(dict(AZ=AZ, geo=geo, ray=ray, paths=paths), open(f'{OUT}/geometry.json', 'w'))
D.to_pickle(f'{OUT}/D_geo.pkl')
print(pd.DataFrame(AZ).round(1).to_string()); print(geo)
print("I(0) rho=0.85:", round(ray['rho085'][0], 3), " I(30):", round(ray['rho085'][12], 3), " I(45):", round(ray['rho085'][18], 3), " I(60):", round(ray['rho085'][24], 3))
