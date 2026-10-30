"""Veri yükleme ve su özellikleri (Word dosyasındaki korelasyonlar, Eş. 4-9)."""
import numpy as np, pandas as pd
from config import *

def load():
    raw = pd.read_excel(XLSX, header=None)
    frames = []
    for k, c0 in enumerate([1, 9, 17]):
        d = raw.iloc[2:, c0:c0 + 7].copy()
        d.columns = ['G', 'Ta', 'Tt', 'Tin', 'Tfo', 'Tpi', 'Tpo']
        d = d.apply(pd.to_numeric, errors='coerce').dropna()
        d['day'] = k + 1
        d['min'] = np.arange(len(d)) * 10
        frames.append(d)
    D = pd.concat(frames).reset_index(drop=True)
    D['clock'] = 10 + D['min'] / 60.0
    D['dTf'] = D.Tfo - D.Tin             # düz kolektör sıcaklık artışı
    D['dTp'] = D.Tpo - D.Tin             # parabolik kolektör sıcaklık artışı
    assert (D.Tin == D.Tpi).all()        # iki kolektörün girişi ortak ölçüm
    return D

cp  = lambda T: 4.2174 - 3.72e-3*T + 7.28e-6*T**2 - 8.66e-9*T**3                  # kJ/kgK
rho = lambda T: (999.84+16.95*T-7.99e-3*T**2-46.17e-6*T**3+105.56e-9*T**4-280.54e-12*T**5)/(1+16.88e-3*T)
mu  = lambda T: 2.414e-5*10**(247.8/(T+133.15))                                  # Pa s
kf  = lambda T: 0.567+1.9e-3*T-3.2e-6*T**2                                       # W/mK
beta = lambda T: 0.000207+0.0000012*T                                            # 1/K
