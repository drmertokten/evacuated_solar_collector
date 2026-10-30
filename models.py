"""Gri-kutu (fizik temelli) doğal dolaşım modeli ve ML yardımcıları."""
import numpy as np
from scipy.optimize import brentq, least_squares
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import *; from data import cp

def ss_dT(G, Ta, Tin, A, eta_t, U, K, n=1.0):
    """Yarı-kararlı ΔT:  K cp ΔT^(n+1) + (A U/2) ΔT = A [G η0 m − U (Tin − Ta)]   (ṁ = K ΔT^n)."""
    out = np.zeros(len(G))
    for i in range(len(G)):
        rhs = A*(G[i]*eta_t[i]-U*(Tin[i]-Ta[i]))
        if rhs <= 0: continue
        f = lambda dT: K*cp(Tin[i]+dT/2)*1000*dT**(n+1)+A*U/2*dT-rhs
        out[i] = brentq(f, 1e-6, 120)
    return out

def simulate(d, A, eta0, U, K, tau, mfun):
    """Gün gün: ΔT_ss(t) -> birinci mertebe gecikme (τ, saat). mfun(rows)->optik modifikatör dizisi."""
    pred = np.zeros(len(d))
    for dy in d.day.unique():
        idx = np.where(d.day.values == dy)[0]; g = d.iloc[idx]
        ss = ss_dT(g.G.values, g.Ta.values, g.Tin.values, A, eta0*mfun(g), U, K)
        a = 1-np.exp(-(10/60.0)/tau) if tau > 0 else 1.0
        y = np.zeros(len(g)); y[0] = ss[0]
        for i in range(1, len(g)): y[i] = y[i-1]+a*(ss[i]-y[i-1])
        pred[idx] = y
    return pred

def fit_model(d, col, A, eta0, U, kind, ray_fun=None):
    """kind: qs | lag | time | geo | geo_lag | ray_lag"""
    y = d['dT'+col].values
    def unpack(p):
        K = np.exp(p[0]); tau = np.exp(p[1]) if kind in ('lag', 'time', 'geo_lag', 'ray_lag') else 0.0
        return K, tau, p[2], p[3]
    def mfun_from(p):
        _, _, a, b = unpack(p)
        if kind == 'time': return lambda g: np.clip(1-a*(g.clock.values-b)**2, 0.05, 1.0)
        if kind in ('geo', 'geo_lag'): return lambda g: np.clip(1-(g.thT.values/a)**2, 0.05, 1.0)
        if kind == 'ray_lag': return ray_fun
        return lambda g: np.ones(len(g))
    def res(p):
        K, tau, _, _ = unpack(p); return simulate(d, A, eta0, U, K, tau, mfun_from(p))-y
    x0 = [np.log(3e-3), np.log(0.3), 0.02 if kind == 'time' else 45.0, 13.25]
    lb = [-12, np.log(0.02), 0.0 if kind == 'time' else 20.0, 11.5]; ub = [-2, np.log(3), 0.3 if kind == 'time' else 90.0, 15.0]
    r = least_squares(res, x0, bounds=(lb, ub), diff_step=1e-3)
    K, tau, a, b = unpack(r.x); return dict(K=K, tau=tau, a=a, b=b, mfun=mfun_from(r.x))

def metrics(y, p):
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))), mae=float(np.mean(np.abs(y-p))), r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))

def ml_models():
    return {'Linear': LinearRegression(), 'Ridge': make_pipeline(StandardScaler(), Ridge(1.0)),
            'SVR': make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.2)),
            'RF': RandomForestRegressor(200, min_samples_leaf=3, random_state=0),
            'GBR': GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)}
