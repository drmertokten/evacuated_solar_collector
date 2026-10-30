"""Güneş geometrisi, geliş açıları ve 2B ışın izleme (parabolik oluk + Ø60 mm cam tüp)."""
import numpy as np
from config import *

def sun(doy, clock):
    dec = np.radians(23.45*np.sin(np.radians(360*(284+doy)/365)))
    B = np.radians(360*(doy-81)/364)
    eot = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)             # dakika
    solar = clock + ((LON-45)*4 + eot)/60.0                         # UTC+3
    h = np.radians(15*(solar-12))
    lat = np.radians(LAT)
    cz = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(h)
    return dec, h, cz, solar

def cos_inc(dec, h, beta, gam):
    lat = np.radians(LAT)
    return (np.sin(dec)*np.sin(lat)*np.cos(beta)-np.sin(dec)*np.cos(lat)*np.sin(beta)*np.cos(gam)
            +np.cos(dec)*np.cos(lat)*np.cos(beta)*np.cos(h)+np.cos(dec)*np.sin(lat)*np.sin(beta)*np.cos(gam)*np.cos(h)
            +np.cos(dec)*np.sin(beta)*np.sin(gam)*np.sin(h))

def angles(doy, clock, beta=np.radians(BETA_DEG), gam=np.radians(GAMMA_DEG)):
    """Enine (theta_T) ve boyuna (theta_L) geliş açıları [derece] ve cos(theta). Oluk ekseni eğim boyunca."""
    lat = np.radians(LAT)
    dec, h, cz, _ = sun(doy, clock); h = np.atleast_1d(h); one = np.ones_like(h)
    sE = -np.cos(dec)*np.sin(h)
    sN = np.cos(lat)*np.sin(dec)-np.sin(lat)*np.cos(dec)*np.cos(h)
    sU = np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(h)
    hE, hN = -np.sin(gam), -np.cos(gam)
    n = np.array([np.sin(beta)*hE*one, np.sin(beta)*hN*one, np.cos(beta)*one])
    a = np.array([-np.cos(beta)*hE*one, -np.cos(beta)*hN*one, np.sin(beta)*one])
    t = np.cross(n.T, a.T).T; s = np.array([sE, sN, sU])
    sn, sa, st = (s*n).sum(0), (s*a).sum(0), (s*t).sum(0)
    return np.degrees(np.arctan2(st, sn)), np.degrees(np.arctan2(sa, sn)), sn

# ---------------- 2B ışın izleme ----------------
R_TUBE = D_C/2

def tangent_zc():
    """Tüpün oluğa yan duvarlara teğet oturduğu merkez yüksekliği (tepe noktasından, m)."""
    xs = np.linspace(-W_AP/2, W_AP/2, 24001); zs = xs**2/(4*F_TR)
    for zc in np.arange(0.0, 0.2, 0.0001):
        if np.min(np.hypot(xs, zs-zc)) >= R_TUBE-1e-9: return float(zc)

def trace(theta_deg, zc, N=2001, rho=1.0, maxb=8, return_paths=False, xs_sel=None):
    th = np.radians(theta_deg); d0 = np.array([np.sin(th), -np.cos(th)])
    xs = np.linspace(-W_AP/2, W_AP/2, N) if xs_sel is None else np.asarray(xs_sel)
    absorbed = 0.0; paths = []
    for x0 in xs:
        p = np.array([x0, H_TR]); d = d0.copy(); wgt = 1.0; pts = [p.copy()]
        for b in range(maxb):
            oc = p-np.array([0, zc]); B = np.dot(oc, d); Cc = np.dot(oc, oc)-R_TUBE**2; disc = B*B-Cc; tc = np.inf
            if disc >= 0:
                t1 = -B-np.sqrt(disc)
                if t1 > 1e-9: tc = t1
            a = d[0]**2; bq = 2*p[0]*d[0]-4*F_TR*d[1]; c = p[0]**2-4*F_TR*p[1]; tp = np.inf
            if abs(a) < 1e-14:
                if abs(bq) > 1e-14:
                    t = -c/bq
                    if t > 1e-9: tp = t
            else:
                dis = bq*bq-4*a*c
                if dis >= 0:
                    for t in sorted([(-bq-np.sqrt(dis))/(2*a), (-bq+np.sqrt(dis))/(2*a)]):
                        if t > 1e-9:
                            q = p+t*d
                            if abs(q[0]) <= W_AP/2+1e-12 and q[1] <= H_TR+1e-9: tp = t; break
            if min(tc, tp) == np.inf:
                pts.append(p+0.05*d); break
            if tc < tp:
                absorbed += wgt; pts.append(p+tc*d); break
            q = p+tp*d; nrm = np.array([-q[0]/(2*F_TR), 1.0]); nrm /= np.linalg.norm(nrm)
            d = d-2*np.dot(d, nrm)*nrm; p = q; wgt *= rho; pts.append(p.copy())
        paths.append(np.array(pts))
    frac = absorbed/len(xs)
    return (frac, paths) if return_paths else frac
