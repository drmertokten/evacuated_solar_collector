"""Adım 3: enerji dengesi, debi tayini (Senaryo A/B), depo kapanışı, ekserji, duyarlılık, boyutsuz sayılar."""
import json, numpy as np, pandas as pd
from config import *; from data import *
D = pd.read_pickle(f'{OUT}/D_geo.pkl'); geo = json.load(open(f'{OUT}/geometry.json')); angs = np.array(geo['ray']['angs'])
I085 = np.array(geo['ray']['rho085']); I1 = np.array(geo['ray']['rho1'])
D['I085'] = np.interp(np.abs(D.thT.values), angs, I085); D['I1'] = np.interp(np.abs(D.thT.values), angs, I1)
D['hour'] = (D['min']//60).clip(upper=4).astype(int)
TO = {'f': 'Tfo', 'p': 'Tpo'}; AR = {'f': A_F, 'p': A_P}

def heat(D, scen='A', scale=1.0, U=UL, I='I085'):
    """Faydalı ısı (W) -- Senaryo A: nominal η0 ve her kolektör kendi alanı için U_L;
       Senaryo B: η_p = η_f·I(θ_T), tüp kayıp iletkenliği iki kolektörde aynı (U_L·A_f)."""
    Tmf = (D.Tin+D.Tfo)/2; Tmp = (D.Tin+D.Tpo)/2
    Qf = A_F*(D.G*ETA_F*scale-U*(Tmf-D.Ta))
    if scen == 'A': Qp = A_P*(D.G*ETA_P*scale-U*(Tmp-D.Ta))
    else:           Qp = A_P*D.G*ETA_F*scale*D[I]-U*A_F*(Tmp-D.Ta)
    return Qf, Qp

def mdot(Q, c, D, idx=None):
    g = D if idx is None else D.loc[idx]
    Tm = (g.Tin+g[TO[c]])/2
    return Q.loc[g.index].mean()/(cp(Tm).mean()*1000*g['dT'+c].mean())*1000          # g/s (ortalamaların oranı)

def tank_Q(g, Ut):
    Qt = M_TANK*cp(g.Tt.mean())*1000*(g.Tt.iloc[-1]-g.Tt.iloc[0])/((len(g)-1)*600.0)
    return Qt+Ut*A_TANK*np.mean(g.Tt-g.Ta)
out = {}
# --- senaryolar: günlük + toplam ---
scen = {}
for sc, I in (('A', None), ('B085', 'I085'), ('B1', 'I1')):
    s = 'A' if sc == 'A' else 'B'; Qf, Qp = heat(D, s, I=I or 'I085'); rows = []
    for d, g in list(D.groupby('day'))+[('all', D)]:
        Tmf = (g.Tin+g.Tfo)/2; Tmp = (g.Tin+g.Tpo)/2
        rows.append(dict(day=d, mf=mdot(Qf, 'f', D, g.index), mp=mdot(Qp, 'p', D, g.index), Qf=Qf.loc[g.index].mean(), Qp=Qp.loc[g.index].mean(),
            etaf=(Qf.loc[g.index]/(A_F*g.G)).mean(), etap=(Qp.loc[g.index]/(A_P*g.G)).mean(), Qsum=(Qf+Qp).loc[g.index].mean()))
    scen[sc] = rows
    if sc == 'A': QfA, QpA = Qf, Qp
    if sc == 'B085': QfB, QpB = Qf, Qp
out['scen'] = scen
# --- depo kapanışı ---
clos = []
for d, g in D.groupby('day'):
    for Ut in (0, 1):
        Qt = tank_Q(g, Ut); clos.append(dict(day=d, Ut=Ut, Qtank=Qt, Qinc=(g.G*(A_F+A_P)).mean(), A=(QfA+QpA).loc[g.index].mean(), B=(QfB+QpB).loc[g.index].mean()))
out['closure'] = clos; C = pd.DataFrame(clos)
for Ut in (0, 1):
    s = C[C.Ut == Ut]; out[f'cl_tot_{Ut}'] = dict(A=s.A.sum()/s.Qtank.sum(), B=s.B.sum()/s.Qtank.sum(), sys_eta=s.Qtank.sum()/s.Qinc.sum())
# --- duyarlılık: η0 ölçeği × U_L ---
sens = []
for s_ in ('A', 'B'):
    for sc in (0.8, 0.9, 1.0, 1.1):
        for U in (0.5, 1.0, 2.0, 3.0):
            Qf, Qp = heat(D, s_, scale=sc, U=U); r = dict(scen=s_, scale=sc, U=U, mf=mdot(Qf, 'f', D), mp=mdot(Qp, 'p', D), etaf=(Qf/(A_F*D.G)).mean(), etap=(Qp/(A_P*D.G)).mean())
            for Ut in (0, 1): r[f'cl{Ut}'] = (Qf+Qp).mean()*3/C[C.Ut == Ut].Qtank.sum()
            sens.append(r)
out['sens'] = sens
# --- saatlik debi (Senaryo A ve B) ---
H = []
for (d, h), g in D.groupby(['day', 'hour']):
    H.append(dict(day=d, hour=10+h, mfA=mdot(QfA, 'f', D, g.index), mpA=mdot(QpA, 'p', D, g.index), mpB=mdot(QpB, 'p', D, g.index), mf=mdot(QfA, 'f', D, g.index)))
out['hourly'] = H
# --- ekserji ---
Ts = 5777.0; ex = []
for sc, Qf, Qp in (('A', QfA, QpA), ('B', QfB, QpB)):
    for c, Q in (('f', Qf), ('p', Qp)):
        for d, g in list(D.groupby('day'))+[('all', D)]:
            m = mdot(Q, c, D, g.index)/1000; Ti = g.Tin+273.15; To = g[TO[c]]+273.15; Ta = g.Ta+273.15; cpm = cp((g.Tin+g[TO[c]])/2)*1000
            exg = m*cpm*((To-Ti)-Ta*np.log(To/Ti)); exin = AR[c]*g.G*(1-4/3*Ta/Ts+1/3*(Ta/Ts)**4); ex.append(dict(scen=sc, c=c, day=d, psi=float(exg.sum()/exin.sum())))
out['exergy'] = ex
# --- boyutsuz sayılar (Senaryo A, saatlik blok debisi; B için de) ---
d_i, L_b, Lt = D_BI, L_B, L_CHAR
rows = []; DL = {}
for sc, Qf, Qp in (('A', QfA, QpA), ('B', QfB, QpB)):
    for c, Q in (('f', Qf), ('p', Qp)):
        Tm = (D.Tin+D[TO[c]])/2
        mb = pd.DataFrame([dict(day=d, hour=h, m=mdot(Q, c, D, g.index)/1000) for (d, h), g in D.groupby(['day', 'hour'])])
        m = D.merge(mb, on=['day', 'hour'], how='left')['m'].values; nu = mu(Tm)/rho(Tm)
        Re = 4*m/(np.pi*d_i*mu(Tm)); Pr = mu(Tm)*cp(Tm)*1000/kf(Tm); Gz = Re*Pr*d_i/L_b
        Grd = G_ACC*beta(Tm)*D['dT'+c]*d_i**3/nu**2; Ri = Grd/Re**2
        Gr = G_ACC*beta(Tm)*(Tm-D.Ta)*Lt**3/nu**2; Ra = Gr*Pr; Nu = np.where(Ra < 1e9, 0.59*Ra**0.25, 0.10*Ra**0.33); h = Nu*kf(Tm)/Lt
        Rab = Gr*Pr/1000; Nub = np.where(Rab < 1e9, 0.59*Rab**0.25, 0.10*Rab**0.33)       # çalışma dosyasındaki Pr hatası (c_p kJ ile)
        for nm, v in (('Re', Re), ('Pr', Pr), ('Gz', Gz), ('Gr_d', Grd), ('Ri', Ri), ('Gr', Gr), ('Ra', Ra), ('Nu', Nu), ('h', h), ('Ra_bug', Rab), ('Nu_bug', Nub)):
            v = np.asarray(v, float); rows.append(dict(scen=sc, c=c, var=nm, mean=float(np.nanmean(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v))))
        DL[f'{sc}_{c}'] = dict(Re=Re.tolist(), Gz=Gz.tolist(), Ri=Ri.tolist(), Grd=Grd.tolist(), Pr=Pr.tolist())
out['dimless'] = rows

# --- Gr, Ra, Nu, h zincirinin karakteristik uzunluğa (L) duyarlılığı ---
dimL = {}
for L in (D_BO, D_BI, H_TANK, L_C):
    o = {}
    for c in ('f', 'p'):
        Tm = (D.Tin+D[TO[c]])/2; nu = mu(Tm)/rho(Tm); Pr = mu(Tm)*cp(Tm)*1000/kf(Tm)
        Gr = G_ACC*beta(Tm)*(Tm-D.Ta)*L**3/nu**2; Ra = Gr*Pr; Nu = np.where(Ra < 1e9, 0.59*Ra**0.25, 0.10*Ra**0.33); h = Nu*kf(Tm)/L
        o[c] = {n: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for n, v in (('Gr', Gr), ('Ra', Ra), ('Nu', Nu), ('h', h))}
        o[c]['dT_air'] = float(np.mean(Tm-D.Ta)); o[c]['loss_if_h_W'] = float(np.mean(h)*AR[c]*np.mean(Tm-D.Ta))
    dimL[f'{L:.4g}'] = o
out['dimless_L'] = dimL
json.dump(out, open(f'{OUT}/energy.json', 'w'), default=float); json.dump(DL, open(f'{OUT}/dimless_series.json', 'w'))
D['QfA'], D['QpA'], D['QfB'], D['QpB'] = QfA, QpA, QfB, QpB; D.to_pickle(f'{OUT}/D_en.pkl')
for sc in scen: print(sc, pd.DataFrame(scen[sc]).round(3).to_string())
print(pd.DataFrame(out['closure']).round(1).to_string()); print({k: out[k] for k in ('cl_tot_0', 'cl_tot_1')})
print(pd.DataFrame(ex).pivot_table(index=['scen', 'c'], columns='day', values='psi').round(4))
print(pd.DataFrame(rows).pivot_table(index='var', columns=['scen', 'c'], values='mean').round(3).to_string())
