"""Step 3: energy balance, model-based flow rate (scenarios S1/S2), tank closure, exergy,
sensitivity and flow-regime indicators.

Subscripts used internally: 'f' = conventional collector (C in the paper), 'p' = PTC-assisted collector (P).
Scenarios: S1 = nominal optical efficiencies (0.60 / 0.75); S2 = geometry-consistent (eta_P = eta_o,C * I(theta_tr))."""
import json, numpy as np, pandas as pd
from config import *; from data import *
D = pd.read_pickle(f'{OUT}/D_geo.pkl'); geo = json.load(open(f'{OUT}/geometry.json')); angs = np.array(geo['ray']['angs'])
I085 = np.array(geo['ray']['rho085']); I1 = np.array(geo['ray']['rho1'])
D['I085'] = np.interp(np.abs(D.thT.values), angs, I085); D['I1'] = np.interp(np.abs(D.thT.values), angs, I1)
D['hour'] = (D['min']//60).clip(upper=4).astype(int)
TO = {'f': 'Tfo', 'p': 'Tpo'}; AR = {'f': A_F, 'p': A_P}

def heat(D, scen='S1', scale=1.0, U=UL, I='I085'):
    """Useful heat (W). S1: nominal eta_o and each collector's own area and loss term.
       S2: eta_P = eta_o,C * I(theta_tr); the tube loss conductance U_L*A_C is the same in both collectors."""
    Tmf = (D.Tin+D.Tfo)/2; Tmp = (D.Tin+D.Tpo)/2
    Qf = A_F*(D.G*ETA_F*scale-U*(Tmf-D.Ta))
    if scen == 'S1': Qp = A_P*(D.G*ETA_P*scale-U*(Tmp-D.Ta))
    else:            Qp = A_P*D.G*ETA_F*scale*D[I]-U*A_F*(Tmp-D.Ta)
    return Qf, Qp

def mdot(Q, c, D, idx=None):
    """Mean flow rate (g/s) as a ratio of mean values: mean(Q) / (mean(c_p) * mean(dT))."""
    g = D if idx is None else D.loc[idx]
    Tm = (g.Tin+g[TO[c]])/2
    return Q.loc[g.index].mean()/(cp(Tm).mean()*1000*g['dT'+c].mean())*1000

def tank_Q(g, Ut):
    Qt = M_TANK*cp(g.Tt.mean())*1000*(g.Tt.iloc[-1]-g.Tt.iloc[0])/((len(g)-1)*600.0)
    return Qt+Ut*A_TANK*np.mean(g.Tt-g.Ta)
out = {}
# --- scenarios: daily and three-day values ---
scen = {}
for sc, s, I in (('S1', 'S1', 'I085'), ('S2', 'S2', 'I085'), ('S2_rho1', 'S2', 'I1')):
    Qf, Qp = heat(D, s, I=I); rows = []
    for d, g in list(D.groupby('day'))+[('all', D)]:
        rows.append(dict(day=d, m_C=mdot(Qf, 'f', D, g.index), m_P=mdot(Qp, 'p', D, g.index), Q_C=Qf.loc[g.index].mean(), Q_P=Qp.loc[g.index].mean(),
            eta_C=(Qf.loc[g.index]/(A_F*g.G)).mean(), eta_P=(Qp.loc[g.index]/(A_P*g.G)).mean(), Q_sum=(Qf+Qp).loc[g.index].mean()))
    scen[sc] = rows
    if sc == 'S1': QfS1, QpS1 = Qf, Qp
    if sc == 'S2': QfS2, QpS2 = Qf, Qp
out['scen'] = scen
# --- tank energy closure ---
clos = []
for d, g in D.groupby('day'):
    for Ut in (0, 1):
        Qt = tank_Q(g, Ut); clos.append(dict(day=d, Ut=Ut, Qtank=Qt, Qinc=(g.G*(A_F+A_P)).mean(), S1=(QfS1+QpS1).loc[g.index].mean(), S2=(QfS2+QpS2).loc[g.index].mean()))
out['closure'] = clos; C = pd.DataFrame(clos)
for Ut in (0, 1):
    s = C[C.Ut == Ut]; out[f'cl_tot_{Ut}'] = dict(S1=s.S1.sum()/s.Qtank.sum(), S2=s.S2.sum()/s.Qtank.sum(), sys_eta=s.Qtank.sum()/s.Qinc.sum())
# --- sensitivity: eta_o scale x U_L ---
sens = []
for s_ in ('S1', 'S2'):
    for sc in (0.8, 0.9, 1.0, 1.1):
        for U in (0.5, 1.0, 2.0, 3.0):
            Qf, Qp = heat(D, s_, scale=sc, U=U); r = dict(scen=s_, scale=sc, U=U, mf=mdot(Qf, 'f', D), mp=mdot(Qp, 'p', D), etaf=(Qf/(A_F*D.G)).mean(), etap=(Qp/(A_P*D.G)).mean())
            for Ut in (0, 1): r[f'cl{Ut}'] = (Qf+Qp).mean()*3/C[C.Ut == Ut].Qtank.sum()
            sens.append(r)
out['sens'] = sens
# --- hourly flow rates ---
H = []
for (d, h), g in D.groupby(['day', 'hour']):
    H.append(dict(day=d, hour=10+h, m_C=mdot(QfS1, 'f', D, g.index), m_P_S1=mdot(QpS1, 'p', D, g.index), m_P_S2=mdot(QpS2, 'p', D, g.index)))
out['hourly'] = H
# --- exergy efficiency ---
Ts = 5777.0; ex = []
for sc, Qf, Qp in (('S1', QfS1, QpS1), ('S2', QfS2, QpS2)):
    for c, Q in (('f', Qf), ('p', Qp)):
        for d, g in list(D.groupby('day'))+[('all', D)]:
            m = mdot(Q, c, D, g.index)/1000; Ti = g.Tin+273.15; To = g[TO[c]]+273.15; Ta = g.Ta+273.15; cpm = cp((g.Tin+g[TO[c]])/2)*1000
            exg = m*cpm*((To-Ti)-Ta*np.log(To/Ti)); exin = AR[c]*g.G*(1-4/3*Ta/Ts+1/3*(Ta/Ts)**4); ex.append(dict(scen=sc, c=c, day=d, psi=float(exg.sum()/exin.sum())))
out['exergy'] = ex
# --- flow-regime indicators (hourly block flow rates; Gr uses dT = T_out - T_in) ---
d_i, L_b = D_BI, L_B
rows = []; DL = {}
for sc, Qf, Qp in (('S1', QfS1, QpS1), ('S2', QfS2, QpS2)):
    for c, Q in (('f', Qf), ('p', Qp)):
        Tm = (D.Tin+D[TO[c]])/2
        mb = pd.DataFrame([dict(day=d, hour=h, m=mdot(Q, c, D, g.index)/1000) for (d, h), g in D.groupby(['day', 'hour'])])
        m = D.merge(mb, on=['day', 'hour'], how='left')['m'].values; nu = mu(Tm)/rho(Tm)
        Re = 4*m/(np.pi*d_i*mu(Tm)); Pr = mu(Tm)*cp(Tm)*1000/kf(Tm); Gz = Re*Pr*d_i/L_b
        Gr = G_ACC*beta(Tm)*D['dT'+c]*d_i**3/nu**2; Ri = Gr/Re**2
        for nm, v in (('Re', Re), ('Pr', Pr), ('Gz', Gz), ('Gr', Gr), ('Ri', Ri)):
            v = np.asarray(v, float); rows.append(dict(scen=sc, c=c, var=nm, mean=float(np.nanmean(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v))))
        DL[f'{sc}_{c}'] = dict(Re=Re.tolist(), Gz=Gz.tolist(), Ri=Ri.tolist(), Gr=Gr.tolist(), Pr=Pr.tolist())
out['dimless'] = rows
json.dump(out, open(f'{OUT}/energy.json', 'w'), default=float); json.dump(DL, open(f'{OUT}/dimless_series.json', 'w'))
D['Q_C'], D['Q_P_S1'], D['Q_P_S2'] = QfS1, QpS1, QpS2; D.to_pickle(f'{OUT}/D_en.pkl')
for sc in scen: print(sc, pd.DataFrame(scen[sc]).round(3).to_string())
print(pd.DataFrame(out['closure']).round(1).to_string()); print({k: out[k] for k in ('cl_tot_0', 'cl_tot_1')})
print(pd.DataFrame(ex).pivot_table(index=['scen', 'c'], columns='day', values='psi').round(4))
print(pd.DataFrame(rows).pivot_table(index='var', columns=['scen', 'c'], values='mean').round(3).to_string())
