"""Adım 6: tüm şekiller (İngilizce etiketler) ve fotoğraf paneli."""
import os, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from config import *; from data import *; from geometry import *
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100})
D = pd.read_pickle(f'{OUT}/D_en.pkl'); geo = json.load(open(f'{OUT}/geometry.json')); st = json.load(open(f'{OUT}/stats.json'))
en = json.load(open(f'{OUT}/energy.json')); mo = json.load(open(f'{OUT}/models.json')); un = json.load(open(f'{OUT}/uncertainty.json')); dls = json.load(open(f'{OUT}/dimless_series.json'))
C = {'Tin': '#1f77b4', 'Tfo': '#2ca02c', 'Tpo': '#d62728', 'Tt': '#7f7f7f', 'Ta': '#ff7f0e', 'f': '#2ca02c', 'p': '#d62728'}
DAYC = {1: '#1f77b4', 2: '#ff7f0e', 3: '#9467bd'}
def save(fig, name): fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png', dpi=200); plt.close(fig)

# ---- Fig 2: ölçülen zaman serileri (2 satır: geleneksel / PTC; 3 sütun: gün) ----
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True, sharey='row')
for r, (key, ttl, col) in enumerate((('Tfo', 'Conventional', C['Tfo']), ('Tpo', 'PTC-assisted', C['Tpo']))):
    for k in (1, 2, 3):
        g = D[D.day == k]; a = ax[r, k-1]; t = g.clock
        a.plot(t, g.Tin, color=C['Tin'], label='Inlet'); a.plot(t, g[key], color=col, label='Outlet'); a.plot(t, g.Ta, color=C['Ta'], ls=':', label='Ambient')
        b = a.twinx(); b.fill_between(t, 0, g.G, color='#999999', alpha=.18, lw=0); b.plot(t, g.G, color='#999999', lw=.8); b.set_ylim(0, 1600); b.grid(False); b.spines['right'].set_visible(True)
        if k == 3: b.set_ylabel('Irradiance in collector plane (W/m²)')
        else: b.set_yticklabels([])
        a.set_title(f'{ttl} – Day {k}')
        if r == 1: a.set_xlabel('Local time (h)')
    ax[r, 0].set_ylabel('Temperature (°C)')
    h1, l1 = ax[r, 0].get_legend_handles_labels(); ax[r, 0].legend(h1+[plt.Rectangle((0, 0), 1, 1, color='#999999', alpha=.3)], l1+['Irradiance (right axis)'], fontsize=7, loc='upper left')
ax[0, 0].set_ylim(25, 90); save(fig, 'fig_timeseries')

# ---- Fig 2b: saatlik model tabanlı verim (Senaryo A ve B) ----
D['hour'] = (D['min']//60).clip(upper=4).astype(int); rows = []
for (d_, h_), g in D.groupby(['day', 'hour']):
    G_ = g.G.mean(); rows.append(dict(day=d_, hour=10+h_, eta_f=g.QfA.mean()/(A_F*G_), eta_pA=g.QpA.mean()/(A_P*G_), eta_pB=g.QpB.mean()/(A_P*G_)))
HE = pd.DataFrame(rows); fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
STY = (('eta_f', C['f'], 'o', 'Conventional'), ('eta_pA', C['p'], 's', 'PTC – Scenario A'), ('eta_pB', '#ff9896', '^', 'PTC – Scenario B'))
for k in (1, 2, 3):
    h = HE[HE.day == k]
    for c_, col_, mk, lb in STY: ax[k-1].plot(h.hour+.5, h[c_], mk+'-', color=col_, label=lb)
    ax[k-1].set_title(f'Day {k}'); ax[k-1].set_xlabel('Local time (h, block centre)')
mm = HE.groupby('hour')[[s_[0] for s_ in STY]].agg(['mean', 'min', 'max'])
for c_, col_, mk, lb in STY: ax[3].plot(mm.index+.5, mm[c_]['mean'], mk+'-', color=col_, label=lb); ax[3].fill_between(mm.index+.5, mm[c_]['min'], mm[c_]['max'], color=col_, alpha=.15)
ax[3].set_title('3-day mean (band: day range)'); ax[3].set_xlabel('Local time (h, block centre)'); ax[0].set_ylabel('Model-based thermal efficiency (–)'); ax[0].set_ylim(0.3, 0.8); ax[0].legend(fontsize=7, loc='lower left')
save(fig, 'fig_eta_hourly')

# ---- Fig 3: eşleştirilmiş ΔT karşılaştırması ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for k in (1, 2, 3):
    g = D[D.day == k]; ax[0].scatter(g.dTf, g.dTp, s=16, color=DAYC[k], label=f'Day {k}')
ax[0].plot([0, 18], [0, 18], 'k--', lw=.8); ax[0].text(9.5, 8.2, '1:1 line', rotation=38, fontsize=7); ax[0].set_xlabel('ΔT conventional (K)'); ax[0].set_ylabel('ΔT PTC-assisted (K)'); ax[0].legend(fontsize=7); ax[0].set_title('(a) Paired observations (n = 93)')
for k in (1, 2, 3):
    g = D[D.day == k]; ax[1].plot(g.clock, g.dTp-g.dTf, color=DAYC[k], marker='.', ms=3, lw=1, label=f'Day {k}')
P = st['paired']; ax[1].axhline(P['mean_diff'], color='k', lw=1); ax[1].axhspan(P['ci_block'][0], P['ci_block'][1], color='k', alpha=.12, label=f"Mean {P['mean_diff']:.2f} K, 95% block-bootstrap CI {P['ci_block'][0]:.1f}–{P['ci_block'][1]:.1f}")
ax[1].axhspan(P['ci_t'][0], P['ci_t'][1], color='tab:green', alpha=.25, label=f"Naïve t-interval {P['ci_t'][0]:.2f}–{P['ci_t'][1]:.2f}")
ax[1].set_xlabel('Local time (h)'); ax[1].set_ylabel('ΔT$_{PTC}$ − ΔT$_{conv}$ (K)'); ax[1].legend(fontsize=6.3, loc='upper right'); ax[1].set_ylim(0, 19); ax[1].set_title('(b) Paired difference through the day')
data = [D[D.day == k].dTf for k in (1, 2, 3)]+[D.dTf]; data2 = [D[D.day == k].dTp for k in (1, 2, 3)]+[D.dTp]
pos = np.arange(4); bp1 = ax[2].boxplot(data, positions=pos-.18, widths=.3, patch_artist=True); bp2 = ax[2].boxplot(data2, positions=pos+.18, widths=.3, patch_artist=True)
for b in bp1['boxes']: b.set_facecolor(C['f']); b.set_alpha(.6)
for b in bp2['boxes']: b.set_facecolor(C['p']); b.set_alpha(.6)
ax[2].set_xticks(pos); ax[2].set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'All']); ax[2].set_ylabel('ΔT (K)'); ax[2].set_title('(c) ΔT distributions'); ax[2].legend([bp1['boxes'][0], bp2['boxes'][0]], ['Conventional', 'PTC-assisted'], fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 1.0)); ax[2].set_ylim(0, 24)
save(fig, 'fig_paired')

# ---- Fig 4: indirgenmiş sıcaklık ----
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
for a, c, nm in zip(ax, ('f', 'p'), ('Conventional', 'PTC-assisted')):
    x = (D.Tin-D.Ta)/D.G*1000; y = D['dT'+c]/D.G*1000
    for k in (1, 2, 3): m = D.day == k; a.scatter(x[m], y[m], s=15, color=DAYC[k], label=f'Day {k}')
    r = st['reduced'][c]; xs = np.linspace(x.min(), x.max(), 20); a.plot(xs, (r['intercept']+r['slope']*xs/1000)*1000, 'k-', lw=1)
    a.set_title(f"{nm}: slope = {r['slope']:.3f}, R² = {r['r2']:.2f}"); a.set_xlabel('(T$_{in}$ − T$_a$)/G  (10$^{-3}$ K m²/W)'); a.set_ylabel('ΔT/G  (10$^{-3}$ K m²/W)')
ax[0].legend(fontsize=7); save(fig, 'fig_reduced')

# ---- Fig 5: korelasyon ----
cols = st['corr_cols']; lab = ['G', 'T$_a$', 'T$_{tank}$', 'T$_{in}$', 'ΔT$_{conv}$', 'ΔT$_{PTC}$', 'θ$_T$']
pr = np.array(st['pearson']); sp = np.array(st['spearman'])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
for a, M_, tt in zip(ax, (pr, sp), ('Pearson r', 'Spearman ρ')):
    im = a.imshow(M_, vmin=-1, vmax=1, cmap='RdBu_r'); a.set_xticks(range(7)); a.set_yticks(range(7)); a.set_xticklabels(lab, rotation=45, ha='right'); a.set_yticklabels(lab); a.grid(False); a.set_title(tt)
    for i in range(7):
        for j in range(7): a.text(j, i, f'{M_[i,j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if abs(M_[i, j]) > .6 else 'black')
fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(f'{OUT}/fig_corr.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# ---- Fig 6: yönelim ve geliş açıları ----
AZ = pd.DataFrame(geo['AZ']); beta_r = np.radians(BETA_DEG)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
ax[0].plot(AZ.bearing, AZ.rmse, 'o-', color='#1f77b4'); ax[0].axvline(188, color='#2ca02c', ls='--'); ax[0].text(193, 215, 'Konya direction\n(188°)', fontsize=7, color='#2ca02c')
ax[0].set_xlabel('Front-face bearing (° from north)'); ax[0].set_ylabel('RMSE vs. measured G(t) (W/m²)'); ax[0].set_title('(a) Orientation inferred from irradiance')
from scipy.optimize import lsq_linear
g = D[D.day == 2]; dec, h, cz, _ = sun(DOY0+1, g.clock.values)
ax[1].plot(g.clock, g.G, 'k.-', ms=3, lw=1, label='Measured G (day 2)')
for gam, ls, lb in ((0, '-', 'South-facing model (γ = 0°)'), (85, '--', 'West-facing model (γ = 85°)')):
    ci = np.clip(cos_inc(dec, h, beta_r, np.radians(gam)), 0, None); X = np.c_[ci, np.full(len(g), (1+np.cos(beta_r))/2)]
    r = lsq_linear(X, g.G.values, bounds=([0, 0], [1100, 200])); ax[1].plot(g.clock, X@r.x, ls, label=lb)
ax[1].set_xlabel('Local time (h)'); ax[1].set_ylabel('G (W/m²)'); ax[1].legend(fontsize=6.8); ax[1].set_title('(b) Measured vs clear-sky plane model')
t = np.linspace(10, 15, 61); tt = np.array([angles(DOY0+1, np.array([x]))[0][0] for x in t]); cs = np.array([angles(DOY0+1, np.array([x]))[2][0] for x in t])
ax[2].plot(t, tt, color='#555555', label='θ$_T$ (transverse incidence angle)'); ax[2].set_ylabel('θ$_T$ (°)'); ax[2].set_xlabel('Local time (h)')
a2 = ax[2].twinx(); a2.plot(t, cs, color='#d62728', label='cos θ (plane normal)'); a2.set_ylim(0, 1.05); a2.grid(False); a2.set_ylabel('cos θ'); ax[2].legend(fontsize=7, loc='upper left'); a2.legend(fontsize=7, loc='lower right'); ax[2].set_title('(c) Sun angles on the 45° south-facing plane')
save(fig, 'fig_orientation')

# ---- Fig 7: oluk geometrisi, ışın izleme, IAM ----
fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.8), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
zs = np.linspace(-W_AP/2, W_AP/2, 200); zp = zs**2/(4*F_TR)
for a, ang in zip(ax[:3], ('0', '30', '50')):
    a.plot(zs*100, zp*100, 'k-', lw=2); a.add_patch(Circle((0, geo['geo']['zt']*100), R_TUBE*100, fill=False, ec='#d62728', lw=1.5)); a.plot(0, F_TR*100, 'b+', ms=7)
    for p_ in geo['paths'][ang]:
        p_ = np.array(p_); a.plot(p_[:, 0]*100, p_[:, 1]*100, color='#ff9900', lw=.7)
    a.set_aspect('equal'); a.set_xlim(-8, 8); a.set_ylim(-1, 11.5); a.set_title(f'θ$_T$ = {ang}°'); a.set_xlabel('x (cm)')
    a.text(-7.6, 10.4, f"Intercept = {np.interp(float(ang), geo['ray']['angs'], geo['ray']['rho1']):.2f} (ρ = 1)", fontsize=7)
ax[0].set_ylabel('z (cm)'); ax[0].text(0.4, F_TR*100-1.3, 'focus', color='b', fontsize=7)
angs_ = np.array(geo['ray']['angs']); I085 = np.array(geo['ray']['rho085']); I1 = np.array(geo['ray']['rho1']); fin = mo['fin']; th = np.linspace(0, 75, 151)
ax[3].plot(angs_, I085/I085[0], color='#1f77b4', label='Ray tracing, ρ = 0.85'); ax[3].plot(angs_, I1/I1[0], ':', color='#1f77b4', label='Ray tracing, ρ = 1')
ax[3].plot(th, np.clip(1-(th/fin['p']['tha'])**2, 0, 1), color='#d62728', label=f"Fitted to data, θ$_a$ = {fin['p']['tha']:.0f}°"); ax[3].axvspan(0, 45, color='gray', alpha=.07)
ax[3].axvline(geo['geo']['acc_ideal'], color='k', ls='--', lw=.8); ax[3].text(geo['geo']['acc_ideal']+1, 0.05, 'ideal limit 30°', fontsize=7)
ax[3].set_xlabel('|θ$_T$| (°)'); ax[3].set_ylabel('Relative capture / modifier'); ax[3].set_ylim(0, 1.15); ax[3].legend(fontsize=6.8, loc='upper right'); ax[3].set_title('Angle modifier')
save(fig, 'fig_raytrace')

# ---- Fig 8: debi ve duyarlılık ----
H = pd.DataFrame(en['hourly']); S = pd.DataFrame(en['sens'])
fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.2))
for k, ls in zip((1, 2, 3), ('-', '--', ':')):
    h = H[H.day == k]; ax[0, 0].plot(h.hour+.5, h.mfA, ls, marker='o', ms=3, color=C['f'], label=f'Day {k}'); ax[0, 1].plot(h.hour+.5, h.mpA, ls, marker='o', ms=3, color=C['p'], label=f'Day {k} – Scenario A'); ax[0, 1].plot(h.hour+.5, h.mpB, ls, marker='s', ms=3, color='#ff9896', label=f'Day {k} – Scenario B' if k == 1 else None)
ax[0, 0].set_title('(a) Conventional: hourly ṁ'); ax[0, 1].set_title('(b) PTC-assisted: hourly ṁ (A dark, B light)')
for a in ax[0]: a.set_xlabel('Local time (h, block centre)'); a.set_ylabel('ṁ (g/s)'); a.set_ylim(0, 8)
from matplotlib.lines import Line2D
ax[0, 0].legend(fontsize=7); ax[0, 1].legend([Line2D([0], [0], color='k', ls=l_) for l_ in ('-', '--', ':')], ['Day 1', 'Day 2', 'Day 3'], fontsize=7)
sc_ = sorted(S.scale.unique()); us = sorted(S.U.unique())
for a, key, ttl in ((ax[1, 0], 'mp', '(c) ṁ$_{PTC}$ (g/s), Scenario A'), (ax[1, 1], 'etap', '(d) η$_{PTC}$, Scenario A')):
    Z = np.array([[S[(S.scen == 'A') & (S.scale == s) & (S.U == u)][key].iloc[0] for s in sc_] for u in us]); im = a.imshow(Z, origin='lower', cmap='viridis', aspect='auto')
    a.set_xticks(range(len(sc_))); a.set_xticklabels([f'×{s}' for s in sc_]); a.set_yticks(range(len(us))); a.set_yticklabels(us); a.set_xlabel('η$_0$ scale'); a.set_ylabel('U$_L$ (W/m²K)'); a.grid(False); a.set_title(ttl)
    for i in range(len(us)):
        for j in range(len(sc_)): a.text(j, i, f'{Z[i,j]:.2f}', ha='center', va='center', fontsize=7, color='white')
    fig.colorbar(im, ax=a, shrink=.8)
save(fig, 'fig_flow')

# ---- Fig 9: enerji dengesi, verim, ekserji ----
Cl = pd.DataFrame(en['closure']); Cl0 = Cl[Cl.Ut == 0]; EX = pd.DataFrame(en['exergy']); sc = {s['day']: s for s in en['scen']['A']}
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5)); x = np.arange(3); w = .2
ax[0].bar(x-1.5*w, Cl0.Qinc, w, color='#cccccc', label='Incident (both collectors)'); ax[0].bar(x-.5*w, Cl0.Qtank, w, color='#7f7f7f', label='Tank energy gain'); ax[0].bar(x+.5*w, Cl0.A, w, color='#d62728', label='Σ Q, Scenario A'); ax[0].bar(x+1.5*w, Cl0.B, w, color='#ff9896', label='Σ Q, Scenario B')
ax[0].set_xticks(x); ax[0].set_xticklabels(['Day 1', 'Day 2', 'Day 3']); ax[0].set_ylabel('Mean power (W)'); ax[0].legend(fontsize=6.3, ncol=2, loc='upper center'); ax[0].set_ylim(0, 430); ax[0].set_title('(a) Tank energy closure (U$_{tank}$ = 0)')
lab = ['Conv.', 'PTC (A)', 'PTC (B)']; sA = en['scen']['A'][3]; sB = en['scen']['B085'][3]
eta = [sA['etaf'], sA['etap'], sB['etap']]; ax[1].bar(lab, eta, color=[C['f'], C['p'], '#ff9896']); ax[1].set_ylabel('Model-based thermal efficiency'); ax[1].set_ylim(0, .85); ax[1].set_title('(b) Efficiency (aperture-area basis)')
for i, v in enumerate(eta): ax[1].text(i, v+.01, f'{v:.2f}', ha='center', fontsize=8)
exv = [EX[(EX.scen == 'A') & (EX.c == 'f') & (EX.day == 'all')].psi.iloc[0], EX[(EX.scen == 'A') & (EX.c == 'p') & (EX.day == 'all')].psi.iloc[0], EX[(EX.scen == 'B') & (EX.c == 'p') & (EX.day == 'all')].psi.iloc[0]]
ax[2].bar(lab, np.array(exv)*100, color=[C['f'], C['p'], '#ff9896']); ax[2].set_ylabel('Model-based exergy efficiency (%)'); ax[2].set_title('(c) Exergy efficiency')
for i, v in enumerate(exv): ax[2].text(i, v*100+.1, f'{v*100:.1f}', ha='center', fontsize=8)
save(fig, 'fig_energy')

# ---- Fig 10: boyutsuz sayılar ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3)); t_ = D.clock.values; dayv = D.day.values
for key, c, col_ in (('A_f', 'Conventional', C['f']), ('A_p', 'PTC (A)', C['p']), ('B_p', 'PTC (B)', '#ff9896')):
    for k, ls in zip((1, 2, 3), ('-', '--', ':')):
        m = dayv == k
        for a, nm in zip(ax, ('Re', 'Gz', 'Ri')): a.plot(t_[m], np.array(dls[key][nm])[m], color=col_, ls=ls, lw=1, label=c if k == 1 and a is ax[0] else None)
ax[0].axhline(2300, color='k', ls='--', lw=.8); ax[0].text(10.1, 2380, 'Re = 2300', fontsize=6.5); ax[0].set_ylim(0, 2900)
ax[0].set_ylabel('Re'); ax[1].set_ylabel('Gz = Re·Pr·d/L$_b$'); ax[2].set_ylabel('Ri = Gr$_d$/Re²'); ax[2].set_yscale('log')
for a, tt in zip(ax, ('(a) Reynolds number', '(b) Graetz number', '(c) Richardson number')): a.set_xlabel('Local time (h)'); a.set_title(tt)
ax[0].legend(fontsize=7, loc='lower right'); ax[1].legend([Line2D([0], [0], color='k', ls=l_) for l_ in ('-', '--', ':')], ['Day 1', 'Day 2', 'Day 3'], fontsize=7); save(fig, 'fig_dimless')

# ---- Fig 11: LODO ΔT zaman serileri ----
preds = mo['preds']; fig, ax = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
for r, (c, nm, pk) in enumerate((('p', 'PTC-assisted', 'Phys geo'), ('f', 'Conventional', 'Phys lag'))):
    for k in (1, 2, 3):
        m = (D.day == k).values; a = ax[r, k-1]; t = D.clock[m]; a.plot(t, D['dT'+c][m], 'k.-', lw=1, ms=4, label='Measured')
        a.plot(t, np.array(preds[c][pk])[m], color='#d62728', label='Gray-box (LODO)'); a.plot(t, np.array(preds[c]['ML Ridge'])[m], color='#1f77b4', ls='--', label='ML Ridge (LODO)'); a.plot(t, np.array(preds[c]['Hyb Ridge'])[m], color='#2ca02c', ls=':', label='Hybrid (LODO)')
        a.set_title(f'{nm} – Day {k}')
        if r == 1: a.set_xlabel('Local time (h)')
    ax[r, 0].set_ylabel('ΔT (K)')
ax[0, 0].legend(fontsize=7, loc='lower right'); save(fig, 'fig_lodo_series')

# ---- Fig 12: model başarımı ----
dT = mo['dT']; names = ['Train mean', 'ML Ridge', 'ML Ridge+thT', 'ML GBR', 'ML RF', 'Phys qs', 'Phys time', 'Phys geo', 'Hyb Ridge']
labs = ['Day-mean\nbaseline', 'ML\nRidge', 'ML Ridge\n+θ$_T$', 'ML\nGBR', 'ML\nRF', 'Gray-box\nstatic', 'Gray-box\ntime m(t)', 'Gray-box\nθ$_T$', 'Hybrid\nRidge']
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
v = [dT['p'][n]['rmse'] for n in names]; ax[0].bar(range(9), v, color=['#bbbbbb']+['#1f77b4']*4+['#d62728']*3+['#2ca02c']); ax[0].set_xticks(range(9)); ax[0].set_xticklabels(labs, fontsize=6); ax[0].set_ylabel('LODO RMSE of ΔT (K)'); ax[0].set_title('(a) PTC-assisted ΔT')
for i, x_ in enumerate(v): ax[0].text(i, x_+.05, f'{x_:.2f}', ha='center', fontsize=6.5)
to = mo['tout']; tn = ['Naive (Tout=Tin)', 'Linear', 'Ridge', 'SVR', 'RF', 'GBR', 'Gray-box (m(θT)+lag)']; tl = ['Naive\nT$_{out}$=T$_{in}$', 'Linear', 'Ridge', 'SVR', 'RF', 'GBR', 'Gray-box']
v2 = [to[n]['r2'] for n in tn]; ax[1].bar(range(7), v2, color=['#bbbbbb']+['#1f77b4']*5+['#d62728']); ax[1].set_xticks(range(7)); ax[1].set_xticklabels(tl, fontsize=7); ax[1].set_ylabel('LODO R² of T$_{out}$'); ax[1].set_ylim(0, 1.05); ax[1].set_title('(b) Outlet temperature (both collectors)')
for i, x_ in enumerate(v2): ax[1].text(i, x_+.01, f'{x_:.3f}', ha='center', fontsize=6.5)
S_ = pd.read_pickle(f'{OUT}/S_stack.pkl'); TP = pd.read_pickle(f'{OUT}/tout_preds.pkl')
ax[2].scatter(S_.Tout, TP['Linear'], s=8, color='#1f77b4', alpha=.6, label=f"Linear (R² = {to['Linear']['r2']:.3f})"); ax[2].scatter(S_.Tout, TP['Gray-box (m(θT)+lag)'], s=8, color='#d62728', alpha=.6, label=f"Gray-box (R² = {to['Gray-box (m(θT)+lag)']['r2']:.3f})")
lim = [S_.Tout.min()-2, S_.Tout.max()+2]; ax[2].plot(lim, lim, 'k--', lw=.8); ax[2].set_xlabel('Measured T$_{out}$ (°C)'); ax[2].set_ylabel('Predicted T$_{out}$ (°C)'); ax[2].legend(fontsize=7); ax[2].set_title('(c) Predicted vs measured (LODO)')
save(fig, 'fig_models')

# ---- Fig 13: belirsizlik ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
g = D[D.day == 2]; sm = g.dTf.rolling(7, center=True, min_periods=1).mean(); u = un['u_single']
ax[0].plot(g.clock, g.dTf, 'k.-', lw=1, ms=4, label='Measured ΔT, conventional (day 2)'); ax[0].fill_between(g.clock, sm-u, sm+u, color='#1f77b4', alpha=.25, label=f'Thermocouple uncertainty ±{u:.2f} K (u(ΔT), moving mean)')
ax[0].set_xlabel('Local time (h)'); ax[0].set_ylabel('ΔT (K)'); ax[0].legend(fontsize=7, loc='lower right'); ax[0].set_title('(a) Conventional-collector ΔT vs. sensor noise')
items = [('ΔT$_{conv}$', 'dTf'), ('ΔT$_{PTC}$', 'dTp'), ('ΔT$_{PTC}$/ΔT$_{conv}$', 'ratio'), ('ṁ$_{conv}$', 'mf'), ('ṁ$_{PTC}$ (A)', 'mpA'), ('ṁ$_{PTC}$ (B)', 'mpB')]
for i, (lb, k) in enumerate(items):
    b = un['base'][k]; lo, hi = un['both'][k][2]/b, un['both'][k][3]/b; ro, rh = un['random'][k][2]/b, un['random'][k][3]/b
    ax[1].errorbar(1, 5-i, xerr=[[1-lo], [hi-1]], fmt='o', color='#d62728', capsize=3, label='Random + systematic' if i == 0 else None); ax[1].errorbar(1, 5-i+.22, xerr=[[1-ro], [rh-1]], fmt='s', ms=3, color='#2ca02c', capsize=2, label='Random only' if i == 0 else None)
    ax[1].text(hi+.03, 5-i, f"{un['both'][k][2]:.2f}–{un['both'][k][3]:.2f}", fontsize=7, va='center')
ax[1].axvline(1, color='k', lw=.8); ax[1].set_yticks(range(6)); ax[1].set_yticklabels([lb for lb, _ in items][::-1]); ax[1].set_xlim(0.4, 2.3); ax[1].set_xlabel('95% interval relative to nominal value'); ax[1].legend(fontsize=7, loc='lower right'); ax[1].set_title('(b) Monte Carlo, 95% interval')
save(fig, 'fig_uncertainty')

# ---- Fotoğraf paneli ----
if not os.path.exists(os.path.join(BASE, 'veri', 'sistem_bilgisi.docx')):
    print('veri/sistem_bilgisi.docx yok: fotoğraf paneli atlandı')
else:
    from PIL import Image, ImageDraw, ImageFont
    import zipfile, io
    src = os.path.join(BASE, 'veri', 'sistem_bilgisi.docx')      # deney fotoğraflarının bulunduğu Word dosyası
    z = zipfile.ZipFile(src); ph = {i: Image.open(io.BytesIO(z.read(f'word/media/image{i}.jpeg'))).convert('RGB') for i in range(1, 9)}
    ROT = {3: -90, 5: -90, 6: -90, 7: -90, 8: 90}   # kaynak Word dosyasındaki döndürme (derece, saat yönü negatif)
    ph = {i: (im.rotate(ROT[i], expand=True) if i in ROT else im) for i, im in ph.items()}
    def fit_h(im, h): return im.resize((int(im.width*h/im.height), h))
    rowA = [fit_h(ph[3], 420), fit_h(ph[1], 420), fit_h(ph[2], 420)]; wA = sum(i.width for i in rowA)+20*2
    rowC = [fit_h(ph[i], 260) for i in (5, 6, 7, 8)]; wC = sum(i.width for i in rowC)+20*3
    W = max(wA, wC); wide = ph[4].resize((W, int(ph[4].height*W/ph[4].width)))
    H_ = 420+20+wide.height+20+260; P = Image.new('RGB', (W, H_), 'white'); dr = ImageDraw.Draw(P)
    try: fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    except: fnt = None
    def put(im, x, y, lab):
        P.paste(im, (x, y)); dr.rectangle([x, y, x+46, y+46], fill='white'); dr.text((x+10, y+3), lab, fill='black', font=fnt)
    x = 0
    for im, lab in zip(rowA, 'abc'): put(im, x, 0, lab); x += im.width+20
    put(wide, 0, 440, 'd'); x = 0; y = 440+wide.height+20
    for im, lab in zip(rowC, 'efgh'): put(im, x, y, lab); x += im.width+20
    P.save(f'{OUT}/fig_photos.png'); 

print('figures done')
