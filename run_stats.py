"""Adım 2: ölçülen veriler üzerinde istatistik (eşleştirilmiş fark, korelasyon, indirgenmiş sıcaklık)."""
import json, numpy as np, pandas as pd
from scipy import stats
from config import *
D = pd.read_pickle(f'{OUT}/D_geo.pkl'); rng = np.random.default_rng(2023)
res = {}
# --- günlük özet ---
res['daily'] = []
for d, g in list(D.groupby('day')) + [('all', D)]:
    res['daily'].append(dict(day=d, G=g.G.mean(), Gmax=g.G.max(), Ta=g.Ta.mean(), Tt0=g.Tt.iloc[0], Tt1=g.Tt.iloc[-1], Tin0=g.Tin.iloc[0], Tin1=g.Tin.iloc[-1],
        dTf=g.dTf.mean(), dTp=g.dTp.mean(), dTf_max=g.dTf.max(), dTp_max=g.dTp.max(), Tfo_max=g.Tfo.max(), Tpo_max=g.Tpo.max(), Tpo_mean=g.Tpo.mean(), Tfo_mean=g.Tfo.mean()))
# --- eşleştirilmiş karşılaştırma ---
diff = (D.dTp - D.dTf).values; n = len(diff)
t, p_t = stats.ttest_rel(D.dTp, D.dTf); w = stats.wilcoxon(D.dTp, D.dTf)
se = diff.std(ddof=1)/np.sqrt(n); ci_t = (diff.mean()-stats.t.ppf(.975, n-1)*se, diff.mean()+stats.t.ppf(.975, n-1)*se)
# otokorelasyon-bilinçli: gün/saat bloklarıyla bootstrap
blk = (D.day.astype(str)+'_'+(D['min']//60).clip(upper=4).astype(int).astype(str)).values; keys = np.unique(blk)
bs = []
for _ in range(5000):
    ks = rng.choice(keys, len(keys)); bs.append(np.mean(np.concatenate([diff[blk == k] for k in ks])))
ci_b = tuple(np.percentile(bs, [2.5, 97.5]))
r1 = np.corrcoef(diff[:-1], diff[1:])[0, 1]; n_eff = n*(1-r1)/(1+r1)
ratio_bs = []
for _ in range(5000):
    ks = rng.choice(keys, len(keys)); idx = np.concatenate([np.where(blk == k)[0] for k in ks]); ratio_bs.append(D.dTp.values[idx].mean()/D.dTf.values[idx].mean())
res['paired'] = dict(n=n, mean_diff=diff.mean(), sd=diff.std(ddof=1), t=float(t), p_t=float(p_t), wilcoxon_p=float(w.pvalue), ci_t=ci_t, ci_block=ci_b,
    dz=float(diff.mean()/diff.std(ddof=1)), n_pos=int((diff > 0).sum()), lag1=float(r1), n_eff=float(n_eff), ratio=float(D.dTp.mean()/D.dTf.mean()), ratio_ci=tuple(np.percentile(ratio_bs, [2.5, 97.5])),
    ci_t_neff=(diff.mean()-stats.t.ppf(.975, n_eff-1)*diff.std(ddof=1)/np.sqrt(n_eff), diff.mean()+stats.t.ppf(.975, n_eff-1)*diff.std(ddof=1)/np.sqrt(n_eff)))
res['paired_day'] = []
for d, g in D.groupby('day'):
    df = (g.dTp-g.dTf).values; res['paired_day'].append(dict(day=d, mean=df.mean(), sd=df.std(ddof=1), n_pos=int((df > 0).sum()), p=float(stats.ttest_rel(g.dTp, g.dTf).pvalue)))
# çıkış sıcaklığı farkı
dTo = (D.Tpo-D.Tfo).values; res['outlet_diff'] = dict(mean=float(dTo.mean()), max=float(dTo.max()), min=float(dTo.min()))
# --- korelasyon ---
D['dTdiff'] = D.dTp-D.dTf
cols = ['G', 'Ta', 'Tt', 'Tin', 'dTf', 'dTp', 'thT']
pear = D[cols].corr(method='pearson'); spear = D[cols].corr(method='spearman')
res['corr_cols'] = cols; res['pearson'] = pear.values.tolist(); res['spearman'] = spear.values.tolist()
res['corr_day'] = {str(d): D[D.day == d][['G', 'dTf', 'dTp']].corr().loc['G', ['dTf', 'dTp']].tolist() for d in (1, 2, 3)}
# --- indirgenmiş sıcaklık: dT/G ~ a - b (Tin-Ta)/G ---
red = {}
for c in ('f', 'p'):
    x = ((D.Tin-D.Ta)/D.G).values; y = (D['dT'+c]/D.G).values
    lr = stats.linregress(x, y); red[c] = dict(slope=lr.slope, intercept=lr.intercept, r2=lr.rvalue**2, p=lr.pvalue)
res['reduced'] = red
json.dump(res, open(f'{OUT}/stats.json', 'w'), default=float)
D.to_pickle(f'{OUT}/D_geo.pkl')
P = res['paired']; print({k: (np.round(v, 3) if not isinstance(v, tuple) else np.round(v, 2).tolist()) for k, v in P.items()})
print(pd.DataFrame(res['paired_day']).round(3).to_string()); print(pear.round(2).to_string()); print(red)
