# Natural-circulation U-tube evacuated tube collector: effect of a parabolic trough reflector — analysis code

All calculations and figures of the paper are produced from this repository (Python 3.10+; results were generated with Python 3.12 and the package versions in `requirements.txt`).

    pip install -r requirements.txt
    python run_all.py          # outputs are written to ./out (about 1 min)

## Input
- `veri/deney_verileri.xlsx`: three consecutive July days, 10:00–15:00 local time, 10 min interval (3 × 31 rows). One block of columns per day: irradiance in the collector plane (G), ambient temperature, tank temperature, common inlet temperature, conventional-collector outlet, PTC-collector inlet (identical to the common inlet) and PTC-collector outlet.
- `veri/sistem_bilgisi.docx` (optional): read only to build the photo panel (`fig_photos.png`); the step is skipped if the file is absent.

## Scripts (run in this order by `run_all.py`)
| Script | Task | Main output |
|---|---|---|
| `config.py`, `data.py` | Dimensions, model parameters, uncertainties, location; data loading and water properties | – |
| `geometry.py`, `run_geometry.py` | Solar geometry, orientation check, trough geometry, 2D ray tracing | `geometry.json` |
| `run_stats.py` | Paired comparison (t-test, Wilcoxon, block bootstrap), correlations, reduced temperature | `stats.json` |
| `run_energy.py` | Scenarios S1/S2 (heat, flow rate, efficiency), tank closure, sensitivity, exergy, flow-regime indicators | `energy.json` |
| `run_facts.py` | Daily peak values quoted in the figure descriptions | `facts.json` |
| `run_uncertainty.py` | Monte Carlo propagation of measurement uncertainty (±0.75 °C; ±1.001 W/m²) | `uncertainty.json` |
| `models.py`, `run_models.py` | Grey-box model, machine learning, hybrids; leave-one-day-out (LODO) validation | `models.json` |
| `make_figures.py` | All figures (English labels) | `fig_*.png` |

## Assumptions (all in `config.py`)
- The mass flow rate was not measured. Useful heat, efficiency, flow rate and exergy are model-based estimates that depend on the inputs below.
- **S1 (nominal):** optical efficiency eta_o,C = 0.60 (conventional) and eta_o,P = 0.75 (PTC), each collector with its own area and U_L = 2 W/m²K. Since eta_o,P exceeds the geometric limit for the same absorber tube, S1 is an upper-bound scenario.
- **S2 (geometry-consistent):** eta_P = eta_o,C · I(theta_tr), where I is the ray-traced capture fraction (reflectance 0.85; 1.0 for sensitivity). Both collectors share the same tube, so the tube loss conductance U_L·A_C is identical.
- Front face towards the south (azimuth 180°), tilt 45°, trough axis along the tilt direction; Ankara, 39.93° N, 32.86° E.
- Grey-box model: nominal eta_o of S1 is kept fixed and the flow coefficient kappa is fitted, so kappa also absorbs the uncertainty of the optical scale.

## Notation map (code → paper)
| Code | Paper |
|---|---|
| `A_F`, `A_P` | A_C, A_P |
| `ETA_F`, `ETA_P`, `UL`, `RHO_REFL` | eta_o,C, eta_o,P, U_L, rho_r |
| `Tin`, `Tfo`, `Tpo`, `Ta`, `Tt`, `G` | T_in, T_out,C, T_out,P, T_a, T_tank, G |
| `dTf`, `dTp` | ΔT_C, ΔT_P |
| `thT`, `thL` | theta_tr, theta_lo |
| scenario keys `S1`, `S2`, `S2_rho1` | S1, S2 (reflectance 0.85), S2 (reflectance 1) |
| `Gr` (in `energy.json`) | Gr, with ΔT = T_out − T_in |

## Notes
- Random seeds are fixed; the outputs are reproducible with the versions listed in `requirements.txt`.
- No licence file is included; the choice of licence is left to the repository owner.

# Türkçe özet
Makaledeki tüm hesaplar ve şekiller `python run_all.py` ile üretilir (çıktılar `./out`). Debi ölçülmemiştir; verim, debi ve ekserji model tabanlı tahmindir ve varsayımlar `config.py` içindedir.
