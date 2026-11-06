"""Shared constants and paths (all scripts read from here).

Notation map (code -> paper): A_F -> A_C, A_P -> A_P, ETA_F -> eta_o,C, ETA_P -> eta_o,P (S1 only), UL -> U_L,
RHO_REFL -> rho_r, W_AP -> W, H_TR -> H, F_TR -> f, D_C -> d_c, L_C -> l_c, L_B -> l_b, D_BI -> d_i, D_BO -> d_o.
Data columns: Tin, Tfo (T_out,C), Tpo (T_out,P), Ta, Tt (T_tank), G, dTf (dT_C), dTp (dT_P), thT (theta_tr), thL (theta_lo)."""
import os
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(BASE, "veri", "deney_verileri.xlsx")
OUT = os.path.join(BASE, "out")                 # intermediate results and figures
os.makedirs(OUT, exist_ok=True)

# --- Experimental dimensions ---
L_B, D_BI, D_BO = 1.8, 0.008, 0.010     # copper U-pipe: length, inner and outer diameter (m)
L_C, D_C = 1.85, 0.06                    # glass tube: length, diameter (m)
W_AP, H_TR = 0.12, 0.08                  # trough aperture width and depth (m)
F_TR = (W_AP / 2) ** 2 / (4 * H_TR)      # focal length, y^2 = 4 f x (y: half aperture) -> 0.01125 m
A_F = L_C * D_C                          # conventional collector area (projected glass tube) = 0.111 m2
A_P = L_B * W_AP                         # PTC-assisted collector area (aperture)             = 0.216 m2
V_TANK, H_TANK, M_TANK = 0.050, 1.0, 50.0
D_TANK = float(np.sqrt(4 * V_TANK / (np.pi * H_TANK)))
A_TANK = float(np.pi * D_TANK * H_TANK + 2 * np.pi * D_TANK ** 2 / 4)
G_ACC = 9.81

# --- Model parameters ---
ETA_F, ETA_P, UL = 0.60, 0.75, 2.0       # optical efficiencies (conventional / PTC, S1 only) and U_L (W/m2K)
RHO_REFL = 0.85                          # reflector reflectance (assumed; sensitivity with 1.0)

# --- Location / tilt / orientation ---
LAT, LON = 39.93, 32.86                  # Ankara
BETA_DEG = 45.0
GAMMA_DEG = 0.0                          # front face towards south
DOY0 = 200                               # day 1 = day of year 200 (a day in July)

# --- Measurement uncertainties ---
U_TC = 0.75                              # K-type thermocouple, °C (limit value)
U_PYR = 1.001                            # pyranometer, W/m2 (limit value)
