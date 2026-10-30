"""Ortak sabitler ve yollar (tüm betikler buradan okur)."""
import os
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(BASE, "veri", "deney_verileri.xlsx")
OUT = os.path.join(BASE, "out")                 # ara sonuçlar ve şekiller
os.makedirs(OUT, exist_ok=True)

# --- Deney boyutları (Word dosyası / kullanıcı düzeltmeleri) ---
L_B, D_BI, D_BO = 1.8, 0.008, 0.010     # bakır U-boru: uzunluk, iç/dış çap (m)
L_C, D_C = 1.85, 0.06                    # cam tüp: uzunluk, çap (m)
W_AP, H_TR = 0.12, 0.08                  # oluk açıklığı ve derinliği (m)
F_TR = (W_AP / 2) ** 2 / (4 * H_TR)      # odak uzaklığı, y^2 = 4 f x (y: yarı açıklık) -> 0.01125 m
A_F = L_C * D_C                          # düz kolektör alanı (cam tüp izdüşümü) = 0.111 m2
A_P = L_B * W_AP                         # parabolik kolektör alanı (açıklık)     = 0.216 m2
V_TANK, H_TANK, M_TANK = 0.050, 1.0, 50.0
D_TANK = float(np.sqrt(4 * V_TANK / (np.pi * H_TANK)))
A_TANK = float(np.pi * D_TANK * H_TANK + 2 * np.pi * D_TANK ** 2 / 4)
G_ACC = 9.81
L_CHAR = D_BO                            # Gr, Ra, Nu, h zincirinde karakteristik uzunluk (U-boru dış çapı, 0.010 m)

# --- Benimsenen model parametreleri ---
ETA_F, ETA_P, UL = 0.60, 0.75, 2.0       # optik verimler (düz/parabolik) ve U_L (W/m2K)
RHO_REFL = 0.85                          # yansıtıcı yansıtma katsayısı (varsayım; duyarlılık için 1.0)

# --- Konum / eğim / yönelim ---
LAT, LON = 39.93, 32.86                  # Ankara (şehir merkezi)
BETA_DEG = 45.0
GAMMA_DEG = 0.0                          # ön yüz güneye (Konya yönü) bakıyor
DOY0 = 200                               # 19 Temmuz 2023 = yılın 200. günü

# --- Ölçüm belirsizlikleri (kullanıcı) ---
U_TC = 0.75                              # K tipi termokupl, °C (sınır değer)
U_PYR = 1.001                            # pyranometre, W/m2 (sınır değer)
