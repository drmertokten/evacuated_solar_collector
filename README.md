# Doğal dolaşımlı U-borulu vakum tüplü kolektör: parabolik oluk (PTC) etkisi — analiz kodu

Makaledeki tüm hesaplar ve şekiller bu depodan üretilir (Python 3.10+).

    pip install -r requirements.txt
    python run_all.py          # çıktılar ./out klasörüne yazılır (~1 dk)

## Girdi
- `veri/deney_verileri.xlsx`: Ankara, 19–21 Temmuz 2023, 10:00–15:00, 10 dk aralık (3 × 31 satır). Her gün için bir sütun bloğu: ışınım (kolektör düzleminde), çevre, depo, ortak giriş, geleneksel çıkış, PTC giriş (giriş ile aynı), PTC çıkış.
- `veri/sistem_bilgisi.docx` (isteğe bağlı): yalnızca fotoğraf panelini (`fig_photos.png`) üretmek için okunur; yoksa bu adım atlanır.

## Betikler (sırayla)
| Betik | İş | Ana çıktı |
|---|---|---|
| `config.py`, `data.py` | Boyutlar, model parametreleri, belirsizlikler, konum; veri yükleme ve su özellikleri | – |
| `geometry.py`, `run_geometry.py` | Güneş geometrisi, yönelim çıkarımı, oluk geometrisi, 2B ışın izleme | `geometry.json` |
| `run_stats.py` | Eşleştirilmiş fark (t, Wilcoxon, blok bootstrap), korelasyon, indirgenmiş sıcaklık | `stats.json` |
| `run_energy.py` | Senaryo A/B, depo kapanışı, duyarlılık, ekserji, boyutsuz sayılar | `energy.json` |
| `run_facts.py` | Şekil açıklamalarında kullanılan günlük tepe değerleri | `facts.json` |
| `run_uncertainty.py` | Monte Carlo belirsizlik yayılımı (±0.75 °C; ±1.001 W/m²) | `uncertainty.json` |
| `models.py`, `run_models.py` | Gri-kutu model, makine öğrenmesi, hibrit; gün-dışarıda-bırakma (LODO) | `models.json` |
| `make_figures.py` | İngilizce şekiller (6 panelli Şekil 2, saatlik verim dahil) | `fig_*.png` |

## Varsayımlar (hepsi `config.py` içinde)
- Optik verim η₀ = 0.60 (geleneksel), 0.75 (PTC, Senaryo A); U_L = 2 W/m²K; yansıtıcı yansıtma katsayısı ρ = 0.85.
- Senaryo B: aynı tüp için η_p = η_f·I(θ_T), tüp kayıp iletkenliği iki kolektörde aynı.
- Ön yüz güneye (γ = 0°), eğim 45°, oluk ekseni eğim boyunca; Ankara 39.93°K, 32.86°D.
- Gr–Ra–Nu–h zincirinde karakteristik uzunluk `L_CHAR` = U-boru dış çapı (0.010 m); duyarlılık için diğer uzunluklar `energy.json` içinde `dimless_L` altında.
- Debi ölçülmemiştir; faydalı ısı, verim, debi ve ekserji model çıktısıdır.

## Notlar
- Rastgele tohumlar sabittir; çıktılar yinelenebilirdir.
- Lisans seçimi depo sahibine aittir (bu depoda lisans dosyası yoktur).

# English summary
Reproduces all calculations and figures of the paper comparing a conventional and a parabolic-trough-assisted U-tube evacuated tube collector under natural circulation (Ankara, 3 days, 93 paired observations). Run `python run_all.py`; outputs go to `./out`. Efficiencies, flow rates and exergy are model-based (no flow meter); assumptions are in `config.py`.
