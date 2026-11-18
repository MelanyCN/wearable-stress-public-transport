# Protocolo (fijado ANTES de correr `analisis_avanzado.py`)

**Datos:** `salida/gold.csv` (ventanas de 30 s, paso 15 s, solo bus). A = sin EDA (4 personas); B = con EDA tipo B (3 personas).

**3. EDA y calentamiento.** B' = B excluyendo ventanas a < 300 s del inicio del EDA de la sesión. Sensibilidad: 600 s. EDA transformado con log y normalizado por participante.

**2. Fisiología ajustada por movimiento.** Por participante, regresión robusta (Huber) de FC y ln(RMSSD) sobre log(acc_mag_std_g) y log(acc_dif_media_g); se usa el residual (`hr_res`, `lnrmssd_res`). Conjunto de features "res" = hr_res, lnrmssd_res, temp_media_c, temp_pend_c_min (+EDA en B').

**1. Modelo mixto.** Ventanas SIN traslape (una de cada dos por tramo). Resultados: hr_picos_bpm, ln(rmssd), temp_media_c y (B') ln(eda). Efectos fijos (estandarizados por participante salvo hora): ruido_leq_dbfs, frac_detenido, ln(acc_mag_std_g), hora_local. Intercepto aleatorio por sesión. Se reportan coeficiente, IC 95 % y p; sin ajuste a posteriori de la lista de variables.

**4. Límite superior supervisado.** Random Forest (300 árboles, balanced) y regresión logística; objetivos: gold_trafico, gold_ruido, gold_exposicion. Validación dejando un participante fuera (LOPO) y dejando una sesión fuera (LOSO). Métrica principal: exactitud balanceada. Línea base: permutación del GOLD dentro de cada participante (100 permutaciones, RF 100 árboles) y azar = 1/3.

**Clustering v2.** KMeans / GMM / Ward, k = 2..6, normalización por participante únicamente. Criterio de selección fijo: mayor NMI LOPO contra gold_trafico. Se reporta toda la parrilla (mediana y máximo), no solo la mejor.

**Reglas:** no se añaden ni quitan variables, umbrales ni conjuntos de datos después de ver resultados. Si algo se cambia, se reporta como análisis adicional.
