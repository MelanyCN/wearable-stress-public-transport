# Protocolo de exploración (fijado antes de correr)
Familia de hipótesis: SOLO H1 y H2 (2 hipótesis primarias). Todo lo demás se marca como exploratorio.

**H2 (detector de parada con acelerómetro de muñeca).** Objetivo: detenido (frac_detenido >= 0.5) vs en marcha (< 0.1), verdad = velocidad GPS del celular.
Features solo de ACC (potencias en bandas 0.3-1, 1-3, 3-6, 6-10, 10-16 Hz, desvío por eje), normalizadas por participante. Sin orientación de la mano.
Validación: dejando un participante fuera (LOPO) y una sesión fuera (LOSO). Métrica: AUC. Nulo: permutación de etiquetas dentro de participante. Éxito = AUC LOPO >= 0.75 y p < 0.05.

**H1 (respuesta cardíaca a estallidos de ruido).** Eventos: segundos donde ruido supera su mediana móvil de 60 s en >= 6 dB y está en el 10 % superior de la sesión; separación >= 30 s. 
Resultado primario: ΔFC = media(FC 3..15 s después) - media(FC -10..-2 s antes). Controles: 20 instantes aleatorios por evento, de la misma sesión, a > 30 s de cualquier evento.
Contraste: media(ΔFC eventos) - media(ΔFC controles), IC 95 % por bootstrap de sesiones y p por desplazamiento circular del ruido dentro de cada sesión. Sensibilidad: ajustar por ΔACC.
No se cambian umbrales después de ver resultados; cualquier variante se reporta como secundaria.
