# Equipos sintéticos: análisis

- Generados válidos: 2000; descartados por el validador de Showdown: 0; intentos totales: 2244.
- Reemplazos por equipo: k=1: 918, k=2: 761, k=3: 321

## Uso de especies (fracción de equipos)

| Especie | Reales (`fit`) | Sintéticos | Diferencia |
|---|---|---|---|
| Rillaboom | 64.7 % | 54.1 % | -10.6 |
| Sneasler | 56.5 % | 51.2 % | -5.3 |
| Salamence | 45.9 % | 42.4 % | -3.4 |
| Incineroar | 40.0 % | 37.6 % | -2.4 |
| Basculegion | 34.7 % | 32.6 % | -2.2 |
| Floette-Eternal | 32.9 % | 33.8 % | +0.9 |
| Gholdengo | 27.1 % | 24.9 % | -2.2 |
| Kingambit | 24.7 % | 25.1 % | +0.3 |
| Arcanine-Hisui | 18.8 % | 16.7 % | -2.1 |
| Raichu | 15.9 % | 18.1 % | +2.2 |
| Pelipper | 17.1 % | 15.7 % | -1.4 |
| Golisopod | 15.3 % | 16.7 % | +1.4 |
| Archaludon | 14.7 % | 16.0 % | +1.3 |
| Indeedee-F | 12.4 % | 14.3 % | +2.0 |
| Farigiraf | 11.8 % | 11.2 % | -0.5 |

Especies distintas: 81 en `fit`, 81 en los sintéticos (no puede haber nuevas).

## Parecido con el equipo real más cercano (especies en común, de 0 a 6)

| Grupo | Referencia | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| sintéticos | `fit` | 0 % | 0 % | 0 % | 7 % | 40 % | 53 % | 0 % |
| `eval` (real) | `fit` | 0 % | 0 % | 5 % | 14 % | 29 % | 52 % | 0 % |
| `val` (real) | `fit` | 0 % | 0 % | 0 % | 10 % | 33 % | 57 % | 0 % |
| sintéticos | `eval` | 0 % | 0 % | 6 % | 32 % | 48 % | 13 % | 0 % |

Lectura: si los sintéticos se parecen a `fit` tanto como `eval` se parece a `fit`, cubren el tipo de novedad que el agente encuentra en la evaluación (equipos que comparten casi todas las piezas del metajuego pero en otra combinación).
