# Baselines: mid-price direction, symbols QQQ, PYPL, NVDA

Train Dec 8-10 2025, validate Dec 11 (LightGBM early stopping only), test Dec 12. Classes: down / flat / up in mid over the horizon.

| Horizon | Model | Accuracy | Macro-F1 | Dir. acc. (both move) | IC (regression) |
|---|---|---|---|---|---|
| 1s | prior | 0.359 | 0.176 | - | - |
| 1s | logreg | 0.548 | 0.530 | 0.534 | 0.044 |
| 1s | lgbm | 0.549 | 0.536 | 0.533 | 0.054 |
| 5s | prior | 0.425 | 0.199 | 0.510 | - |
| 5s | logreg | 0.463 | 0.468 | 0.518 | 0.015 |
| 5s | lgbm | 0.469 | 0.476 | 0.516 | 0.009 |
| 10s | prior | 0.453 | 0.208 | 0.507 | - |
| 10s | logreg | 0.458 | 0.429 | 0.514 | 0.025 |
| 10s | lgbm | 0.461 | 0.415 | 0.513 | 0.009 |

Per-symbol macro-F1 (LightGBM):

| Horizon | QQQ | PYPL | NVDA |
|---|---|---|---|
| 1s | 0.335 | 0.290 | 0.409 |
| 5s | 0.328 | 0.425 | 0.325 |
| 10s | 0.334 | 0.400 | 0.324 |

Test class balance (down / flat / up):

- 1s: 0.33 / 0.36 / 0.31
- 5s: 0.42 / 0.17 / 0.41
- 10s: 0.45 / 0.11 / 0.44
