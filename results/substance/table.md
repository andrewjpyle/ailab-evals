| candidate | n | accuracy | macro-F1 | 95% CI | ECE | error rate | $ / 1k | p50 ms | bar |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| majority | 40 | 0.500 | 0.333 | [0.26, 0.39] | 0.500 | 0.000 | 0.0000 | 0 | baseline |
| random | 40 | 0.575 | 0.575 | [0.42, 0.72] | 0.075 | 0.000 | 0.0000 | 0 | baseline |
| gemma3-27b-local | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.053 | 0.000 | 0.0000 | 4942 | **PASS** ✅ cheapest passing |
| qwen3-32b-local | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.060 | 0.000 | 0.0000 | 3617 | **PASS** |
| claude-haiku-4.5 | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.039 | 0.000 | 0.7701 | 1571 | **PASS** |
| claude-sonnet-4.5 | 40 | 1.000 | 1.000 | [1.00, 1.00] | 0.035 | 0.000 | 1.8878 | 3055 | reference |

**Cheapest model that passes the bar: `gemma3-27b-local`.**
