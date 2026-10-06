# State duration evaluation

| State | Ground truth | Predicted | Absolute error | % of GT |
|---|---:|---:|---:|---:|
| Lying In Bed | 157.0s | 153.4s | 3.6s | 2.3% |
| Sitting On Bed | 82.0s | 84.8s | 2.8s | 3.4% |
| Standing | 46.0s | 17.2s | 28.8s | 62.6% |
| Walking | 57.0s | 61.0s | 4.0s | 7.0% |
| Sitting Outside Bed | 40.0s | 68.0s | 28.0s | 70.0% |
| Out Of Bed | 19.0s | 16.8s | 2.2s | 11.6% |
| Unknown | 0.2s | 0.0s | 0.2s | 100.0% |
| Lying On Floor | 0.0s | 0.0s | 0.0s | n/a |

- Mean absolute error across 8 states: 8.7s.
- Total misattributed time: 34.8s (8.7% of video; sum of absolute state errors / 2).

| Bed summary | Ground truth | Predicted |
|---|---:|---:|
| Time In Bed Sec | 239.0s | 238.2s |
| Time Out Of Bed Sec | 162.2s | 163.0s |
| Exit Count | 2 | 1 |
| Return Count | 2 | 1 |
| Longest Out Of Bed Period Sec | 113.0s | 113.4s |

**Duration sums:** predicted 401.2s; video 401.2s; within 1s: True.

**Caution:** Duration error can hide mistakes. A 5s overcount of walking and a 5s undercount of standing can leave other duration totals unchanged. Read the confusion matrix next to this table.
