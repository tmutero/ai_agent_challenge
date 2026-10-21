# Retrieval evaluation (20 labelled questions, k=5)

| Configuration | recall | precision | hit_rate | mrr | ndcg |
|---|---:|---:|---:|---:|---:|
| dense only | 0.850 | 0.170 | 0.850 | 0.800 | 0.813 |
| hybrid (dense+BM25, RRF) | 0.950 | 0.190 | 0.950 | 0.863 | 0.885 |
| hybrid + lexical rerank | 0.850 | 0.170 | 0.850 | 0.692 | 0.733 |
| shipped config + query planner/routing | 1.000 | 0.200 | 1.000 | 1.000 | 1.000 |

Per-case results for the shipped configuration:

| Question | Hit@k | Reciprocal rank |
|---|---:|---:|
| What is the TER of the Kestrel Bond Fund? | 1 | 1.00 |
| Who manages the global equity feeder fund? | 1 | 1.00 |
| minimum monthly debit order for the money market fund | 1 | 1.00 |
| When was the infrastructure income fund launched? | 1 | 1.00 |
| Which fund is benchmarked against the FTSE/JSE Capped SWIX? | 1 | 1.00 |
| What is the TER of the fund that invests in listed property? | 1 | 1.00 |
| What documents does a foreign national need to open an account? | 1 | 1.00 |
| How often are politically exposed clients reviewed? | 1 | 1.00 |
| How long does onboarding take for a private company? | 1 | 1.00 |
| What is the annual contribution limit on a TFSA? | 1 | 1.00 |
| What drawdown range is allowed on a living annuity? | 1 | 1.00 |
| What is the maximum advice fee on a retirement annuity? | 1 | 1.00 |
| Maximum equity exposure for a moderately aggressive client | 1 | 1.00 |
| What is the limit on hedge funds in a retirement portfolio? | 1 | 1.00 |
| How much of an RA can be in listed property? | 1 | 1.00 |
| When must a client's risk profile be re-assessed? | 1 | 1.00 |
| What happens if an adviser recommends a fund riskier than the client's profile allows? | 1 | 1.00 |
| How much gold exposure is allowed? | 1 | 1.00 |
| Which fund has the highest 1-year return? | 1 | 1.00 |
| Which fund has the lowest TER? | 1 | 1.00 |
