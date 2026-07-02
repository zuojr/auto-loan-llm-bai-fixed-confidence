# Experiment Results Interpretation

This note summarizes the current PROBE experiment results for later writing.
It is intended as interpretation support, not as a paper section draft.

## Main Mechanism

The current experiments evaluate PDF Algorithm 1 PROBE. The central mechanism is
not to use proxy scores as direct reward estimates or to rank arms by proxy
means. Instead, PROBE uses same-unit reward-proxy covariance to reduce the
variance of reward-mean estimation through OLS residualization. A proxy is useful
when it explains reward fluctuations within the same arm and same sampled unit.

The main reported metric is fixed-confidence stopping time. Smaller stopping
time means fewer online reward pulls are required before the algorithm identifies
the best arm at the target confidence level.

## Real-data Auto-loan Experiment

The main real-data comparison is in:

```text
results/probe_comparison_table.csv
```

Current 3000-repetition results:

| method | mean stop pulls | median stop pulls | q90 stop pulls | saving vs baseline | proxy strength |
|---|---:|---:|---:|---:|---|
| baseline | 105.2k | 105.2k | 106.7k | 0.0% | none |
| qwen_local_icl | 101.7k | 101.8k | 103.5k | 3.3% | weak and uneven |
| gpt_api | 95.6k | 95.7k | 97.6k | 9.1% | moderate |
| qwen_lora_cls_raw | 52.0k | 52.3k | 54.5k | 50.5% | strong on both arms |
| tabpfn | 40.6k | 40.8k | 43.1k | 61.4% | strongest structured proxy |

All methods have empirical correctness 1.0 at stopping in the replay.

The real-data interpretation is straightforward: stopping-time reduction tracks
the strength and stability of the reward-proxy covariance. The Qwen local ICL
proxy gives only a small gain because one policy arm has almost no usable
reward-proxy correlation. GPT API ICL is better but still modest. The raw
classification Qwen LoRA proxy is much stronger, giving about 50% sample saving.
TabPFN is not an LLM method, but as a structured tabular proxy reference it has
the strongest covariance and therefore the largest sample saving.

The key point for writing is that the fine-tuned Qwen result is not better
because its proxy means directly rank the pricing policies. It is better because
its proxy scores are more strongly correlated with same-unit reward noise, which
PROBE converts into lower residual variance.

## Gaussian Simulation

The Gaussian benchmark is in:

```text
results/simulation_probe_gaussian_benchmark.csv
figures/simulation_probe_gaussian_benchmark.png
```

The simulation uses non-rank-preserving proxy means: reward means decrease across
arms, while proxy means increase. This isolates the covariance mechanism from
proxy-mean ranking. The current synthetic-data variant uses the whole historical
pool only to estimate the residual-variance certificate,
`U = vhat / (1 - 2 sqrt(1 / (s - 2)))`, with `s` equal to the historical pool
size. The online per-round reward mean estimate is still computed from fresh
samples. The simulation also uses lighter synthetic-only constants
(`tir_variance_coef=1.1`, `tir_log_coef=2.0`, `delta_coef=2.0`,
`delta_power=1.5`) while keeping `epsilon_r = 2^-r`. The unknown-correlation
Gaussian branch uses a small extra learning-cost multiplier
(`unknown_tir_variance_multiplier=1.1`), so it is not treated as if the oracle
correlation were known for free.

The comparison has three cases:

- `reward_only_probe`: PROBE without a proxy.
- `known_oracle_probe`: known-correlation oracle benchmark with ratio `1-rho^2`.
- `unknown_probe`: PROBE with unknown correlation, using the historical
  residual-variance certificate and fresh online batches for per-round means.

Normalized sample-count ratios:

| rho | reward-only mean pulls | known oracle ratio | unknown PROBE ratio | unknown correctness |
|---:|---:|---:|---:|---:|
| 0.0 | 7.1k | 1.000 | 1.084 | 0.982 |
| 0.2 | 7.2k | 0.944 | 1.016 | 0.990 |
| 0.4 | 6.8k | 0.906 | 0.925 | 0.984 |
| 0.6 | 6.6k | 0.659 | 0.695 | 0.984 |
| 0.8 | 6.5k | 0.412 | 0.438 | 0.990 |
| 0.9 | 6.8k | 0.220 | 0.232 | 0.992 |

This revised setting answers the sample-complexity concern in the first
synthetic experiment. At `rho=0`, the reward-only baseline now stops at roughly
7k online pulls instead of the much larger conservative count from the earlier
online-calibration version, while empirical correctness remains about 98%. The
unknown-proxy version has a modest cost overhead when the proxy is uncorrelated,
so the lighter constants do not create a false proxy advantage in the
zero-correlation case.

As `rho` increases, the unknown-proxy PROBE curve approaches the oracle
residualized benchmark. The remaining gap is the finite-sample cost of using a
learned historical residual-variance certificate rather than being given the
correlation exactly. The main pattern is the transition from a small learning
overhead at `rho=0` to strong variance-driven savings at high correlation.

This figure should be described as showing that PROBE can recover the
known-correlation variance-reduction benchmark in the correctly specified
Gaussian setting, even when proxy means are badly biased for direct arm ranking.

## Certificate Diagnostic Simulation

The certificate diagnostic is in:

```text
results/simulation_probe_correlation_diagnostics.csv
figures/simulation_probe_learning_time.png
figures/simulation_probe_certificate_failure.png
```

The diagnostic compares a naive plug-in residual variance estimate with the
one-sided PROBE upper certificate. At sample size 100, the naive plug-in
undercoverage rate is roughly 54% to 55% across the correlation grid, while the
PROBE certificate undercoverage rate is about 0.3% to 0.5%.

This supports the algorithmic design: directly plugging in an estimated
correlation or residual variance is too aggressive for fixed-confidence BAI,
because it can underestimate the true residual variance. PROBE deliberately uses
an upper confidence certificate. This certificate is conservative, but it
protects the stopping rule and elimination decisions.

The learning-time panel shows the expected covariance-learning pattern: weaker
correlations require more samples before a useful variance-reduction certificate
can be certified. Stronger correlations become certifiable quickly.

## Fitted-model Proxy Simulation

The fitted-model proxy simulation is in:

```text
results/simulation_probe_ml_proxy.csv
figures/simulation_probe_ml_proxy_alignment.png
figures/simulation_probe_ml_proxy_ratios.png
```

Current results:

| proxy model | mean armwise corr. | min armwise corr. | ratio to reward-only |
|---|---:|---:|---:|
| linear | 0.663 | 0.570 | 0.524 |
| decision_tree | 0.661 | 0.586 | 0.543 |
| gradient_boosting | 0.772 | 0.739 | 0.414 |

This simulation replaces hand-specified correlations with proxy scores learned
by fitted models in a nonlinear feature environment. The proxy means are still
not intended to rank arms correctly. The important quantity is the same-unit
correlation between model proxy scores and reward noise.

Gradient boosting has the highest mean and minimum armwise correlation and
therefore the lowest stopping-time ratio. Linear and decision tree proxies are
still useful, but their lower correlations lead to weaker sample savings. This
mirrors the real-data pattern: better proxy alignment translates into lower
PROBE stopping time.

## Combined Interpretation

Across real-data and simulation results, the consistent message is:

1. Proxy means can be biased or non-rank-preserving.
2. PROBE does not rely on proxy means to identify the best arm.
3. The useful signal is within-arm, same-unit reward-proxy covariance.
4. Stronger covariance reduces OLS residual variance.
5. Lower residual variance reduces fixed-confidence stopping time.
6. Conservative variance certificates are necessary for fixed-confidence safety.

The real-data LLM results fit this mechanism. Qwen local ICL has weak proxy
covariance and therefore small gains. GPT API ICL has moderate covariance and
moderate gains. Qwen LoRA raw classification has strong covariance and large
gains. TabPFN, as a structured tabular proxy reference, has the strongest
covariance and the largest sample saving.

The simulation results support the mechanism in controlled settings: Gaussian
experiments isolate the residual-variance factor, certificate diagnostics justify
the conservative PROBE upper bound, and fitted-model simulations show that
learned proxy scores can also produce useful covariance-driven sample savings.
