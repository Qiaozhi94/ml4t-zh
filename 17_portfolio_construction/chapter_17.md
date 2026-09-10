# Chapter 17: Portfolio Construction

Portfolio construction translates forecasts into positions. The predictive models developed in *Chapters 11* to *14* produce expected returns, class probabilities, rankings, or alpha scores. These outputs become investable only after an allocator decides how much capital to assign to each asset, how much risk to concentrate, how often to rebalance, and which constraints must remain binding while the portfolio is held.

This translation is itself a model. It has inputs, assumptions, hyperparameters, and failure modes. A strong predictive signal can underperform when naive sizing concentrates risk, raises turnover, or amplifies estimation error. Conversely, a disciplined allocator can preserve the value of weak but diversified signals by controlling leverage, covariance exposure, and implementation costs.

After completing this chapter, you will be able to:

- Formalize the allocation problem in terms of inputs, constraints, and decision variables.
- Evaluate allocators across multiple dimensions rather than relying on the Sharpe ratio alone.
- Implement baseline heuristics and understand why they are surprisingly hard to beat.
- Apply mean-variance optimization with robust fixes and diagnose the Markowitz curse.
- Construct hierarchical risk-parity portfolios that avoid matrix inversion and yield stable allocations across regimes.
- Compare allocators under a common protocol without introducing a new source of overfitting.
- Explain how deep learning enables end-to-end portfolio weight prediction.

The chapter begins by defining the allocation problem and its inputs, then introduces a structured workflow. *Section 17.3* develops the multi-dimensional evaluation framework. *Sections 17.4–17.6* present classical allocators in order of complexity: baseline heuristics, mean-variance optimization with robust fixes, and hierarchical risk parity. *Section 17.7* compares them under a common protocol. *Section 17.8* turns to end-to-end deep learning, where a single network maps features to positions under a portfolio-level objective.

## 17.1 Defining the allocation problem

Portfolio allocators combine forecasts, risk estimates, and constraints to translate signals into posi𝑠௧, asset rankings, or return forecasts from *Chapters 11* to *14*. Allocators rarely use raw model output tion weights. The first input is a view on **expected returns**, represented in this book by model scores

directly. Signals are typically standardized, winsorized, ranked, or decayed across horizons to prevent extreme predictions from dominating the portfolio. Rank transformations usually produce more even position sizes, while z-scores allow conviction to affect sizing but can concentrate risk in names with the most extreme forecasts. Even modest distortions in relative expected returns can materially change allocations (Lopez de Prado, 2016).

The second input is a description of **expected risk**. Even simple diversification rules require volatility 𝛴௧. Its diagonal records individual asset risk; its off-diagonal elements describe how positions reinforce estimates, while mean-variance and risk-budgeting methods rely explicitly on a covariance matrix,

or offset one another.

The third input is the investor’s definition of **admissible risk**. Volatility targets, leverage caps, position limits, concentration rules, drawdown tolerances, and liquidity requirements define the feasible region from the start. The same forecast can justify a very different portfolio for a long-only pension fund than for a leveraged market-neutral mandate. Constraints are therefore part of the allocation model, not an implementation afterthought.

### The decision problem – Weights, exposure, and timing The allocator’s output appears simple: a vector of portfolio weights, 𝑤௧= (𝑤௧ 1, 𝑤௧ 2, … , 𝑤௧ ே) drawn from

a feasible set 𝑊௧, where each 𝑤௧ ௜ denotes the fraction of capital assigned to asset 𝑖 at time 𝑡. Formally, the allocator aims to maximize investor utility 𝑈:

𝑤௧∈argmax ೟ܷ(𝑤Ǣ ݏ௧, ߑ௧, 𝑤௧ିଵ) ௪∈ࣱ But this weight vector 𝑤௧ embeds several distinct decisions:

**Long-only versus long-short**. A long-only constraint, 𝑤௜≥0, substantially shrinks the feasi-

• ble region and often prevents the allocator from fully expressing relative views. A long-short mandate can hedge out market exposure and express cross-sectional signals more sharply, but it also introduces leverage, stock-borrow costs, and practical constraints on the short book. Even the scaling of the long and short legs matters. French (2024) argues that equal-notional dollar neutrality is usually not the best way to size long-short factor portfolios. In his sample, short legs are typically more volatile and less diversified than long legs, so volatility-matched scaling of the short side improves both absolute and risk-adjusted performance and yields **Gross and net exposure**. Gross exposure, ∑|ݓ௜| ,measures the total amount of risk taken, remarket exposure closer to zero than standard equal-notional construction. gardless of direction. Net exposure, ∑ݓ௜ ,measures directional bias. A market-neutral portfolio ௜ • ௜

may target zero net exposure while still maintaining substantial gross exposure through offsetting long and short positions. This distinction becomes especially important once leverage, margin requirements, or derivatives enter the problem.

- **Leverage**. Whenever gross exposure exceeds 100%, the portfolio is leveraged, whether through borrowing, derivatives, or embedded financing. Leverage can improve the use of weak but diversified signals, but it also amplifies estimation error, drawdowns, and implementation risk. Position sizing and leverage, therefore, cannot be separated.
- **Rebalance frequency**. More frequent rebalancing keeps the portfolio aligned with current signals but increases turnover and trading costs. Less frequent rebalancing reduces implementation costs but allows the actual holdings to drift away from target weights. There is no universal optimum. The right schedule depends on signal decay, market frictions, liquidity, and the rate at which the covariance structure itself changes.

### Constraints as first-class modeling choices

Constraints are modeling choices with direct economic consequences: straint such as 𝑤௜≤𝑤max protects the portfolio not only from estimation error but also from

- **Position limits** prevent excessive concentration in individual names. A maximum-weight con-

liquidity risk, if properly calibrated: a position in a small-cap stock that appears attractive may be difficult to enter or exit at scale.

- **Sector and factor caps** limit aggregate exposure to correlated groups of assets. Without such caps, an optimizer may present a highly concentrated macro or style bet as if it were diversified stock selection. new െݓ௜ ∑|ݓ௜ old| ൑ܶ Ǥ Once turnover is constrained, portfolio choice becomes path-depen-
- **Turnover limits** make the path from yesterday’s holdings to today’s target explicit: ௜

dent. Today’s optimal allocation depends not only on today’s signals, but also on the inherited portfolio and the cost of moving away from it.

- **Cash and leverage requirements** matter for the same reason. Minimum cash balances may be needed to meet redemptions, margin calls, or internal liquidity policies. Conversely, leverage caps limit how aggressively the allocator can scale a signal, even when the signal looks attractive in isolation.

The mathematical form of a constraint also changes behavior:

- A **hard cap** at 5% creates a literal boundary in the feasible region
- A **soft** **concentration penalty** merely makes large positions progressively more expensive

Hard constraints are appropriate when a limit reflects a true operational, regulatory, or mandated restriction. Soft penalties are better interpreted as preferences about diversification, turnover, or stability.

### How signal quality becomes portfolio performance

The **Fundamental Law of Active Management** (FLAM) (Grinold and Kahn, 2000) links the quality of the ML signals produced by *Chapters 11* to *14* to the portfolio performance realized here: IR ≈IC × √BR Where:

- IC is the information coefficient (see *Chapter 7*)
- BR is breadth, measured as the number of independent bets
- IR is the information ratio, the benchmark-relative performance per unit of benchmark-relative risk

When the benchmark is cash or a passive allocation, the IR is closely related to the Sharpe ratio; when the benchmark is an active reference portfolio, the distinction becomes more important.

The ETF case study’s linear specification with the highest validation IC delivers a daily IC of about positions, **nominal breadth** is 𝐵= 20 × 12 = 240, which implies 𝐼≈0.054 × √240 ≈0.84 (before 0.054. On its own, this seems low. But with a universe of roughly 100 ETFs rebalanced monthly to 20

costs). Breadth compensates for weak individual forecasts: a small edge, repeated across many assets and many periods, can still support an attractive allocator.

The relevant concept, however, is **effective breadth**, not the naive count of assets times rebalancing largely driven by roughly five common risk factors, effective breadth is closer to 5 × 12 = 60, which periods. Correlated assets do not provide independent opportunities. If those 20 ETF positions are yields IR ≈0.054 × √60 ≈0.42. This gap between nominal and effective breadth is one of the main

reasons realized performance falls short of naive FLAM projections. IC equals the regression 𝑅2. An IC of 0.03, therefore, explains only about 0.09% of the cross-sectional In a univariate cross-sectional weighted regression with an intercept, the squared weighted Pearson

variation in returns: in terms of variance explanation, the raw signal is extremely weak. Its potential economic value comes not from single-name predictability, but from repeated application across many assets and periods: if the signal is sufficiently stable, diversified, and inexpensive to trade, portfolio construction can aggregate a small predictive edge into an investable strategy.

The FLAM should therefore be treated as a **mental model** and as an **order-of-magnitude calibration**. It assumes unbiased forecasts, a correct covariance model, independent bets, and negligible trading frictions. In practice, none of these conditions holds exactly. Still, the approximation provides the right intuition: doubling IC has the same effect as quadrupling breadth, and both benefits can easily be overwhelmed by high correlation among bets, poor risk estimation, or implementation costs. Allocation quality can amplify a weak signal, but it can also destroy it.

**Implementation**: `02_mean_variance_optimization` traces the efficient frontier and contrasts five objective functions (max-Sharpe, min-vol, ERC, inverse-volatility, equalweight) on a 30-ETF universe, showing how each maps a noisy expected-return vector and covariance matrix into very different allocations.

## 17.2 A portfolio construction workflow

Portfolio construction requires a specification before it can use an optimizer. The allocator term sheet records the objective, inputs, constraints, rebalancing rule, cost treatment, and evaluation plan. It prevents allocation from becoming an unlogged search layer after model selection. A structured workflow makes these allocation choices explicit and testable. *Figure 17.1* summarizes it: signals feed a risk-and-constraint layer, the allocator turns them into positions, and diagnostics feed back into the next specification pass.

![Figure 17.1](assets/figure_17_1.png)

*Figure 17.1: Portfolio-construction workflow. Allocation sits between signal generation and implementation, with diagnostics feeding back into the next specification pass*

### The allocator term sheet

Before implementing any allocation method, the specification should be documented in a compact term sheet that records the objective, inputs, constraints, trading protocol, and evaluation design.

The **objective function** should be explicit enough that two practitioners would produce the same optimizer from the same description. Mean-variance utility, maximum Sharpe, minimum variance, and risk parity all imply different trade-offs, and those trade-offs should not remain implicit in notebook code.

The same discipline applies to the **inputs**:

- **Expected returns** may come from predictive models developed in *Chapters 11-14*, historical averages, or equilibrium-style priors.
- **Covariance** may be estimated with a sample matrix, a shrinkage method, or a factor model.
- Each input also carries a **horizon choice**, and those choices shape behavior at least as much as the optimization routine itself.
- **Constraints** belong on the same term sheet with their numerical values and enforcement rules: long-only restrictions, maximum position sizes, sector caps, turnover penalties, leverage limits, and any distinction between hard feasibility constraints and soft objective penalties.

The documentation should also contain the **rebalancing protocol** and the **evaluation plan**. Rebalancing frequency, drift thresholds, and the treatment of transaction costs determine whether the allocator is testing a realistic implementation or an idealized exercise.

Evaluation rules determine which metrics, subperiods, and regime slices will be reported. Once these choices are written down in advance, poor performance can be traced to a forecast, a covariance estimate, a constraint set, or an execution assumption.

### Estimation windows and leakage prevention

mated on day 𝑡 with data through day ݐ൅݇ is enough to produce an invalid backtest. Three methods Allocation can leak future information just as easily as signal generation. A covariance matrix esti-

protect against leakage:

- **Strict temporal ordering**: expected returns, covariance estimates, risk budgets, and execution assumptions must all be constructed only from information available when the portfolio is formed.
- **Matched estimation windows**. An allocator that uses next-week forecasts and a 10-year covariance matrix combines a short-horizon view of expected return with a long-horizon view of risk structure. That mismatch can be defensible, but it should be a conscious modeling choice rather than an accident of default settings. Short-horizon signals usually call for shorter, more adaptive risk estimates, even though that choice increases estimation noise.
- **Out-of-sample validation** of the allocator itself. Shrinkage intensity, risk-aversion parameters, turnover penalties, and constraint values are hyperparameters in exactly the same sense as the model choices in the predictive pipeline. If those levers are tuned by repeatedly inspecting the final test sample, the allocator has been fitted to the answer key, even if the forecasting model was held fixed.

## 17.3 Portfolio evaluation metrics

*Chapter 16* established the core backtest report: return, volatility, Sharpe, drawdown, turnover, and gross versus net performance. Allocator comparison requires an additional layer. Once the forecast stream and backtest protocol are held fixed, the main questions become benchmark-relative performance, concentration, diversification, and trading stability.

### Benchmark-relative performance

The most important addition is the distinction between absolute and active performance. A new allocator is usually judged against a benchmark allocator rather than against cash. Equal weight, inverse volatility, or a strategic policy portfolio all serve as defensible references. In that setting, the active = 𝑟௧ portfolio −𝑟௧ key quantities are **active return**, **tracking error** (TE), and the **information ratio** (IR): 𝑟௧

benchmark TE = √൫ݎ௧ active)

IR = ܧ[ݎ௧ active]

 TE

The Sharpe ratio remains useful, but it no longer answers the full question. An allocator can improve Sharpe simply by lowering total risk, while still failing to add value relative to a simple benchmark. The information ratio asks the more relevant question: given the benchmark already available, did the new allocator earn enough active return to justify its deviations? Tracking error makes the source of allocator risk visible. A high tracking error may reflect deliberate, well-paid active bets, or it may reflect unnecessary churn and unstable positions. That is why benchmark-relative metrics should be read together with concentration and trading metrics rather than in isolation.

For benchmarked portfolios, **active share** is often informative as well: Active Share = 1 portfolio െݓ௜ 2 ∑|ݓ௜ benchmark|

௜

Active share measures how different the holdings are from the benchmark in weight space. It does not replace tracking error because it ignores covariance, but together the two metrics distinguish a portfolio that is genuinely different from one that is only cosmetically active.

### Concentration and diversification

Allocators also differ in how they distribute capital and risk across the investable universe. The **Herfindahl-Hirschman Index**: ே HHI ൌ෍ݓ௜ 2 ௜ୀଵ

is a compact summary of **capital concentration**. Its reciprocal, often interpreted as the **effective number of bets** , gives an intuitive measure of how many meaningful positions the portfolio is really holding. Two allocators can hold the same number of names and still look very different on this metric if one spreads risk broadly while the other concentrates capital in a few dominant positions.

Capital concentration is only part of the picture. **Risk contribution** asks how much each position contributes to total portfolio volatility: (ߑݓ)௜ ܥ௜ൌݓ௜⋅ ߪ௣ 

This quantity is especially important for allocators advertised as balanced or diversified. Equal capital weights do not imply equal risk contributions, and inverse-volatility weights do not imply equal portfolio-risk contributions once correlations are present. A portfolio that looks diversified in capital space can still be dangerously concentrated in risk space.

### Stability and implementation metrics

Allocator comparisons also need metrics that connect portfolio design to implementation. *Chapter 16* introduced **turnover** as a trading-cost diagnostic; here, it takes on an additional interpretation. High turnover can mean that the allocator is unstable, reacting aggressively to small changes in forecasts or covariance estimates. That instability matters even before costs are modeled in detail because it signals that the portfolio may be fitting noise rather than a durable structure. **Realized gross exposure and leverage** stability serve a similar role. A strategy with a modest average leverage target can still be fragile if it experiences sharp swings in gross exposure as market conditions change. Monitoring the time path of leverage, sector weights, or factor exposures often reveals behavior that a single summary statistic hides. **Diversification** has a ceiling when assets share the same risk driver. In the stylized case where 𝑛 assets have identical standalone Sharpe ratio 𝑠, identical volatility, and constant pairwise correlation 𝜌, the

equal-weight portfolio has: ௡ൌݏ√݊1 + (݊ െ1)ߩ

As 𝑛 grows, this converges to 𝑠√ߩ. The calculation is not a trading rule; it is a diagnostic. When cor-

relations are high, adding more assets from the same opportunity set contributes little independent risk. The allocator must either find less correlated bets, reduce concentration in the common factor, or accept that diversification cannot rescue the strategy.

This is also where *Chapter 16*’s **regime slicing** still matters, but only as a secondary view. The allocator-specific metrics above should be established first in aggregate. If an allocator then appears acceptable, the *Chapter 16* regime framework can be reused to ask whether concentration, leverage, or benchmark-relative losses deteriorate in stress periods.

### Evaluating the risk model through the portfolio

Because covariance estimates drive many allocators, they should be judged by portfolio outcomes rather than by abstract matrix fit alone. A practical test is to form comparable minimum-variance or risk-budgeted portfolios using competing covariance estimates and then compare realized out-ofsample risk. If a covariance model is genuinely better for allocation, it should deliver lower realized variance or more stable risk contributions under the same target exposure.

A portfolio-based criterion can be more useful than a generic matrix norm because **the allocator never trades the covariance matrix** directly. It trades the portfolio implied by that matrix. *Chapter 19* returns to this issue when tail risk and stress constraints become explicit; the principle applies here as well – allocator inputs should be judged by the portfolios they generate.

**Implementation**: `01_portfolio_metrics` runs the full `ml4t.diagnostic` evaluation suite on the ETF case-study allocation backtest, resolved at runtime via `resolve_best_` `backtest_runs(...)` so the metrics always reflect the allocator that achieves the best Sharpe ratio during the validation period. The strategy series is benchmarked against SPY across alpha, beta, tracking error, information ratio, up/down capture, drawdown, VaR/ CVaR, rolling metrics, and stress-period comparisons.

## 17.4 Defining baseline allocators

Before reaching for sophisticated optimization, consider what simple allocation rules can achieve. Equal weighting, inverse volatility, and score-based weighting require minimal estimation, are robust to input errors, and frequently outperform complex alternatives out of sample. These heuristics serve as rigorous baselines: any proposed allocator must improve on them to justify its complexity.

*Figure 17.2* illustrates the methods discussed as a trade-off between information use and estimation sensitivity: methods relying on fewer inputs are typically more robust, while methods exploiting richer information pay for that expressiveness with increased noise. Risk parity and hierarchical risk parity occupy a middle ground: both consume the full covariance matrix but impose structural constraints that dampen sensitivity to estimates.

![Figure 17.2](assets/figure_17_2.png)

*Figure 17.2: Allocation methods trade information use against estimation sensitivity*

The equal-weight portfolio assigns identical weight to each of 𝑁 assets: Equal weight – The null model

𝑤௜= 1ܰ

This seemingly naive approach has surprising virtues. It requires no estimation of expected returns or covariance, eliminating the primary source of optimization error. It maintains maximum diversification across available assets. Empirically, equal-weight portfolios are famously hard to beat out of sample because **estimation error often overwhelms the theoretical gains from optimization** (DeMiguel, Garlappi, and Uppal, 2009).

Optimization trades diversification for concentration based on estimated inputs. When those estimates are noisy (and they always are), the concentration may add more risk than the “optimal” tilt adds return. In competitive markets where no asset offers an obvious free lunch, equal weighting’s agnosticism is wisdom.

### Inverse volatility weighting

Inverse volatility weighting scales positions inversely to an estimate of each asset’s ex ante volatility: 1Ȁߪ௜ 𝑤௜= ∑ 1 Ȁߪ௝ ே ௝ୀଵ  where 𝜎௜ is the volatility estimate for asset 𝑖. Simple implementations base 𝜎௜ on recent realized returns.

More generally, it can be any forward-looking estimate, including an exponentially weighted estimate, portfolio equalizes standalone volatility exposure, 𝑤௜ߪ௜, across assets. This approach sits between or a GARCH-style conditional volatility forecast. More volatile assets receive smaller weights, so the

equal weighting and full optimization. It uses volatility estimates but ignores expected returns and, in its plain form, correlations.

**Note the distinction from risk parity**: inverse volatility equalizes standalone volatility exposure, not marginal contribution to total portfolio risk. When asset correlations differ, some positions can still contribute disproportionately to portfolio volatility. The appeal is in estimation reliability: volatility is sufficiently persistent that recent or model-based volatility estimates often contain useful information, whereas short-window expected-return estimates are usually dominated by noise.

### Score-weighted portfolios

When the predictive models from *Chapters 11-14* produce alpha scores or expected return forecasts, ݏ௜ one natural allocation is to weight proportionally: 𝑤௜= ∑ |ݏ௝| ே ௝ୀଵ  where 𝑠௜ is asset 𝑖’s score. Positive scores receive long positions; negative scores short positions. This

formula normalizes gross exposure but not net exposure. Unless scores sum to zero by construction, the portfolio will have directional bias. For dollar-neutral portfolios, normalize long and short legs separately.

Score-weighted allocation works best when the predictive model already accounts for risk, and the correlation structure is relatively uniform. It fails when extreme scores concentrate exposure in correlated assets. Combining score-weighting with position limits or volatility targeting can address this weakness.

### Conformal position sizing

Score-weighted portfolios use the *magnitude* of a point prediction to size a position. Conformal posiany forecaster in a calibration step that produces a prediction interval [ݕො௜െݍ௜ǡ ݕො௜൅ݍ௜] whose width tion sizing uses the *uncertainty* of that prediction instead. Conformal prediction (*Chapter 11*) wraps 𝛥௜ൌʹݍ௜ is calibrated so the realized return falls inside the interval at the chosen coverage level (80%

in our implementation). Narrower intervals indicate higher model confidence for that asset at that time. Inverse-width weights translate that confidence directly into a position size: 1Ȁ߂௜ǡ௧ 𝑤௜ǡ௧= ∑ 1 Ȁ߂௝ǡ௧ ௄ ௝ୀଵ  For inverse-width weights to differ from equal weights, the widths 𝛥௜ must vary across assets. Pooled

split-conformal calibration produces a single quantile per fold and would collapse the rule to equal **formal calibration** under strict walk-forward ordering: fold-𝑘’s width uses only validation folds that weight. The `07_conformal_position_sizing` notebook therefore uses **per-symbol Mondrian split-con-** end before fold 𝑘 begins, and the chronologically earliest fold falls back to pooling residuals from all *other* folds for that symbol. On the holdout window, each symbol’s width is the (ͳ െߙ)-quantile of |ݕtrue െݕscore| over the full validation history, with the trailing ℎ timestamps embargoed (where ℎ is the

forward-return horizon in data-step units) so the calibration cannot peek across the holdout boundary.

We compare the conformal allocator against the cross-stage validation baseline (the highest-Sharpe non-conformal configuration on the same label, pooling signal, allocation, and risk-overlay stages) across seven of nine engine case studies in *Table 17.1*. Two case studies are excluded for structural reasons. The S&P 500 options case study is excluded because its cohort engine realizes PnL on overlapping 30-day option positions, which the per-prediction Mondrian residual definition does not extend to. NASDAQ-100 microstructure is excluded because its selected configuration is a signal-stage slot strategy whose slot mechanism is itself the position-sizing rule, leaving no cross-sectional allocation step for the conformal inverse-width overlay to replace; for the same structural reason, its allocation stage is reported not applicable in *Chapter 20*. On validation, conformal sizing trails the baseline in all seven cases, by margins ranging from negligible to about 1.2 Sharpe, because the baseline is selection-maximized in-sample. On holdout, conformal sizing improves the comparison on four of six cases with a non-conformal comparator on the same prediction set, though only one move is material: it lifts Sharpe substantially on `sp500_equity_option_analytics` (−0.73 → +0.08) and is essentially flat on `etfs` (+1.00 → +1.02), `crypto_perps_funding` (−0.13 → −0.13), and `us_firm_characteristics` (+1.77 → +1.79). It trails `cme_futures` and `us_equities_panel` by 0.24 to 0.41 in Sharpe ratio.

The `fx_pairs` row illustrates the flat-signal case. The cross-stage baseline (HRP top-5 on the linear-ridge predictions) barely clears zero on validation (+0.05), and conformal sizing on the same thin signal is worse (−0.11): with no real dispersion in predictive confidence to exploit, the inverse-width weights only add noise. Only that baseline strategy was taken to the holdout, where it earns +0.19 yet remains statistically indistinguishable from an equal-weight book of the same currencies. Conformal sizing carries no embedded direction-of-effect (it amplifies whatever per-symbol confidence the underlying model reports), so it can neither rescue nor distort a signal that has no resolved edge to begin with.

| Case study | Label | Val baseline → conformal | Holdout baseline → conformal | Δ holdout |
| --- | --- | --- | --- | --- |
| etfs | fwd_ret_21d | +1.36 → +1.04 | +1.00 → +1.02 | +0.02 |
| crypto_perps_funding | fwd_ret_24h | +2.57 → +2.41 | −0.13 → −0.13 | +0.00 |
| sp500_equity_option_analytics | fwd_ret_risk_adj_5d | +2.39 → +1.23 | −0.73 → +0.08 | +0.81 |
| us_firm_characteristics | fwd_ret_1m_win | +2.75 → +2.73 | +1.77 → +1.79 | +0.02 |
| cme_futures | fwd_ret_5d | +1.36 → +1.07 | +1.11 → +0.70 | −0.41 |
| us_equities_panel | fwd_ret_5d | +2.03 → +1.56 | −0.49 → −0.73 | −0.24 |
| fx_pairs | fwd_ret_21d | +0.05 → −0.11 | +0.19 → - | n/a |

*Table 17.1: Conformal sizing (the uncertainty-driven inverse-width overlay defined above) versus the crossstage baseline across seven engine case studies. Sharpe ratios are queried from each case study’s registry. db. The fx_pairs holdout has no conformal comparator because only the baseline HRP strategy was taken to the holdout*

The pattern is consistent with the conformal allocator’s mechanism: a position-sizing rule driven by calibrated uncertainty trades some in-sample Sharpe for tighter alignment between sizing and model confidence. It does not substitute for a properly tuned risk overlay, and it cannot rescue a selection signal whose predictive content decays out of sample.

### Risk parity – Equal risk contribution

**Risk parity allocates to equalize each asset’s contribution to portfolio risk** (Maillard, Roncalli, and Teiletche, 2008; for an accessible introduction, see Hurst, 2010). Traditional allocation concentrates rather than bonds. Risk contribution for asset 𝑖 is: risk in the most volatile assets: a 50/50 stock-bond portfolio typically carries most of its risk in equities (ߑݓ)௜ ܥ௜ൌݓ௜⋅ ߪ௣ 

tween minimum variance and equal weighting, with portfolio volatilities satisfying 𝜎ெ௏≤𝜎ாோ஼≤𝜎ଵȀ௡. Risk parity seeks weights to achieve equal risk contribution (ERC) across assets. Risk parity lies be-

Risk parity proponents frame ERC as a practical compromise: less risk-concentrated than minimum variance, more risk-balanced than equal weighting, and often competitive on the diversification-turnover trade-off.

Risk parity requires leverage to achieve competitive returns; whether leveraged risk parity outperforms depends on financing costs and the stability of correlations.

### Volatility targeting

Volatility targeting is an *overlay* that wraps any base allocator, not a standalone method. First, construct ߪtarget a base portfolio using a method of your choice, then scale its exposure to achieve a target volatility: 𝑤scaled = ⋅𝑤base ߪportfolio

If the base portfolio has an annualized volatility of 15% and the target is 10%, scale all positions by 0.67. When realized volatility exceeds the target, reduce exposure; when volatility falls, increase it. Volatility targeting produces a more stable risk profile over time, naturally reduces exposure during high-volatility periods, and prioritizes explicit control over portfolio risk, subject to the accuracy of estimates. The main design choice is how to estimate volatility: trailing windows are simple but slow to adapt; EWMA or GARCH models respond faster but introduce additional parameters.

The preceding allocation rules are deliberately simple. Equal weighting ignores both forecasts and covariance. Score-based weighting uses forecasts but does not define an explicit growth or utility objective. Volatility scaling and risk parity incorporate risk, but they largely avoid expected-return estimates. These heuristics are often robust precisely because they ask little of the data. The Kelly criterion introduces the missing optimization principle: when a signal implies a positive edge, how much capital should be committed to it?

### The Kelly criterion – Optimal sizing from signal quality The Kelly criterion starts from a repeated-betting problem. Suppose you stake a fixed fraction 𝑓 of

current wealth each round. Kelly’s key insight is that the relevant objective is not expected wealth after one bet, but the expected growth rate of wealth over many repeated bets. Because wealth compounds, For a binary bet that gains 𝑏 on a win and loses 𝑓 on a loss, where 𝑏 denotes the payoff odds and 𝑝 this objective is the expected change in log wealth.

the probability of winning, the expected log-growth rate is: 𝑔(݂) ൌ݌log(1 +ܾ݂ ) + (1 െ݌)log(1 െ݂ )

Maximizing this expression gives the familiar **edge-over-odds Kelly fraction**: ݌−(ͳ −݌)ܾ 𝑓∗=



The numerator is the expected advantage of the bet, while the denominator converts that edge into the appropriate fraction of wealth to stake. Betting too little leaves long-run growth on the table, while betting too much increases the probability of large losses enough to reduce expected log growth, even when the bet has positive expected value. If the edge is negative and shorting the bet is not allowed, **The same logic extends to financial returns**. For one risky asset with excess return 𝑟, expected excess the constrained Kelly stake is zero. return 𝜇, and variance 𝜎2, the expected log-growth objective is:

𝔼[log(1 +݂ݎ)]

For small returns, a second-order approximation gives: ߤ−1 𝔼[log(1 +݂ݎ)] ≈݂ 2ߪ2 2݂

𝑓∗= ߤ So the growth-maximizing exposure is: ߪ2 This expression is useful because it links signal quality directly to position size. A larger expected excess return increases the optimal exposure, while greater variance reduces it. The criterion, therefore, does not merely ask whether a signal is positive. It asks how large the edge is relative to the risk of harvesting it.

In a multi-asset setting, the same quadratic approximation yields: 𝔼[log(1 ൅ݓ⊤ݎ)] ൎݓ⊤ߤെ1 2 ݓ⊤ߑݓ

where 𝑤 is the vector of risky-asset weights, 𝜇 is the vector of expected excess returns, and 𝛴 is the

covariance matrix. The unconstrained Kelly allocation is then: 𝑤∗ൌߑ−1ߤ

The same result reappears in the next section on mean-variance optimization. Under the standard approximation, the Kelly portfolio points in the same direction as the tangency portfolio. Both reward higher expected excess returns and penalize covariance risk. The difference is that a normalized tangency portfolio typically fixes the scale of risky exposure, while the Kelly solution also determines the leverage implied by the estimated opportunity set. In that sense, Kelly is an integrated allocation criterion rather than just a forecasting rule.

Kelly gives a principled answer to the sizing problem when the return distribution is known (as in sively with the estimated edge, even modest overestimation of 𝜇 can produce excessive leverage and many games), but in trading applications, the inputs are estimated. Because full Kelly scales aggres-

severe drawdowns. The criterion is therefore highly sensitive to the same estimation errors that make forecast-based (classical) mean-variance optimization unstable.

Practitioners usually treat full Kelly as an upper bound rather than as a literal operating rule. **Fractional Kelly** scales the theoretical allocation by a constant, such as one-half or one-quarter, to reduce leverage, drawdowns, and sensitivity to estimation error. This sacrifices some expected log growth under the assumed model, but it can improve robustness when the model is misspecified or the estimated edge is uncertain.

This connection makes Kelly a useful bridge from heuristic sizing rules to formal portfolio optimization. The same moment estimates that make Kelly allocations principled also make them fragile: expected returns and covariances must be estimated, and errors in those inputs are amplified when they are converted into portfolio weights. The next section turns this observation into the classical Markowitz framework and shows why regularization is essential in practice. **Implementation**:

- `04_kelly_criterion` derives the binary, continuous, and multi-asset Kelly fractions in SymPy and shows the full-versus-fractional trade-off on real ETF returns, including raw Kelly’s leverage (typically 30×+) before shrinkage.
- `05_factor_allocation_evidence` documents the empirical case for diversify-Harvey-Liu-Zhu 𝑡 thresholds, value-momentum negative correlation across ing across factor sources using Fama-French and AQR Century-of-Premia data:

asset classes, TSMOM crisis alpha, and crisis-period factor performance.

- `07_conformal_position_sizing` implements Mondrian split-conformal calibration on the ETF and CME-futures case-study GBM predictions and computes the inverse-width weights compared in this section.
- `09_allocator_comparison` provides the inverse-volatility, equal-weight, and score-weighted reference implementations used elsewhere in the chapter.

## 17.5 Mean-variance optimization and the Markowitz curse

**Mean-variance optimization** (**MVO**) remains the standard framework for portfolio construction. Markowitz (1952) showed that rational investors should choose portfolios on the **efficient frontier**: the set of allocations offering the highest expected return for each level of risk. The framework is elegant, but naïve implementations often produce unstable, concentrated portfolios that perform poorly outof-sample.

The standard formulation is: ௪ߤ்ݓെߣ max 2 ݓ்ߑݓ

subject to 𝟏்ݓൌͳ, where 𝜇 denotes expected returns, 𝛴 the covariance matrix, 𝑤 the portfolio weights, and 𝜆 risk aversion. Smaller values of 𝜆 tilt the solution toward return-seeking portfolios; larger values

place more weight on risk control and move the solution toward the minimum-variance end of the frontier. Equivalent formulations maximize the Sharpe ratio or minimize variance for a target return, but all depend on the same estimated moments.

*Figure 17.3* places the minimum-variance and maximum-Sharpe portfolios on the same frontier. The figure clarifies why the **tangency portfolio** is especially exposed to estimation error. The tangency point is determined by the slope of the capital allocation line, so small changes in estimated expected returns or covariances can materially shift the optimum and produce large changes in portfolio weights. Portfolios near the minimum-variance region are typically less fragile because they depend less on precise estimates of expected return.

![Figure 17.3](assets/figure_17_3.jpeg)

*Figure 17.3: Efficient frontier with allocation zones*

This quadratic objective is exact under **constant absolute risk aversion** (**CARA**) utility and normally distributed returns. Under CARA, an investor’s aversion to a given dollar amount of risk does not vary with wealth. That assumption matters because, with normal returns, it implies that expected utility depends only on mean and variance, yielding the MVO objective directly. More generally, the same expression can be read as a second-order approximation to expected utility over moderate horizons. to log utility, 𝜆 reproduces the full-Kelly scale. Larger 𝜆 values correspond to fractional-Kelly-style In the unconstrained risky-asset formulation with excess returns and a second-order approximation

scaling.

The central implementation problem appears in the unconstrained solution: 𝑤∗= 1 ߣߑ−1ߤ

Both 𝜇 and 𝛴 must be estimated, and both are noisy. MVO combines these noisy inputs through the

inverse covariance matrix, which can amplify small errors into large allocation changes. That sensitivity is the central practical weakness of the classical framework.

### Why MVO fails out of sample

Antonov, Lipton, and Lopez de Prado (2024) argue that MVO underperforms out-of-sample for three linked reasons:

1. **Expected returns are hard to estimate**, so optimizer inputs carry substantial noise.
2. **Covariance inversion becomes unstable** when the sample size is not large relative to the universe size.
3. The optimizer tends to **concentrate on assets whose alphas are overestimated** by chance, creating an error-maximization effect (Lopez de Prado, 2016).

The practical symptom is familiar: strong in-sample Sharpe and weak out-of-sample stability. The `02_mean_variance_optimization` notebook reproduces this gap on the ETF universe. A useful nuance is that allocation quality depends more on alpha ranking than on alpha calibration. If the forecast scale is uniformly miscalibrated, portfolio composition changes little, and leverage absorbs most of the error. Ranking mistakes, not pure scale error, do most of the damage.

### Making MVO more robust with covariance shrinkage

In practice, the question is not whether MVO is theoretically correct, but how aggressively one should regularize it. One source of fragility is covariance estimation. **Ledoit-Wolf shrinkage** (Ledoit and Wolf, 2003) replaces the sample covariance matrix with a weighted average of the sample estimate and a structured target: 𝛴̂ = (ͳ െߙ)ܵ൅ߙܨ

where 𝑆 is the sample covariance, 𝐹 is a structured target, and 𝛼 is a data-driven shrinkage intensity.

This trades a small amount of bias for a substantial reduction in estimation variance and numerical instability. A related approach models **covariance through factor structure**: ߑൌܤߗ௙ܤ்൅ߗఌ

thereby reducing dimensionality by separating common-factor risk from idiosyncratic residual risk. Shrinkage and factor structure are best viewed as complementary tools rather than competing alternatives.

**Constraints play a similarly important role**. Position caps limit concentration, long-only constraints suppress fragile short positions supported only by noisy return estimates, and turnover penalties reduce the tendency to convert forecast error into trading costs. In this sense, constraints are not secondary implementation details but part of the estimator itself: they regularize the mapping from noisy inputs to portfolio weights.

As Paleologo (2025) argues, many common constraints admit a robustness reading. Quadratic penalties reflect uncertainty about alpha magnitudes, sparse allocations reflect a preference for parsimony, conservative covariance treatments act like higher effective risk aversion, and turnover penalties reflect uncertainty about net alpha after costs. Framed this way, MVO is not a claim of precise optimality. It is a disciplined procedure for translating uncertain beliefs into tradable weights under explicit guardrails.

Accordingly, MVO is most defensible when the investable universe is modest, covariance estimation is tractable, and signal quality is strong enough that return tilts survive implementation costs. Under those conditions, shrinkage, realistic constraints, and turnover control often matter more than the particular form of the optimizer.

### Shrinkage hedging as robust optimization

The same robustness logic applies when the optimizer is used to remove risk rather than to allocate toward expected return. **A hedge ratio is an allocation decision**: it chooses how much of a hedge instrument to add to an existing portfolio in order to reduce an unwanted exposure. If the exposure estimate is noisy, full neutralization can be as fragile as an unconstrained maximum-Sharpe portfolio. **Beta hedging provides the simplest example**. With a known beta, the variance-minimizing hedge offsets the portfolio’s exposure to the hedge instrument. In practice, however, beta is estimated from finite data. Paleologo (2025, *Chapter 12*) shows that when the beta estimate is noisy, full beta-neutral hedging can increase realized variance rather than reduce it. The hedge overreacts to estimation error: it trades against noise as if it were true exposure.

**The remedy is shrinkage**. The hedge ratio should be scaled down when the estimation error in beta is large relative to the portfolio’s aggregate exposure. In Paleologo’s formulation, the shrinkage factor declines with the portfolio-weighted covariance of beta-estimation errors and rises with the squared estimated aggregate beta. The intuition is the same as for robust MVO: when the signal-to-noise ratio is weak, the optimizer should move less.

**This point is especially important for concentrated portfolios**, where beta-estimation errors do not diversify across many positions. A concentrated book should usually hedge more conservatively, while a highly diversified book can move closer to full beta neutrality if the aggregate exposure is estimated with greater precision. The practical rule is therefore not “always hedge to zero,” but “hedge in proportion to the reliability of the exposure estimate.”

Viewed this way, hedging is not a separate topic from portfolio optimization. It is the defensive version of the same problem: translating uncertain estimates into tradable weights without letting estimation error dominate the allocation. The next subsection extends this idea from single hedge instruments to factor-mimicking portfolios.

### Factor-mimicking portfolios

A **factor-mimicking portfolio** (**FMP**) isolates a single factor exposure with minimal residual risk. tor-model derivation in *Chapter 9*. Given factor loadings 𝑏 and idiosyncratic covariance 𝛺ఌ, the mini-Grinold and Kahn (2000) give the active-management framing, and Paleologo (2025) provides a fac-

mum-residual-risk factor-mimicking portfolio is: ߗఌ 𝑤ிெ௉= −1 ⊤ߗఌ−1ܾ

FMPs are useful because they convert feature-level hypotheses into tradable portfolios and provide a direct portfolio-level test of the incremental value of factors. In practice, factors are evaluated by the out-of-sample marginal Sharpe of orthogonalized FMPs, which aligns factor admission with the same model-accounting discipline used elsewhere in the chapter. **Implementation**:

- `02_mean_variance_optimization` traces the efficient frontier on a 30-ETF universe, and reproduces the Markowitz curse – the unconstrained max-Sharpe solution collapses to two assets (XLK 78.9%/GLD 21.1%) and the unconstrained min-variance solution to one (SHY ~98%).
- `03_robust_optimization` runs six allocators side-by-side via Riskfolio-Lib (max-Sharpe, min-variance, risk-parity, HRP, min-CDaR, equal-weight) on an 11-ETF multi-asset universe under Ledoit-Wolf covariance shrinkage, and quantifies the equal-capital-versus-equal-risk gap with risk-contribution diagnostics.

## 17.6 Optimizing for stability with Hierarchical Risk Parity

**Hierarchical Risk Parity** (**HRP**) takes a hierarchy-based approach to the same covariance-driven allocation problem that makes naïve MVO fragile. Rather than estimating expected returns and inverting the full covariance matrix, HRP uses hierarchical clustering to order assets based on the empirical correlation structure, then allocates capital through recursive inverse-variance splits (Lopez de Prado, 2016). HRP avoids covariance-matrix inversion and tends to produce less concentrated and more stable portfolios.

The intuition is that **not every asset should compete with every other asset for portfolio weight**. In MVO, small changes in expected returns or covariances can alter the entire allocation. HRP instead imposes a nested structure on the universe: assets that appear similar under the chosen distance metric are grouped together before capital is allocated. This does not eliminate estimation error, but it localizes much of its effect and reduces the chance that a small covariance perturbation produces a global reallocation.

Consider a portfolio containing technology and utilities stocks. MVO compares all names simultaneously. HRP uses the correlation structure to place similar names near one another, then allocates across and within the resulting branches. Errors in estimating the relationship between Apple and Microsoft primarily affect the technology branch and its estimated cluster variance; they are less likely to propagate through the entire portfolio through a global covariance inverse.

The hierarchical structure also follows a simple rule: diversify first across dissimilar groups, then within similar ones. This produces portfolios that better capture true diversification rather than the spurious diversification MVO often creates.

### From correlations to weights – The HRP algorithm

HRP has three computational steps. First, **convert correlations into distances**, commonly: 𝑑௜௝= √ͳ െߩ௜௝ 2  Second, **apply hierarchical clustering** to this distance matrix and order assets by traversing the dendrogram. This quasi-diagonalization places similar assets near one another, making the covariance matrix more block-like.

![Figure 17.4](assets/figure_17_4.png)

*Figure 17.4: HRP workflow from clustering to recursive allocation. The hierarchy stabilizes diversification by grouping similar assets before assigning weights*

Third, allocate capital by recursive bisection. Start from the whole universe, repeatedly split each cluster into two sub-clusters and allocate capital between them in inverse proportion to the variance of the each branch until the process reaches individual assets. If 𝐿 and 𝑅 denote two sub-clusters, the split is: cluster. The lower-variance cluster receives the larger share, and the same logic is then applied within

𝑤௅= , 𝑤ோ= ௅൅ܸ ோ ௅൅ܸ ௅ ோ ோ  where 𝑉௅ and 𝑉ோ are cluster variances. The important feature is not the algebra alone, but how it is

applied locally within the hierarchy.

Diversification is imposed in stages, first across broad groups and then within finer groups, so estimation errors are less likely to propagate across the entire portfolio.

### Why HRP can be more stable than naive MVO

The survey evidence summarized by Marti et al. (2021) and the subsequent HRP literature point to four practical sources of stability. First, HRP avoids matrix inversion entirely, eliminating the numerical instability that undermines MVO. Second, grouping similar assets provides hierarchical regularization: estimation errors within a cluster partially cancel, so individual-asset misestimates have less impact on the final portfolio. Third, the hierarchical structure itself is relatively stable, yielding more consistent weight assignments across rebalancing dates. Fourth, by ignoring expected returns, HRP eliminates error-maximization at the cost of not expressing directional views.

### Limitations and extensions

HRP is not a universal solution. Its basic form does not use return forecasts, so any genuine predictive content in the signal is intentionally ignored. That is often an acceptable trade when forecast error dominates, but it becomes a real opportunity cost when the model has stable directional information. Vanilla HRP is also long-only by construction because the recursive bisection step assigns positive capital shares at every level. Readers working on long-short strategies should treat HRP as a stability template rather than a plug-in allocator. One workaround is to apply hierarchical allocation separately to long and short candidate pools before combining them, though nested clustered optimization provides a cleaner way to express views inside the same general framework.

**The method is also sensitive to the clustering recipe**. Raffinot (2016) compares several hierarchical variants using bootstrap-based model confidence sets and finds strong out-of-sample performance for the class as a whole, but not a single universally dominant specification. That finding is more useful than a single ranking because it identifies where the method is robust and where design choices still matter. Standard HRP also treats the hierarchy as a sequence of point-in-time clusterings rather than as a dynamic structure with explicit control over turnover, so changes in cluster membership can create a different kind of instability.

**Nested Clustered Optimization** (**NCO**) extends the same clustering intuition in a different direction. HRP uses the hierarchy to avoid global optimization and allocate capital through recursive risk splits. NCO instead uses clustering to decompose the optimization problem: it first optimizes within clusters, then across the resulting cluster portfolios. This preserves the ability to express expected-return views, constraints, and alternative objectives, while reducing the dimensionality and instability of the covariance-driven optimization problem. NCO is therefore closer to a clustered, regularized version of MVO than to vanilla HRP.

**The trade-off is complexity**. HRP is transparent and difficult to overfit because it ignores expected returns and avoids matrix inversion. NCO is more flexible, but it reintroduces the optimizer assumptions and can still overfit noisy expected-return estimates if the inner and outer optimizers are not carefully constrained or validated.

A related line of work, **Schur Complementary Allocation** (Cotton, 2024), connects hierarchical allocation to minimum-variance portfolios by using Schur-complement adjustments to retain selected off-block covariance information. This reframes the HRP-versus-minimum-variance choice as a continuum rather than a binary choice: the practitioner can decide how much covariance information to admit, depending on the reliability of the estimate.

### ETF walk-forward results

Running the four allocators on the same GBM-selected ETF universe (top-5 picks at each monthly rebalance, 252-day covariance window, 2010–2023) yields a narrow but consistent ordering in favor of the covariance-aware methods. Ledoit-Wolf shrinkage minimum variance delivers the highest realized Sharpe at 0.8437, HRP follows at 0.8293, inverse-volatility at 0.8074, and equal weight trails at 0.7763. The spread is 0.067, and all four allocators share the same -28% maximum drawdown because the GBM-ranked top-5 selection dominates the overall path. The cross-section is narrow enough that allocator differences are bounded. With only five names per rebalance, the correlation structure has limited room to express itself, and equal weighting is already close to the inverse-variance solution when the universe is small. Monthly rebalancing also limits how quickly the hierarchy can re-cluster as regimes shift, and GBM-selection concentrates capital in names the signal already ranks highly, so allocator differences reduce to how each weights an already-filtered set. Even so, the covariance-aware methods extract a small edge: the renormalized long-only projection used by Min Variance (LW) keeps the shrinkage signal in the realized weights, and HRP’s recursive bisection captures most of that same structure without the matrix inversion. Marti et al. (2021) document how thin out-of-sample margins typically are across hierarchical and non-hierarchical methods, consistent with a 0.067 Sharpe spread on a five-asset universe. HRP’s structural advantages (hierarchical regularization, no matrix inversion, no return forecasts) matter most on larger, more heterogeneous universes than the one tested here.

| Method | Annual Return | Sharpe | Max DD | Calmar |
| --- | --- | --- | --- | --- |
| Min Variance (LW) | 11.83% | 0.8437 | -28% | 0.4215 |
| HRP | 11.41% | 0.8293 | -28% | 0.4063 |
| Inverse Volatility | 10.92% | 0.8074 | -28% | 0.3893 |
| Equal Weight | 10.38% | 0.7763 | -28% | 0.3698 |

*Table 17.2: Walk-forward backtest on the GBM-selected top-5 ETF universe (2010–2023)*

**Implementation**: `06_hierarchical_risk_parity` shows the end-to-end HRP pipeline (clustering, quasi-diagonalization, and recursive bisection) with dendrograms, quasi-diagonal covariance heatmaps, and weight-evolution time-series. The walk-forward backtest resolves the selected GBM prediction set from the ETF case-study registry at runtime and applies HRP, inverse-volatility, minimum-variance, and equal-weight sizing to that selection. An `ml4t-diagnostic` tear sheet packages the cumulative return, drawdown, rolling Sharpe ratio, monthly heatmap, and return distribution panels for the HRP allocator.

Now we turn to using deep learning for end-to-end portfolio learning, building on the material from *Chapter 13*.

## 17.7 Comparing allocator performance

A fair comparison of the allocation approaches introduced so far (heuristics, robust MVO, and hierarchical methods) requires identical inputs, identical backtest protocols, and careful attention to the degrees of freedom each method consumes, while guarding against the subtle overfitting that can arise from “allocator shopping.”

### Experimental design

**A controlled allocator comparison holds everything constant** except the allocation method. All allocators should receive the same expected-return forecasts; here, the common input is a ridge-based cross-sectional ETF model, so performance differences can be attributed to sizing and diversification rather than to changes in the predictions. The same principle applies to the backtest protocol. Rebalancing frequency, transaction-cost assumptions, treatment of dividends, and corporate-action handling should all remain fixed.

Constraints and estimation windows require the same discipline. Long-only rules, maximum position sizes, sector caps, leverage limits, and turnover penalties should be aligned across methods wherever the comparison is meant to be head-to-head. Covariance estimation should also be matched. If robust MVO uses a Ledoit-Wolf covariance estimate within a particular lookback window, the competing riskbased allocators should use the same underlying risk information unless the experiment explicitly tests the covariance model itself.

The notebook `09_allocator_comparison` **compares four methods** that represent increasing modeling structure:

- **Equal weight** is the null model: capital is divided evenly across assets regardless of forecast strength or risk, and any sophisticated allocator should clear that hurdle before it is taken seriously.
- **Inverse-volatility** weighting adds a risk estimate but still ignores directional views and cross-asset correlations.
- **Robust MVO** introduces forecasts, covariance structure, position limits, and turnover-aware regularization.
- **HRP** maintains the risk focus while introducing diversification via clustering rather than matrix inversion.

### Empirical results for an ETF momentum strategy

The controlled comparison uses identical ML signals on the ETF universe from January 2018 through December 2023. The backtest runs each allocator on the same dates, with the same rebalancing schedule and cost assumptions.

The following table compares key performance metrics:

| Method | Annual Return | Annual Vol | Sharpe | Max DD | Avg Turnover |
| --- | --- | --- | --- | --- | --- |
| MVO (Ledoit-Wolf) | 6.7% | 18.0% | 0.450 | -21.7% | 10.5% |
| Inverse Volatility | 6.8% | 19.2% | 0.439 | -23.3% | 8.3% |
| HRP | 6.1% | 19.3% | 0.404 | -25.6% | 10.1% |
| Equal Weight | 4.6% | 19.5% | 0.327 | -19.7% | 6.8% |

*Table 17.3: Allocator performance metrics*

Robust MVO and inverse-volatility achieve similar Sharpe ratios, separated only by a gap of 0.011 (see *Table 17.3*), while equal-weight delivers the shallowest drawdown despite lower Sharpe. Allocator choice changes risk shape and turnover, but often by narrower margins than expected.

The comparison notebook also simulates performance under realistic transaction cost assumptions for the inverse volatility method. In this simulation, the weight-based return (+46.6%) drops to +1.5% after accounting for commission, slippage, and fill timing. *Chapter 18* details this transaction-cost channel.

The cross-case summary (the best allocator per case study, the best-minus-worst Sharpe spread, and the structural reading of when allocator choice is load-bearing) is consolidated in *Section 20.5*, where allocator outcomes are read jointly with cost survival (*Section* *20.6*) and risk overlays (*Section 20.7*). The conclusion that carries over from the ETF comparison above is unchanged: allocator gains cannot rescue a weak signal-cost combination.

*Figures 17.5* through *17.7* give a cross-case-study preview of the *Section 20.5* analysis. *Figure 17.5* is a Sharpe heatmap across allocators × case studies, where the row-by-row variation shows which case studies are allocator-sensitive and which are not. *Figure 17.6* plots allocator uplift over equal weight against signal strength: the slope answers “when does optimization help?” Uplift correlates positively with signal quality, with the strongest evidence on case studies where the underlying signal already had measurable predictive power. *Figure 17.7* ranks the best allocator Sharpe per case study, showing where allocator choice is load-bearing for the reported result.

![Figure 17.5](assets/figure_17_5.png)

*Figure 17.5: Allocator performance heatmap. Each cell shows the Sharpe ratio of one allocator on one case study; darker cells mark stronger allocators on that case study*

*Figure 17.6* plots allocator uplift over equal weight against signal strength: the slope answers “when does optimization help?” Uplift correlates positively with signal quality, with the strongest evidence on case studies where the underlying signal already had measurable predictive power.

![Figure 17.6](assets/figure_17_6.png)

*Figure 17.6: When does allocation optimization help? Signal-strength on the x-axis (case-study IC), allocator uplift over equal weight on the y-axis. Allocator choice matters more on case studies with stronger underlying signals*

*Figure 17.7* ranks the best allocator Sharpe per case study, showing where allocator choice is load-bearing for the reported result.

![Figure 17.7](assets/figure_17_7.png)

*Figure 17.7: Best allocator Sharpe by case study. Horizontal bars show the headline Sharpe achieved by the best-performing allocator on each case study, with the allocator name annotated*

### The broader lesson

The “best” allocator depends on signal quality, transaction costs, and investor objectives. Strong and stable forecasts can justify score-based tilts or constrained optimization because there is something worth expressing. Weak or noisy forecasts usually call for robust heuristics that resist overreacting to estimation error. HRP occupies a middle ground: more structured than inverse-volatility weighting, less directionally expressive than forecast-driven optimization, and often attractive when the priority is diversification stability rather than maximum aggressiveness.

The broader lesson is humility. Allocation methods often live in narrower performance bands than the predictive models that feed them. In the ETF comparison above, the spread between best and worst Sharpe is only 0.123, and the top two methods differ by 0.011. Those are not meaningless differences, but they are smaller than many researchers implicitly assume before running the test. In practice, more value is often created by improving signal quality and execution realism than by repeatedly re-parameterizing allocator geometry. *Chapter 20* compares allocators across the nine case studies.

**Implementation**:

- `08_library_comparison` complements the comparison by showing `skfolio`’s `WalkForward` cross-validation and comparing `PyPortfolioOpt`, `Riskfolio-Lib`, and `skfolio` under the same 11-ETF universe – useful when treating the allocator itself as an estimator with proper out-of-sample folds.
- `09_allocator_comparison` implements the controlled allocator comparison reported in this section, with the Ridge-based ETF signal, identical inputs and rebalances across allocators, and the production replay through the `ml4t-backtest` Engine. A `ml4t-diagnostic` tear sheet for the selected allocator (resolved at runtime, not hard-coded) is saved to `output/allocator_comparison/allocator_` `comparison_<method>_tear_sheet.html`, packaging the cumulative-return, drawdown, rolling-Sharpe, monthly-heatmap, and returns-distribution panels in a single self-contained file.

## 17.8 Deep learning for portfolio construction

Classical allocators separate prediction from sizing. A model estimates returns or ranks assets; a risk model estimates covariance; an allocator turns both into weights. This separation improves diagnosis, but the losses are not aligned. A model can reduce forecast error while producing signals that are expensive to trade, poorly diversified, or fragile under leverage.

End-to-end portfolio learning replaces this modular pipeline with a policy that maps features directly to positions and trains on a portfolio-level objective. The advantage is alignment: gradients reward the model for improving portfolio returns after risk scaling and costs. The cost is diagnostic opacity. Prediction error, sizing, turnover, and exposure control become entangled in one loss surface. For that reason, learned allocators should be evaluated on a separate evidence track. They do not consume the same forecast stream as equal weight, inverse volatility, MVO, or HRP, so a direct headto-head comparison confounds forecasting architecture with allocation logic. The appropriate question is not whether a neural allocator outperforms MVO in isolation, but whether it clears simple heuristics after leakage checks, seed variation, turnover costs, and regime-sliced diagnostics.

### The end-to-end pipeline

The end-to-end allocators in this section share a common five-stage structure with gradients flowing backward through every stage: **Inputs**: For each asset 𝑖 and decision time 𝑡, the network receives a fixed-length lookback window 𝐗௜ǡ௧ൌൣݔ௜ǡ௧ି௅ାଵǡ ǥ ǡ ݔ௜ǡ௧] where 𝑥௜ǡ௧∈ℝி is a per-day feature vector. Standard choices 1.

are vol-scaled returns over multiple horizons, an EWMA realized-volatility estimate (typically with a 63-day half-life), and a small number of technical indicators, such as multi-scale MACD or normalized momentum. DeePM deliberately restricts itself to closing prices to isolate the contribution of its architectural priors; the Saly-Kaufmann benchmark uses a slightly richer **Sequence encoder**: A sequence model 𝑔థ processes the lookback window and produces a feature set centered on stationarized returns and ticker embeddings. fixed-dimensional hidden state ℎ௜ǡ௧∈ℝு. The encoder is the swappable component of the 2.

pipeline. Any of the *Chapter 13* architectures (LSTM, xLSTM, PatchTST, iTransformer, TFT, shared weights (channel-independence): the same 𝑔థ runs on every asset’s lookback window. Mamba, Mamba2, or hybrids) can occupy this slot. The encoder is applied per asset with

This is a deliberate regularization that prevents the model from memorizing asset identities through its temporal weights; identity is reintroduced separately, typically through learned ticker embeddings concatenated to the hidden state.

1. **Signal projection**: A small head (a linear layer followed by a hyperbolic tangent) projects the hidden state to a scalar signal: 𝑠௜,௧ൌ൫ݓlin ⊤ℎ௜,௧൅ܾ lin) ∈[−1,1]

Reading 𝑠௜ǡ௧ as a directional conviction is a deliberate design choice: the interval [−1,1] allows

long-short positioning, and the tanh squash bounds the projection so the position layer cannot ask for unbounded exposure. Notebook 11 (Zhang, Zohren, and Roberts, 2020) is the only exception in this section: it ends in a softmax over assets and produces a long-only weight vector. We treat it as a baseline precisely because it sits one architectural choice away from the long-short template the later notebooks adopt.

1. **Position layer**: A volatility-targeting transformation converts the signal into a position size: 𝑤௜ǡ௧ൌݏ௜ǡ௧⋅ߪ∗ ߪො௜ǡ௧  where 𝜎∗ is a portfolio volatility target ( typically 5–15% annualized depending on the universe;

benchmark, while the closed-form rule is most often stated at 10%) and 𝜎௜ǡ௧ is an ex-ante asset the VLSTM and DeePM ablations in this chapter use 15% in line with the Saly-Kaufmann

volatility estimate. The position layer does two things at once. It equalizes risk contributions across assets so that learning is not dominated by the highest-volatility markets, and it makes gross exposure self-scaling: when realized volatility rises, position sizes mechanically shrink, limiting drawdowns without requiring an explicit regime label.

1. **Portfolio aggregation and loss**: Realized portfolio returns are computed within the same computational graph by combining positions with realized asset returns, minus a cost term proportional to the absolute weight changes. Under the equal-risk-capital convention used by the Saly-Kaufmann benchmark, the cross-sectional aggregate is: port = 1 ௄೟ ௄೟ 𝑅௧ [෍ݓ௜ǡ௧ିଵ ݎ௜ǡ௧െܿ ෍|ݓ௜ǡ௧ିଵെݓ௜ǡ௧ିଶ| ] ܭ௧ ௜ୀଵ ௜ୀଵ where 𝐾௧ is the number of active assets and 𝑐 is a per-unit cost coefficient. The training loss is then a differentiable function of 𝑅௧ port, typically the negative annualized Sharpe ratio, introduced below.

*Figure 17.8* shows how these five stages compose into a single end-to-end graph.

![Figure 17.8](assets/figure_17_8.png)

*Figure 17.8: The end-to-end portfolio-learning pipeline*

Per-asset features pass through a shared sequence encoder (any *Chapter 13* architecture) into a tanh signal head; vol-scaling turns the signal into a position; the cross-sectional aggregator produces a realized portfolio return; the loss is computed on that return and backpropagates through the entire graph.

### Training a portfolio objective directly

The standard training loss is the negative annualized Sharpe ratio computed across the training horizon: ॱ̂ൣܴ port(ߠ)] ℒSR(ߠ) = −√ܣ⋅ ௧ √Var̂ൣܴ port(ߠ)] ൅ߝ  ௧ where 𝐴 is the annualization factor (252 for daily returns) and 𝜀 is a small numerical stabilizer. Minimizing ℒSR optimizes the economic quantity the allocator is judged on rather than a proxy that may

or may not be aligned with it.

Two refinements are standard practice:

- **Pooled Sharpe** computes the statistic across all training returns concatenated into a single sequence rather than per asset or per training window: per-asset Sharpe encourages the model to specialize in a few easy names, while pooled Sharpe forces the policy to perform across the full universe and is the better proxy for out-of-sample Sharpe (an empirical observation both DeePM and Saly-Kaufmann confirm).
- **Transaction-cost-aware loss** subtracts a turnover penalty inside the numerator so the network internalizes implementation friction during training, rather than producing a strategy that survives only on gross returns.

A subtlety arises in how the loss is computed across mini-batches. **Sharpe is a non-separable objective**: its gradient depends on global statistics, namely the sample mean, and standard deviation of the returns being optimized. Therefore, naive microbatch gradient accumulation yields the wrong gradient: it optimizes the average of mini-batch Sharpe, a different objective from pooled Sharpe over the full batch. DeePM handles this with an exact two-pass procedure:

- A first pass accumulates sufficient statistics (sums of returns and squared returns) without storing activations
- A second pass uses those statistics to compute the correct analytical gradients

Details matter when effective batch sizes need to span multiple years to stabilize the Sharpe estimator.

Average Sharpe over a long sample can also mask periods when the allocator is fragile, even when the long-run statistic is favorable. The DeePM subsection below introduces a SoftMin penalty over rolling sub-windows as the regime-robust counterpart to plain pooled Sharpe.

### Architecture under a portfolio objective

The pipeline is encoder-agnostic, which raises an empirical question: which encoders do best when the loss is a portfolio Sharpe rather than a forecast error? Saly-Kaufmann et al. (2026) answer this with a unified benchmark that runs linear baselines, LSTM and xLSTM variants, PatchTST, iTransformer, TFT, Mamba and Mamba2, and several hybrids (VLSTM, VxLSTM, LPatchTST, VSN-Mamba2) through the same volatility-targeted long-short layer and pooled-Sharpe loss on roughly fifteen years of futures, FX, bonds, equity indices, and energy products. All models face the same target, the same evaluation universe, and the same transaction-cost stress test, so Sharpe differences isolate the effect of the temporal inductive bias rather than of the data. The benchmark’s central finding is that **inductive bias matters more than raw capacity**. Linear baselines are unreliable across regimes; generic Transformers are mixed (iTransformer in particular trades too little, posting low turnover but weak economic performance); plain state-space models do not consistently outperform the recurrent class. Recurrent and hybrid architectures lead. The top performer, VLSTM, couples a TFT-style variable-selection network to a shared-weight LSTM encoder; LPatchTST and TFT are close behind, and xLSTM variants offer a favorable balance of return and turnover.

A second, equally important finding is that **robustness reorders the architectures**. VLSTM leads on average Sharpe but is not dominant on every axis: LPatchTST and VxLSTM are comparatively attractive on drawdown and tail-risk measures, and xLSTM often achieves the highest breakeven transaction cost because it trades less aggressively. The “best” architecture therefore depends on whether the objective is maximum average return, downside protection, or implementation robustness.

### Variable selection – VLSTM as the case study

The benchmark’s top performer, VLSTM, is the chapter’s natural architectural case study because the mechanism that distinguishes it from a plain LSTM (the variable-selection network) is also the simplest example of an inductive bias paying its way under a portfolio loss.

Financial features have very uneven signal strength across markets and through time. A 21-day momentum feature can carry a strong signal in one regime and pure noise in another; a volatility-of-volatility feature might matter for FX but not for equities; a slow-moving carry feature might dominate in calm periods and disappear in crises. A plain LSTM that receives all features stacked into its input vector spends part of its capacity learning to ignore the noisy channels at every time step. The TFT-style variable-selection network provides the model with an explicit gating mechanism: each feature is embedded through its own gated residual network (a per-feature non-linear projection with a learned skip), and a softmax gate over features computes a per-time, per-asset weighted combination. The mixture that reaches the LSTM has weights that adapt to each asset’s current state rather than being held fixed across the universe.

Notebook `12_vlstm_portfolio` implements the Saly-Kaufmann VLSTM on the ETF universe under the same volatility-targeted long-short layer, the same pooled-Sharpe training loss, and a 5 bps turnover cost internalized inside the loss. On the test window, the VLSTM attains a gross Sharpe of 0.66 with a 17.6% maximum drawdown, closes to 0.55 at the 5 bps cost level used in training, and degrades to 0.44 at 10 bps and 0.22 at 20 bps. Equal-weight is effectively cost-invariant (Sharpe 0.65 → 0.64 across 0–50 bps) because rebalance-to-equal carries near-zero turnover; inverse-volatility decays modestly across the same 0/5/10/20/50 bps grid (0.62 → 0.60 → 0.59 → 0.55 → 0.45), so the VLSTM’s relative gap widens monotonically with cost.

Variable selection therefore recovers most of the gap between a softmax long-only LSTM and the benchmark heuristics at gross terms but does not yet clear them at realistic cost. The training diagnostics show the characteristic failure mode: the pooled Sharpe on the training window climbs past 12, while the validation Sharpe collapses within a few dozen epochs. Early stopping protects the out-of-sample metric, but the underlying fragility (overfitting to training-window statistics that do not generalize) is exactly what the next architecture is designed to counter.

### DeePM – Structural priors and a regime-robust loss

DeePM (Wood, Roberts, and Zohren, 2026) extends the same end-to-end pipeline with four additional structural components, each motivated by a specific failure mode of the simpler architecture above.

**Asynchronous closes corrupt naive cross-sectional attention → Directed Delay (Causal Sieve)**. A cross-sectional attention layer that lets each asset’s representation see all other assets’ contemporaneous closes is fine when markets close synchronously, but global futures and FX markets do not. When the Nikkei closes at 06:00 UTC and the S&P closes at 21:00 UTC, an attention layer that gives the Nikkei representation access to the same-date S&P close uses information from the Nikkei’s future filtration.

DeePM’s Directed Delay shifts the allowed information flow per pair by an amount that respects each market’s closing calendar, so each asset’s representation only sees other assets’ returns realized before its own close. This is a cross-sectional analog of the look-ahead-bias discipline that runs through the entire book. It is what makes a cross-asset attention mechanism deployable in a multi-region universe at all.

**Channel-strength heterogeneity beyond what plain VSN handles → V-VSN temporal backbone**. DeePM uses a hybrid VSN-LSTM-Attention encoder per asset. The variable-selection block is “vectorized” in that it operates on each lookback time step independently before the LSTM receives the gated feature vector, and the attention block on top of the LSTM lets the per-asset representation attend to its own past at variable horizons.

The combination is a richer per-asset temporal encoder than the VLSTM of the previous subsection: the temporal-encoding equivalent of moving from “weighted feature mixture into an LSTM” to “weighted feature mixture into an LSTM with self-attention on top.”

**Cross-sectional attention overfits in low-signal regimes → Macro Graph Prior implemented as a GAT**. Free cross-sectional attention is permutation-equivariant (the right inductive bias for a portfolio set), but it allows attention weights to form between economically implausible asset pairs and amplifies noise when correlation estimates are unstable.

DeePM constrains attention to a fixed economic adjacency: intra-group cliques inside each macro group (equity indices, rates, energy, metals, agricultural commodities, FX, and more.) plus a small set of cross-group edges capturing economically motivated linkages such as safe-haven flows and supply-chain relationships. Inside that fixed adjacency, a Graph Attention Network learns dynamic edge weights that can upweight the bonds-equity link in risk-off regimes and downweight it in growth regimes. The structural prior fixes *which* edges are admissible while leaving *how strongly* each edge transmits information to the data: the cross-sectional counterpart of how a variable-selection network reasons over features.

**Lucky-window overfit on long pooled-Sharpe horizons → SoftMin-augmented loss**. A plain pooled Sharpe over a long training horizon can be inflated by a few favorable periods, leaving the policy fragile during the worst windows it has seen. DeePM augments the pooled Sharpe with a SoftMin penalty over rolling sub-windows: ℒ(ߠ) = −SRpool(ߠ) −ߣڄ Sఛ({SR௕(ߠ)}௕ୀଵ ) ஻ where SR௕ are Sharpe ratios computed on 𝐵 non-overlapping training sub-windows (quarterly blocks

in the paper) and: SoftMinఛ({SR௕}) ൌ−߬o (1 ஻ ܤ∑exp (−SR௕Ȁ߬))

௕ୀଵ As 𝜏, the SoftMin approaches the worst sub-window’s Sharpe, and the loss reduces to a pure worst-case objective; at moderate 𝜏, it weights the weak sub-windows without collapsing onto a sin-

gle episode. The objective, therefore, asks the network to perform well on average *and* be tolerable in the worst regime it has seen during training. The mechanism has a distributionally robust optimization interpretation: the SoftMin term is mathematically equivalent to taking the expectation of the negative Sharpe under an adversarial reweighting of the training sub-windows, constrained to a Kullback-Leibler ball around the uniform distribution. The effect on the policy is the same regardless of which interpretation one prefers. The gradient asks the network to invest some of its capacity in the regimes where average-case Sharpe would have let it coast.

`13_deepm_regime_robust` puts these four pieces together on the ETF test window: The full DeePM model achieves a Sharpe of 0.98 with a maximum drawdown of −7.5%.

• Sharpe 0.74 with a −8.9% drawdown.

- A no-SoftMin ablation that keeps the structural priors but drops the regime-robust loss reaches

of −18.4% and inverse-volatility with −14.7%.

- The two heuristic baselines have Sharpe ratios of 0.69: equal-weight with maximum drawdown The regime-sliced comparison shows the full model’s calm-vs-crisis Sharpe gap is +0.21 (calm 1.21, crisis 1.00), compared with +0.69 for the no-SoftMin ablation (calm 1.26, crisis 0.57). •

SoftMin substantially narrows the regime gap, mostly by lifting crisis-window performance. both heuristics and 0.74 for the no-SoftMin ablation, a maximum drawdown of −7.5% against −18.4% On this test window, full DeePM leads on every margin that matters: a Sharpe of 0.98 against 0.69 for and −14.7% for equal-weight and inverse-volatility, and a calm-crisis Sharpe gap of 0.21 against 0.69

for the no-SoftMin variant. The contribution of SoftMin is sharply localized: the calm-window Sharpe barely moves (1.26 to 1.21), while the crisis-window Sharpe lifts from 0.57 to 1.00, a +0.43 gain that accounts for almost the entire regime-gap reduction. The result reinforces a recurring lesson of this chapter: **regime-robust training can deliver a Sharpe lift and a drawdown improvement simultaneously**, and the two effects need to be read together rather than traded against one another.

*Figure 17.9* shows the resulting drawdown behavior through volatile periods.

![Figure 17.9](assets/figure_17_9.png)

*Figure 17.9: DeePM drawdown comparison. A regime-robust allocator should improve the path of losses, not only the average return*

The three notebooks together decompose the end-to-end paradigm into separable contributions:

- `11_dl_portfolio_allocation` (Zhang et al.) achieves a Sharpe ratio of 0.48, a faithful softrealized maximum drawdown of the three (−10.9%) at less than half the heuristics’ volatility. max-long-only LSTM baseline that trails the heuristics on this universe despite the lowest
- `12_vlstm_portfolio` reaches a gross Sharpe of 0.66 (effectively tying the heuristics in gross terms) but degrades to 0.55 at the 5 bps cost level used in training and 0.22 at 20 bps, while equal-weight is essentially flat (0.65 → 0.65) and inverse-volatility loses 0.07 over the same `13_deepm_regime_robust` reaches Sharpe 0.98 with a −7.5% maximum drawdown, exceeding range (0.62 → 0.55). • The 0.48 →0.66 improvement measures the architectural contribution of variable selection; the the heuristics on Sharpe by roughly 0.29 while cutting drawdown by more than half. 0.66 →0.98 move (together with the much-improved drawdown profile) measures the joint contri-

bution of the structural priors (Directed Delay, Macro Graph) and the SoftMin regime-robust loss, with the gain showing up in both Sharpe and risk shape on this particular test window.

### Evaluating learned allocators

Because end-to-end training optimizes a portfolio objective, learned allocators should be evaluated on allocator metrics rather than forecasting accuracy. *Section 17.3*’s multi-dimensional framework, including Sharpe, drawdown, turnover, concentration, regime-sliced performance, and transaction-cost breakeven, applies without modification.

Three considerations are specific to the learned setting:

- **Seed sensitivity** **becomes a first-class metric**: The loss surface is non-convex, the gradient depends on global batch statistics, and the variance across random initializations can dominate small differences in performance. Saly-Kaufmann report this explicitly, and an architecture ranking that ignores it can be reordered by changing the random-seed budget.
- **Turnover stops being a post hoc diagnostic**: Architectures with similar gross Sharpe ratios can differ by a factor of two in breakeven cost, so the cost-stress curve matters as much as the headline number.
- **The heuristic comparison stays non-negotiable**: An end-to-end allocator that cannot clear equal weight and inverse volatility on its own test window is not yet competitive, however sophisticated its encoder.

Deep learning enters portfolio construction as a class of methods rather than a single model. Classical allocators remain the right default when signals are stable, and the covariance structure is the binding constraint. End-to-end allocators are the right default when the objective is hard to factor cleanly into forecasting plus optimization: regime robustness, cost-aware sizing, or cross-asset interaction with known structure. DeePM is the chapter’s clearest example of the second case.

All three notebooks train end-to-end on raw ETF prices over a sub-universe of the ETF case study; only `13_deepm_regime_robust` uses asset-class group labels for its macro graph. Unlike `06_hierarchical_risk_parity` and `07_conformal_position_sizing`, they do *not* consume the ETF case study’s GBM walk-forward predictions. The goal here is to isolate the architectural and loss-design contribution of end-to-end portfolio learning, not to reuse a fixed signal as input. Reusing case-study predictions on top of these encoders is a natural extension but would conflate signal quality with allocator architecture.

**Implementation**:

- `11_dl_portfolio_allocation` implements the Zhang, Zohren, Roberts softmax long-only LSTM baseline.
- `12_vlstm_portfolio` shows the Saly-Kaufmann VLSTM (TFT-style variable-selection block + LSTM encoder) under the volatility-targeted long-short layer with a 5 bps cost-aware Sharpe loss and a cost-stress sweep over 0/5/10/20/50 bps.
- `13_deepm_regime_robust` shows the full DeePM implementation with Directed Delay, Macro Graph Prior, SoftMin-augmented loss, and regime-sliced (calm versus crisis) evaluation.

## 17.9 Summary

Portfolio construction translates forecasts into tradable positions: the point at which a signal becomes profit or loss. This chapter established a rigorous framework for making allocation decisions explicit, testable, and comparable. It formalized the allocation problem as a constrained optimization over weights, leverage, and rebalancing frequency, and then introduced the allocator term sheet to ensure every design choice remains auditable. Multi-dimensional evaluation, including Sharpe, drawdown, turnover, concentration, and regime-sliced diagnostics, replaced single-metric comparisons.

The empirical results reinforce a humbling message: simple heuristics set a high bar. Equal weighting and inverse volatility require minimal estimation and frequently match or beat sophisticated optimization out-of-sample. Mean-variance optimization fails without robust fixes. Ledoit-Wolf shrinkage, position constraints, and turnover penalties partially mitigate it but cannot fully overcome the instability inherent in inverting a noisy covariance matrix. Hierarchical Risk Parity avoids matrix inversion entirely and produces portfolios with better drawdown characteristics, at the cost of ignoring return forecasts. Regime-aware adaptation of inputs and constraints offers a middle path between static allocation and fragile regime switching.

End-to-end deep learning allocators bypass the predict-then-optimize pipeline entirely, training directly on the portfolio objective, but sacrifice the modularity and interpretability that make classical approaches debuggable. The chapter-local comparison shows that allocator performance spreads are often narrower than model-performance spreads, while the cross-case results already show that the leading allocator varies by dataset rather than converging on a single universal method. *Chapter 20* provides the full cross-case allocator synthesis.

*Chapter 18* develops the transaction-cost dimension that allocation decisions must ultimately confront. *Chapter 19* adds the risk overlays that determine whether allocator differences survive stress, tail events, and leverage constraints. The turnover generated by different allocators has real dollar costs, and the portfolios that look similar on gross Sharpe can separate sharply once costs and risk controls are imposed.
