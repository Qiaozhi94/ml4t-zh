# Chapter 7: Defining the Learning Task

A machine learning model can only be as good as the learning problem you define. Before you select an algorithm, you need a label that captures an outcome relevant to your strategy, preprocessing conventions that keep inputs comparable across trials, and an evaluation framework that can distinguish genuine signal from artifacts of timing, overlap, or search. This chapter builds that scaffolding.

*Chapters 8–10* provide the candidate features (financial, model-based, and text-derived) that this framework evaluates. The definitions, diagnostics, and decision gates built here govern every candidate in that sequence.

After completing this chapter, you will be able to:

- Build a split-aware preprocessing pipeline that produces stable, auditable inputs for label and feature computation.
- Define execution-consistent labels and diagnose their overlap and implied trading intensity.
- Evaluate feature–label bundles using fold-aware diagnostics for continuous and binary targets.
- Apply feasibility screens to filter out candidates who cannot survive implementation costs.
- Account for selection bias by defining searched sets, separating exploration from confirmation, and adjusting fold-level summaries.
- Apply mechanism plausibility checks to distinguish stable mechanisms from confounded proxies.

Throughout, diagnostics are computed within each walk-forward fold and aggregated across folds to surface instability that pooled statistics hide.

## 7.1 Data preprocessing and encodings

Data preprocessing turns validated datasets into stable inputs for label and feature computation. Earlier chapters handled point-in-time correctness, corporate actions, and calendar alignment; *Chapter 6* defined tradeable universes. This section covers the steps that are new at the feature-engineering stage: split-aware fitting, outlier treatment, representation choices, and missing-data protocols. **Box 7.1: Protocol-safe definitions and computation**

This chapter treats labels, features, and preprocessing transforms as data products. To keep results comparable across iterations and to prevent silent leakage, every run must satisfy five invariants:

- **Observability**: Every raw input must be observable under the trading setup at decision time. Reporting lags, update times, eligibility rules, and calendar conventions are part of the definition.
- **Train-only fitting**: Transforms that estimate parameters from data must be fit using only the training portion of each walk-forward split.
- **Overlap-aware evaluation**: Overlapping label horizons induce dependence. Handle this using a split design (*Chapter 6*) and report overlap diagnostics alongside the metrics.
- **Versioned metadata**: A “feature” or “label” is a fully specified recipe. Log the definition with the run record; changes create new trial families.
- **Auditable masks**: Eligibility masks are part of the learning problem. Store the mask definition (not just the resulting rows) and report effective sample sizes by fold and asset group.

### Split-aware preprocessing

Any preprocessing step that estimates parameters from data (means, standard deviations, quantiles, imputed values) must be fit only to the training split. Fitting on the full dataset before splitting leaks future information into the training representation, inflating performance.

In walk-forward evaluation (*Chapter 6*), this means refitting the preprocessor at each fold boundary using only data available up to that point. Steps that are **stateless (**dropping rows with invalid values, applying a fixed unit conversion, computing a boolean calendar flag) do not require split-aware fitting, but anything that computes a summary statistic from a population of observations does.

**When in doubt, treat the step as fitted**. Record which preprocessing steps are fitted and which are stateless in the pipeline configuration. Ensure that the same pipeline object handles both training and inference, so that no transform is accidentally omitted or double-applied (*Box 7.1*, invariant 2). See `02_preprocessing_pipeline` for a `SplitAwarePreprocessor` that enforces train-only fitting and shows the consequences of leakage: test mean 0.110 under train-only fitting versus 0.101 with full-panel fit.

### Outliers and heavy tails

The first distinction is between **invalid** and **rare**. Invalid observations violate domain constraints (negative volume, impossible timestamps, or crossed bid-ask quotes) and should be removed. Single-interval spikes that revert immediately are often data errors; field-aware spike checks that compare the magnitude of the change and reversion pattern against the field’s distribution help separate errors from genuine moves. Valid observations that fall in the tails require a different treatment. Aggressive truncation discards real information; the goal is to limit the influence of extreme values on downstream transforms without eliminating them. **Winsorization** or clipping at a fixed percentile prevents single-point blowups during scaling and normalization. **Robust scaling** (centering on the median and dividing by the interquartile range) is generally more stable than mean-variance standardization when the distribution is skewed or leptokurtic. `02_preprocessing_pipeline` applies winsorization and domain filters to the US Equities panel, with before-and-after diagnostics.

The important exception: when the label itself targets extreme moves or barrier events, clipping returns removes precisely the outcomes the model is meant to learn. Preserve tails when tails are the prediction target.

### Scaling and representation choices

Representation is part of the definition of a learning task. For level-like series (prices, yields, spreads), models learn more reliably from changes (returns or differences) than from raw levels. Stationarity diagnostics and the transformations that enforce it are covered in *Chapter 9*; return calculations for labels follow in *Section 7.2*.

**Risk scaling** (dividing by realized volatility) stabilizes features and labels across regimes. Treat the volatility estimate (its definition and lookback) as a fixed part of the representation rather than a parameter tuned between trials.

**Ranks and percentile ranks** are often the most robust encoding for cross-sectional features because they reduce sensitivity to outliers and vendor-specific units. Specify whether you rank across assets at each timestamp (cross-sectional) or within an asset’s history (time-series percentile), and keep that choice stable.

For **categorical encodings**, use one-hot encoding for low-cardinality fields, ordinal encoding when ordering is meaningful (for example, credit rating tiers), and hashing or learned embeddings for high-cardinality identifiers. Fit the encoder on the training split.

### Handling missing data

Missing values are not one problem but three. Your response depends on *why* the data is missing and what your model can accept:

- **Missing values as noise**. Values are absent for reasons unrelated to the asset, the time, or the value itself (random gaps). If your model requires complete inputs, fill in a robust default (for example, the cross-sectional median). If your model can handle missing values, you can often leave them missing.
- **Missing values due to observed coverage rules**. Missingness is explained by fields you do observe (for example, vendor coverage depends on size, exchange, region, or listing age). Either impute from the observed fields and add a “was missing” indicator, or keep the missing values and rely on a model that can handle missingness and the indicator.
- **Missing values that are themselves informative**. Absence is linked to the unobserved value or an unobserved driver (for example, non-reporting when metrics are poor; sparse prints precisely when liquidity is low). Here, “missing” can be signal or bias. Preserve it explicitly (using an indicator or a separate “not reported” category), and treat imputation as a modeling assumption, not a default.

Two practical defaults include

- If the model requires complete inputs, impute robustly and add a missingness indicator
- Use models that handle missingness natively (for example, tree-based methods, see *Chapter 12*)

Avoid silent imputations that make missingness disappear without a trace. *Table 7.1* lists some dataset-specific considerations:

| Dataset | Key preprocessing concerns |
| --- | --- |
| ETF Universe | Adjustment artifacts, ticker changes, coverage gaps |
| Crypto Premium/Perps | Funding rate alignment, liquidation outliers, delistings mid-panel |
| CME Futures | Roll-adjusted continuity, vacation calendar alignment |
| FX Pairs | Weekend gaps, rollover liquidity, timezone conventions |
| US equities | Survivorship bias, split/dividend adjustments, penny-stock filtering |
| Firm Characteristics | Pre-cleaned panel; missingness patterns and structural breaks |
| NASDAQ-100 Bars | Session boundaries, auction prints (pre-cleaned by AlgoSeek) |
| S&P 500 Bars | Index rebalancing efefcts, consistency with ETF universe |
| S&P 500 Options | Moneyness/expiry filtering, Greeks checks, alignment with underlying |

*Table 7.1: Case study preprocessing concerns*

**Implementation**:

- `01_data_quality_diagnostics` summarizes distribution characteristics across all datasets, including tail statistics that inform the choice between scaling methods.
- `02_preprocessing_pipeline` reports coverage by column, asset, and time period, and shows cross-dataset alignment.

The next section turns to engineering labels.

## 7.2 Label engineering

Labels define the learning task: what outcome you want to predict, over what horizon, and when that outcome becomes known. A label that is even slightly misaligned (starting or ending at an untradeable price) can create “predictability” that disappears in live trading. Label definitions must be **execution-consistent** and evaluated under the time-series protocol from *Chapter 6*; see also *Figure 7.1*.

![Figure 7.1](assets/figure_7_1.jpeg)

Let 𝑡 denote the decision time, 𝑡௘ the anchor at which the position is established, 𝑎 the asset, and 𝑝௧ǡ௔ Figure 7.1: Label and Feature Horizons the tradable price at 𝑡; every price in a label formula must be tradable under the chosen execution assumption. The horizon 𝐻 counts forward bars from 𝑡௘ to the resolution time 𝑡௘൅ܪ, in time or vol-

ume-related units (*Chapter 3*). `03_label_methods` demonstrates the methods in this section.

### Execution conventions

The execution convention determines which prices are tradable at the start and end of the label window. For bar data, two conventions dominate: **Close-to-close**: The signal is generated at the close of bar 𝑡; the return is measured from ݈ܿ݋ݏ݁௧ to ݈ܿ݋ݏ݁௧ାு. This is the simplest convention and the default in most academic work. • **Next-open-to-open**: The signal is generated at the close of bar 𝑡, but the position is entered at the open of bar 𝑡 and exited at the open of bar ݐ൅ܪ൅ͳ. This reflects the fact that most •

end-of-bar signals cannot be executed at the closing price.

The difference is not cosmetic. On daily equity data, the close-to-close label can differ from the nextopen-to-open label by 50–100 basis points per trade in either direction, depending on overnight moves and opening gaps. The effect is large: over the past 30 years, nearly all U.S. equity market returns have accrued overnight rather than intraday (Glasserman et al., 2025).

Choose the convention that matches your execution assumption, document it, and keep it consistent between label computation and backtest PnL. `03_label_methods` quantifies the close-to-close versus next-open gap on SPY for a 21-day horizon.

The default label is a forward return computed over a fixed number of bars 𝐻. Two common return Fixed-horizon labels

conventions are: *Simple (arithmetic) return*: 𝑟௧ǡ௧ାுǡ௔ = −1 ௦ ௣೟శಹǡೌ ௣೟ǡೌ • *Log return*: 𝑟௧ǡ௧ାுǡ௔ ൌ݈݋݃൬ ) ௟௢௚ ௣೟శಹǡೌ ௣೟ǡೌ •

For small moves, the two are nearly identical. Log returns are additive across periods; simple returns align with PnL arithmetic. Choose one, use it consistently, and record the choice.

Fixed-horizon labels align best with strategies that act on a stable cadence with a roughly fixed holding be measured in calendar time: you can define 𝐻 in bar counts over volume bars or information bars period, whether you are predicting a continuous value or a discrete direction. The horizon need not

ݐ൅ܪ, which simplifies overlap analysis and fold construction. (*Chapter 3*), so the horizon adapts to activity rather than the clock. The outcome is always known at

The key decision is whether to train on a continuous metric or discrete categories. Many long/flat/short strategies model continuous returns and discretize model scores using a threshold to generate entries.

The most common continuous target is the raw forward return 𝑟௧ǡ௧ାுǡ௔ itself. Alternative represen- Continuous targets

tations include **transformations** that are robust to outliers, such as (percentile) ranks relative to the cross-section or an asset’s own history, **excess returns** relative to a peer group, a benchmark, or a financing series, and risk measures such as realized volatility. The choice depends on the task: directional prediction, ranking, or risk estimation.

#### Discrete targets

Discrete labels are useful when small returns are economically irrelevant relative to costs, when you want to learn a directional or event-like outcome, or when your evaluation and model selection benefit from classification metrics: **Binary labels**: The simplest discretization flags a directional outcome: set 𝑦௧ǡ௔= 1 if 𝑟௧ǡ௧ାுǡ௔൐߬ else 0. Common variants include flagging extreme returns (top decile, meaning values above the ௧, •

**Multi-class labels**: A common three-class construction assigns +1 (long), 0 (neutral), and −1 top 10 percent of historical returns), volatility spikes, or drawdown events. • (short) based on upper and lower thresholds: 𝑦௧ǡ௔ൌሼ൅ͳݎ௧ǡ௧ାுǡ௔൐߬ ௧Ͳȁݎ௧ǡ௧ାுǡ௔ȁ ൑߬ ௧െͳݎ௧ǡ௧ାுǡ௔൏െ߬௧

The threshold 𝜏௧ controls two things simultaneously: the **magnitude** of the event being labeled and

the **base rate** (fraction of positives). A threshold that is too tight produces a near-50/50 split and labels noise; one that is too wide produces a rare-event problem and implies a trading frequency that may be infeasible. Choose thresholds that are economically meaningful and consistent with your turnover and cost budget. Three rules cover most cases: **Fixed absolute threshold**: Set 𝜏 to a constant (for example, 1%). Simple and interpretable, but

1. **Volatility-scaled threshold**: Set 𝜏௧ൌ݇ Ԝߪො௧ǡ௔√ܪ or 𝜏௧ൌ݇ Ԝܣܴܶ̂ the implied base rate drifts with volatility. ௧ǡ௔, where the volatility estimate 2. uses past-only inputs. This stabilizes the base rate across regimes. **Percentile thresholds**: Time-series percentiles set 𝜏௧ to be the rolling percentile of the asset’s recent return distribution (for example, the 75th percentile of |ݎ| over the past 252 bars), adapt-3.

at each 𝑡 and assign labels by quantile membership (for example, top quintile = +1, bottom ing to asset-specific volatility. Cross-sectional percentiles rank all assets by forward return = −1). Cross-sectional labels have the useful property that *class proportions are stable by con-*

*struction*. Time-series percentiles must be estimated within the training fold; global percentiles leak information about the future distribution.

Most multi-asset strategies use cross-sectional percentiles; single-asset strategies favor volatility-scaled or time-series percentile thresholds. Treat threshold selection as part of the label definition.

### Variable-horizon labels

Fixed horizons are simple and well-behaved, but many real trading problems do not have a single natural horizon.

#### Adaptive horizons via trend scanning

When the appropriate horizon is uncertain, **trend scanning** provides an adaptive alternative: for each observation, it evaluates candidate horizons (typically 5–60 bars) and selects the one with the highest t-statistic for a linear trend fit (López de Prado, 2018).

The adaptive horizon introduces **selection bias**: by picking the “best” horizon for each observation, trend scanning systematically selects extreme outcomes that may not persist outside the sample. You can mitigate this by applying **Bonferroni correction** to the selected t-statistic (dividing by the number of candidate horizons) or by constraining the horizon range to match strategy turnover constraints. *Section 7.4* generalizes this per-observation correction to the search-level problem of evaluating many candidates.

Because the prediction horizon varies across observations, you cannot compute a single IC. Report the distribution of selected horizons alongside performance metrics, and verify that the distribution is stable across folds.

**Implementation**: `03_label_methods` demonstrates trend scanning on SPY using 5–20-bar windows and compares raw versus Bonferroni-corrected t-statistics.

#### Event-style labels

Event-style labels let the price **path** determine when the outcome resolves. The broader risk-management context - stop-loss design, position sizing, execution - is in *Chapter 20*; here, the goal is a precise, auditable label definition.

![Figure 7.2](assets/figure_7_2.jpeg)

*Figure 7.2: Triple barrier labels*

price at 𝑡, define an upper barrier 𝜋௧> 0, a lower barrier ℓ௧> 0, and a vertical barrier at ݐ൅ܪ௠௔௫. The The standard construction is the **triple-barrier label** (López de Prado, 2018). Starting from the tradable cumulative return from the entry at 𝑡 to any subsequent bar 𝑢 is 𝑟௧ǡ௨ǡ௔. Assign the event label as follows:

+1 if 𝑟௧ǡ௨ǡ௔ first reaches ൅ߨ௧ −1 if 𝑟௧ǡ௨ǡ௔ first reaches −ℓ௧ • 0 if neither barrier is hit before ݐ൅ܪ௠௔௫ • Unlike fixed-horizon labels, where the outcome is always known at ݐ൅ܪ, event labels resolve at a • variable **resolution time**, namely the moment a price barrier is hit, or ݐ൅ܪ௠௔௫ otherwise. We denote this as 𝑡௥௘௦ when we need to refer to it explicitly - it becomes the key quantity in the overlap analysis

later in this chapter.

With bar data, when both barriers are crossed in the same bar, specify a resolution rule and log it: for example, consider the event a loss (conservative) or exclude ambiguous bars.

Barrier widths can be set using the same volatility-scaling logic as threshold design, with asymmetric multipliers when the strategy has directional conviction.

#### Maximum favorable and adverse excursion diagnostics

Barrier widths are not arbitrary. They shape class balance, time to resolution, and the amount of label overlap. A practical way to calibrate them is to examine **the maximum favorable** and **adverse excursions** (**MFE**/**MAE**) over the candidate holding period:

- MFE measures the best return reached before the event resolves
- MAE measures the worst drawdown over the same window

Together, they show how far prices typically move for and against a position before a barrier is hit or the horizon expires. These diagnostics help set profit-taking and stop-loss barriers at levels that are economically sensible rather than arbitrary:

- Narrow barriers tend to resolve quickly but can be dominated by noise
- Wide barriers increase time to resolution, reduce the number of usable observations, and create more overlap across labels

In practice, the goal is to choose widths that yield a workable class balance, plausible upper- and lower-barrier hit rates, and reasonable resolution-time distributions.

**Implementation**: `04_maximum_favorable_adverse_excursion` demonstrates MFE/MAE visualizations, barrier calibration, and validation using hit fractions and resolution-time distributions.

### Overlap, sample dependence, and effective sample size Overlap is structural: adjacent 𝐻-bar labels share most of the same price increments, and event-style

labels overlap irregularly because resolution times vary. Any price shock affects every label alive at A label generated at time 𝑡 is alive from 𝑡 until its resolution time 𝑡௥௘௦ - the interval 𝐿௧ǡ௔= [ݐǡԜݐ௥௘௦]. The that moment, creating mechanical dependence. **concurrency** at bar 𝑢, denoted 𝑐(ݑ), counts how many labels are alive at 𝑢. The **average uniqueness**

of a label summarizes its overlap: 1 1 𝑤௧ǡ௔= |ܮ௧ǡ௔| ∑ (ݑ) ௨א௅೟ǡೌ  where |ܮ௧ǡ௔| is the number of bars in the label’s lifetime. A uniqueness of 1.0 means no overlap; values near ͳȀܪ indicate near-maximal overlap. The **effective sample size** is approximately 𝑁௘௙௙= ∑ ݓ௧ǡ௔ fixed-horizon labels sampled at every bar, 𝑁௘௙௙ൎ𝑁Ȁܪ (ignoring cross-section and time-series correlation ௧ǡ௔ . For

As a concrete example, `03_label_methods` measures uniqueness for SPY with 𝐻21 over roughly that may further reduce the effective sample size).

fidence intervals based on the nominal count are roughly 4.6 × too narrow. Standard errors should 10 years: 248,436 nominal observations collapse to an effective sample size of about 11,830, so conuse 𝑁௘௙௙, not 𝑁.

Overlap creates three distinct problems:

1. Training and test labels that share price increments leak information across the split.
2. Gradient updates and evaluation metrics overweight high-concurrency periods.
3. Serial dependence in labels inflates apparent statistical significance.

addresses problem 2; and reporting 𝑁௘௙௙ (or block-bootstrap standard errors) addresses problem 3. The **time-series protocol** (*Chapter 6*) addresses problem 1 through purge gaps; sample weighting

#### Dealing with overlap

Four complementary strategies address overlap: **Protocol-correct splits** are non-negotiable: leave at least 𝐻௠௔௫ bars between training and

• **Sample weighting** assigns a uniqueness weight 𝑤௧ǡ௔ to each observation, so that high-concurvalidation/test windows (*Chapter 6*). • **Subsampling** reduces overlap by sampling every 𝐻 bars or starting a new label only after the rency periods do not dominate loss or metrics. • previous one resolves.

- A more data-efficient alternative is the **sequential bootstrap** (López de Prado, 2018). Instead of drawing each index with equal probability, the sequential bootstrap updates selection probabilities after every draw so that observations with higher expected uniqueness, that is, with less overlap with the labels already selected, become more likely to enter the sample. Intuitively, labels that overlap heavily with the current sample contribute less new information and are less likely to be drawn, while labels from less crowded periods are more likely to be selected. The result is a resampled dataset with lower effective redundancy, even though the original labels may overlap substantially.

Weighting retains all data; subsampling is simpler but discards information. Most setups combine at least two of these four.

### Label diagnostics before modeling

Before training, verify labels are correctly defined and stable. Inspect distribution and tail behavior by time and asset group. Outliers often signal broken adjustments. Track base rates over time (class fractions should be stable across regimes), examine resolution-time distributions for event labels, and compute implied trading intensity. If trading intensity is infeasible, revise the label before modeling.

Close each label definition with a short audit record: anchor convention, horizon definition, resolution-time rule (including bar-data tie-breaking), overlap summary, and base-rate summary. These fields keep trials comparable under the *Chapter 6* research ledger (see `03_label_methods` and `04_maximum_favorable_adverse_excursion` for reusable diagnostics).

### Meta-labels for confidence-weighted sizing

A different way to structure the labeling problem is to **decompose direction and confidence** into two stages (López de Prado, 2018). A primary model (often a simple rule or an existing strategy) generates entry signals; the secondary label records only whether each signal was profitable after costs. A separate **meta-model** then learns *when* the primary signal is trustworthy, and its calibrated probability is mapped to a position size, for example through a sigmoid that scales conviction into a fraction of the maximum bet. `03_label_methods` includes a self-contained illustration: a 20-day-momentum signal on SPY is meta-labeled by its forward-return outcome, and a sigmoid bet-size mapping converts the confidence score into a position. The case studies in this book do not adopt this two-stage structure: each case study trains a single model on a single label horizon and sizes positions via an allocator (*Chapter 17*). The technique is most useful when a directional model is already in place and the goal is to filter weak signals rather than re-engineer the entry rule.

Label diagnostics here focus on *the quality of definitions*; the next section shows how to evaluate the relationship between a label and a candidate feature.

## 7.3 Univariate feature–label evaluation

The first layer of evaluation is univariate feature–label screening: each candidate is assessed against the label one at a time. Preprocessed inputs and execution-consistent labels are described in *Sections 7.1–7.2*; candidate features appear in *Chapters 8–10*. The screens apply in rough order:

1. **Correctness**: Can you trust the definition at decision time?
2. **Association**: Does the feature carry information about the label?
3. **Shape**: Is the relationship compatible with a plausible mapping from signal to positions?
4. **Feasibility**: Could anything implementable survive the trading setup’s cost and capacity constraints?

A feature must clear each screen to advance to the next. All diagnostics are computed within each fold and aggregated across folds: medians, interquartile ranges, and worst-fold views rather than a single pooled statistic.

Univariate triage is necessary but not sufficient. A feature that looks promising in isolation may duplicate another candidate, and a feature with weak marginal association may become valuable once conditioned on the right companion. Multivariate model selection (*Chapters 11–12*) makes the decisive judgments. *Section 7.4* addresses the multiple-testing problem that arises when many candidates pass screening.

### Correctness screens – The non-negotiable first pass

Before evaluating predictive power, verify that the feature and label are usable under the stated protocol. Run these checks on each feature–label bundle:

1. **Coverage**: What fraction of eligible (asset, decision-time) pairs have non-null feature values? Sparse coverage changes the effective sample size and may indicate stale or lagged inputs.
2. **Timing and lag consistency**: Verify that the feature uses only information available at the time of decision. Check reporting lags for fundamental data, publication times for third-party signals, and update schedules for derived quantities.
3. **Mask alignment**: Confirm that the eligibility mask is applied identically to feature computation, label computation, and evaluation. Misaligned masks are a common source of leakage.
4. **Staleness**: For features that update infrequently (for example, quarterly fundamentals), check that the feature value changes at appropriate intervals. A feature that never updates may be dominated by asset-specific intercepts rather than a time-varying signal.

These checks do not establish economic value; they prevent the costliest research failure: promoting a definition you cannot trust. See `05_signal_evaluation` for reusable diagnostics of correctness and coverage applied to the case-study ETF universe.

### Evaluating continuous labels

For continuous labels (forward returns, excess returns, realized risk targets), the workhorse diagnostic at triage is the **information coefficient** (**IC**): the cross-sectional association between a feature observed at decision time and a label that resolves later.

#### The Information Coefficient

The default is **rank IC** (Spearman), which equals the Pearson correlation of cross-sectional ranks: ܫܥ௧᩸׷ൌ᩸ܿ ݋ݎݎ௔ቀݎܽ݊݇ ൫ݔ௧ǡ௔)ǡ᩸ݎܽ݊݇ ൫ݕ௧ǡ௔))

where 𝑥௧ǡ௔ is the feature value for asset 𝑎 at time 𝑡, 𝑦௧ǡ௔ is the label anchored at 𝑡, and ܿ݋ݎݎ௔(⋅) denotes correlation across assets at time 𝑡. Computing ܫܥ௧ for each decision time yields an IC time series ሼܫܥ௧}௧.

IC is a learnability diagnostic for the definition bundle (feature, label, masks) under the evaluation protocol. It is not, by itself, a claim about tradable performance.

**Rank IC versus Pearson IC**

Rank IC is the conservative default: it aligns with ranking-based portfolio construction and is less sensitive to heavy tails and nonlinear monotone relationships. Pearson IC is appropriate when you care about linear association on raw values - for example, a signal that will enter a linear forecast where scale matters. Keep the representation consistent: if you store a feature as ranks, evaluate it with rank IC.

![Figure 7.3](assets/figure_7_3.png)

*Figure 7.3: IC time series with fold boundaries*

*Figure 7.3* shows ܫܥ௧ over the full sample period. Vertical dashed lines mark fold boundaries. A cen-

tered rolling average highlights the trend. Two patterns to watch for: (i) IC that concentrates in a single episode - the feature may be exploiting a transient shock or protocol artifact; (ii) IC that flips the sign across folds - the feature’s usefulness depends on state variables not yet modeled.

Because ܫܥ௧ observations are not IID, overlapping horizons induce serial dependence, and cross-sec- Fold-level summarization – ICIR and sign consistency tional universes share common shocks - treat the fold as the unit of summary. Within each fold 𝑘, compute a fold mean 𝐼‾(௞) and a dispersion estimate 𝜎(௞). Summarize across folds using the median and interquartile range (IQR) of 𝐼‾(௞) plus a worst-fold view (see also *Figure 7.3*).

If you want a compact stability ratio, compute a fold-level IC information ratio: 𝐼(௞) ∶= 𝐼‾(௞) ߪ(௞) 

where 𝐼‾(௞) shares the sign of the overall median - is the key stability gate. A momentum feature and report it by fold rather than as a single pooled statistic. **Sign consistency** - the fraction of folds

might show a mean IC of 0.04 across five folds but achieve 0.08 in two trending folds and −0.02 in three mean-reverting folds - the aggregate masks a regime switch that would devastate a static allocation. In the ETF universe, 21-day momentum has a pooled IC of 0.001 (near zero) but a fold-level mean IC of 0.064 with ICIR of 0.79 and 75% of folds positive - the pooled statistic hides a signal that is real in most regimes but cancelled by a few adverse episodes (see `05_signal_evaluation`).

#### IC across horizons and rebalancing frequency

Computing IC across a grid of forward-return horizons reveals the signal’s useful life. A signal that peaks at 5 days and halves by 10 should not be held for a month; one flat from 5 to 21 days offers rebalancing flexibility. Report the IC decay curve alongside fold-level statistics (see `05_signal_evaluation` for the ETF momentum signal).

If you use inference, apply it to fold summaries, not daily ܫܥ௧: (1) **block bootstrap** within each fold, Inference at triage

with block lengths reflecting horizon overlap; (2) **within-time permutation** that shuffles the asset–label assignment within each cross-section, preserving cross-sectional dependence while breaking feature–label pairing. Treat p-values as descriptive at this stage; *Section 7.4* covers multiplicity adjustment.

`05_signal_evaluation` demonstrates walk-forward fold construction and reports per-fold IC, spread, and monotonicity. See `06_ic_inference` for HAC-adjusted inference and block bootstrap confidence intervals applied to IC time series. For the 21-day ETF momentum IC, HAC inflates the naive standard error by 2.54× and effective sample size drops from 4,989 to 773, an 85% efficiency loss that anchors minimum-track-record planning. **Box 7.3: IC and portfolio performance**

Under simplifying assumptions, Grinold’s Fundamental Law of Active Management (1989) suggests an approximate relation between forecast skill and portfolio performance: 𝐼 ≈ 𝐼ܥ× √ܤ

where 𝐼 measures forecast skill and 𝐵 is the effective number of independent bets

(“breadth”). The approximation is not a performance forecast - bets are rarely independent, and a universe of 100 ETFs with shared sector exposure may offer effective breadth of 20–30. The triage lesson: small but persistent ICs matter when breadth is genuinely

#### Kendall’s 𝜏 and mutual information

large; peak IC is less informative than stable IC across folds.

**Kendall’s** 𝜏 and **mutual information** (**MI**) measure different kinds of dependence. Kendall’s 𝜏 is a

**rank-based measure of association:** it assesses whether pairs of assets are consistently ordered by the signal and the realized outcome. More formally, it compares the number of *concordant* and *discordant* it is insensitive to scale, but it is often more stable in very small cross-sections (𝑛30) because it is pairs, so it is interpretable as a measure of **monotone association**. Like Spearman’s rank correlation, built directly from pairwise orderings rather than rank moments. In practice, 𝜏 is usually smaller in

magnitude than Spearman for the same relationship, so the two should not be compared numerically as if they were interchangeable.

**Mutual information** (**MI**) is different: it is an **information-theoretic measure of dependence**, not a a result, it can detect **nonlinear** and **non-monotonic** relationships that Kendall’s 𝜏, Spearman’s, or rank correlation. MI asks how much knowing the signal reduces uncertainty about the outcome. As

Pearson’s may miss. However, MI is harder to estimate reliably in finite samples, typically requires is also nonnegative, so unlike Kendall’s 𝜏, it does not indicate direction. discretization or other density-estimation choices, and is less directly interpretable than a rank IC. It

**dall’s** 𝜏 when cross-sections are small, and you want a robust monotone dependence check based on For routine triage of large cross-sections with a monotone prior, rank IC remains the default. Use **Ken-**

pairwise ordering. Use MI only when quantile plots or other diagnostics suggest a clear nonlinear or non-monotone pattern that rank-based measures miss. `05_signal_evaluation` computes cross-sectional IC at multiple horizons and visualizes the resulting time series.

### Discrete labels – Evaluating features as classifiers For binary labels with 𝑦௧,௔∈{0,1}, evaluate whether a feature can act as a score that separates positives

from negatives. As with continuous labels, all metrics should be computed fold-by-fold, respecting masks.

#### Threshold-dependent evaluation Given a score threshold or top-𝑘 action set, report precision, recall, and specificity. In trading terms,

**false positives** are costly entries (round-trip costs plus adverse move); **false negatives** are foregone gains.

#### Threshold-free metrics

Start with the **Receiver Operating Characteristic** (ROC) area under the curve (AUC), which measures how well the feature ranks positives above negatives across decision thresholds, and the **Precision-Recall** (PR) AUC. PR AUC is often more informative when the positive class is rare because precision directly reflects how many predicted positives are true positives.

Under strong **class imbalance**, a feature can have a high ROC AUC while still producing many false positives relative to true positives at practically relevant thresholds, which the PR curve makes more visible. Compute both metrics fold-by-fold and report the median and inter-quartile range (IQR).

![Figure 7.4](assets/figure_7_4.png)

*Figure 7.4: ROC and PR curves, side by side. Left: ROC curve for a feature with moderate discrimination (AUC ≈ 0.65), with the diagonal reference. Right: PR curve for the same feature, baseline at the prevalence rate.*

As shown in *Figure 7.4,* the PR curve reveals that at high-recall operating points, precision drops sharply, a pattern hidden by the ROC curve when the base rate is low.

For multi-class event labels (profit, stop, time-out), report base rates by class and fold, which shift across regimes. If you collapse multi-class labels into binary actions, explicitly record the collapse rule and verify that it aligns with your intended action set.

**Implementation**: `05_signal_evaluation` computes ROC AUC and PR AUC with fold-level breakdowns and Wilson confidence intervals for precision and recall.

### Nonparametric diagnostics – Quantile and decile analysis Correlation is compact, but it can hide non-monotone structure. At each decision time 𝑡, sort assets

by 𝑥௧ǡ௔ within the eligibility mask, split into 𝑄 bins, and compute the mean label per bin 𝜇௧ǡ௤. Track the spread 𝑆௧׷ൌߤ௧ǡொെߤ௧ǡଵ and summarize within folds.

![Figure 7.5](assets/figure_7_5.png)

*Figure 7.5: Quantile bar charts: monotone versus non-monotone. Left: mean forward return by feature decile increases monotonically, consistent with a ranking-based mapping. Right: a U-shaped pattern where both extreme deciles show elevated returns.*

*Figure 7.5* shows that a monotone mapping will miss the signal in the bottom decile; this implies either Monotone increasing patterns are compatible with ranking-based approaches (top-𝑘 selection, long– a nonlinear mapping or a confounder.

short spreads). Non-monotone patterns may require nonlinear mappings or suggest an interaction with market state. At this stage, you are validating shape, not designing the mapping.

### Preliminary feasibility checks

simplest possible mapping - rank assets by score and select the top 𝑘 - as a stress test. Three checks Feasibility screens ask whether any plausible mapping could survive implementation. These use the

cover most early failures: **Turnover proxies**: Measure entry and exit rates in the top-𝑘 set across decision times within each fold. A feature in which the top-𝑘 set turns over completely at each rebalance is a warning; 1.

one in which positions persist has a structural cost advantage.

1. **Break-even cost sanity checks**: Compare typical fold spreads to conservative cost estimates. If spread is 5 bps - barely above noise. A signal whose gross spread does not clear 2 × estimated the median fold spread is 20 basis points (0.2 %, bps) and round-trip costs are 15 bps, the net

costs warrants caution; one that fails to clear the cost hurdle is a stop.

1. **Capacity warnings**: Recompute IC or spreads by liquidity bucket. A signal confined to the least liquid bucket is a stop for the current setup; smooth degradation from liquid to illiquid suggests the signal is broad-based.

Standardize triage decisions so later stages can trust the results:

| Decision | Condition | Action |
| --- | --- | --- |
| Proceed | Correctness screens pass; primary diagnostic directionally consistent across folds; quantile/ confusion-matrix supports plausible mapping | Hand off to the modeling chapters |
| Revise | Idea plausible, but definition falwed (anchor mismatch, horizon mismatch, unstable base rates) | Change definition, re-run triage, record delta |
| Stop | Sign unstable across folds; spread too small to survive costs; signal confined to untradeable liquidity | Archive and document the failure mode |

*Table 7.2: Triage decisions standards*

Read triage output in order: correctness first, then association, then shape and feasibility, then search accounting. Record the decision, diagnostic summaries, and searched-set metadata in the research ledger alongside the definition bundle (see *Box 7.1*).

Feature evaluation is not model selection, hyperparameter search, or a backtest. Its output is an auditable decision (proceed, revise, or stop) supported by fold-level summaries and plots that can be regenerated from the run record. The next section addresses the risks associated with an extensive search for predictive features.

**Implementation**: `05_signal_evaluation` covers correctness screens, fold-level IC and ICIR, IC decay, turnover, and break-even costs for the ETF momentum signal. `06_ic_inference` covers HAC inference, stationary block bootstrap, and minimum-track-record planning.

## 7.4 Search accounting and multiple testing

Trend scanning (*Section 7.2*) applied Bonferroni correction to a single observation’s t-statistic when the best horizon is selected from many candidates. The same logic (correcting for the number of comparisons made) applies at the search level. When triage screens many candidate features, labels, or variants, the best-observed IC, spread, or AUC come with a positive bias: some candidates will look good by chance.

### Define the searched set

The starting point is to record what you actually evaluated. Define the **search set** as all candidates tested under the same decision rules. This typically includes

- Label-definition variants (anchor, horizon, threshold, or barrier rules)
- Feature-family variants (lookbacks, transforms, reference frames)
- Any conditioning templates you have tried

Record the searched-set size and the generation rules (grid, deduplication, exclusions). Without this, “significance” claims are uninterpretable.

For example, evaluating 5 label horizons × 3 threshold rules × 10 feature variants gives 150 bundles. Even if each bundle has only a 5% chance of passing a naive significance test under the null, we would expect roughly 7 - 8 false positives. The searched-set size is 150, and that number must accompany any reported p-value. See `07_multiple_testing` for a simulation showing how selection bias inflates the best IC when all factors are noise.

### Separate exploration from confirmation

There are two phases to evaluation:

1. **Exploration pass**: Evaluate many candidates, but promote primarily on fold stability rather than peak performance (sign consistency, lack of single-episode dominance, robustness to timing and coverage checks).
2. **Confirmation pass**: Freeze promoted candidates and re-run triage with a minimized comparison set to produce the summaries you will cite later.

The exploration pass is where we learn; the confirmation pass is where we commit. Mixing the two is the most common cause of failure in multiple testing: promoting on peak performance, then “confirming” on the same data.

### Adjustment methods

If you report p-values or confidence intervals, compute them on fold summaries and adjust over the searched set you actually evaluated. Two families of correction dominate:

**Family-wise error rate** (**FWER**) controls the probability of *at least one* false positive across all tests. Use FWER when promoting a small number of candidates from a large pool: the cost of a single false The standard procedure is **Holm–Bonferroni**: sort the 𝑚 p-values in ascending order, 𝑝(1) ≤𝑝(2) ≤⋯≤𝑝(௠) positive (a strategy built around a spurious feature) is high. , and reject 𝑝(௜) only if 𝑝(௜) ൑ߙȀ(݉ െ݅൅ͳ) for all ݆൑݅. This is uniformly more powerful than the original Bonferroni (𝑝(௜) ൑ߙȀ݉

) while controlling the same error rate.

**False discovery rate** (**FDR**) controls the expected fraction of false positives among all rejections. FDR is appropriate for large screens where you expect many true positives and can tolerate some false discoveries: for example, when screening 200 feature variants and expecting to advance 20–30.

![Figure 7.6](assets/figure_7_6.png)

*Figure 7.6: Selection bias under the null*

*Figure 7.6* shows three graphs. (a) Among 100 noise factors (50 assets, 252 days), the best achieves an IC IC is 0.022  -  a spurious signal that would pass most screens. (c) Naive testing (𝑝05) yields 5 false of 0.020, close to the theoretical maximum of 0.023. (b) Across 200 simulations, the median best-factor

The standard procedure is **Benjamini–Hochberg**: sort p-values and reject 𝑝(௜) if 𝑝(௜) ≤(݅Ȁ݉ discoveries per simulation; BH-FDR control reduces this to near zero. ) ڄ ݍ, where 𝑞 is the target FDR (for example, 0.05). The notebook `07_multiple_testing` applies both BH and

statistical power. Applied to 13 ETF features, zero survive BH-FDR or Holm-Bonferroni at 𝛼05: Holm–Bonferroni to a simulated factor zoo and compares discovery rates, false-positive counts, and RVol 10d leads with HAC 𝑡36 but is rejected after correction.

The choice depends on the **downstream cost structure**:

- If each promoted feature incurs expensive modeling and backtesting costs, prefer FWER
- If you are building a feature pool where a few false positives are tolerable because model selection will weed them out, FDR is more practical nificance threshold to 𝑡 (from 2.0) for new factor discovery  -  a practical shortcut that accounts Harvey, Liu, and Zhu (2016) surveyed over 300 published factors and recommended raising the sig-

for the accumulated search across the literature.

When candidates are correlated (lookback variants or parameter sweeps around a single factor), independence-based corrections overstate the penalty. **Rademacher complexity** provides sharper bounds by measuring the effective richness of the hypothesis class (see *Chapter 16*).

### Report effect sizes, not only significance

Adjustment methods control error rates but say nothing about economic magnitude. For any candidate carried forward, report:

- Searched-set size and generation rules
- Fold-level summaries (median and worst fold) for the primary diagnostic
- A stability indicator, such as sign consistency and time concentration
- Whether the numbers come from the exploration or confirmation pass

**Box 7.4: Adaptive choices are additional trials**

If a threshold, lookback grid, missingness rule, or interaction was chosen or modified after reviewing results, treat it as a new trial family and include it in the search-set accounting. These are legitimate research moves, but they increase the multiple-comparison burden. This is the single most common accounting failure: optimizing a parameter after peeking at results and not counting it as a trial.

#### Strategy-level selection-bias diagnostics

When the outcome is a Sharpe ratio rather than an IC, three analogous diagnostics apply:

- The **Deflated Sharpe Ratio** (Bailey and López de Prado, 2014) estimates the probability that the best Sharpe exceeds chance
- the **Probability of Backtest Overfitting** (Bailey et al., 2015) measures how often the in-sample best-ranked configuration becomes the out-of-sample worst
- **Minimum Track Record Length** quantifies the amount of data required before promoting a strategy

**Implementation**: `07_multiple_testing` implements the full pipeline: HAC-adjusted p-values, BH-FDR, and Holm–Bonferroni. It also previews Rademacher adjustments and the Deflated Sharpe Ratio (for more detail, see *Chapter 16*).

The next section provides a first introduction to causality.

## 7.5 From correlation to causality

The association diagnostics in *Section 7.3* establish that a feature carries information about the label. They do not explain why. A feature can appear predictive because it proxies for a market state characterized by volatility, liquidity, or risk appetite that also drives returns, rather than because it captures a stable, exploitable relationship.

This section introduces causal thinking as a **falsification filter**: state a mechanism, encode it as a graph, and then check whether the data are consistent with the assumptions implied by the mechanism. The point is not to prove causality but to rule out stories that fail to meet their own basic implications before committing to more complex modeling.

In practice, this means asking whether the observed association survives simple checks suggested by the assumed mechanism, or whether it disappears once plausible confounders, timing effects, or selection effects are taken into account. These are robustness diagnostics guided by **mechanism reasoning**; they test one feature at a time and cannot detect multivariate confounding. *Chapter 15* supplies that more comprehensive toolkit.

### Why causal thinking matters for features

Most trading data is observational. Nobody runs randomized experiments on asset prices. A feature that correlates with future returns may do so for reasons that will not persist. Causal thinking surfaces three common traps:

- **Confounding**: A feature and a label share a common driver; the correlation disappears when that driver shifts. For example, both momentum and forward returns may respond to the volatility regime: the observed correlation reflects a shared exposure, not a direct link.
- **Conditioning traps**: Adding the wrong “control” can create spurious relationships (collider bias) or remove the effect you care about (conditioning on a mediator). Whether a variable should be included as a control depends on the causal structure, not on its statistical significance.
- **Construction artifacts**: The pipeline inadvertently encodes information about selection or timing rather than about the market: for example, a rolling window that overlaps the label horizon, creating spurious correlation through shared data points.

### What is a directed acyclic graph?

A **directed acyclic graph** (**DAG**) is a diagram that encodes causal assumptions. Each node represents a variable (a feature, a label, or a third quantity), and each arrow represents a hypothesized direction of influence, one variable affecting another. “Acyclic” means no path through the arrows leads back to its starting point; there are no feedback loops. *Figure 7.7* illustrates the basic anatomy.

![Figure 7.7](assets/figure_7_7.png)

*Figure 7.7: Directed Acyclic Graph*

Why draw a DAG? Because it forces you to state your assumptions explicitly. A claim that “momentum predicts returns” is vague; a DAG that places an arrow from momentum to returns - and specifies that volatility regime affects both - is a testable structure. The arrows commit you to specific implications: which variables should be correlated, which should become independent once you condition on a third, and which conditioning choices are safe. If the data contradict those implications, the assumed mechanism is wrong, and you have saved yourself from modeling a spurious relationship.

In financial applications, DAGs are typically small (three to five nodes) and encode a single hypothesis about one feature. The goal is not a comprehensive model of the economy but a minimal structure sufficient to identify what could go wrong with a specific feature-label association. Three recurring patterns - confounders, mediators, and colliders - determine what you should and should not condition on.

### Three structural roles

Every variable adjacent to a feature-label pair plays one of three structural roles. Getting this right determines whether conditioning on a variable clarifies or distorts the relationship you are trying to evaluate. *Figure 7.8* illustrates each pattern with a minimal DAG:

![Figure 7.8](assets/figure_7_8.png)

*Figure 7.8: Fundamental causal relationships*

The three components shown in this diagram are:

- **Confounder: A common cause.** A confounder is a variable that influences both the feature and the label. In the DAG, it sits above both, with arrows pointing down to each. For example, the volatility regime (measured by the VIX) affects both the strength of the momentum signal and return dispersion. If you do not account for it, the feature-label correlation is inflated by the shared exposure. *Conditioning on a confounder blocks the spurious path* and isolates whatever direct relationship may exist. In the worked example below, the regime heterogeneity check partitions by VIX precisely to assess this.
- **Mediator: An intermediate mechanism.** A mediator lies on the causal chain between feature and label: the feature affects the mediator, which in turn affects the label. For example, momentum may partly operate through liquidity: past winners attract flows, improving liquidity and supporting further price appreciation. If you condition on liquidity, you remove the indirect channel and may eliminate the very signal you are measuring. *Do not condition on a mediator* unless your goal is specifically to test whether the feature has any effect beyond the mediated path.
- **Collider: A common effect.** A collider is a variable caused by both the feature and the label. In the DAG, arrows from both the feature and the label converge on it. For example, both momentum and forward returns drive fund flows: strong-momentum funds with high returns attract the most capital. Unconditionally, there is no distortion. But if you *condition on the collider* (say, by restricting your analysis to high-flow funds) you induce a spurious negative correlation between momentum and returns that does not exist in the full population. The notebook `08_causal_sanity_checks` includes a synthetic simulation showing this effect: conditioning on fund flows (the collider) creates an approximately −0.25 correlation between two independent variables.

Before controlling for any variable, decide which role it plays. If you cannot decide, flag the feature for the multivariate toolkit in *Chapter 15*.

### From DAG vocabulary to falsification checks

Drawing a DAG is the first step; testing its implications is the payoff. Each arrow in a DAG implies specific, testable patterns in the data: for example, that conditioning on a confounder should reduce a correlation, or that shifting a feature in time should cause its predictive power to decay. The four checks below probe these implications cheaply, using tools you already have from *Section 7.3*.

The diagnostics produce one of three outcomes, matching the triage gates introduced in *Section 7.3*:

- **PROCEED** (the mechanism survives all checks)
- **REVISE** (partial survival: the feature has a signal, but with caveats that inform downstream modeling)
- **STOP** (the mechanism fails: redirect effort to other features)

In efficient markets, cross-sectional ICs are small, and few single features will cleanly pass every check. REVISE is the expected and most informative outcome: it tells you *where* and *when* a feature works, which is precisely what downstream model design needs to know.

![Figure 7.9](assets/figure_7_9.png)

*Figure 7.9: The mechanism plausibility workflow from hypothesis to triage decision*

### Mechanism plausibility checks

Each check has a clear protocol: what you compute, what you expect if the mechanism is real, and what constitutes a red flag. The notebook `08_causal_sanity_checks` first runs a feature × horizon scan - expanding on the null result from *Section 7.4* - and then applies the first three checks to two Try a **timing placebo.** Shift the feature backward by 𝛥 bars (for example, 5, 21, 63, 126, 252 days) and selected features.

0 and decay as the feature becomes stale. For a rolling-window feature with lookback 𝐿, a 𝛥-shifted recompute IC at each lag. If the feature contains timely information, IC should be strongest near lag version shares roughly (ܮെ߂)Ȁܮ of its inputs with the original, creating a mechanical floor on IC

persistence; focus on decay at lags beyond the lookback window. A red flag: IC that remains flat or *increases* at distant lags suggests the feature proxies a slow-moving state variable rather than timely information.

A **shared-driver check** replaces the label with an outcome that has no plausible link to the feature’s mechanism: for example, Treasury returns for a microstructure-driven equity feature, or a shuffled cross-sectional assignment. IC should be indistinguishable from zero. This tests whether the feature’s information is specific to the hypothesized channel or leaks into unrelated outcomes via a shared driver. A second control uses permutation: shuffling forward returns cross-sectional data at each date, producing a null distribution; the observed IC should lie far outside it.

**Regime heterogeneity** is important. Partition the sample by a candidate state variable (for example, VIX terciles) and recompute IC within each partition. IC may attenuate in some regimes, but should partition (HAC |ݐ| > 2) and a near-zero unconditional IC - the unconditional association is a pure maintain sign and rough magnitude. If IC flips sign across partitions - with a significant opposite-sign

aggregation artifact. This check cannot distinguish confounding from genuine effect modification; a feature whose IC varies by regime may be confounded or may have a mechanism that operates differently across states. For features motivated by a discrete event (earnings, FOMC, rebalancing), compute IC in event-time windows. IC should concentrate where the mechanism predicts an effect. Symmetric pre- and postevent ICs, or ICs peaking before the event, usually indicate anticipation, leakage, or a confound. The example below omits this check because neither feature is event-driven.

#### Worked example – 12-1 momentum versus short-term reversal

The multiple-testing scan in *Section 7.4* found that no short-lookback feature survived BH-FDR at a 5-day horizon on ETFs. The notebook expands the search: a scan of 10 features across three horizons (5d, 21d, 63d) reveals that long-lookback momentum features (126d+) carry significant cross-sectional information at monthly horizons, confirming academic findings on cross-asset momentum (Asness et al., 2013). Two features are selected for the mechanism checks, using a 21-day forward return label.

**Feature A: 12-1 Momentum (REVISE)**

| Check | Result | Interpretation |
| --- | --- | --- |
| Timing placebo | IC peaks at lag 0 (0.053); persists to lag 252d (0.034), HAC | IC strongest at lag 0—partly mechanical for 231d lookback, but timely |
| Shared-driver check | Treasury HAC 𝑡 perm | No shared driver; IC well above permutation null |
| Regime heterogeneity | Low-VIX IC = +0.086 (HAC 𝑡); high-VIX IC = +0.017 (𝑡) | Sign stable; magnitude varies 5× across VIX regimes |
![Figure 7.10](assets/figure_7_10.png)
![Figure 7.11](assets/figure_7_11.png)
*Table 7.3: Momentum analysis*

tional information (IC = 0.053, HAC 𝑡) that does not predict Treasury returns, but the timing Momentum earns REVISE: only the shared-driver check passes cleanly. It carries genuine cross-sec-

placebo and regime heterogeneity checks both flag caution - IC persists without clear decay, and the effect concentrates in low-volatility markets, attenuating during high-VIX episodes, consistent with the well-documented “momentum crash” phenomenon (Daniel and Moskowitz, 2016). This is an actionable finding that motivates regime-conditional modeling in later chapters.

**Feature B: 1-day Reversal (STOP)**

| Check | Result | Interpretation |
| --- | --- | --- |
| Timing placebo | No timely signal (IC₀ = 0.002, HAC 𝑡 = 0.4) | Near-zero IC at all lags - no information to decay |
| Shared-driver check | Treasury HAC 𝑡 perm | IC indistinguishable from shuffled noise |
| Regime heterogeneity | Low-VIX IC = −0.019 (HAC 𝑡); high-VIX IC = +0.028 (𝑡) | Sign flips (HAC sig.) with near-zero unconditional IC - aggregation artifact |
![Figure 7.12](assets/figure_7_12.png)
*Table 7.4: Reversal analysis*

Reversal earns STOP: all three checks fail. The unconditional IC is near zero (0.002, HAC 𝑡)

with no timely signal at any lag, and the regime heterogeneity check reveals a textbook aggregation artifact - the feature encodes opposite information in low-VIX and high-VIX markets, producing a near-zero unconditional IC.

### Plausibility scorecard

The following table summarizes the decision criteria for each check:

| Check | Pass | Caution | Stop |
| --- | --- | --- | --- |
| Timing placebo | IC significant (HAC) with meaningful decay | IC persists without clear decay | No timely signal; IC increases at distant lags |
| Shared-driver check | IC ≈ 0 on control outcome (HAC) | IC is small but nonzero | IC is significant on control (HAC) |
| Regime heterogeneity | IC is stable across partitions | IC varies; sign flip not sig. (HAC) | Sign flip sig. (HAC) AND unconditional IC ≈ 0 |
| Event-time alignment | IC concentrates post- event | IC spread across windows | IC peaks pre-event |

*Table 7.5: Plausibility scorecard*

This is visualized in the following figure:

![Figure 7.13](assets/figure_7_13.png)

*Figure 7.10: Mechanism plausibility scorecard*

*Figure 7.10* shows the timing placebo decay curves, regime-conditional IC, and permutation null for the two features. Momentum IC decays gradually from 0.053 to 0.034 over 252 days, consistent with a slow-moving signal that is still strongest when most current. Momentum IC concentrates in low-vol𝑝001) lies far outside the permutation null; reversal IC of 0.002 (permutation 𝑝535) is indisatility markets (IC = 0.086 in low-VIX versus 0.017 in high-VIX). Momentum IC of 0.042 (permutation

tinguishable from noise. All t-statistics use **Newey–West** (**HAC**) standard errors to account for serial dependence in overlapping IC series. Features that fail multiple checks should be investigated before proceeding; features that pass all checks have survived a first filter but have not been shown to be causal: these are bivariate diagnostics that cannot detect multivariate confounding.

These checks can reject proposed mechanisms but cannot confirm them. The natural next step is to move from single-feature diagnostics to the multivariate toolkit: structural models, double- or debiased machine learning, and sensitivity analysis. That machinery is reserved for the survivors and lives in *Chapter 15*.

**Implementation**: `08_causal_sanity_checks` runs the collider simulation, the feature × horizon scan, and the timing-placebo, shared-driver, and regime-heterogeneity checks on ETF returns.

## 7.6 Summary

Defining the learning task well is one of the highest-leverage steps in the quantitative workflow. This chapter developed five interlocking disciplines that aim to separate reliable features from research noise. Clean, auditable preprocessing (*Section 7.1*) ensures that every input is comparable across trials and free of encoding surprises. Execution-consistent labels (*Section 7.2*) anchor predictions to tradeable prices with explicit timing, lag, and horizon conventions; a mismatch of even one bar can fabricate or erase a possible edge. Fold-aware evaluation (*Section 7.3*) surfaces instability that pooled statistics hide: a feature whose IC flips sign across walk-forward folds is a lot less useful, regardless of its aggregate statistic. Search accounting (*Section 7.4*) records the number of candidates tested and applies multiple-testing corrections so that the best-observed IC is not simply the luckiest. Finally, mechanism plausibility checks (*Section 7.5*) encode causal assumptions as directed acyclic graphs and test their implications cheaply - catching confounded proxies and aggregation artifacts before they consume modeling effort.

Together, these disciplines feed a single decision framework. The triage gates (proceed, revise, or stop) convert diagnostics into auditable decisions at every stage. A candidate that advances carries a definition bundle (label specification, feature configuration, preprocessing choices), fold-level diagnostics (IC, ICIR, sign consistency, quantile spreads), a searched-set record (how many variants were tested, which correction was applied), and a mechanism assessment (which plausibility checks it passed and where its signal concentrates). This documentation also serves as the minimum audit trail to make results reproducible and failures diagnosable.

The shared `ml4t.*` ecosystem wires this framework together under the canonical `timestamp + symbol` schema: `ml4t.data` for loaders, `ml4t.engineer` for feature recipes, `ml4t.diagnostic` for evaluation utilities, and `ml4t.backtest` for execution; see `10_ml4t_library_ecosystem` for the API tour.

With the framework set, *Chapter 8* translates strategy hypotheses into concrete feature specifications and applies the evaluation and triage gates built here to each family.
