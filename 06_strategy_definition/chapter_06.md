# Chapter 6: Strategy Research Framework

Every trading strategy starts with an idea. The hard part is turning that idea into trustworthy evidence: results that survive realistic costs and constraints, remain interpretable after iterative search, and translate into a repeatable process that performs out-of-sample rather than a one-off in-sample backtest. We introduce the strategy research framework that anchors the ML4T workflow: a research loop we develop in *Chapters 7-20*, from feature engineering and modeling to strategy backtesting, before taking a strategy live in *Chapter 25*.

The framework requires us to define the trading setup early: what is traded, when decisions are made, how signals become positions, and how positions produce net outcomes after costs, constraints, and risk controls. Some parameters are known from the outset (for example, the universe of investable assets, or explicit fees); others require further research (for example, market impact or risk management). The goal is to make dependencies visible, state conservative assumptions for early feasibility screens, and record open issues for later chapters.

By the end of the chapter, you will be able to:

- **Place an idea on the strategy map,** connecting it to a strategy family, a plausible source of recurring profits, and the dominant feasibility constraints.
- **Specify the trading setup** in decision-time terms: what is tradable, when decisions are made, what is admissible, and how scores become positions.
- **Define “better” economically**, and separate metric roles so diagnostics guide iteration without substituting for strategy evaluation.
- **Adopt a time-series evaluation protocol** that separates selection from performance estimation.
- **Define a baseline checkpoint** that is narrow by design and supports early feasibility screens before broad search.
- **Keep search countable and recoverable** using automatic run logging and a simple trial taxonomy.

In practice, these choices live in code: a small set of versioned configuration objects and a pipeline that consumes them. Every run automatically emits a record (config, splits, metrics, outputs) so that results are reproducible and iterations are countable.

## 6.1 From idea to evidence with the ML4T workflow

A trading strategy is an executable process. Once live, it runs on a schedule, uses the information available at decision time to produce model inputs, converts model outputs into positions, and realizes outcomes after accounting for costs and constraints. This workflow is shown in *Figure 6.1*.

![Figure 6.1](assets/figure_6_1.png)

*Figure 6.1: Live trading and the research loop*

During research, we simulate this process using historical data. The goal is to optimize the process by iteratively refining each component while keeping comparisons meaningful and interpretable, so that historical simulation matches the live system’s behavior at decision time.

### The live trading loop

Every systematic strategy - whether it uses a hand-built signal or an ML model - can be described as a sequence of steps that repeat on a fixed cadence:

1. **Observe a decision snapshot**. At each decision time, the strategy takes a snapshot of the available information that defines what it “knows” when it commits to trades.
2. **Compute features and produce a score**. This data is transformed into a set of features for a model (or a rule-based signal) to predict rankings, probabilities, return forecasts, or other scores useful for trading decisions.
3. **Translate the score into positions**. A trading policy converts these predictions into target positions, subject to risk limits and constraints, and places orders accordingly.
4. **Execute and realize net outcomes**. Orders become fills (completed trades) with slippage, spreads, fees, financing, funding, roll costs, and other market-specific mechanics. Positions generate profit and loss (PnL) and exposures that are measured and monitored.
5. **Monitor, update, and repeat**. Performance, risk, and data quality are tracked; operational exceptions are handled; the loop repeats at the next decision time.

This loop is only meaningful when its timing is well defined. It is impossible to use information that is not available when trading live, but deceptively easy to do so during research on historical data. We must therefore clearly define the timing of trading decisions and keep these conventions stable, so that improvements can be attributed to deliberate changes rather than to shifting assumptions.

### The research loop

Research builds evidence about how the live loop will behave without “moving the goalposts” as we iterate. We start with a baseline version of the live loop, evaluate it using a time-series protocol, diagnose weaknesses, and change one component at a time, while accounting for measurement noise. A practical research loop looks like this:

1. **Define the evaluation environment**. Declare the tradable universe, the decision cadence, basic constraints, and the cost components treated as material. This setup makes results comparable across iterations and avoids surprises during deployment.
2. **Build a baseline pipeline**. Implement the simplest end-to-end version that is consistent with the setup: minimal features, a baseline label, a baseline model class, and a simple mapping from scores to positions.
3. **Evaluate under a time-series protocol**. Compare candidates using time-aware validation and reserve a holdout test set for confirmation once the pipeline is stable. This is where we prevent leakage and selection bias from dominating conclusions.
4. **Diagnose, then revise**. Use diagnostics to decide what to change next: data definitions, feature families, model class, hyperparameters, mapping class parameters, constraints, or cost assumptions. Make changes deliberately and keep the rest fixed.
5. **Log what happened**. Record enough metadata to reproduce each run and to count the alternatives tried. Without this, iteration becomes untraceable, and selection bias becomes invisible.

The loop does not assume a single crisp hypothesis at the outset. In ML-driven research, we typically explore a **hypothesis class:** a family of measurements that might, in combination, yield better trading decisions. The discipline is not “never explore” but to explore within a stable setup and robust protocol so that results remain interpretable.

### Illustrating the research framework with case studies

Nine case studies illustrate the strategy research workflow throughout our remaining chapters. Each is a scaffold for timing, feasibility, and evaluation, not a recommended trading system. They span multiple asset classes and data frequencies, drawing on the datasets from *Chapters 2* and *3*. For each case study, we highlight the asset class, data frequency, and coverage in terms of the number of symbols and the time period.

- `etfs:` *Multi-asset · Daily · ~100 symbols over 20 years.* Cross-asset rotation strategies that highlight turnover economics and regime dependence across equities, fixed income, and commodities.
- `us_equities_panel:` *Equities · Daily · ~3,000 symbols over 50 years.* The workhorse cross-sectional dataset for signal construction, universe filtering, and capacity analysis at scale.
- `us_firm_characteristics:` *Equities · Monthly · ~2,500 stocks · 1996**2016.* Factor models built on ~57 firm-level characteristics, emphasizing point-in-time discipline to avoid survivorship and lookahead bias.
- `fx_pairs:` *FX · 4-hour bars / daily NY 5PM decisions · 20 pairs · 2011**2025.* Currency strategies that confront close definitions, bar aggregation choices, and tradable return construction across time zones.
- `cme_futures:` *Futures · Daily · 30 contracts over 15 years.* Trend and carry strategies that require careful session alignment, roll mechanics, and term-structure modeling.
- `crypto_perps_funding:` *Crypto · 8-hour · ~20 symbols over 5 years.* Perpetual futures, where funding schedules, exchange-specific market mechanics, and transaction costs dominate signal economics.
- `nasdaq100_microstructure:` *Equities · Minute (downsampled to 15 Minutes) · ~114 symbols over 2 years.* High-frequency features that expose timing conventions, execution realism, and the gap between theoretical and realized fills.
- `sp500_equity_option_analytics:` *Equities + Options · Daily · S&P 500 over 5 years.* Equity strategies enriched with option-derived features such as implied volatility, skew, and put-call ratios.
- `sp500_options:` *Options · Daily · S&P 500 over 5 years.* Options strategies that foreground payoff accounting, hedging turnover, and the financing assumptions behind net returns.

The next section presents strategy families and sources of edge, using the case studies to make the strategy map concrete.

## 6.2 Mapping strategies and sources of edge

Most strategy ideas start as narratives: “prices underreact,” “carry is a premium for bearing risk,” or “flows create predictable pressure.” A map turns those narratives into a usable design tool. Before building models or running large backtests, the map should answer two questions in operational terms:

- **What is the strategy’s structure?** What is traded, how often are decisions made, how long is risk held, and which frictions dominate at that cadence?
- **Why should this earn net returns, and why should it persist?** What economic force could plausibly generate returns after costs and constraints, and what would have to remain true for the effect to survive competition and regime change?

The map uses two complementary lenses. **Strategy families** classify ideas by cadence, information source, and the frictions that become first order at that cadence. They identify what must be fixed early. **Sources of edge** are economic narratives about why a premium might exist and what persistence depends on. They guide stress tests and failure-mode hypotheses. A credible story with an infeasible implementation is not a strategy. A feasible implementation without a credible story is a backtest whose failure modes remain uninterpretable. **Box 6.1: Time notions to keep separate**

The recurring activity of a trading strategy implies several periods and points in time that are important to define:

- **Decision cadence:** How often the strategy is allowed to change its position. This is the schedule for decisions and rebalancing. A strategy might enter only once per day, for example, but monitor exit signals intraday.
- **Holding period:** How long risk is typically held once a position is entered. It can be fixed (every 5 days, for example) or variable (exit on a signal or risk control).
- **Forecast horizon:** The future interval the model is trained to predict. It does not need to match the holding period when predictions span a given horizon and positions are held until a reason to exit manifests. A strategy can also use multiple models for entry and exit signals, each with a different horizon.
- **Lookback window:** How far back features look when formed at decision time. It determines the historical span available to features. Different model inputs can use different windows, irrespective of holding and forecast periods.

![Figure 6.2](assets/figure_6_2.png)

*Figure 6.2: Strategies illustrated in edge and holding-period terms, spanning the strategy families (cadence axis) and sources of edge developed in the rest of this section*

### Strategy families as feasibility filters

Families overlap. The same label (for example, “momentum”) can refer to both monthly cross-sectional sorting and intraday trend-following. The family does not uniquely determine the cadence; it identifies what will break first once one is chosen. Use the family whose dominant constraints match the intended cadence, and treat the family as a feasibility checklist: what must be fixed *early enough* to make results interpretable.

#### Price-based strategies

These strategies generate signals from past prices, returns, or realized volatility, including time-series trend-following, cross-sectional momentum, and short-horizon mean reversion (Moskowitz et al., 2012; Hurst et al., 2017; Jegadeesh and Titman, 1993; Asness, Moskowitz, and Pedersen, 2013). At short cadences, turnover and execution realism become first order. At longer cadences, risk concentrates in episodic drawdowns and regime shifts (momentum crashes are a known stress case; Daniel and Moskowitz, 2016).

To make a price-based idea testable, fix the holding horizon and rebalance cadence, lookback window, and score-to-position mapping. First screens: does performance hold up to modest changes in lookback/rebalance choices, and does it remain plausible under conservative, turnover-aware costs?

**Case studies**: `etfs`, `us_equities_panel`, and `fx_pairs` illustrate turnover economics, exposure control, and regime sensitivity at daily-to-monthly horizons.

#### Fundamental and valuation strategies

These strategies trade on cross-sectional differences in valuation, quality, growth, or profitability, typically at monthly to quarterly cadences. Their defining constraint is not speed but the **decision-time meaning** of the inputs. The most common failure mode is accidental use of information unavailable at decision time (see *Chapters 2* and *4* on **point-in-time conventions** and **coverage rules)**. When signals move slowly, the **effective sample size** is much smaller than the row count suggests.

**Case studies**: `us_firm_characteristics` is the primary anchor for point-in-time discipline, coverage rules, and the interaction between signal definitions and universe construction.

#### Flow and microstructure strategies

These strategies target small edges per trade using order-flow proxies, quote dynamics, or microstructure-motivated features. As cadence speeds up, the problem shifts from “forecast returns” to “capture microstructure effects under realistic trading rules.” Feasibility depends on timing discipline, execution assumptions, and market access. Precision matters about what is observable at decision time, what latency is assumed, and how fills scale with size relative to depth and volume.

**Case studies**: `nasdaq100_microstructure` prioritizes timing conventions and execution realism as first-order concerns.

#### Market mechanics and payoff strategies

These strategies depend on instrument mechanics that form part of the economic payoff: futures roll and term structure, perpetual funding, margin and financing, option payoffs and hedging turnover, settlement conventions, and contract eligibility rules. Testability requires an explicit **return decomposition** and explicit **lifecycle rules** (contract selection, roll schedule). First screens: component attribution, rule stability, and stress behavior under binding constraints (margin, funding, forced deleveraging).

**Case studies**: `cme_futures` and `crypto_perps_funding` are mechanics-first examples. `sp500_equity_option_analytics` and `sp500_options` anchor options research, where payoff accounting, surface timing/coverage, hedging turnover, and financing assumptions dominate interpretation. **A note on “machine learning strategies.”**

Machine learning is not a family of strategies. It is a modeling and decision-support layer that can be applied to any family. Because ML expands the effective degrees of freedom (more features, more model classes, more hyperparameters), it increases, rather than reduces, the value of a bounded strategy definition and disciplined evaluation.

### Defining edge, alpha, and capacity

An **edge** is a repeatable advantage that makes the distribution of **net** trade outcomes favorable under a given strategy, reflecting both the likelihood of gains versus losses and their magnitudes after costs. Two related terms matter:

- **Alpha** is performance relative to a benchmark (an index or factor model) representing alternative opportunities or compensated risk exposures. A strategy that cannot outperform a passive baseline with similar exposures and constraints may not be worth pursuing.
- **Capacity** is the capital that can be deployed before the edge degrades from impact, fees, financing constraints, or opportunity-set limits. Capacity does not determine whether an edge exists, but often determines whether it is economically usable at scale.

### Sources of edge as durability filters

Feasibility addresses whether the idea can be tested credibly. Durability addresses what must remain true for the edge to persist. McLean and Pontiff (2016) found that published predictors lose roughly half their predictive power post-publication, partly due to overfitting and partly to arbitrage. The implication: edges that depend on information alone tend to erode; edges that depend on risk tolerance, constraints, or capacity limits can persist even when widely known. Real strategies often combine multiple sources of edge; the goal is to be explicit about the dominant ones and to test their corresponding failure modes.

A useful organizing framework distinguishes six sources of excess returns based on *why* the opportunity persists (Paleologo, 2025):

| Source | Mechanism | Durability | ML4T examples |
| --- | --- | --- | --- |
| Risk compensation | Bearing exposures others avoid (tail, volatility, illiquidity risk) | High — persists as long as investors are risk averse | cme futures carry, _ sp500 options volatility _ risk premium |
| Liquidity provision | Earning a premium for supplying immediacy or absorbing inventory | High — structural demand for immediacy | nasdaq100 _ microstructure intraday reversals |
| Funding constraints | Exploiting dislocations when capital is scarce or leverage is restricted | Moderate — episodic, strongest during stress | Short-horizon mean reversion after forced liquidations |
| Flow predictability | Front-positioning around mandated or rules-based demand (index rebalancing, hedging programs) | Moderate — erodes with crowding but recurs with each event | Index reconstitution trades, ETF creation/ redemption flows |
| Informational advantage | Better inference from public data: cleaner features, faster processing, superior models | Low to moderate — erodes as competitors replicate methods | ML-based feature engineering across case studies |
| Pure arbitrage | Exploiting identical- asset mispricing across venues or structures | Very low — fleeting, capacity-constrained, operationally intensive | Not represented in our case studies |

*Table 6.1: Comparing six sources of excess*

The following subsections expand on several of these sources and related mechanisms. Most real strategies blend two or more; the table helps identify which ones anchor a thesis and which failure modes to test first.

**Box 6.2: Why most published anomalies fail in** **practice**

The academic literature documents hundreds of cross- al predictors, yet very few survive as tradable strategies. Three gaps account for most of the attrition:

- **Post-publication decay.** McLean and Pontiff (2016) show that anomaly alphas decline roughly 50% after publication, partly due to overfitting corrections, partly due to arbitrage capital flowing in.
- **Implementation gap.** Academic studies typically use simplified portfolio construction (equal-weight long-short deciles, monthly rebalancing, no costs), which overstates live performance. Turnover, bid-ask spreads, and borrow costs often consume the reported alpha.
- **Detail sensitivity.** Small definitional choices (universe filters, rebalancing timing, winsorization thresholds) can flip the sign of a backtest result. A backtested anomaly is a hypothesis, not evidence; the evidence comes from out-of-sample evaluation under a fixed protocol (*Sections 6.5* and *6.6*).

The defense is not to avoid published ideas but to treat them as starting points that require independent validation within a researcher’s own trading setup and cost assumptions.

#### Slow adjustment and behavioral responses

Prices may incorporate information gradually due to limited attention, delegated decision-making, or slow portfolio rebalancing, motivating underreaction narratives behind momentum and intermediate-horizon trend effects (Jegadeesh and Titman, 1993). **Early failure-mode tests**: Examine drawdown and rebound episodes (momentum crashes; Daniel and Moskowitz, 2016) and check whether performance concentrates under benign conditions but reverses when volatility spikes or liquidity dries up, a pattern that often reveals that alpha partly compensates for state-contingent risk.

#### Risk compensation

Some returns compensate for bearing exposures others avoid: liquidity risk, tail risk, volatility risk, or balance-sheet-intensive exposures. Persistence can coexist with widespread awareness because the constraint is risk tolerance and leverage capacity, not information.

**Early failure-mode tests**: Performance during tail episodes when constraints bind (margin/funding shocks, forced deleveraging) and performance conditional on volatility and liquidity proxies.

#### Market segmentation and limits to arbitrage

Segmented markets and institutional constraints create persistent wedges: funding differences, capital charges, short-selling constraints, inventory limits, and client-flow internalization. The edge persists because the arbitrage is balance-sheet intensive or operationally constrained.

**Early tests:** Stability under stressed financing and fee regimes, realism of access assumptions, and capacity/crowding behavior.

**Box 6.3: Why prediction alone is insufficient**

A correct forecast does not guarantee a profitable trade. When 3Com announced the IPO of its Palm subsidiary in 2000, Palm’s implied valuation quickly exceeded that of the parent - a textbook mispricing visible to every market participant. Yet the trade was nearly impossible to execute: Palm shares available to borrow were scarce, borrowing fees spiked to extreme levels, and uncertainty about the spin-off timeline created open-ended funding risk. The mispricing persisted for months despite being obvious. For ML-driven research, the lesson is concrete: before engineering features for a mispricing hypothesis, verify that the short side is accessible, that borrowing and funding costs do not consume the spread, and that the holding-period risk is bounded. A model that identifies unexploitable opportunities is operationally useless regardless of its predictive accuracy.

#### Mechanical and institutional flows

Predictable demand from mandates, hedging, index rebalancing, roll schedules, or systematic execution programs creates transient or recurrent price pressure, driven by rules and constraints rather than beliefs.

**Early tests:** Calendar dependence and event alignment, decay under crowding, and execution sensitivity (flow edges often live near the spread). Index reconstitution is the standard example. When an index provider announces an addition or deletion, passive funds tracking that index must trade by the effective date, creating a demand shock whose direction, approximate magnitude, and timing are publicly known in advance. The edge is not in having private information but in positioning ahead of the mandated flow: predicting *which* securities will be added or removed, and *when* the demand will materialize. The risk is real: holding-period exposure to company-specific losses, potential event cancellations, and crowding as more participants exploit the same signal. As passive investment continues to grow (estimated at 35-40% of U.S. equity market capitalization), these flow events create larger dislocations and attract more competition, illustrating the tension between opportunity size and crowding that recurs across flow-based strategies. In our case studies, futures roll effects in `cme_futures` and ETF creation/redemption dynamics in `etfs` create analogous flow-driven patterns at different cadences.

#### Information and measurement advantages

Net returns can come from better inference: cleaner definitions, better timing discipline, more reliable labeling, and a more faithful mapping from observables to actions. This is rarely “free money”; it is often what turns a fragile backtest into a robust implementation within another category.

Early tests emphasize robustness to definition choices and conservative timing, sensitivity to coverage/ missingness, and the extent to which incremental signal quality survives realistic costs and constraints.

### Using the map

A strategy is well posed when the following questions can be answered:

- **Family and cadence:** Which frictions dominate?
- **PnL decomposition:** Which payoffs generate returns, and which costs are first order?
- **Edge narrative:** What is the primary economic source, and why should it persist?
- **Top failure modes:** Two or three ways the strategy could fail.

Answering these before expanding the search helps focus the next design decision: what to fix in the trading setup, which diagnostics to run first, and which stress tests are non-negotiable. We discuss the trading setup components next.

## 6.3 Defining the trading setup

Comparability breaks when the evaluation environment drifts without notice: the instrument set changes, the rebalance snapshot shifts, execution timing moves, or costs are treated differently. The **trading setup** is the fixed part of that environment: the invariants the code treats as constant within a research line, and every run automatically records. We can explore aggressively inside those invariants, but changing them defines a new setup and requires bumping the version.

### What the trading setup must fix

The trading setup fixes the decision-time snapshot, mapping form, and friction regime, rather than enumerating every lookback or threshold. It should cover the following topics:

#### Universe and tradability rules

Define eligibility and membership mechanics that make pricing and execution economically meaningful. Specify filters (liquidity, listing venue, contract type), and how listings, delistings, and stale or missing prices are handled. For example, `etfs` need explicit investability filters and rules for ETF launches and closures; `crypto_perps_funding` needs venue and contract eligibility and a rule for how convention changes (such as fees, funding caps, margining) are treated.

### Decision schedule and admissible information

Specify the trading snapshot and what is knowable at that moment. This is the guardrail against leakage, so it must be explicit: the timestamp convention, the bar frequency (if any), and any mechanical execution delay. `etfs` might use a weekly or month-end snapshot; `crypto_perps_funding` aligns decisions with the funding timestamp; `nasdaq100_microstructure` requires a fixed bar schedule and a stated observation-to-execution delay because a one-bar shift can change results.

#### Cadence and horizon feasibility

The cadence choice is intimately linked to costs: at each horizon, what fraction of typical price moves exceeds round-trip transaction costs? This principle determines feasibility before considering any signal.

Each case study includes a setup notebook with a horizon feasibility analysis that shows return distributions across multiple cadences and a cost reference line. These identify three regimes:

- A **hard floor** where costs clearly dominate (for example, `nasdaq100_microstructure` at 15-minute bars)
- A **gray zone** where feasibility depends on signal strength and turnover (for example, `etfs` for daily trading)
- A **comfortable zone** where costs are not the binding constraint (for example, `etfs` with monthly rebalancing)

The cadence decision is made in the following chapters, when signal diagnostics reveal whether features are sufficiently predictive, either on a standalone basis or in a model context, to justify trading at a given frequency.

#### Score-to-trade mapping

Define how a model’s score translates into orders, positions, and exits. This mapping determines trade frequency, holding-period distribution, and where costs and constraints bite. At a minimum, determine the following design parameters to ensure results are comparable:

- **Position state space:** For example, long-only, long/short, or multi-leg (spread/hedge).
- **Entry logic:** For example, rank selection or threshold gating.
- **Exit logic:** For example, time-, signal-, or risk-based exit.
- **Position sizing:** How scores translate into exposures (for example, equal weight, volatility targeting).
- **Order timing:** When orders are placed (at open, intraday window, execution delay).

The specifics will likely evolve, but when they do, the reference for comparison will as well.

#### Constraints and risk controls in scope

Specify the constraints treated as binding: exposure and leverage limits, concentration rules, and any liquidity or capacity limits enforced. Risk controls that alter trading actions (for example, a de-risking overlay or a kill switch) must be explicit and versioned, as they alter the effective strategy.

#### Cost model class and components

Early research does not need a perfect simulator, but it does require consistency about which frictions are treated as material. State the cost components included (fees, spreads, slippage/impact, financing or funding, borrow, roll) and how conservative the assumptions are. Later chapters expand the cost model; early screens use conservative, coarse assumptions.

*Example (perpetual futures)*: in `crypto_perps_funding`, funding is a position-dependent cash flow that is part of the total return stream: a cost when paid and a benefit when received. Because it accrues only when a position is held at the funding timestamp and is settled under venue-specific conventions, the trading setup must specify (i) how trades and position changes are aligned to the funding snapshot, (ii) which pricing and funding formula (including any caps/floors) and fee schedule are assumed, and (iii) how convention changes are versioned. Otherwise, the same backtest can imply different effective exposures and produce materially different PnL.

*Example*: The **ETF cross-asset momentum setup (v1)** defines the following invariants:

- **Universe:** 100 cross-asset ETFs with point-in-time eligibility based on trailing annual ADV ≥ $10M. The lenient threshold preserves cross-sectional breadth in early sample years (70–95 eligible ETFs per year). Note that the universe composition has survivorship bias that cannot be fully resolved without historical constituent data, while within-universe eligibility is pointin-time correct.
- **Decision schedule:** Monthly month-end close snapshot with execution at the next-day bar open. This aligns with momentum literature for benchmark comparability. Weekly (5-day) cadence is tested as a variant.
- **Mapping:** Long-only, rank-select top-N, equal-weight, rebalanced on cadence. Long-only because ETFs are expensive to short; equal-weight because it isolates the ranking signal from sizing optimization that would confound evaluation.
- **Constraints:** Fully invested long-only with no leverage, a minimal constraint set appropriate for a pedagogical baseline.
- **Cost model:** Material is a per-share commission ($0.0035/share, IBKR Pro Tiered) plus tiered half-spread slippage (0.5¢ for the most liquid mega-ETFs, 1¢ for sector ETFs, 2¢ default for thematic and regional ETFs). The horizon feasibility analysis uses a 5–15 bps-per-leg reference cost line to screen cadences and confirms that monthly moves exceed 30 bps round-trip costs in over 90% of observations; daily trading operates in the gray zone where feasibility depends on signal strength and turnover.

The setup notebook writes these choices to versioned YAML artifacts (`setup.yaml`) that downstream chapters consume programmatically. Changing the mapping from long-only to long/short would require `setup_version: v2` and a new baseline checkpoint; parameter changes, such as adjusting top-N from 10 to 20, remain within `v1`.

### When a change requires a new setup version

Treat the trading setup as versioned. Bump the setup version on any change that alters the strategy’s decision-time meaning or shifts its dominant frictions and constraints. Parameter tuning stays within a setup; changes to mechanics define a new setup. A change is a new setup version if it changes at least one of:

- **Universe or tradability rules**, for example, eligibility, membership mechanics, instrument scope.
- **Decision schedule or admissible information**, for example, rebalance snapshot, cadence, execution delay, and input availability rules.
- **Score-to-trade mapping** at the level of state space, entry or exit form, sizing normalization, or order timing.
- **Constraint regime**, such as binding exposure/leverage limits, concentration rules, capacity or liquidity constraints, action-changing overlays.
- **Cost model class or included components**, for example, adding funding or roll mechanics, or changing the slippage model in a way that affects feasibility.

To keep this boundary unambiguous, separate **mechanics** from **parameters**:

| Same setup version (parameter tuning) | New setup version (mechanics changed) |
| --- | --- |
| Threshold 1.8 → 2.0 | Rank selection → threshold gating |
| Top 10% → top 20% | Long-only → long/short |
| 20-day → 60-day normalization window | Adding a stop rule when none existed |
| Difefrent lookback horizon | Changing rebalance cadence or execution delay |
| Cost assumption 10 bps → 15 bps per leg | Adding funding/roll as a cost component |

*Table 6.2: Comparing mechanics and parameters*

Results across setup versions can still be informative, but they are not direct competitors within a single research program. The next section defines what counts as improvement within a fixed setup, separating development diagnostics from strategy-level evaluation.

**Implementation**: `02_case_study_overview` inventories the trading setup for each of the nine case studies (universe, decision schedule, mapping, constraints, cost model) and is the source of the worked examples above.

## 6.4 Setting objectives and evaluation metrics

A research program needs an explicit definition of “better” before running large experiment grids. Here, “better” means **economic value under a fixed trading setup**: portfolio returns produced by mapping model outputs into positions and applying constraints and costs. This section specifies three important rules:

- Evaluate strategy candidates in **economic terms**, based on a consistent PnL definition
- **Keep metric roles separate** for model and strategy evaluation
- Choose **one primary selection metric for development** and a **separate reporting set** for strategy outcomes

Specific metrics appear in later chapters on features, models, and backtests; here, we establish the framework for how these metric layers interact.

### Three metric layers, each with a different role

Confusion in ML-driven strategy research often arises from relying on a single metric to address multiple questions. Treat metrics as three distinct layers, each tied to a specific decision:

- **Model diagnostics** answer a narrow question: can the model learn the label in a way that generalizes across time splits? Typical checks include loss or error, calibration, and stability across folds. If these fail, treat downstream trading results as difficult to interpret because the predictor is unstable. These checks do not require a cost model, position sizing, or portfolio simulation; they serve as an early gate that prevents overinterpreting noise.
- **Signal diagnostics** test whether the model output behaves like a tradable signal under the chosen mapping class. A common choice is the **information coefficient (IC),** the Spearman rank correlation between predicted scores and subsequent realized returns (see *Chapter 7*). Signal diagnostics treat the output **as a score to be traded**; model diagnostics treat it **as a predictor to be trusted**. *Chapter 7* covers both in detail.
- **Strategy outcomes** evaluate the end-to-end process under costs and constraints, in terms of risk and return. Typical reporting includes risk-adjusted returns, volatility and drawdowns, exposure and concentration behavior, and realized turnover (see *Chapter 17*). The failure mode is to use these outcomes as the objective for every small design choice during development, which encourages overfitting to the simulator’s degrees of freedom and will likely disappoint out-of-sample.

A complementary diagnostic, **conformal prediction intervals,** provides calibrated uncertainty estimates without distributional assumptions. *Chapter 11* introduces conformal methods for time-series prediction; here, note that well-calibrated prediction intervals can flag when a model’s confidence is misaligned with realized outcomes, serving as an early warning alongside the three layers mentioned earlier. Use model diagnostics to assess learnability, signal diagnostics to assess tradability under the mapping, and strategy outcomes to determine whether the declared setup produces economic value. Keep strategy outcomes “late-stage”: when they drive every micro-decision, the risk of overfitting to implementation details rises, masking a robust edge. We now turn to evaluating our target metrics in a time-series context.

## 6.5 Evaluation protocol for time series

Evaluation answers a simple question: how will a procedure learned from historical data behave on data it has not seen? We estimate this by testing decisions on data unavailable when those decisions were made. The notebook `01_cv_foundations` illustrates these concepts using the `ml4t-diagnostic` library. The core discipline: separate selection from performance estimation, and enforce a fixed decision time so no future information leaks into training or simulation.

### Data leakage and estimation bias

**Look-ahead bias** occurs when any component of the pipeline uses information that would not have been available at the time of the trading decision. It is the most dangerous form of data leakage because it produces spectacular backtest results that cannot be replicated in live trading. Look-ahead bias manifests in several forms in the context of time-series evaluation (in addition to source data problems like survivorship bias or point-in-time errors, as discussed in *Chapters 2* and *4*):

- **Label leakage:** Using future returns in the same observation’s features (for example, computing 5-day-forward returns and accidentally including next-day returns in a feature).
- **Standardization leakage:** For derived features, computing inputs on the full dataset, instead of the training set alone. For example, using the mean and standard deviation to compute z-scores from the full data may reveal a distribution shift during the validation period.
- **Threshold leakage:** Setting thresholds using statistics computed from validation or test data. For example, choosing “enter when the z-score exceeds 2.0” after observing that 2.0 maximizes returns on the test period embeds future information in the trading rule.

Every component of the ML pipeline must respect point-in-time constraints: labels (*Chapter 7*), features (*Chapters 8-10*), validation splits, and signal calibration. A backtest that assumes access to information before it was knowable produces fictional performance.

### From k-fold cross-validation to time series evaluation

Standard *k-fold cross-validation* trains on, say, 80% of randomly selected observations and validates on the remaining 20%, repeating k times (Kohavi, 1995). This works when observations are IID, which implies they are exchangeable draws from a stable distribution. Financial time series break that assumption in two ways:

- **The live procedure is directional.** In production, at time t, only information available at or before *t* can be used. A random split almost always trains on timestamps after the validation point, which means it estimates a procedure that cannot be executed in real time. Nearby observations are also dependent due to autocorrelation and slow-moving regimes, so random shuffling inflates apparent performance.
- **Rows are coupled through windows.** Features and labels are computed from lookback and forward windows (rolling volatility, trailing returns, h-day forward returns). Even with chronological splits, overlapping windows across the boundary can leak information.

Time-series evaluation requires split rules that preserve chronology and prevent window overlap, matching a decision-time procedure that could actually be run. Bergmeir, Hyndman, and Koo (2018) show that standard k-fold CV can produce unbiased error estimates for autoregressive models when residuals are uncorrelated, a condition that often fails when model misspecification leaves exploitable structure in the errors.

### Walk-forward cross-validation as the baseline

**Walk-forward CV** mirrors deployment: fit on a past window, predict on a future window, score, roll forward, and repeat. *Figure 6.3* compares k-fold with walk-forward CV. A minimal CV design:

1. Choose a **training window**.
2. Choose a **validation window** (a future block for prediction).
3. Enforce **decision-time admissibility** at the boundary: training data must end early enough that every training label is already resolved before validation begins.
4. Fit on the admissible training window.
5. Predict on the validation window.
6. Score with the selection metric.
7. Advance by a fixed step and repeat.

![Figure 6.3](assets/figure_6_3.png)

*Figure 6.3: Comparing k-fold and walk-forward cross-validation*

*available?* A sample indexed at time 𝜏 is admissible for training at decision time t only if its: The protocol can be checked with one question: *At decision time t, which labeled samples were actually*

- **Features** are computable using information available at or before t
- The **label** is already known by t

*Example (FX 4-hour bars)*: For an FX strategy that decides on 4-hour bars, “the close” depends on the venue and bar construction. If the decision snapshot is the 4-hour bar close, features must be computable from prices available at that close, and the execution assumption must specify whether the trade fires at the same close, the next bar open, or with a fixed delay. A one-bar shift can turn a realistic rule into leakage. label for 𝜏 is not known until 𝜏. So, training “as of t” must exclude the most recent h timestamps. The second condition is a common failure mode. If the label is an h-day forward return, then the

This is the **label buffer** (called “purging” by López de Prado, 2018). For example, with 5-trading-day forward-return labels and a five-day trading week, suppose decision time is Tuesday, Feb 10. Then, the latest eligible training timestamp is Tuesday, Feb 3. Interim timestamps are ineligible because their labels have not yet resolved as of Feb 10.

Walk-forward evaluation uses either an **expanding window** (all admissible history, which may have lower variance but may introduce bias from stale data) or a **rolling window** (the most recent L admissible observations, which offers faster adaptation but may produce higher variance).

#### Retraining frequency

Retrain when new data represents at least 1–5% of the training window, or when out-of-sample performance degrades systematically. With a 10-year window, monthly or quarterly retraining suffices; with 20-day windows, daily retraining captures meaningful change. These design choices materially affect results and must be logged.

### Temporal buffers – Label buffer and feature buffer

Walk-forward validation eliminates the grossest form of leakage - training on the future - but leaves a subtler problem. Training observations can “see” validation data through two channels:

- **Labels look forward**: A 20-day return label period begins on the observation date. If a training observation occurs 15 days before validation starts, its label overlaps with five validation days.
- **Features look backward**: A 20-day momentum feature uses data from the past 20 days. If training *follows* validation (as in k-fold or combinatorial schemes; see Combinatorial methods and path dependence below), a training observation immediately after the validation period computes features using prices from the validation period.

The solution is **temporal buffers:** gaps that prevent labels and features from reaching across the boundary. In practice, always count buffer gaps in **trading days**, not calendar days. The difference is material. For example, a 21-trading-day label buffer in January 2024 spans 30–32 calendar days; a naive 21-*calendar*-day purge covers only 15 trading days, leaving 6 days of label leakage (see `01_cv_foundations`):

- The **label buffer** removes training observations whose labels overlap with those in the validation set. Its size equals the label horizon: for 20-day forward returns, exclude training observations within 20 (trading) days of the validation boundary. For one-day-ahead forecasts on daily data, no buffer is needed. López de Prado (2018) calls this **purging**.
- The **feature buffer** removes training observations whose feature lookback windows overlap with the validation period. This buffer is only needed when training data appear after the validation period, which does not occur in standard walk-forward CV but does occur in k-fold and combinatorial schemes. López de Prado (2018) calls this an **embargo** and proposes 1% of the sample length as a rule of thumb; the appropriate size depends on the strategy’s horizon and the strength of temporal dependence.

### Holdout test set and rolling retuning

Walk-forward splits reveal how choices behave when the “fit on the past, predict the future” pattern is repeated. But they do not, by themselves, answer the governance question: *when do we stop selecting and start measuring?*

To make that boundary operational, reserve a **sealed holdout test set**: a final time block that plays no role in any development decision (signal definition, feature selection, hyperparameter tuning, model selection, or reporting choices). The holdout may be inspected only after the pipeline is frozen. During development, we select using the primary development metric computed on walk-forward validation; we report strategy outcomes after freezing the pipeline; and we use the sealed holdout only once for confirmation.

With that rule in place, two distinct evaluation modes are useful:

- **Fixed holdout (tune once, test once)**: Use walk-forward validation on the development period to select a single configuration, freeze the full pipeline (including the score-to-trade mapping and cost/constraint assumptions), and evaluate it once on the sealed holdout to estimate outof-sample performance.
- **Rolling retuning**: To estimate the performance of a procedure that regularly retunes as new data arrives, split the overall test period into smaller windows and repeat the “fixed holdout” procedure for each window. At each test step, tune using only data strictly prior to the next test window, refit on admissible history, and then score on the next test window. Each test window is used for measurement only. This approach is called **nested walk-forward** because it uses an outer test loop and an inner model-selection loop. Bates et al. (2021) show that confidence intervals computed from nested CV have superior statistical properties. *Figure 6.4* illustrates this procedure with expanding and rolling windows.

![Figure 6.4](assets/figure_6_4.png)

*Figure 6.4: Nested walk-forward CV with rolling windows*

Nested walk-forward can capture when model configurations change and can be more realistic. For example, with a two-year test period and a five-year lookback, a live strategy would likely retune the model more than once as assumed by the fixed-holdout approach. Nested walk-forward with many feature sets is computationally demanding; parallelized evaluation, incremental retraining, and feature caching are standard optimizations. The key invariant remains: tuning never uses future data relative to the outer test window.

### Combinatorial methods and path dependence

A standard walk-forward yields a single sequence of out-of-sample windows: a single historical path. **Combinatorial methods** partition time into blocks and evaluate many train–test combinations, producing a distribution of out-of-sample outcomes rather than a single trajectory. The motivation is **path dependence**: a strategy may appear unusually strong or weak depending on which regimes fall along the single test path. groups of contiguous observations from n total groups, creating ( ௞) distinct train/test configurations. López de Prado (2018) proposes **Combinatorial Purged Cross-Validation** (**CPCV**), which selects k ௡

Each configuration applies both label and feature buffers - necessary because validation groups can appear anywhere relative to training blocks.

Combinatorial evaluation is most useful when comparing many variants and seeking a more stable view of selection risk. It does not replace decision-time discipline: overlap leakage must still be preits paired training blocks. Combinatorial evaluation is computationally expensive - ( ௞) configurations vented using the buffers described above, and each test block must be treated as “future” relative to ௡

grow quickly - and offers diminishing returns when the time series is short or when most blocks share the same regime. We will discuss CPCV in more detail in *Chapter 17*.

### Design commitments

Before running experiments, specify and log the evaluation design:

- **Window lengths:** Training, validation (if used), and test
- **Step size:** How far the splits roll forward each iteration
- **Retraining cadence:** How often the model is refit
- **Label horizon:** The forward window used to define labels, which determines the label buffer size
- **Longest feature lookback:** The maximum lookback across all features, which determines the feature buffer size
- **Test period:** The dates reserved for final evaluation, not used for any development decision

These are design commitments, not tuning parameters. Changing them mid-research defines a different evaluation procedure rather than an “improved” result under the same evaluation. Throughout, we encode these commitments in the `setup.yaml` configuration that each setup notebook writes; downstream chapters convert it to a `WalkForwardConfig` object for the ml4t-diagnostic splitters. This makes the evaluation design both explicit and executable. Logging these choices before experimentation enables reproducibility and makes “the same strategy” operationally well-defined. *Figure 6.5* shows the prediction coverage across the case studies. Coverage varies substantially; this heterogeneity motivates strategy-specific protocol choices, and the figure serves as a reference for understanding the statistical depth available for each case study.

![Figure 6.5](assets/figure_6_5.png)

*Figure 6.5: Training, validation, and holdout periods for the nine case studies*

The notebook `02_case_study_overview.py` shows the trading setups and time-series validation configurations for all case studies; the underlying analysis is in the `01_setup.py` notebooks in each case study directory. Next, we discuss how to start research with a focused baseline.

## 6.6 Establishing a baseline checkpoint

Before running large experiment grids, we need a **baseline checkpoint**: the smallest runnable specification that answers an early governance question: under the trading setup and evaluation protocol, is there enough stable structure to justify deeper work, or should the setup itself be revised?

### Three preflight sanity checks

Before investing in feature engineering and model search, run three checks that catch common failures early. These are not evidence of an edge. They are sanity checks that the checkpoint is well-posed. Here are the three checks:

1. **Timing sanity.** Confirm that the baseline measurement and label can be computed using only information available at the declared decision snapshot, respecting data and model pipeline timelines, and execution delays.
2. **Coverage sanity.** Report how much of the intended universe is actually tradable at each decision time under the eligibility rules, and whether that coverage changes materially across history.
3. **Trading-intensity sanity.** Use a coarse proxy for how often the mapping would trade. The goal is to detect definitions that implicitly demand unrealistic turnover before building a detailed cost model.

If these checks fail, the right response is usually to revise the trading setup conventions (snapshot, eligibility, mapping class) rather than to “optimize around” a brittle definition.

### The first reference run is narrow by design

The first reference run produces comparable, interpretable output under the declared setup and protocol. Its inputs (labels, features, and a baseline model) arrive across the next five chapters. Keep it narrow:

- **One label and horizon**. Choose a single forward horizon aligned with the strategy cadence and mapping class. This horizon determines the label buffer required by *Section 6.5*.
- **Compact feature families**. Use a small set of feature families that represent the narrative at the correct cadence. The goal is representative coverage, not exhaustiveness.
- **One simple baseline model class**. Use a stable baseline with minimal tuning. At this stage, the objective is a comparable reference point, not peak performance.
- **One walk-forward design**. Specify training and test window lengths, step size, and the purging rule implied by the label horizon. Reserve the sealed holdout for later confirmation.

The reference run answers a narrow question: Given the declared setup and evaluation protocol, does a simple baseline produce stable, non-pathological behavior without relying on fragile timing conventions or extreme implied turnover?

If the answer is “no,” the next step is often to revisit the setup (cadence, mapping class, tradability filters, or the measurement/label alignment). If the answer is “yes,” the information set and the model class can be expanded in later chapters, while keeping the baseline comparable. The next section addresses the bookkeeping that makes this comparison trustworthy: how to log every run so that search breadth is countable and selection bias is visible.

## 6.7 Search accounting and run logging

A **run log** is automatic metadata, along with pointers to the artifacts produced by each execution. When the same configuration can be rerun and the same inputs, splits, metrics, and outputs recovered, results can be compared reliably. Otherwise, the discussion of outcomes proceeds without verifying what produced them. The multiplication of research opportunities, amplified by LLM-assisted hypothesis generation (*Chapters 22–24*), further strengthens the case for countable search. A run log supports three practical needs:

- **Comparability**: Performance differences can be attributed to a change in the pipeline only when everything else is held constant and recorded, including the trading setup version, data snapshot, split design, and cost model class.
- **Selection transparency**: Once variants are compared, selection bias enters. We must know how many alternatives were tested, which metric was used to choose among them, and whether any results were used solely for confirmation. Bailey and López de Prado (2014) introduced the Deflated Sharpe Ratio, which adjusts performance metrics to account for the probability of overfitting, based on the number of trials. Applying these corrections requires knowing the trial count, which, in turn, requires logging.
- **Recovery**: A figure, a backtest, or a table can be reproduced from a concrete run identifier, without guessing which notebook cell or intermediate file produced it.

### Trial taxonomy

Use a small, consistent taxonomy so the search is countable without being bureaucratic. A workable four-level structure is:

1. **Strategy**. A named trading setup and objective with fixed invariants (universe, decision cadence, position mapping class, and cost model class), for example, `etfs`.
2. **Trial family**. A group of comparable alternatives that share the same setup and evaluation design but differ along one intended axis, for example, baseline signal definitions for the same horizon, or alternative feature sets for the same model class.
3. **Trial**. One fully specified pipeline configuration within a family, including the exact signal definition, feature specification, model class, hyperparameters (if any), and position-sizing rules.
4. **Run**. One execution of a trial tied to a concrete code version, data snapshot, random seed, and compute environment, producing metrics and artifacts.

This taxonomy answers “how many signal definitions did we compare?” and “which run produced *Figure 6.3*?” without turning research into a document workflow.

### Minimum contents of the run log

The goal of a run log is to make results **reproducible, comparable, and gateable**. There is no single way to organize research. However, the following five categories of records are non-negotiable, while the specific representations may vary:

- **Provenance (what ran):** Strategy setup version, trial IDs, code revision (git hash), timestamp, random seeds.
- **Data and evaluation (what was tested):** Dataset and point-in-time conventions, dev and holdout ranges, split scheme (windowing, step, label, and feature buffers), label horizon, and admissibility constraints.
- **Configuration (how it was produced):** Baseline definition and timing, feature and preprocessing spec, model class and hyperparameters, position mapping, plus risk and cost model parameters.
- **Artifacts (where outputs live):** References to features and labels, predictions, and backtest outputs (positions, trades, PnL, diagnostics).
- **Decision and gates (why it was kept):** Selection metric and rule, reported acceptance metrics, and the holdout gate outcome.

This list is intentionally small. If a field does not affect reproducibility, comparability, or the holdout rule, do not add it by default. Tools such as MLflow and Weights & Biases automate much of this logging; *Chapter 26* discusses MLOps governance in detail (*Section 26.6*).

One structural way to reduce the effective number of strategies tested is to pre-register the hypothesis: specify the economic thesis, signal definition, and evaluation criteria before running the backtest. Pre-registration does not preclude exploration, but it draws a clear line between confirmatory tests (which count toward the deflation adjustment) and exploratory analyses (which inform future hypotheses). *Chapter 16* formalizes this idea via the Rademacher Anti-Serum, which quantifies the penalty for searching over multiple strategies (*Section 16.7*). The summary draws these commitments together into the research contract that *Chapters 7-20* will carry forward.

## 6.8 Summary

We have turned a strategy idea into a research program that produces trustworthy evidence under iterative search. We distinguished the live trading loop from the research loop, then used a strategy map, built from families (dominant frictions by cadence) and sources of edge (durability requirements and failure modes), to force early clarity on the strategy’s structure and what must remain true for an effect to persist.

Discipline rests on four commitments: a versioned trading setup that fixes the evaluation environment; explicit economic objectives that separate development metrics from reporting outcomes; a time-series evaluation protocol that enforces chronology and decision-time admissibility with a sealed holdout; and a compact baseline checkpoint with run logging that keeps the search auditable, reproducible, and countable.

The next chapter is devoted to defining our learning task.
