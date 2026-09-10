# Chapter 2: The Financial Data Universe

The financial data landscape is expanding, and the surface area for analytical error grows in parallel. Alternative data providers have proliferated over the past decade alongside the broader expansion of large-scale data infrastructure, adding satellite imagery, credit card panels, and social sentiment feeds. Exchanges publish once-proprietary microstructure feeds. New providers are broadening access to institutional-grade market data. Prediction markets such as Polymarket and Kalshi add a new class of event-contract data. This expansion creates both opportunity and complexity.

Rigorous data management determines whether that complexity becomes insight or noise. Point-intime errors, survivorship bias, and mishandled corporate actions have always invalidated backtests; each new source expands the failure surface. The value alternative data adds depends on the strategy, but it increases the number of ways a pipeline can fail. Vendor evaluation requires discipline across quality, legal, and technical dimensions. Storage decisions between parquet files, time-series databases, and lakehouse formats shape research velocity and production feasibility.

*Chapter 1* established the ML4T workflow and emphasized process discipline. Before specifying strategies, we need to understand the data landscape - its possibilities, constraints, and failure modes. This chapter maps that universe, focusing on *three questions*: what kinds of data exist, which specific data challenges arise across asset classes, and which decisions matter along the dimensions of quality, sourcing, and storage. The following chapters deepen this foundation: market microstructure (*Chapter 3*), fundamental and alternative data (*Chapter 4*), and synthetic data generation (*Chapter 5*). Strategy specification (*Chapter 6*) builds on these data foundations. At the end of this chapter, you will be able to:

- Distinguish between Market, Fundamental, and Alternative data and their quality challenges.
- Evaluate the trade-offs between data availability and transparency across major asset classes.
- Apply a five-dimensional data-quality framework and identify specific failure modes in financial data.
- Conduct systematic vendor due diligence across technical, legal, and commercial factors.
- Select storage technologies based on access patterns and operational requirements.

This chapter and the next three focus on **dataset characteristics** - what data exists, how it behaves, and what can go wrong. They cover the raw material for strategy development, a process that begins in *Chapter 6*.

## 2.1 A Modern taxonomy of financial data

A dataset is a set of measurements, along with the rules that make those measurements comparable over time. It is a particular way of measuring real-world activity. That measurement comes with built-in definitions: timestamp conventions, trading calendars, adjustment policies, identifier schemes, and rules for revisions or restatements. These are not optional details; understanding them is necessary to understanding what the data *means*.

Because every dataset embeds definitions, the first question is what real-world process it measures: trading, fundamentals, or proxies. Financial data falls into three categories:

- **Market data**: Records what the market did - quotes, trades, and derived aggregates
- **Fundamental data**: Captures economic drivers of value (definitions vary by asset class)
- **Alternative data**: Signals not in standard market or fundamental feeds, often proxies for fundamentals

This taxonomy helps you reason about the kinds of real-world activity a dataset represents, what it omits, and the errors it can introduce. When you download “daily prices,” you are adopting the provider’s definitions. Define “close”: last trade, auction close, or a timestamped snapshot (and in which timezone)? Are splits and dividends reflected point-in-time or back-adjusted? If you get any of these wrong, you may not see an obvious failure. Instead, you get silent misalignment and biased results that propagate through feature engineering, labels, and backtests.

Before you model anything, lock down four items: timestamp semantics, corporate-action adjustment methodology, identifier stability, and the handling of revisions and restatements. Make these choices explicit in your pipeline configuration and carry them forward as metadata, so downstream code cannot accidentally reinterpret the data. These issues are the focus of this chapter and the next two.

### Market data – A hierarchy of aggregation

Market data reflects trading activity in financial markets. It embodies the institutional setting that shapes this activity, from centralized exchanges to over-the-counter environments. It originates in real time with message-level events related to individual orders and executions. Vendors typically aggregate this data in real time over a given interval to produce the familiar open, high, low, close, and volume summaries. Moving up the aggregation spectrum makes data easier to store and model, but discards micro-level information about supply and demand dynamics.

At the base, raw event streams reflect the mechanics of the trading venue: how orders are posted and matched, when trades are reported, and what trading rules apply. In modern markets, trading is split across multiple venues, including off-exchange venues such as dark pools, internalizing broker-dealers, and other alternative trading systems. As a result, “the best price” you see depends on which venues your feed covers and how it merges them.

Exchanges sell high-quality data products with strong completeness guarantees and validation fields; free feeds are often delayed, filtered, or missing metadata needed to verify correctness. Sourcing and validation discipline applies even to “basic” price data, regardless of cost. *Chapter 3* covers working with market data across levels of aggregation, including order-book reconstruction from tick data and the construction of analytical bars at multiple frequencies.

### Fundamental data – Drivers of value with release lags

Fundamentals are economic variables that justify valuations. What counts as “fundamental” depends on asset class and institutional framework:

- **Equities**: Financial statements, corporate actions, guidance
- **Commodities**: Inventories, production, shipping, weather
- **Foreign Exchange (FX)**: Interest rates, inflation, growth, policy
- **Crypto**: Issuance schedules, fees, on-chain activity, liquidity structure

Two properties drive most engineering errors:

- **Release-time data, not event-time data.** Fundamentals arrive with lags, schedules, embargoes, and revisions. The relevant question is not “what is the value,” but “what was knowable at decision time.”
- **Shaped by rules.** Accounting standards determine what equities disclose. Government agencies define commodity reporting. Statistical agencies revise macro series and publish vintages. Protocol rules govern which crypto variables are observable. Interpreting fundamentals requires understanding institutional constraints, not just downloading a series.

*Chapter 4* works with three primary sources of fundamental data:

- For **equities**, we use regulatory filings to extract balance sheet and income statement data, using point-in-time filing dates to support bitemporal queries that respect announcement timing.
- For **macro**, we use the Federal Reserve’s FRED data service to retrieve Treasury yields, unemployment claims, inflation measures, and volatility indices, with explicit handling of release schedules and vintage corrections.
- For **commodities**, we use CFTC Commitment of Traders reports to track institutional and speculator positioning in futures markets.

### Alternative data – High variety, high validation burden

Alternative data extends observations beyond standardized price feeds and structured public information, such as regulatory filings. The rise of alternative data over the last decade reflects the broader trend toward digitalization (Ekster and Kolm, 2020).

Many categories of alternative data give timelier or more granular views of economic activity, which makes them useful complements to fundamental data:

- **Geospatial/mobility**: Satellite imagery, foot traffic, shipping AIS
- **Consumer analytics**: Credit/debit panels, app usage, web traffic
- **Corporate exhaust**: Job postings, procurement, supply-chain signals
- **Text/events**: News, earnings calls, filings, social media
- **ESG/disclosures**: Emissions, workforce metrics, governance ratings

While alternative data may provide novel measurements, it comes with its own interpretation challenges. For each category, the validation questions are: How stable is coverage over time? What selection effects exist in the sample? Can you reproduce the methodology? Do usage rights permit your intended deployment?

*Chapter 4* applies the alternative data evaluation framework to three categories:

- **Prediction markets** - both CFTC-regulated Kalshi (economic events: Fed decisions, CPI, unemployment) and crypto-based Polymarket - show how to extract probability time series and engineer features from event contracts.
- **On-chain crypto data** from DeFi Llama provides Total Value Locked (TVL) metrics for protocol activity, evaluated as a forward-return predictor across major chains (see *Chapter 4* for the panel construction and IC analysis).
- **Text data** from SEC EDGAR filings demonstrate extraction pipelines that feed into NLP features, covered in more detail in *Chapter 10*.

The next section surveys market data across asset classes, lists the datasets used throughout the book, and points to the notebooks that explore them.

## 2.2 The asset-class market data landscape

Data characteristics vary across asset classes because market structures and economic determinants of value differ. The organization of trading - exchanges versus over-the-counter (OTC), electronic versus voice, consolidated versus fragmented - determines what market data you can obtain, how informative this data is, and where engineering choices affect results.

*Table 2.1* summarizes observability, key failure modes, and engineering priorities across asset classes:

| Asset Class | Observability | Key Failure Modes | Engineering Priority |
| --- | --- | --- | --- |
| Equities | High (consolidated) | Corp actions, fragmentation, identifiers | Adjustment methodology |
| Futures | High (exchange) | Roll rules, continuity method | Continuous series construction |
| Options | High (but sparse tails) | Illiquidity noise, surface construction | Surface representation |
| Digital Assets | Medium (venue-specific) | Volume integrity, 24/7 sessionization | Venue screening |
| FX | Low (OTC) | No consolidated tape, close conventions | Aggregation rule definition |
| Fixed Income | Low (OTC, sparse) | Matrix pricing, indicative versus firm | Liquidity inference |
| Swaps/OTC | Low (reported) | Curve construction, convention mismatches | Curve-based representation |
| Commodities | Medium (futures high) | Spot ambiguity, delivery specs | Contract spec in metadata |

*Table 2.1: Asset class overview*

The subsections move from exchange-traded instruments (equities, futures, options, digital assets), where data is richer and more standardized, to OTC markets (foreign exchange, fixed income, swaps, commodities), where data is sparser and conventions more fragmented.

### Equities

Global equity market capitalization was approximately $126.7T (WFE, 2024); U.S. equity market cap was about $67.7T (SIFMA, Q3 2025). Listed equities trade on multiple lit exchanges and off-exchange venues (ATS/dark pools and retail wholesalers/internalizers). In the U.S., consolidated reporting defines a market-wide best bid and offer (NBBO) and publishes consolidated quotes and trades; direct venue feeds can be faster and expose richer detail, but cost more.

#### What you observe

Equity market structure varies along three axes that affect what you can observe and how you should define “the price”:

- **Consolidation versus fragmentation.** Some jurisdictions provide consolidated reporting, while others have historically been fragmented across venues. The EU’s consolidated tape for equities launches in 2026.
- **Auction intensity.** Many markets rely heavily on opening and closing auctions, which can dominate end-of-day liquidity and benchmark pricing. “Close” often means “auction close,” not “last trade.”
- **Trading constraints and conventions.** Tick sizes, short-selling constraints, price limits/volatility interruptions, and settlement conventions differ across markets and can alter the meaning of intraday liquidity measures and transaction cost estimates.

Quotes, trades, and derived bars are widely available for liquid names, but “best price” and “available liquidity” depend on feed scope and aggregation rules. Treat each venue feed as a partial view of the market; any consolidated view is a constructed aggregate with its own timestamps, inclusion rules, and latency profile.

#### Failure modes

Fragmentation creates ambiguity about timestamps and aggregation (especially when stitching venues or mixing consolidated and direct feeds). Corporate actions (splits, dividends, spinoffs) create discontinuities that require consistent adjustment and return definitions. International datasets add additional pitfalls: currency conversion and FX timestamps, withholding taxes for dividend-inclusive series, and cross-listing/ADR mapping that can break identifier joins.

#### Engineering decisions

Before research, clarify and document:

- Your definition of “close” (auction versus last trade; local close convention)
- Whether you use consolidated versus venue feeds for quotes
- Your corporate-action and total-return methodology
- Identifier policy for cross-listed names

These are dataset design choices with material impact on results.

#### Datasets

We use free daily data on 3,199 US equities (1962–2018) sourced from NASDAQ (the original Quandl WIKI price data, no longer updated after the Quandl acquisition). We also use two commercial datasets generously provided by Algoseek: daily OHLCV bars for S&P 500 constituents (2017-21) and minute bars for NASDAQ 100 constituents (2020-21), enriched with several dozen ML-focused market microstructure features.

**Notebooks**: `01_us_equities_eda` explores US equity characteristics; `02_corporate_actions` validates adjustment factors and compares unadjusted versus adjusted returns. We will explore the Algoseek NASDAQ 100 data in *Chapter 3* on Market Microstructure.

### Exchange-traded products

**Exchange-traded products** (**ETPs**) are *exchange-listed wrappers that package exposures* to one or more underlying markets. They trade like equities - with quotes, trades, and order books - but holdings, index rules, and replication mechanics. ETPs may wrap bond, commodity, FX, volatility, or multi-asset exposures, so their trading and risk characteristics reflect both the wrapper and the underlying assets. ETPs also differ across common structures:

- **Exchange-traded funds (ETFs)** use a **creation/redemption** mechanism with authorized participants that links secondary-market prices to underlying asset values.
- **Exchange-traded notes (ETNs)** are unsecured debt instruments; they embed **issuer credit risk** and may track an index via an issuer promise rather than a fund structure.
- **Closed-end funds (CEFs)** generally lack continuous creation/redemption and can trade at **persistent premiums/discounts** to NAV.

#### What you observe

Consolidated quotes and trades, bars, and (from direct feeds) deeper order book information. Unlike single-name equities, ETP research requires a second layer of data:

- **NAV** (typically end-of-day) and, for many ETFs, an **intraday indicative value**.
- **Holdings and weights** (or index constituents) are often updated daily, but sometimes disclosed with or as representative baskets.
- **Fund actions** such as distributions, splits, and (for leveraged/inverse products) periodic rebalancing that alter exposure through time.

#### Failure modes

ETPs bundle a range of different products; therefore, the failure modes multiply:

- **Return definition.** Distributions like dividends or coupons are often material. Price-only series can understate returns and distort volatility; the “adjusted close” embeds vendor-specific adjustment methodology (see *Section 2.3* and *Chapter 4*). Be also explicit about (i) total versus price return, (ii) reinvested versus cash distributions, and (iii) taxes and currency treatment where relevant.
- **Premium/discount.** Traded prices can deviate from NAV, especially when the underlying market is closed, illiquid, or stressed. Avoid treating NAV returns as tradable.
- **Liquidity mismatch.** A liquid ETF can wrap illiquid underlyings. Secondary-market liquidity can overstate true capacity when flows are transmitted to constituents through creation/redemption or dealer hedging, particularly in stress regimes.
- **Look-through leakage.** Holdings and baskets change through rebalances and at the manager’s discretion, and public holdings data may not be point-in-time. Using today’s holdings to compute yesterday’s exposures creates subtle forward-looking bias.
- **Replication complexity.** Many ETPs hold complex products: commodity products may be **futures-based**, leveraged/inverse products are **path-dependent** due to daily rebalancing, and options-overlay products embed *volatility surface* dynamics.

#### Engineering decisions

Treat ETPs as a distinct dataset type. At minimum, lockdown:

- **Return methodology:** Price versus total return; distribution handling; currency conversion.
- **Valuation anchor:** Price, NAV, or both, and which is used for signals versus evaluation.
- **Exposure mapping:** Holdings versus index constituents versus factor proxies; point-in-time and disclosure-lag policy.
- **Liquidity model:** ETF-level costs versus look-through capacity constraints for large orders/ stress regimes.
- **Replication metadata:** Physical versus synthetic; futures-based; leveraged/inverse; options overlay.

**Datasets and notebooks**: We use daily data on 100 ETFs (2006–2025) sourced from Yahoo Finance, with 50 categorized by asset class for the rotation case study. Liquidity spans a 23× range between SPY (126M shares/day) and the Commodities bucket (5.5M); 41 of the 100 ETFs launched after 2006, constraining lookback for features and evaluation. The notebook `03_etfs_eda` introduces this dataset.

### Futures

Global exchange-traded futures volume in 2024 was about ~28B contracts (FIA). Futures trade on exchanges with transparent contract-level data. The complication is identity over time: a market is a sequence of expiring contracts with shifting liquidity.

#### What you observe

High-quality prices, volume, and open interest per contract. No perpetual ticker exists - you must construct one.

#### Failure modes

Roll rules (calendar, volume, open-interest) and continuity methods (ratio adjustment, difference adjustment) are backtest-defining choices. Different strategies need different roll logic.

Back-adjusted continuous series typically preserve futures price PnL but do not embed collateral return (margin interest). Back-adjusted continuous series often behave like **excess-return** proxies unless you explicitly model collateral return. When comparing to spot assets or total-return indices, be explicit about whether you model futures PnL, collateral return, or their sum.

#### Engineering decision

Store raw contract histories, along with one or more continuous variants so you can test and reconstruct roll decisions.

#### Datasets

We use hourly data on 30 CME futures for a diverse set of underlyings covering 2011-25, sourced from Databento.

**Notebooks**: `04_cme_futures_eda` explores the futures data; `05_futures_session_aggregation` shows how to align the data with CME trading sessions; and `06_futures_continuous` demonstrates volume-based roll detection and the two mainstream back-adjustment methods (ratio and difference), validated against the vendor’s pre-built continuous series.

### Options

Global exchange-traded options volume in 2024 was about ~177B contracts (FIA). Listed options are electronically quoted, but the instrument space explodes across strikes, expiries, and types. Liquidity is highly uneven - concentrated in near-money, near-dated contracts.

#### What you observe

Rich quote and trade data for liquid strikes; sparse and noisy data elsewhere.

#### Failure modes

Storing and modeling the whole chain is expensive and mostly illiquid noise. Treating the chain as a single “dataset” conflates liquid and illiquid instruments.

#### Engineering decision

Represent options markets through derived surfaces - implied volatility surfaces, ATM vol, skew, term structure - rather than raw chains. This representation is a dataset design decision: the surface is the dataset, not a summary of it. Store volume and open interest for liquid contracts; ignore or aggregate the illiquid tail.

#### Datasets

S&P 500 daily options prices and analytics covering 2017-2021 provided by AlgoSeek. The notebook `07_sp500_options_eda` explores this dataset; the notebook `08_options_greeks_computation` derives Black-Scholes pricing and Greeks from first principles - Delta, Gamma, Vega, Theta, and Rho - and validates Delta, Gamma, Vega, and Theta against vendor-computed values; `09_options_continuous` demonstrates roll contamination in constant-maturity option series and constructs backtestable continuous returns.

### Digital assets

Total crypto market capitalization was about ~$3T on 18 January 2026 (CoinMarketCap). Digital assets combine centralized exchanges (CEX) with familiar order books and decentralized exchanges (DEX) with automated market makers. The on-chain state is publicly observable, while exchange microstructure is venue-specific.

#### What you observe

CEX data resembles equities: spreads, depth, order flow - the microstructure concepts we will discuss in *Chapter 3* apply directly, and order book reconstruction works as expected. DEX data differs fundamentally: AMMs replace order books with reserve-based pricing (for example, constant-product AMMs; concentrated-liquidity designs generalize this), where price emerges from reserve ratios rather than bid-ask matching. There is no order book to reconstruct; “depth” becomes slippage at a given trade size, computable directly from pool reserves.

#### Failure modes

24/7 trading complicates sessionization. Reported volume on some venues is unreliable. **Maximal extractable value** (**MEV**) refers to the profit that validators or other actors can capture by reordering, inserting, or censoring transactions within a block; together with impermanent loss and on-chain latency, it materially affects DEX execution. Venue fragmentation makes cross-exchange comparison difficult.

#### Engineering decision

Screen venues for integrity before including them. Handle 24/7 timestamps explicitly. For DEXs, derive slippage and liquidity from reserves rather than treating AMM quotes as order-book depth.

#### Datasets

We use Perpetual Futures prices at hourly frequency and Premium Index at 8-hour intervals for 19 symbols covering 2020-25, sourced from Binance. The notebooks `10_crypto_perps_eda` and `11_crypto_premium_analysis` explore this data; the funding-rate premium analysis is relevant to funding arbitrage strategies.

### Foreign exchange

Global daily average FX turnover was about ~$7.5T/day (April 2022, Fed NYC). FX is predominantly OTC with no single consolidated tape. Even “daily close” (for example, 5 pm New York) is a convention, not an exchange-defined event. Different platforms display different liquidity and spreads.

#### What you observe

Venue-specific quotes without centrally reported volume. In FX, “the price” is a convention: you choose a venue and an aggregation rule (mid, best-of, VW-mid).

#### Failure modes

Assuming a single authoritative price when none exists. Stale quotes, crossed markets, and outlier filtering all require explicit rules.

#### Concrete example

A “4 pm London close” and a “5 pm New York close” for EUR/USD often differ by several pips on volatile days. If your strategy mixes close conventions across currencies, you may be comparing non-comparable timestamps.

#### Engineering decision

Specify which price you can realistically trade: best-of aggregation, volume-weighted mid, or a specific venue. Document the close convention in your dataset metadata.

#### Datasets

We use 20 currency pairs at 4-hour intervals from 2011–2025 via OANDA and explore this data in the notebook `12_fx_pairs_eda`.

### Fixed income

The outstanding amount in the global fixed-income market was approximately ~$145T (2024, SIFMA). Fixed income is predominantly OTC, with a vast instrument universe - thousands of corporate bonds and tens of thousands of municipal bonds. Many instruments trade infrequently; quoted markets may be indicative rather than firm.

#### What you observe

Sparse transaction prints, indicative quotes, and model-derived valuations. When the last trade is stale, research pipelines operate on yields, curves, and inferred prices rather than clean price series.

#### Failure modes

Liquidity estimation is an inference problem, not a measurement. Matrix pricing - mapping illiquid bonds to similar liquid instruments - introduces model assumptions. Still, any approach requires distinguishing between observed prints and indicative quotes, and between model outputs and model inputs.

#### Engineering decision

Decide how to handle illiquid instruments: exclude them, impute prices via matrix pricing, or model them separately. Document the methodology because “price” means different things for liquid treasuries versus thin municipals.

### Swaps and OTC derivatives (rates, credit)

OTC derivatives notional amounts are enormous (~$846T in mid-2025, BIS), but **notional is not exposure**; it mainly reflects contract scaling. **Interest rate derivatives** accounted for **~79%** of OTC derivatives notional (~$670T). Swaps markets sit “behind” much of modern fixed income and macro trading. Interest rate swaps, OIS, cross-currency swaps, and CDS are primarily OTC (with varying degrees of central clearing), and their data is shaped by reporting regimes and market conventions rather than a single consolidated tape.

#### What you observe

Post-trade reporting (where available), indicative dealer runs, cleared pricing where accessible, and curve-building inputs rather than a single authoritative “price series.” Even when transactions are reported, timestamps, block-size masking, and venue fragmentation can affect what is observable at research granularity.

#### Failure modes

Treating OTC quotes as firm executable prices; mixing venues/reporting feeds with incompatible conventions; and building curves without consistent day-count, calendars, collateral/discounting conventions, or roll rules. Many downstream “returns” are artifacts of choices in curve construction.

#### Engineering decision

Decide whether your dataset is (1) transaction-based, (2) quote-based, or (3) curve/surface-based - and treat that choice as fundamental. For rates, store curve snapshots with explicit conventions; for credit, separate index products from single-name CDS; and keep a clear line between observed inputs and model-derived marks.

### Commodities

Exchange-traded commodity derivatives volume was about ~8B contracts in 2024 (options and futures, FIA). Commodity datasets sit at the intersection of financial markets and physical constraints. “Spot” is often not a single tradable object (and can be reported, assessed, or location-specific), while futures curves are standardized, liquid, and exchange-observable for major commodities.

#### What you observe

Futures prices and term structures are typically the most reliable market data. Physical-market indicators (inventories, shipping, weather, refinery runs) often drive fundamentals but are subject to reporting lags and coverage bias.

#### Failure modes

Treating assessed spot series as if they were exchange-traded prices; ignoring contract specs (delivery location/grade) that determine basis; and mixing curve snapshots from different timestamps when computing roll yield or calendar spreads.

#### Engineering decision

Decide whether your “commodity price” is (1) a specific futures contract, (2) a continuous futures series (with explicit roll/adjustment), or (3) a curve representation (front-to-back term structure). Store the contract spec metadata in the dataset, not as documentation.

#### Microstructure tooling applicability

*Chapter 3* discusses order-book reconstruction and bar-sampling methods. They apply directly to exchange-traded instruments (equities, futures, options, CEX crypto). They do not apply to OTC markets (much of fixed income many swaps) where quotes are indicative and fragmented across dealers, nor to DEX AMMs where no order book exists. For these markets, alternative approaches - such as quote aggregation, curve construction, matrix pricing, and reserve-based liquidity computation - are required.

Next, we discuss how to evaluate data sources and providers.

## 2.3 A due diligence framework for data sourcing

Data sourcing discipline catches defects before they become “alpha” (risk-adjusted trading gains) in a backtest and turn into production losses. Errors are systematic, not random: microstructure artifacts, reporting lags, revisions, corporate actions, and vendor backfills create error patterns that can inflate or deflate apparent performance. A single undetected defect can invalidate months of research. Luo et al. (2014) catalog systematic errors that inflate backtest returns; López de Prado (2018) emphasizes that many ML strategy failures trace to data issues rather than modeling.

### General data quality dimensions

Data quality assessment typically spans five dimensions:

- **Timeliness**: Lag/latency from release/event to availability
- **Completeness**: Coverage gaps across instruments, dates, and fields
- **Accuracy**: Values match ground truth
- **Consistency**: Stable schemas and definitions over time
- **Validity**: Constraints satisfied (timestamps ordered; values within plausible bounds; field-level invariants hold)

**Implementation**: `13_data_quality_framework` implements an automated validation pipeline using the `ml4t-data` library to check these dimensions on OHLCV data. The 100-ETF panel from `03_etfs_eda` illustrates the scale to expect: 473 OHLC invariant violations (~0.1% of rows) trace to independent split and dividend adjustment of the four price fields, an artifact small enough to ignore for most uses but worth flagging in a vendor-comparison report. Prioritize checks that create lookahead bias (PIT, backfills, revisions) and universe leakage (survivorship).

This section covers data quality issues applicable to *all* financial data. For evaluation criteria specific to alternative data, see *Section 4.4*.

### Finance-specific failure modes

Four failure modes commonly result in manufactured alpha:

#### Point-in-time correctness

**Point-in-time** (**PIT**) correctness means that every value used in historical research reflects what was *knowable at the time of the decision*, under the definitions and reporting conventions in force . PIT violations occur when datasets are revised after the fact, and you unknowingly train or backtest on the revised version.

Common culprits include revisions and restatements (fundamentals, macro), corporate action backfills, delayed or amended filings, and vendor database updates that overwrite history.

*Example failure.* A company reports Q4 2019 revenue as $91.8B in early 2020. By 2024, after a restatement, a vendor’s database shows $92.3B for the same quarter. If you query “Q4 2019 revenue” today and use it in a 2020 backtest, you have introduced lookahead bias: the model is trained on information that did not exist at the time of the decision.

Fixing PIT requires **bitemporal storage** and **as-of queries**. A PIT-correct system stores values along with the period they refer to and when each value became publicly observable, so you can reconstruct the historical information set exactly as it existed at time *t*. We use the following terms to describe PIT correctness:

- **Event time**: When the economic event occurred (for example, end of Q4 2019)
- **Knowledge time**: When the information became publicly available (for example, filing acceptance time)
- **As-of time**: The historical moment being reconstructed in your analysis (the decision time)
- **Reference period**: What the value refers to (for example, “Q4 2019 revenue”)

*Figure 2.1* illustrates PIT correctness for fundamental data.

![Figure 2.1](assets/figure_2_1.png)

*Figure 2.1: Point-in-time correctness for fundamental data*

The left panel shows the “lookahead danger zone” between event time and knowledge time; the right panel highlights zero-lag records as a red flag. The companion notebook `14_point_in_time_validation` demonstrates bitemporal query patterns and produces *Figure 2.1*.

For any fundamental or alternative metric, store the *valid time interval* the value describes and the *availability* timestamp when that specific value became observable. Each record should also include a stable `source_id` (for example, SEC accession number or agency release identifier), an `entity_id` at the intended layer (issuer versus security), and a `version` identifier. A downstream feature at decision time *t* may use a record only if `available_at <= t`.

All joins across data sources must be constrained by `available_at <= decision_time`. For identifier mappings (for example, ticker→CIK, CUSIP→ISIN), also require that `effective_date <= decision_time < end_date`, so the mapping was valid at the decision point. Violating either constraint introduces lookahead bias - often silently - through “correct” joins that were impossible at the time.

Before using any dataset with fundamental-like inputs, verify PIT correctness across three dimensions:

| Dimension | Question | Failure mode |
| --- | --- | --- |
| Values | Does each data point have both a valid time and an availability (knowledge) time? | Using restated values for historical decisions |
| Universe | Is index/universe membership tracked historically? | Survivorship bias from current-day selection |
| Identifiers | Are entity mappings time-valid (ticker changes, mergers)? | Joining data across corporate events incorrectly |

*Table 2.2: PIT checklist*

A dataset that passes value-level PIT but fails universe-level PIT can still produce biased backtests, especially for cross-sectional strategies and historical index universes.

#### Survivorship bias

Survivorship bias occurs when the historical universe is filtered using information from the future (for example, “still exists today”). Its effect can be positive or negative depending on what you exclude: excluding *bankruptcies* inflates returns (you miss losses), while excluding merger and acquisition (M&A) targets can deflate returns (because you may miss acquisition premiums). The missing names are not random: they include bankruptcies, delistings, acquisitions, and value traps - precisely the types of situations many strategies are exposed to. Therefore, the entities that “survive” are biased towards cases that did not experience such situations.

![Figure 2.2](assets/figure_2_2.png)

*Figure 2.2: Survivors overstate the full-universe return (2014–2018 US equities panel)*

The *US equities panel* contains 3,199 stocks, 777 of which (24.3%) were delisted before the dataset’s end date - all after 2014, the only window in which the legacy Quandl WIKI feed captured delistings. Quantifying survivorship bias requires simulating terminal returns for delisted stocks by exit reason: distressed delistings, M&A, or voluntary exits, each with its own probability and terminal return. Eckbo and Lithell (2025) tabulate CRSP 2010–2020 outcome shares; the empirical scenario applies those tabulations, assuming 26% compliance failures at −60%, 65% acquisitions at +25%, and 9% other exits at 0%. Terminal returns are conservatively applied on the delisting date itself. In practice, acquisition premiums often materialize gradually as deal rumors circulate and the price converges toward the offer, so by delisting day, much of the +25% may already be priced in; bankruptcy losses, by contrast, tend to be more sudden. Therefore, this calibration overstates the magnitude of bias on M&A-heavy panels.

*Figure 2.2* shows the results across Monte Carlo simulations with 1,000 draws per scenario. Over 2014–2018, survivorship bias inflated perceived returns by 3.5–15.3 percentage points across the three scenarios (+8.0 in the empirical scenario): dropping the delisted names removes the weaker half of the distribution, since the leavers returned a median +13.5% against the survivors’ +30.8%. The sign of the bias depends on the dominant delisting cause: bankruptcy-heavy periods inflate survivor returns, while M&A-heavy periods can deflate them. Shumway (1997) and Beaver, McNichols, and Price (2007) document the underlying patterns of delisting returns. The sign of the bias depends on the dominant delisting cause: bankruptcy-heavy periods inflate survivor returns, while M&A-heavy periods, such as the 2010–2017 sample, deflate them. Shumway (1997) and Beaver, McNichols, and Price (2007) document the underlying patterns of delisting returns.

Another source of bias is using today’s S&P 500 constituents as the “universe” for a historical backtest. Even if those companies survive, the selection criterion is outcome-dependent: current members are firms that have grown large and successful enough to be included in the index. Backtesting on them retroactively bakes that success filter into the experiment.

*The remedy*: point-in-time universe construction. The 2010 backtest uses 2010 membership, with correct handling of delisting returns and corporate actions. CRSP is widely used as a benchmark for survivorship-aware U.S. equity data because it tracks delistings over long horizons, but many institutional vendors provide comparable histories.

**Implementation**: `15_survivorship_bias_detection` quantifies delisting rates and shows how the composition of delisted names (M&A versus distress) affects the magnitude of the bias.

#### Corporate actions

Corporate actions - splits, dividends, spinoffs, mergers - create discontinuities in price series that require adjustment. The failure mode is the mixing of adjustment methodologies or the use of endof-sample factors (covered later in this chapter) in historical research.

![Figure 2.3](assets/figure_2_3.png)

*Figure 2.3: A graph comparing raw and adjusted prices impacted by corporate actions*

A 7:1 stock split makes raw prices appear to have crashed by ~86%. Computing returns from unadjusted prices understates cumulative performance by a catastrophic margin. *Figure 2.3* illustrates this for AAPL: over its multi-decade price history (four splits plus dividends within the Wiki/Quandl panel through 2018), raw prices show a 5.9× cumulative return, while adjusted prices show a 398× return, a roughly 68× understatement.

*Adjustment methodologies* differ. Some vendors provide point-in-time adjustments (as-known-then factors); others offer back-adjusted series (end-of-sample factors applied retroactively). Mixing them introduces lookahead. Vendors also differ in how they handle dividends (total return versus price return), spin-offs, and rights issues.

*The remedy*: document and validate your adjustment methodology. `02_corporate_actions` validates adjustment factors against Apple’s known historical splits. *Chapter 3* covers the mechanics in detail.

#### Identifier integrity

Identifier joins can align the wrong instrument. Ticker symbols get reused (when an old company delists, a new company takes the symbol). CUSIP changes through mergers. Vendor IDs have edge cases.

This creates fake alpha. Suppose your signal table has ticker “XYZ” from 2015, and your price table has “XYZ” from 2020, but these are different companies. The join succeeds; the signal now predicts the returns of a different company. If the second company happened to perform well, you have manufactured correlation from a data error.

*The remedy*: use permanent IDs with effective date ranges (vendor permanent IDs; FIGI/ISIN/CUSIP via crosswalks). `16_provider_comparison` demonstrates multi-source stitching with careful handling of identifiers; `17_complete_pipeline` shows Hive partitioning patterns that preserve identifier integrity across incremental writes.

### The vendor ecosystem

Vendors span a wide range of cost and capability; key attributes that vary with cost include depth (intraday versus daily), history length, handling of corporate actions, PIT support, latency, redistribution rights, and methodology auditability.

| Tier | Typical Profile | PIT Support | Survivorship-aware | Best For |
| --- | --- | --- | --- | --- |
| Free/public | Easy access, limited guarantees | Rare (macro vintages via ALFRED) | Rare | Learning, prototypes, sanity checks |
| Prosumer (API-first) | Good coverage for liquid markets | Partial | Sometimes partial | Individual research |
| Institutional | Deeper history, documented methodology | Often available | Often available | Production, regulated environments |

*Table 2.3: Data provider tiers*

When deciding what to buy and what to build, separate differentiating work (strategy-specific transforms) from commoditized work (clean histories, identifier crosswalks):

- *Build* the strategy-specific parts: ingestion pipelines, storage and versioning, validation checks, canonical identifiers, and the transformations that encode your assumptions (such as sessionization, adjustment policies, roll rules, PIT-aware features).
- *Buy* when the problem is non-differentiating or prohibitively expensive to reproduce: long-horizon cleaned histories, corporate action adjustments at scale, survivorship-aware universes, and standard identifier crosswalks (for example, FIGI mappings, Bloomberg identifiers).

Entity and identifier mapping is often a months-long effort: tracking an instrument across ticker changes, mergers, and cross-vendor conventions. Whether you buy vendor crosswalks or build your own, treat identifier integrity as core risk control.

### Vendor due diligence checklist

Vendor selection is risk management. In practice, diligence comes down to three questions: Can you trust the numbers? Are you allowed to use them as intended? Will the vendor behave like a reliable dependency?

#### Data quality

The first task is to determine whether the dataset supports historically correct analysis rather than just convenient access to current values. That means checking not only what fields are available, but also how the vendor handles time, revisions, identifiers, and corporate events.

Key questions include:

- **PIT support**: Can the vendor provide snapshots or as-of queries? How are revisions handled?
- **Survivorship awareness**: Are delistings included with the correct final prices?
- **Adjustment methodology**: How are splits, dividends, and other corporate actions handled? Is the methodology documented and stable?
- **Coverage and definitions**: What is the true start date, instrument coverage, update cadence, and field definition?
- **Identifier integrity**: Which identifier scheme is used, and how are mappings handled through ticker reuse, mergers, and other entity changes?

These questions establish whether the dataset is analytically credible; the next step is to confirm that it is also permissible and defensible to use.

#### Legal and compliance

Even high-quality data can create problems if the rights and provenance are unclear. This review should establish that the data is lawfully sourced, that usage is permitted for the intended workflow, and that the vendor can document its methods if needed. Focus on four issues:

- **MNPI screening**: Does the vendor have documented provenance controls?
- **Privacy and consent**: For consumer data, is the collection compliant with GDPR, CCPA, and similar rules?
- **Usage rights**: What is permitted for backtests, live trading, redistribution, and model training?
- **Auditability**: Can the vendor provide documentation of its methodology for regulators, clients, or internal control functions?

Once the legal footing is clear, the practical question is whether the vendor can operate as a stable part of your research and production stack.

#### Technical and commercial

A vendor also needs to function as a dependable production dependency. Strong content is not enough if the delivery model is fragile, opaque, or difficult to integrate into research and live systems. In particular, assess the following:

- **Reliability**: Is there a status page, uptime history, or SLA?
- **Change management**: Are schema changes and methodology revisions versioned and communicated clearly?
- **Access patterns**: Are rate limits, bulk exports, and backfill mechanisms compatible with your pipelines?
- **Portability**: Can you export the data in a usable format, and are the exit terms reasonable?

If the vendor looks sound in principle, the next step is a short hands-on validation to confirm that the data behaves as expected in practice.

#### Minimum viable data validation checklist

Once a vendor passes initial due diligence, a brief validation pass should confirm that the dataset is usable for research. At a minimum, verify the following:

- **PIT**: No zero-lag records for delayed data
- **Survivorship**: Historical universe not filtered by future information
- **Corporate actions**: Adjustment methodology documented
- **Identifiers**: Join on permanent IDs + effective dates
- **Timestamps**: Timezone explicit, session assignment clear
- **Outliers**: Stale quote and anomaly detection in place
- **Versioning**: Reproducible as-of queries possible

Vendor diligence, however, is only part of the control framework; internal processes are also required to make data issues visible, traceable, and reversible.

### Data governance

Good vendors do not replace internal governance, which makes problems discoverable and reversible across five areas:

- **Lineage**: Track where each dataset came from and how it was transformed.
- **Versioning**: Version data rather than overwriting. When you fix an error, preserve the ability to reproduce prior results.
- **Logging**: Log cleaning decisions so results are reproducible.
- **Validation automation**: Use tools like Great Expectations or Pandera; fail loudly on violations rather than silently dropping records. `13_data_quality_framework` provides the `AnomalyManager` toolkit, which produces JSON audit trails for each validation run.
- **Parallel runs**: When switching vendors or methodologies, run sources in parallel long enough to quantify differences. Preserve rollback options.

Treat sourcing as risk management: PIT and survivorship checks prevent the most expensive errors. The next section turns to efficient data storage.

## 2.4 Storing data

Storage is the foundation of research velocity and production reliability. There is no best solution independent of access patterns and operational constraints. The right choice depends on data volume, access patterns (scans, point lookups, range queries, joins), concurrency requirements, and operational maturity.

### What we benchmark and why

These benchmarks address two distinct questions:

- **File formats**: If your data lives in files (the most common research pattern), which format gives the best trade-off between size, read speed, and write speed?
- **Database engines**: When you need SQL queries, concurrent access, or time-series operations like ASOF joins (see *ASOF join performance* in this chapter), which engines perform well for financial data workflows?

We separate these because most research workflows start with files (Parquet) and add databases only when the use case demands it. The benchmarks help you decide if and when to make that transition.

### Benchmark context and caveats

The tables and figures in this section summarize benchmarks from the companion code at the L scale (approximately 1 million OHLCV rows, 64 MB of in-memory storage in Polars). The repository includes scripts to run at larger scales (XL: 10 million rows, XXL: 100 million rows) on your hardware.

Keep the following in mind when interpreting benchmark results:

- **Operation definitions matter.** “Read time” can mean opening a file handle, scanning a subset of columns, or fully scanning all data. Arrow IPC supports memory mapping, so opening the file does not necessarily load the entire file into RAM.
- **Warm versus cold cache.** A warm-up run often primes the OS page cache, making subsequent reads far faster than true cold reads.
- **Relative rankings are more portable than absolute times.** Treat the numbers as illustrative for your hardware and workload; use the companion scripts to generate benchmarks on your own systems.

### File-based storage

We benchmarked file formats on a 1-million-row OHLCV dataset (100 symbols × 10,000 bars). *Figure 2.4* traces read-time scaling across formats as the row count grows; columnar formats hold their advantage as size increases, while CSV read time degrades linearly.

![Figure 2.4](assets/figure_2_4.png)

*Figure 2.4: File format read-time scaling with dataset size*

*Table 2.4* summarizes file size, write time, and scan time at L scale for the three formats suitable for persistent storage. Rankings are often more stable than absolute timings, but ingestion paths and caching can materially affect results.

| Format | File Size | Write Time | Full Scan | Notes |
| --- | --- | --- | --- | --- |
| CSV | 136 MB | 0.13s | 0.05s | Baseline: engine-dependent overhead |
| Parquet | 40 MB | 0.07s | 0.02s | Columnar, compressed, selective reads |
| HDF5 | 71 MB | 0.13s | 0.40s | Performance depends on chunking |

*Table 2.4: File storage benchmark results*

When choosing a format:

- **Parquet**: Recommended default for research and production: strong compression (3.4× versus CSV), efficient columnar reads, universal tool support across Python, R, Spark, and cloud platforms.
- **HDF5**: Still common in scientific and legacy finance stacks. It can work well for specific append-and-chunked access patterns, but read performance lags behind columnar formats.
- **CSV**: Use for exports and interoperability, not as a primary analytical store.

**Note on Arrow IPC (Feather)**

**Arrow IPC** is a fast serialization format optimized for interchange. It excels at moving data between notebooks and processes (including memory-mapped reads), but lacks key “data lake” features such as predicate pushdown, partition-aware layouts, and multi-file governance. Use it for short-lived interchange; use Parquet for persistent, queryable storage.

**Lakehouse** formats (such as **Delta Lake**, **Iceberg**, and **Hudi**) store data in Parquet and include a transaction log that provides ACID semantics, schema evolution, and multi-writer governance. Start with plain, partitioned Parquet; adopt a lakehouse format when you need update-heavy workflows, schema evolution with strong guarantees, or concurrent writes across multiple teams.

**Implementation**: `20_storage_benchmark_file` reproduces these benchmarks and extends them to larger scales.

### Embedded analytics databases

Embedded databases provide SQL analytics without running a server:

- **DuckDB**: An in-process analytical database optimized for OLAP workloads. It can query Parquet directly and supports ASOF joins in SQL, making it a strong default for research workflows that want SQL semantics over a Parquet data lake.
- **SQLite**: A reliable embedded relational database for metadata, configuration, and small relational tables. Not competitive with DuckDB for analytical scans or time-series workloads.
- **ArcticDB**: A DataFrame-oriented store with versioning and time-travel. It is licensed under **BSL 1.1** (with time-based conversion to Apache 2.0 per release); verify licensing and operational constraints before adopting.

A useful default architecture:

1. Store raw and curated data as partitioned Parquet.
2. Use DuckDB for SQL analytics and quick joins over those files.
3. Use a DataFrame engine, such as Polars, for feature engineering. See `22_pandas_polars_benchmark` for an empirical pandas-versus-Polars comparison across read, groupby, join, and lazy-evaluation operations.

### Server time-series databases

Server databases are valuable when you need concurrent access, centralized governance, strict durability, and production-service levels.

![Figure 2.5](assets/figure_2_5.png)

*Figure 2.5: Server database comparison (1 million rows)*

*Figure 2.5* compares write and read performance across six server databases at L scale (kdb+/PyKX, ClickHouse, QuestDB, PostgreSQL, TimescaleDB, and InfluxDB). These results are directional; write time is sensitive to ingestion method, schema design, and durability settings. Rerun benchmarks using your intended ingestion path before making decisions.

Server time-series databases include:

- **kdb+/KDB X**: A highly optimized time-series and analytical engine widely used in market data contexts. KX offers free editions with material limits (often non-commercial); verify current license terms before committing.
- **ClickHouse**: A general-purpose OLAP database optimized for high-throughput columnar analytics, with strong SQL support and a large ecosystem.
- **QuestDB**: A time-series database optimized for ingestion and time-series SQL, including ASOFstyle queries. Supports tiered storage patterns with Parquet for colder data.
- **PostgreSQL/TimescaleDB**: PostgreSQL is the default general-purpose relational database; TimescaleDB extends it with hypertables (automatic time partitioning) and continuous aggregates for time-bucketed rollups.
- **InfluxDB**: Designed for metrics and observability workloads. Flux supports equi-joins, but asof style joins are not a native primitive; treat it as a monitoring database rather than a market microstructure engine.

**Implementation**: `21_storage_benchmark_database` provides reproducible server benchmarks. For production-grade orchestration, `18_data_management` composes the DataManager, Universe, and HiveStorage primitives into a working pipeline, while `19_incremental_updates` walks the daily-update lifecycle with weekend-aware gap detection and a health dashboard.

### ASOF join performance

ASOF joins align events that are not perfectly time-synchronized - matching each trade to the most recent prevailing quote, for example. They are typically the dominant cost in panel-data preparation that mixes tick streams with slower-moving reference series. Both pandas and Polars require sorted inputs.

In-memory engines (Polars `join_asof` and pandas `merge_asof`) consistently outperform embedded-SQL engines (DuckDB `ASOF JOIN`) on tick-scale workloads at the sizes we use in this book, with Polars typically the fastest of the three. The absolute gap depends on hardware, sort state, and whether the benchmark counts file I/O, so we report ordering rather than headline times.

A few measurement caveats matter when comparing implementations:

- In-memory engines measure join execution when data is already loaded; SQL engines may include file scans and conversion overhead.
- For fair comparisons, decide whether sorting is part of the benchmark. If the pipeline maintains sorted data, exclude sorting consistently across engines.
- Tick-versus-quote ASOF performance is sensitive to the ratio of left- to right-side rows; benchmark at the ratio that matches the intended workload.

### Strategic decision framework

The following matrix summarizes standard defaults by objective.

| Objective | Strong default | Also consider |
| --- | --- | --- |
| Research velocity | Parquet + Polars | Parquet + DuckDB |
| SQL analytics on files | DuckDB + Parquet | Polars lazy scans |
| Production reliability | PostgreSQL | TimescaleDB |
| Extreme time-series throughput | kdb+/KDB X | ClickHouse |
| High-throughput ingestion | QuestDB | ClickHouse |
| ASOF joins in memory | Polars | pandas |
| ASOF joins with SQL | DuckDB | QuestDB |
| Lowest operational burden | Parquet + DuckDB | Managed Postgres |

*Table 2.5: Recommended default and strong alternative by storage objective*

Start with partitioned Parquet plus Polars or DuckDB. Add server databases when concurrency and governance requirements justify the operational overhead.

## 2.5 Summary

This chapter mapped the financial data universe and the decisions that make research reproducible. Use the taxonomy - market, fundamental, and alternative data - to match datasets to strategy horizons and to surface the assumptions embedded in timestamps, adjustments, identifiers, and revision rules.

Most alpha failures in production trace to a small set of data defects: point-in-time violations, survivorship bias, corporate action mistakes, and identifier join errors. Treat these as default risks rather than edge cases, and design validation checks to catch them early.

Market structure constrains what you can observe across asset classes: exchange-traded markets provide richer event data but require careful sessionization and adjustments; OTC markets often rely on indicative quotes and inferred prices; derivatives require explicit contract identity and roll logic; digital assets add 24/7 trading and venue-integrity concerns. Vendor selection is therefore a risk management process that involves evaluating quality, legal rights, and operational reliability.

Finally, storage choices determine iteration speed. A sensible default is partitioned Parquet queried DuckDB or Polars for most research; add specialized databases only when concurrency, governance, or latency requirements justify the operational cost.

*Chapter 3* develops the details of market data - both as a key input to labels and features for modeling and as the basis for simulating historic strategy performance - and examines the institutional environment of exchanges that shapes how market data should be interpreted.
