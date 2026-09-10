# Chapter 4: Fundamental and Alternative Data

While market data captures *what* happened in the market - prices, volume, and the mechanics of trading - fundamental and alternative data help explain *why*. Turning these sources into model-ready signals is less about clever algorithms than about disciplined pipelines: ingesting messy inputs, tracking revisions, validating identifiers, and producing time-consistent features that hold up in backtests and production.

**Fundamental data** is asset-specific information about the underlying economic drivers of value, rather than trading activity or market microstructure. In practice, fundamentals capture variables that influence expected cash flows, discount rates, or key constraints (for example, supply and demand), and they typically arrive through reporting or measurement rather than through markets. Examples include financial statements and filings for equities, issuer and covenant information for credit, supply-demand and inventory measures for commodities, and protocol and on-chain metrics for crypto. Alongside these sources, **alternative data** has expanded to include digital exhaust, such as web traffic, transactions, geolocation data, satellite imagery, and text, reflecting its growing use in active management (Green and Zhang, 2024).

In practice, two failures dominate the real-world use of fundamentals and alternative data: time consistency and entity consistency.

- **Point-in-time (PIT) accuracy** ensures that research uses only information *available* at the decision date. *Chapter 2* introduces the PIT vocabulary and the bitemporal data model; this chapter applies them to SEC filings, macroeconomic releases, and other revision-prone sources.
- **Entity resolution** ensures that records from disparate sources map to the same real-world issuer and tradable instrument despite naming variants, identifier changes, corporate actions, and mergers, and that the correct security identifier remains aligned over time.

This chapter provides the engineering playbook for both challenges; it will enable you to:

- Implement a bitemporal data model to distinguish valid time from knowledge time (*Section 4.1*).
- Parse SEC EDGAR filings and construct point-in-time corporate fundamental databases (*Section 4.1*).
- Apply various hierarchical entity resolution approaches (*Section 4.2*).
- Characterize fundamental data across asset classes (*Section 4.3*).
- Evaluate alternative data sources along signal, data quality, legal, and commercial/engineering dimensions (*Section 4.4*).
- Extract structured data from EDGAR text filings for NLP analysis (*Section 4.5*).

We’ll start with the practical challenge of creating a data pipeline that is point-in-time correct.

## 4.1 The point-in-time pipeline

*Chapter 2* introduced PIT correctness and the bitemporal vocabulary (valid time versus knowledge time). The question here is operational: given EDGAR filing histories, how do we build a dataset that returns the value eligible at decision time *t*, not the value that exists in a vendor database today? The core complications are amended filings, taxonomy changes, and corporate actions that create multiple “versions” of the same reporting period.

### Why restatements create leakage

Consider a simple **P/E ratio strategy**. A company might report Q3 revenue of $95M in October, then restate it to $98M in the next 10-Q and again to $100M in the next 10-K. A backtest that uses $100M for October decisions has applied the *later* accounting view to an *earlier* decision - this is lookahead bias.

Luo et al. (2014) identify it as one of the “seven sins” of quantitative investing, showing how errors in data handling can completely reverse a strategy’s performance. The others are survivorship bias, storytelling bias, overfitting and data snooping bias, turnover and transaction cost, outliers, and shorting cost.

### Bitemporal storage and as-of queries

The implementation follows the *Chapter 2* PIT contract: each fact must carry both the period it describes and the timestamp when that specific version became observable. A single reporting period may therefore have multiple versions over time: the original filing, an amendment, or a later restatement.

For earnings, markets often react first to the earnings release, typically via Form 8-K Item 2.02, and only later to the detailed Form 10-Q or Form 10-K. Using the 10-Q or 10-K acceptance time as the availability timestamp is conservative for PIT backtests. If event-time precision matters, track both timestamps explicitly.

The **as-of query pattern** is straightforward:

1. Select the latest reporting period eligible at historical time t.
2. Within that period, select the latest version with an availability timestamp no later than t.

A practical caveat is the **SEC XBRL Frames API**. Frames returns the “last filed” value that best matches the requested period, which makes it useful for cross-sectional snapshots and prototyping, but not PITsafe on its own. For PIT research, reconstruct the history from the filing-level records and timestamps. The *Chapter 2* PIT checklist is a precondition here: values must be time-valid, the historical universe must be reconstructible, and identifier mappings must be valid at the decision date. The remainder of this section shows how to satisfy those requirements for large-scale fundamental sourcing.

#### Timestamp authority by modality

The authoritative availability timestamp varies by data source. The conventions below apply uniformly across pipelines:

| Modality | Authoritative Timestamp | Notes |
| --- | --- | --- |
| SEC | accepted at (filing acceptance) _ | Optionally track earlier earnings releases if event-time matters |
| Macro | Official release timestamp (timezone- normalized) | Use ALFRED vintage data for historical accuracy |
| Commodities | Agency release timestamp + contract roll mapping | EIA, USDA, and CFTC each have specific release schedules |
| Crypto | Block time + confirmation/finality policy + vendor ingestion time | Treat observation as available only after N confirmations to avoid reorg lookahead |

*Table 4.1: Timestamp authority by source*

### The EDGAR sourcing pipeline

The SEC’s **Electronic Data Gathering, Analysis, and Retrieval** (**EDGAR**) system provides free, comprehensive access to U.S. corporate filings. EDGAR’s phase-in began in 1993 and became mandatory for domestic issuers by May 1996. The system now offers multiple access channels:

**Bulk Downloads**: The Financial Statement and Notes (FSN) datasets provide parsed XBRL data in tabular format. These cover all 10-K, 10-Q, and 8-K filings back to 2009, organized into eight related tables:

| File | Description |
| --- | --- |
| SUB | Submission metadata—CIK, form type, filing date, accession number |
| NUM | Numeric values extracted from XBRL tags |
| TAG | Taxonomy tag definitions and descriptions |
| DIM | Dimensional breakdowns (by segment, geography, and so on) |

*Table 4.2: Selected FSN tables*

The FSN dataset includes four additional files (TXT, PRE, CAL, REN) for text content and display formatting. **REST APIs**: Real-time access to company filings, specific concepts, and submission histories. No authentication required - callers identify themselves via the user-agent header. The company-concept API returns all historical values for a specific metric (for example, `EarningsPerShareDiluted`) in JSON format.

**Python Libraries**: Packages such as `edgartools`, `sec-edgar`, and `sec-downloader` wrap the API complexity with convenient interfaces. `edgartools` provides direct access to parsed financial statements and text sections. For high-throughput applications, services like `datamule` offer bulk downloads via hosted archives, often with higher throughput than direct EDGAR access (check current rate limits and terms).

For systematic research, the bulk approach is often most efficient: download the compressed FSN archives, extract to columnar Parquet format for fast queries, then build the bitemporal database from the parsed content.

#### XBRL and taxonomy structure

The **eXtensible Business Reporting Language** (**XBRL**) standardizes electronic business reporting. Each numeric value in a filing is tagged with:

- **Concept**: What the number represents (for example, `us-gaap:RevenueFromContractWithCu` `stomerExcludingAssessedTax`)
- **Period**: The time span covered (point-in-time for balance sheet items, duration for income statement)
- **Unit**: Measurement unit (USD, shares, USD/share)
- **Dimensions**: Optional breakdowns by segment, geography, or other categories

The U.S. GAAP taxonomy defines thousands of standardized concepts, but companies can extend it with custom tags. This flexibility enables precise reporting but complicates cross-company analysis - the same economic fact might appear under different tags.

For PIT reconstruction, the filing date (`knowledge_date`) is derived from the submission metadata rather than the financial data itself. For each filing, the valid period is derived from the XBRL content and paired with the filing date from the submission record. This separation makes PIT reconstruction possible.

**Implementation**:

- `01_academic_characteristics` introduces the academic-fundamentals panel and the long-run cross-section that motivates careful PIT construction.
- `02_sec_filing_explorer` introduces EdgarTools for interactive EDGAR exploration.
- `03_sec_form4_insider_transactions` demonstrates how to retrieve and parse Form 4 insider-transaction filings as an example of event-timestamped fundamental data.
- `04_sec_xbrl_fundamentals` consumes the materialized XBRL Frames output and demonstrates bitemporal query patterns.

### Handling data inconsistency

Even with proper PIT tracking, fundamental data presents consistency challenges:

- **Accounting standard evolution**: U.S. GAAP changes over time. A tag like `RevenueFromContractWithCustomerExcludingAssessedTax` might replace `SalesRevenueNet` mid-series, requiring mapping between taxonomy versions.
- **Company-specific choices**: Within GAAP, companies have discretion in categorization. One firm’s “Cost of Revenue” might include items another firm reports separately. Cross-company comparisons require careful normalization.
- **Restatements and amendments**: When companies file amended filings (10-K/A, 10-Q/A) or restate prior periods, multiple versions of the same reporting period coexist. A bitemporal model stores each version with its own availability timestamp; analysis code must be explicit about which version is eligible at each decision date.
- **Corporate actions**: Stock splits require retroactive adjustment of per-share metrics. A 3:1 split means historical EPS must be divided by 3 for comparability. Mergers and spin-offs create discontinuities that simple adjustment factors cannot resolve.

Leakage becomes visible when the same simple strategy is run twice: once using an as-of eligible (“as-known”) fundamentals view and once using a naïve “latest available today” view. The second backtest often appears better - not because the signal is stronger, but because the dataset has silently incorporated information that arrived later.

In practice, even small amounts of leakage can swamp the true signal-to-noise ratio in fundamentals-based strategies. The lesson is methodological rather than a fixed percentage: if a vendor cannot explain how it handles revisions, amendments, and identifier history, presume the dataset is *not* PIT-correct until proven otherwise.

![Figure 4.1](assets/figure_4_1.png)

*Figure 4.1: Lookahead bias from restated fundamentals*

*Figure 4.1* shows a naïve “latest value” fundamentals dataset that assigns the restated Q3 revenue ($100M) to dates before its public release. A point-in-time dataset keeps the original value ($95M) until the restatement date, preventing leakage at the strategy decision date. The next section discusses how to address entity resolution.

## 4.2 Entity resolution and mapping

Before we can combine any data from disparate sources, we must address the universal problem of **entity resolution**. Different data providers, regulatory systems, and internal databases all identify the same company differently. Ekster and Kolm (2020) identify entity mapping as one of the primary technical challenges in alternative data integration. Without correct mapping, any multi-source analysis is compromised from the start.

### Why resolution is critical

Consider a strategy that combines SEC fundamental data with alternative data on government contracts. The SEC identifies companies by Central Index Key (CIK). Federal award data commonly includes the Unique Entity Identifier (UEI) (which replaced DUNS in April 2022), as well as legal names and address fields. Credit card transaction data might use proprietary merchant codes. None of these maps directly to stock tickers without crosswalks and a clear definition of the target layer (issuer, subsidiary, security).

A single false-positive match can poison every downstream join; prioritize precision and escalate uncertain matches. If “ZOOM VIDEO COMMUNICATIONS” (ticker ZM) is confused with “ZOOM TECH-NOLOGIES INC” (formerly ticker ZOOM, now ZTNO), fundamental data from one company gets paired with alternative signals from another. The resulting “alpha” is noise.

The problem multiplies with corporate complexity. IBM’s government contracts might be filed under “INTERNATIONAL BUSINESS MACHINES CORPORATION,” “IBM CORP,” “IBM GLOBAL SERVICES,” or dozens of subsidiary names. These must all map to ticker IBM, but contracts from spun-off, now-independent units like Kyndryl should not.

### A hierarchical resolution approach

Practical entity resolution proceeds in stages, from deterministic to probabilistic to model-based.

#### Stage 1: Deterministic matching

When a data source provides **standard identifiers**, start with deterministic joins. The main pitfall is resolving to the wrong *layer*, because “company identifiers” can refer to different things. More specifically, identifiers can point to:

- A **legal entity** (the corporate parent or filing entity)
- A **security** (a specific tradable instrument, such as a share class, bond, or ADR)
- A **derivatives contract** (a specific listed contract/expiration on an exchange)

Production systems typically maintain three linked master tables - **entities**, **securities**, and (when relevant) **contracts** - with **time-valid mappings** between them. For instance, “Apple” can legitimately resolve to different objects depending on the question:

- A legal entity (via a **Legal Entity Identifier** or **Central Index Key**)
- A security (for example, the common stock versus a specific bond issuance, each with its own identifier)
- An exchange-listed contract (for futures/options)

So “identifier match succeeded” is not the same as “resolved correctly” unless the correct layer (entity, security, or contract) has been selected.

Common standardized identifiers (and what they identify):

- **Legal Entity Identifier (LEI)**: A 20-character identifier for *legal entities* used in financial transactions (ISO 17442).
- **Central Index Key (CIK)**: The SEC identifier for *filing entities* in EDGAR; mapping to traded instruments typically requires additional reference tables (for example, subsidiaries versus holding companies).
- **Financial Instrument Global Identifier (FIGI)**: An *instrument-level* identifier (security-level, not entity-level); OpenFIGI provides lookup and mapping APIs.
- **CUSIP/ISIN/SEDOL**: *Security-level* identifiers used in trading and settlement (CUSIP: U.S.; ISIN: global; SEDOL: U.K.).

When a source includes any of these, resolution is usually a straightforward join against the relevant crosswalk. The harder cases come from sources that omit institutional identifiers - or sources that include one but only at a different layer than the target.

**A wrong-join example**

Alternative data often arrives at the *issuer* level (for example, web traffic for “Amazon.com Inc”), while returns are measured at the *security* level (AMZN common stock). Without an effective-date security master, this join can fail silently: (1) joining issuer-level data to the wrong share class (Class A versus Class B with different voting rights and prices); (2) joining to an ADR instead of the ordinary share; (3) using a ticker that belonged to a different entity before a corporate action. The fix: maintain explicit issuer→security mappings with effective dates, and constrain all joins by `effective_date <= decision_time < end_date`.

#### Stage 2: Probabilistic matching

When identifiers are absent, text matching becomes necessary. String similarity algorithms handle variations in company names:

- **Levenshtein Distance**: Counts character edits (insertions, deletions, substitutions) to transform one string into another. “TESLA INC” and “TESLA, INC.” are one edit apart.
- **Jaro-Winkler Similarity**: Weighted similarity favors matching prefixes. Handles abbreviations well - “INTERNATIONAL BUSINESS MACHINES” scores high against “INTL BUSINESS MACH”.
- **Token-based methods**: Jaccard similarity on word tokens. “APPLE INC” and “APPLE INCOR-PORATED” share the important token despite different suffixes.

Libraries like `rapidfuzz` implement these algorithms efficiently, enabling fuzzy matching against reference lists with configurable similarity thresholds. A typical pattern extracts the best matches above a confidence threshold (often 85-90%), and flags lower-confidence matches for manual review.

Probabilistic matching benefits from human-in-the-loop QA because the failure mode is asymmetric: a missed match costs coverage, but a false match can corrupt every downstream join and backtest. The right objective is usually high precision first, then improved recall via escalation stages and better candidate generation.

The complicated cases tend to fall into repeatable buckets:

- Subsidiaries whose legal names are unrelated to the parent brand
- DBAs and trade names
- Joint ventures and consortia
- Generic names that collide across unrelated entities

For example, “Amazon Web Services” may not fuzzy-match to “Amazon.com Inc” despite the clear relationship, requiring either subsidiary lookup tables or embedding-based semantic matching. Name-only fuzzy matching is rarely sufficient for these cases.

For threshold selection, a tiered scheme is typical: auto-accept only at very high confidence (practitioner thresholds often sit around 95%); route a middle band (commonly 85-95%) to manual review; escalate the tail (below roughly 85%) to richer features such as addresses, websites, or descriptions. Calibrate these cutoffs against the empirical score distribution in a labeled sample rather than treating the numbers as universal constants. Track match quality over time - if post-hoc review reveals incorrect matches, adjust thresholds or add disambiguation features.

#### Stage 3: Manual review and QA

Records in the middle-confidence band - high enough that deterministic and probabilistic matching produced a credible candidate, but not high enough for auto-acceptance - escalate to a human-review queue. Effective workflows separate the queue by failure mode (ambiguous fuzzy match, missing identifier, candidate-set tie) so reviewers can apply the right disambiguation procedure rather than triaging case by case. Each decision is written back to the master entity table as a new alias or a rejection, with the reviewer, timestamp, and rationale. Hence, the audit trail supports both downstream debugging and periodic threshold recalibration.

Embedding-based semantic matching, useful when fuzzy and identifier matching both fail to disambiguate (for example, “Amazon Web Services” versus “Amazon.com Inc”), is covered in *Chapter 10* (text embeddings) and *Chapter 23* (knowledge graphs), where Chen et al. (2023) describe FinKG. This knowledge graph integrates SEC filings and market data for entity linking across U.S. companies.

### Building a master security database

Production systems maintain a master entity table that serves as the canonical reference:
| id | name | ticker | cik | lei | figi |
| --- | --- | --- | --- | --- | --- |
| 1001 | Tesla, Inc. | TSLA | 1318605 | 54930043XZGB27CTOV49 | BBG000N9MNX3 |
| 1002 | Apple Inc. | AAPL | 320193 | HWUPKR0MPOU8FGXBT394 | BBG000B9XRY4 |

*Table 4.3: Master entity example*

All incoming data - regardless of source identifier - resolves to this master `entity_id` before joining with other datasets. The master table accumulates name variants and identifier mappings over time, improving resolution accuracy as more data is processed.

#### Temporal entity mapping

Static mapping is a hidden source of lookahead and survivorship bias. Entity relationships are dynamic: tickers are reused (ZOOM belonged to Zoom Technologies before Zoom Video took the ticker ZM), and identities evolve through rebranding (FB → META) or mergers. A production system must be temporally aware, tracking when each identifier was valid:

| id | identifier_type | identifier_value | effective_date | end_date |
| --- | --- | --- | --- | --- |
| 1095 ticker |  | FB | 2012-05-18 | 2022-06-08 |
| 1095 ticker |  | META | 2022-06-09 | NULL |

*Table 4.4: Temporal mapping example*

Without `effective_date` constraints, a backtest might join 2015 alternative data to 2024 ticker symbols - creating a dataset that could never have existed. Every cross-source join must be constrained by observation timestamp to ensure the mapping was valid at that historical moment.

#### Mapping quality assurance

Production entity resolution requires ongoing validation:

| Check | Method | Frequency |
| --- | --- | --- |
| Confidence Tiers | Segment mappings by confidence using calibrated cutoffs (commonly around 95% auto-accept, 85-95% review, below 85% use another method) | Per batch |
| Sampling Audit | Manually verify 1–5% of auto-accepted mappings for false positives | Monthly |
| Drift Monitoring | Track match rate and confidence distribution over time; investigate sudden changes | Weekly |

*Table 4.5: Quality assurance example*

A declining match rate often signals changes in the data source (for example, new naming conventions or added subsidiaries). Rising false-positive rates indicate threshold drift or coverage gaps in the reference data.

Key QA metrics to track:

- **Match rate by tier**: Percentage of records resolved at each stage (deterministic, fuzzy, embedding, or unresolved)
- **Confidence score distribution**: Histogram of match scores; watch for bimodal patterns indicating threshold issues
- **False positive estimate**: From sampling audit - fraction of auto-accepted matches that are incorrect
- **Temporal drift**: Week-over-week change in match rate and mean confidence; sudden shifts warrant investigation

![Figure 4.2](assets/figure_4_2.png)

*Figure 4.2: Hierarchical entity resolution pipeline*

*Figure 4.2* shows the entity-resolution pipeline. Deterministic identifier matching resolves the bulk of records; probabilistic record linkage absorbs name variants above a confidence threshold; the residual middle-confidence band escalates to manual review, with all decisions written back to a master entity table under time-valid mappings. The next section surveys the fundamental data landscape across asset classes.

**Implementation**: `05_entity_resolution` demonstrates deterministic joins, fuzzy string matching (with normalization), and a simple master database pattern in an end-to-end pipeline.

## 4.3 Fundamentals across the asset-class spectrum

*Sections 4.1–4.2* established the PIT contract and entity mapping for equities. This section generalizes those concepts: every asset class has revision-prone fundamentals, but the *timestamp authority* and *instrument mapping* differ across asset classes. Getting these details wrong introduces the same lookahead bias - the failure mode is universal, even if the mechanics vary. The idea of “fundamentals” extends well beyond corporate equities, but the meaning changes by asset class. Each domain has distinct economic drivers, publication mechanics, and revision behavior, so the engineering problem is not “collect more data” - it is building time-consistent inputs that respect release calendars, vintages, and instrument conventions.

### Macro and sovereign data (FX/Rates)

Rates and FX are driven by macro fundamentals such as growth, inflation, labor conditions, monetary policy, and fiscal sustainability. The engineering challenge is that these series come from multiple agencies and are published on *release schedules* with **lags** and **revisions**, so what was known at each decision time must be modeled explicitly.

#### Key sources

**Federal Reserve Economic Data** (**FRED**) aggregates hundreds of thousands of time series from many sources and provides APIs for retrieval and vintage-aware workflows. For non-U.S. macro, primary sources include Eurostat, IMF/World Bank datasets, national statistics agencies, and central banks. (See *Chapter 2* for a complete data source inventory; this section focuses on timing and revisions.) FRED data show the U.S. 10y-2y yield curve has been inverted on 15.5% of days since 2000, including 36% of days since 2020 - a regime-shift anchor for macroeconomic state variables.

#### Alignment challenges

Macro data presents several engineering challenges:

- **Release cadence variation**: Many key macro series are measured over a period (month/quarter) but published later on scheduled dates. GDP is published quarterly, with multiple monthly vintages (advance/second/third estimates), while claims are weekly and CPI is monthly. Aligning these to a daily decision grid requires a clear policy: treat releases as **event data** (timestamped) and carry forward only after they become available, or treat macro as a slowly evolving **state** updated at release times. Avoid interpolation methods that blend future values into the past.
- **Release time matters**: For daily or intraday models, the release *timestamp* (date, time, and time zone) determines eligibility; “same day” does not mean “pre-trade.” Many U.S. releases are scheduled at 8:30 a.m. ET, but schedules can shift due to holidays or disruptions, so the pipeline should ingest and store the published release timestamp rather than hard-coding assumptions.
- **Time zones and calendars**: A global macro dataset must normalize timestamps (typically to UTC) and track the local release calendar. “Quarterly” series from different jurisdictions can be released on different local dates/times and revised on different schedules, so PIT correctness requires storing both the *period covered* and the *availability timestamp*.
- **Revision histories**: Many macro series are revised after initial release. Backtesting on today’s “final” series gives the model information that was not available at the time. The correct input for realistic backtests is vintage data (“what was known when”), not the latest revision.

FRED supports **vintage-aware retrieval** through **ALFRED** (Archival FRED): observations have real-time validity windows (in other words, the value that was current between two vintage dates), and the API exposes vintage dates and vintage-specific downloads. The Philadelphia Fed Real-Time Data Set provides the canonical reference for vintage-aware macroeconomic research; Croushore (2008) surveys the methods and pitfalls. In practice, the two production patterns are (i) querying specific vintage dates to reconstruct what a model would have seen historically, or (ii) storing the revision ladder explicitly when the downstream strategy is revision-sensitive.

PIT requirements include storing the release timestamp and calendar; maintaining the vintage/revision ladder; normalizing time zones; and distinguishing the period covered from the availability time.

**Implementation**: `06_fred_macro_eda` introduces the macro data; `07_macro_data_alignment` illustrates how to load native-frequency series, model publication lag, and how vintage-aware inputs differ from “latest series” inputs.

### Commodities – Physical market data

Commodity prices are anchored to physical supply-and-demand constraints. However, the data is often **domain-specific** (inventories, shipments, crop conditions, refinery utilization), and the tradable instruments are **contractual** (futures by delivery month). That creates a two-part engineering problem:

1. Ingest scheduled fundamental releases with correct timestamps.
2. Map them to the contract you could actually trade.

#### Energy and agriculture

U.S. government agencies anchor the calendar for energy and agricultural fundamentals. The U.S. Energy Information Administration (EIA), the U.S. Department of Agriculture’s World Agricultural Supply and Demand Estimates (WASDE) report, and the U.S. Commodity Futures Trading Commission’s (CFTC) Commitments of Traders (COT) report are timestamped events; joins should occur only after their published release times. The EIA’s Weekly Petroleum Status Report follows a standard schedule (Wednesdays at 10:30 a.m. ET, with holiday exceptions); the release timestamp must be part of the dataset.

USDA’s WASDE report is released at 12:00 p.m. ET on scheduled dates and is widely viewed as a risk event for grains and oilseeds. Use the release timestamp; avoid any workflow that implicitly treats pre-release expectations as published facts.

Meteorological data (for example, from NOAA) often enter as high-frequency state variables rather than as scheduled releases; treat it differently from “lock-up” economic reports.

#### Mapping to tradable instruments

Physical market data must be mapped **to the instrument you can trade**. When an inventory number is released, the relevant price response may concentrate in the actively traded front-month contract (or the most liquid point on the curve), not in an abstract “continuous series” that blends multiple delivery months. This is where the *futures roll logic* from *Chapter 3* becomes an input dependency: a consistent mapping from calendar time to the active contract. Fundamental event timestamps should be joined to the active contract at their timestamp, then carried into any continuous-series representation using the same roll rules as prices.

Beyond the physical fundamentals, CFTC’s COT reports provide a public snapshot of trader positioning by category. COT is a PIT dataset: positions are as of Tuesday. They are generally released on Friday at 3:30 p.m. ET (with holiday delays), so backtests must treat the report as **available only after the release time**. Positioning extremes and rapid changes are often used as sentiment-style features (contrarian or trend-confirming); any claim about predictability should be evaluated out-of-sample with the release lag enforced.

PIT requirements for this include tracking scheduled release timestamps; mapping fundamental events to the correct contract and curve point at event time; applying roll rules consistent with prices; and enforcing reporting lags.

**Implementation**: `08_futures_positioning` shows COT retrieval, PIT availability timestamps, and positioning z-scores.

### Cryptocurrency and digital assets – On-chain fundamentals

Crypto markets introduce a new category of fundamentals: on-chain data is publicly observable and programmatically verifiable, often at high frequency - the engineering challenge shifts from access to interpretation. Different chains differ in how quickly transactions become final, how vulnerable recent blocks are to reorganization, which token standards they use, and how protocol events are represented in the data. As a result, even the same metric may need to be defined differently across ecosystems. Harvey et al. (2022) provide a practitioner-oriented investor guide that emphasizes that crypto is much broader than just spot bitcoin.

#### Core on-chain metrics

Core on-chain metrics include:

- **Network activity**: Active addresses, transaction counts, and transfer volumes provide a first view of protocol usage. These measures can signal adoption and engagement, but they require careful interpretation because bots, internal transfers, or address-reuse patterns may inflate activity.
- **Security metrics**: In proof-of-work networks, hash rate measures the total computational power securing the chain; in proof-of-stake networks, staked value measures the capital validators have committed and put at risk. In both cases, these are rough proxies for the economic cost of attacking the network, though their interpretation depends on the protocol design and market structure.
- **Economics**: Token supply schedules, inflation mechanics, issuance rules, and fee dynamics shape incentives for users, validators, and investors. These variables also influence valuation narratives by affecting scarcity, dilution, and the distribution of economic rewards. Separately, futures positioning data, including COT-style classifications, are often used to study who trades crypto futures and whether specific trader categories exhibit timing ability (Baur and Smales, 2022).
- **DeFi metrics**: Total value locked, or TVL, is widely used as a proxy for the capital committed to a protocol. It can be informative, but should be interpreted cautiously because it may reflect double-counting, leverage, or token price swings rather than net new economic activity. Protocol fees and revenue offer additional measures of usage and value capture, but definitions vary across aggregators - record the exact definition used.

#### Data sources

When it comes to cryptocurrency data sources, direct access (running nodes) provides maximum control but entails high operational costs. Indexing protocols and analytics platforms (for example, The Graph, Dune) make decoded blockchain data queryable without requiring full infrastructure maintenance.

**Commercial aggregators** provide curated metrics via API, but they should be treated like any vendor feed: document transformations, definitions, and backfills. Alexander and Dakos (2020) examine which cryptocurrency data sources scholars should use, finding that major aggregators provide statistically equivalent data for liquid assets but can diverge for illiquid tokens.

#### AMM and DEX data

Decentralized exchanges using AMMs generate transparent on-chain records of liquidity provision, swaps, and fees. Lehar and Parlour’s (2021) empirical work on Uniswap illustrates how this transparency enables measurement of market structure that is hard to obtain in traditional venues.

PIT requirements include understanding finality and reorg policies; documenting metric definitions and provider transforms; tracking backfills and recomputations; and accounting for differences in chain and token standards. The next section explores alternative data and discusses how to work with this increasingly relevant source.

**Implementation**: `09_onchain_fundamentals` illustrates how to query on-chain metrics and align them with traditional market data.

## 4.4 Understanding alternative data

Alternative data can be a genuine edge, but it is also a high-variance investment: noisy signals, unstable coverage, hidden backfills, and real legal and reputational risk. A disciplined evaluation process is what separates interesting datasets from production-worthy inputs. In *Chapter 2*, we covered data quality issues such as PIT correctness, survivorship bias, and identifier integrity, which apply to *all* financial data. This section focuses on evaluation criteria for alternative data acquisition decisions - specifically, whether to invest in a new dataset.

### The alternative data landscape

Alternative data encompasses information beyond traditional market and fundamental sources. Green and Zhang (2024) provide a useful taxonomy:

- **Text and attention data**: News articles, social media, earnings call transcripts, and search activity. Classic evidence shows that media tone can predict short-horizon market behavior, though effects are sensitive to leakage and model specification (Tetlock, 2007). The *engineering* of text data (extraction, cleaning, PIT storage) is covered in *Section 4.5*; the *featurization* of text (sentiment, embeddings, transformers) is covered in *Chapter 10*.
- **Business process data**: Credit card transactions, point-of-sale records, job postings, and hiring patterns. LinkedIn entry/exit data and Glassdoor ratings can predict operational health, though coverage is uneven and firm-reported numbers in 10-Ks are often too lagged to compete (Green and Zhang, 2024).
- **Sensor and geospatial data**: Satellite imagery of parking lots or construction sites, geolocation tracking, shipping vessel positions, and industrial sensor readings.
- **Prediction markets**: Platforms trading binary event contracts whose prices produce continuously updated probability proxies. Ng et al. (2025) document meaningful price discovery in modern prediction markets, finding that higher-liquidity venues incorporate information more quickly. Limitations include thin liquidity outside headline contracts and regulatory fragmentation.

On the demand side, evidence suggests sell-side analysts increasingly reference non-traditional datasets in their research workflows, and that this adoption is associated with improved forecast accuracy (Chi, Hwang, and Zheng, 2024).

### The evaluation framework

Alternative datasets usually fail for predictable reasons: they are not incremental once existing features are controlled for, they cannot be reconstructed at decision time, or they introduce legal and operational risks that overwhelm their expected value. A practical due diligence focuses on signal content, data quality, legal risks, and commercial/engineering costs with an explicit bias toward stopping early on hard constraints rather than debating marginal score differences.

**Signal content** evaluation asks whether the dataset can plausibly deliver incremental, tradable information:

- **Uniqueness**: Specify the baseline feature set and test incremental value rather than raw correlations (for example, baseline-versus-baseline and dataset-versus-dataset-only). Improvements that vanish under reasonable controls are typically redundant proxies.
- **Decay and durability**: Treat half-life as a design constraint. McLean and Pontiff (2016) show that published stock return predictors lose ~58% of their performance post-publication - a useful baseline for how quickly signals erode. Whether a specific alternative dataset “accelerates diffusion” is an empirical question to test under realistic timestamps and costs (Hong et al., 2000).
- **Common failure modes**: (i) Real signal content but not tradable after costs/capacity, (ii) apparent performance created by revisions or look-ahead.

Data quality checks verify that we can reproduce results under production constraints, especially those related to point-in-time availability.

- **Quality and stability**: Verify versioning, revision exposure, and detectability of changes to the methodology. Backfills and vendor process changes can affect performance; ESG ratings illustrate how methodological differences drive disagreement, making “stability of measurement” central (Berg et al., 2022).
- **Coverage**: Evaluate representativeness across the tradable universe (names, regions, sectors) and across regimes; diagnose systematic bias or missingness (for example, low coverage of small-cap or non-U.S.).
- **Latency**: Align publication lag, cadence, and timestamp conventions with the decision process. If “first observable time” cannot be stated precisely, point-in-time backtests are not defensible.

Legal risks are a hard-fail gate: risk cannot be “optimized away” by model performance.

- **Compliance and rights**: Establish provenance, permitted uses, contractual restrictions, audit/ deletion obligations, and downstream redistribution limits.
- **Red flags**: (i) Material Nonpublic Information (MNPI) by narrowness (small panels, constrained geographies, entity-specific “channel checks”), (ii) privacy risk through linkage even without explicit identifiers (device/ad IDs, granular location/event data) with GDPR/CCPA exposure.

Commercial and engineering gates ensure the dataset fits deployable capacity and does not impose disproportionate operational drag.

- **Technical friction and economics**: Evaluate total cost of ownership (integration, entity resolution, storage, monitoring, schema-change handling, incident load) and compare expected value to roadmap capacity.
- **Operationalization test**: Estimate one-time engineering lift, ongoing maintenance burden, and a conservative value range after realistic latency, costs, and capacity constraints.
- **Decision rule**: Privilege hard constraints (legal, point-in-time integrity) over marginal differences elsewhere; only compare “scores” among datasets that clear the hard-fail gates.

### Build versus buy for alternative data

*Chapter 2* discussed general build-versus-buy tradeoffs for financial data. For alternative data specifically, additional factors apply:

- **Alpha crowding**: Built pipelines using proprietary internal data remain differentiated; purchased datasets erode as adoption spreads.
- **Legal risk asymmetry**: Web scraping and data collection operate in legal gray areas; vendors may provide representations, but the buyer still owns compliance and the interpretation of what use is permitted.
- **Maintenance burden**: Alternative data sources change frequently (API breaks, coverage shifts, and methodology updates); vendors absorb these operational costs.

For most organizations, a hybrid approach works best: buy cleaned, validated data for mainstream alternative sources; build capabilities only for truly proprietary signals where internal data or relationships provide unique access.

### Data sources and implementation

The ecosystem spans terminal platforms (Bloomberg, LSEG Workspace), vendors specializing in different modalities, such as payments, web data, or geospatial data, and marketplaces such as Nasdaq Data Link, Snowflake Marketplace, and AWS Data Exchange. Marketplaces can simplify procurement, but do not eliminate the need for PIT validation and methodology audits.

Several sources provide meaningful alternative data for experimentation (the chapter `README` lists additional sources):

- **GDELT Project**: Large-scale global news event data for geopolitical risk and event detection; just the 2015 dataset weighs 2.5TB; requires NLP pipelines to extract signals (see next section and *Chapter 10*).
- **SEC 13F filings**: Institutional holdings disclosed quarterly; useful for “smart money” positioning and crowded trade detection.
- **On-chain DeFi metrics**: Total Value Locked (TVL) and protocol revenue as crypto-native fundamentals.
- **Prediction markets**: Kalshi (CFTC-regulated) and Polymarket (crypto-based) provide time series of event probabilities.

*Text data*, including corporate filings, news, and earnings calls, represents the largest category of accessible alternative data - but requires substantial engineering before it becomes model-ready. The next section covers the extraction and storage pipeline, and *Chapter 10* covers featurization (sentiment dictionaries, embeddings, and fine-tuned transformers such as FinBERT). **Implementation**:

- `10_institutional_holdings_13f` builds the 2024Q3 manager-stock graph from SEC 13F filings, computes manager-level concentration metrics and co-ownership edges, and surfaces the most widely held issuers.
- `11_defi_tvl_evaluation` walks the DeFi TVL panel (4 chains, 2017-2026) and tests TVL growth as a forward-return predictor (peak IC = 0.12; weak signal).
- `12_kalshi_prediction_markets` extracts event-prior probabilities from Kalshi markets and demonstrates the API pattern for prediction-market consumption.
- `13_polymarket_prediction_markets` covers the Polymarket equivalent, with notes on liquidity and contract-design differences.

## 4.5 Using text data for NLP features

SEC filings are *fundamental* data by source, but the unstructured text they contain, such as Management Discussion and Analysis, Risk Factors, and earnings call transcripts, is, by nature, *alternative* data: it requires NLP pipelines to extract signals that cannot be reduced to accounting line items. This section covers the engineering work that makes text usable: source selection, section extraction, cleaning, and PIT-correct storage.

### Targeting MD&A and risk factors

SEC 10-K annual reports contain standardized narrative sections that provide rich signals about a company’s outlook:

- **Management’s Discussion and Analysis (MD&A, Item 7)**: Narrative discussion intended to provide material information for assessing financial condition, results of operations, liquidity, and cash flows (Regulation S-K Item 303).
- **Risk Factors (Item 1A)**: Required disclosure of *material* factors that make an investment speculative or risky, organized under relevant headings; the SEC discourages generic risk factors (Regulation S-K Item 105). Often lengthy and standardized, but *changes* in structure, emphasis, or newly introduced themes are frequently more informative than the level of boilerplate.

These sections contain information that cannot be reduced to numbers: qualitative context, forward-looking language, and explanations of changes that may not appear in line items. Tetlock (2014) surveys how textual information transmits to financial markets, establishing the theoretical foundation for extracting signals from corporate narratives. Modern NLP approaches (including finance-adapted BERT variants such as FinBERT) can extract more context-sensitive signals than dictionary methods, but only if extraction and timestamps are correct.

### The engineering pipeline

Building a filing-based text dataset requires four engineering steps:

1. **Document selection**: Pick the primary filing document, not an exhibit.
2. **Section extraction**: Identify item boundaries robustly.
3. **Cleaning and normalization**: Remove markup and non-prose artifacts while preserving paragraph structure.
4. **Point-in-time storage**: Store availability timestamps and audit keys.

#### Step 1: Filing access and metadata

EDGAR provides filing packages plus submission metadata. Libraries such as `edgartools` can simplify retrieval, but the pipeline must still capture the filing’s stable identifier (accession number), form type (10-K/10-Q, amendments), period end, and the **SEC acceptance timestamp** (when the filing became public). That metadata becomes part of the PIT contract for downstream NLP features. The accession number is the natural primary key because multiple filings can share the same date, and amendments require separate tracking.

#### Step 2: Section extraction

Extraction via document parsers should be layered (structured access when available; HTML-anchored boundaries otherwise) and instrumented with measurable quality checks (missing sections, collisions, length outliers). Item numbers label sections, but the same filing can contain multiple “Item 7” mentions (for example, in the Table of Contents and cross-references), and formatting varies across filers and over time. Amendments (10-K/A) and older filings can break naive boundary logic - treat extraction quality as a measurable output and sample failures for iterative rule refinement.

#### Step 3: Cleaning and normalization

Cleaning should remove markup and artifacts of repeated filing without destroying linguistic structure. Common issues include:

- **HTML remnants**: Unclosed tags, entity codes (`&nbsp;`, `&amp;`), style attributes.
- **Table structures**: Financial tables embedded in narrative sections. May need separate handling or removal.
- **Boilerplate**: Safe-harbor statements, standard risk language, repeated cross-references. In many corpora, boilerplate accounts for a large fraction of the extracted text. The right approach is to *measure* it (for example, by comparing year-over-year text overlap) and decide whether to remove it (for novelty/change features) or keep it (for level-based sentiment features).

Cleaning should preserve paragraph boundaries (important for downstream chunking and change detection) while removing non-prose artifacts and normalizing whitespace.

#### Step 4: PIT-correct storage

Store each extracted section as a row keyed by a stable filing identifier (accession number) and include both the reporting period and the availability timestamp. The minimal schema includes:

- **Identifiers**: `cik` (company), `accession` (filing), and `section` (Item 7, Item 1A, and so on) together form the primary key - this triple uniquely identifies each extracted section, including amendments.
- **Timestamps**: `filing_date` and `accepted_at` (the SEC acceptance timestamp, to the second) determine *availability* - when this text became public. `period_end` indicates the fiscal period covered.
- **Content**: The `text` column stores the extracted section; optionally, store both raw and cleaned versions.
- **Metadata**: Form type (10-K, 10-Q, amendments), extraction method, pattern version, and character offsets enable auditing and re-extraction when rules change.

Following the bitemporal data model introduced above:

- `filing_date`/`accepted_at` (knowledge time) determines *availability* - when this text became public. Use `accepted_at` for intraday PIT correctness.
- `period_end` (valid time) indicates the reporting period covered.

Store both **the raw extracted text** and **the cleaned text**, along with extraction metadata (method, pattern version, offsets), so the extraction can be audited and rerun when rules change. Persist to Parquet for efficient filtering by section and time.

### Output artifact

The pipeline produces a **text corpus** with the schema described above. Key fields:

- `cik + accession + section` uniquely identify an extracted section version (including amendments).
- `filing_date`/`accepted_at` enable PIT-correct feature construction.
- `section` enables targeted modeling (MD&A versus Risk Factors versus Business).

Text is most useful when aligned with numeric outcomes. The engineering requirement is that joins must be PIT-safe and audit-safe. The accession number is the natural primary key; keeping both `accepted_at` and `period_end` allows the same filing to be queried as of any historical date without leaking later revisions.

This corpus is the input for *Chapter 10*, which covers the full NLP feature engineering pipeline - dictionary-based sentiment, static word embeddings, and fine-tuned transformer models - and demonstrates a ‘news surprise’ alpha factor that measures semantic deviation of today’s news from recent coverage, depending directly on the PIT-correct text storage established here. **Implementation**: `14_text_data_extraction` shows the complete pipeline from EDGAR filings to a structured Parquet dataset. It demonstrates the year-over-year change-analysis primitive on Apple’s 10-K: the 2025 filing’s Risk Factors section runs 9,792 words with 92.9% word overlap relative to the prior year (Jaccard 0.31), surfacing 28 new risk-factor terms.

## 4.6 Summary

We have laid the data engineering foundations for working with fundamental and alternative data. The central theme is point-in-time correctness: every data source - whether SEC filings, macroeconomic releases, futures positioning, or blockchain metrics - must be timestamped with both the date it became available and the period it covers. The bitemporal model introduced here applies uniformly across asset classes and data types, enabling defensible backtests that avoid look-ahead bias.

The companion notebooks demonstrate these principles in practice: extracting XBRL fundamentals with proper vintage handling, aligning macro data by release schedule rather than reference period, building text corpora from SEC filings with robust section extraction, and evaluating alternative data sources through a systematic framework.

These PIT-correct artifacts become the inputs to *Chapter 8*’s feature engineering workflow. The EDGAR text corpus from *Section 4.5* feeds *Chapter 10*, where we build NLP-based features (sentiment, topics, and embeddings) on the storage pipeline established here.

*Chapter 5* turns to synthetic financial data: generative models that augment scarce histories and stresstest backtests against unobserved scenarios.
