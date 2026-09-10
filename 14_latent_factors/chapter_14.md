# Chapter 14: Latent Factor Models

Hundreds of return predictors have been proposed in the empirical asset-pricing literature. The question is not how many exist but which of them reflect a stable common structure, which are priced, and which are artifacts of specification search or portfolio construction.

This chapter approaches that problem from the perspective of latent-factor estimation. It begins with Principal Component Analysis (PCA), then moves to Instrumented PCA (IPCA) and Risk-Premium PCA (RP-PCA), and finally to conditional autoencoders (CAE) and related stochastic-discount-factor (SDF) approaches. The unifying theme: use the data to summarize high-dimensional return variation with fewer factors, while being explicit about what each method can and cannot identify.

These methods differ because they optimize different objects. PCA maximizes explained covariance variation. IPCA allows factor loadings to vary with observable characteristics. RP-PCA trades off variance fit against cross-sectional pricing performance. Conditional autoencoders generalize the loading function nonlinearly, and adversarial SDF methods target pricing restrictions more directly. Keeping these objectives distinct is essential: a factor that explains covariation is not necessarily a priced factor, and a factor that helps price assets need not be the one that explains the most variance.

By the end of this chapter, you will be able to:

- Distinguish covariance-explaining attribution factors from priced factors, and explain why that distinction matters for prediction, risk decomposition, and trading applications.
- Implement PCA on asset returns, interpret principal components as latent risk dimensions or eigenportfolios, and diagnose key practical issues including covariance noise, component selection, and loading instability.
- Explain how IPCA and RP-PCA extend PCA by introducing time-varying characteristic-based betas and pricing-error penalties, and evaluate when these extensions are preferable to plain variance maximization.
- Implement and evaluate Conditional Autoencoders using walk-forward validation, ensemble averaging, and interpretability diagnostics such as SHAP, while recognizing their main failure modes.
- Explain how adversarial SDF estimation enforces no-arbitrage restrictions, how its objective differs from CAE reconstruction, and when direct pricing-error minimization is likely to add value.
- Compare latent factor methods across datasets and modeling objectives, and choose among PCA, IPCA, RP-PCA, CAE, and SDF approaches based on dimensionality, economic goal, and evaluation design.
- Move from the from-scratch teaching implementations to the `ml4t-models` library’s `LatentFactorForecastPipeline` for production use, and recognize when a model (SDF, SAE) does not fit that pipeline.

A **stochastic discount factor** (**SDF**), also called a **pricing kernel**, is the object that links future payoffs to today’s asset prices. An asset’s price equals the expected value of its future payoff weighted Formally, for a gross return 𝑟௜ǡ௧ାଵ, the no-arbitrage condition is 𝐸௧ൣܯ௧ାଵݎ௜ǡ௧ାଵ] = 1, or equivalently by the SDF, so states of the world in which wealth is especially valuable receive a higher weight. 𝐸௧ൣܯ௧ାଵݎ௜ǡ௧ାଵ ] = 0, for excess returns. ௘

The methods in this chapter pursue three different objectives, and the predictive workflow differs accordingly. PCA is a covariance-decomposition tool; eigenportfolios and yield-curve PCA inherit that role. IPCA and CAE are conditional latent-factor models that estimate factor returns together with characteristic-conditioned loadings. RP-PCA also estimates latent factors but tilts the criterion toward priced variation through a pricing-error penalty. The adversarial SDF estimates a pricing kernel directly from no-arbitrage moment restrictions, and the supervised autoencoder learns a return forecast end-to-end without an intermediate factor representation.

Several of these models share a predictive workflow that *Section 14.5* develops as the **three-stage latent-factor forecasting adapter**:

1. **Estimate a factor representation** from realized returns and, where applicable, conditioning information.
2. **Forecast the factor premia** from the estimated factor-return history.
3. **Map factor-premium forecasts to asset-level expected returns** using current exposures.

If stages 2 and 3 are grouped as the forecasting layer, the same workflow can also be described as a two-block structure: factor estimation followed by forecasting and mapping. IPCA, RP-PCA, and CAE pass through the adapter as a matter of routine; PCA can be wrapped in it when factor-timing is the goal, but its main use in this chapter is risk decomposition. The adversarial SDF and the supervised autoencoder sit entirely outside the adapter: the SDF learns a discounting object directly, and the SAE predicts returns end-to-end. The `ml4t-models` library packages the adapter as a single composable object; the teaching notebooks derive each stage from scratch.

*Section 14.1* frames latent factor models as a response to the factor zoo. *Sections 14.2–14.4* develop the linear foundation (PCA on equities, eigenportfolios, yield curves). *Section 14.5* introduces IPCA and RP-PCA and operationalizes the three-stage adapter. *Section 14.6* extends it to the nonlinear CAE and clarifies why the SDF requires a different design. *Section 14.7* walks through a full from-scratch CAE implementation and contrasts it with the supervised autoencoder. *Section 14.8* compares results across five case studies, and *Section 14.9* distills the lessons.

## 14.1 Making the case for latent factors

Empirical asset pricing started with a single ‘market’ factor (CAPM), grew to three (Fama and French, 1993) and then five (Fama and French, 2015), and eventually became what John Cochrane called the “factor zoo” in his 2011 AFA Presidential Address: hundreds of published factors, each claiming to explain the cross-section of expected returns (Cochrane, 2011). By the mid-2010s, the count exceeded 400 (Harvey and Liu, 2019). The question facing every practitioner is how much of this proliferation reflects genuine economic content and how much is a statistical artifact (*Figure 14.1*).

![Figure 14.1](assets/figure_14_1.png)

*Figure 14.1: From Factor Zoo to Latent Factor Models*

Academic research does not support a single, settled verdict on the factor zoo. One strand emphasizes multiple testing, publication bias, and the proliferation of published factors. Another emphasizes that replication rates can remain high once factors are evaluated within consistent frameworks and across broader datasets.

A balanced summary would be: the field has genuine false-positive and specification-risk problems, but the strongest conclusion is not that the entire factor zoo is illusory. Rather, the evidence points to a smaller, more robust set of recurring factor themes embedded in a much larger set of fragile or implementation-sensitive results.

### Three axes of the factor zoo debate

The controversy runs along three axes (statistical validity, identification, and economic interpretation), drawing on work that includes Harvey (2017), Hou, Xue, and Zhang (2020), and Cochrane (2011).

#### Axis 1 – Statistical validity

dreds of factors have been tested across the literature, the conventional t-statistic threshold 𝑡 The primary culprit is **data mining**. Harvey, Liu, and Zhu (2016) quantified the damage. When hun-

is far too lenient. This threshold implies an estimated effect at least two standard errors away from zero, which, in turn, for large samples, corresponds to a (two-sided) p-value below 0.05. Equivalently, than 5%. Instead, the authors suggest a multiple-testing adjustment (see *Chapter 7*) of around 𝑡 if the true effect were zero, the probability of observing an effect this or more extreme would be less

for newly proposed factors, a rule of thumb that, if applied retroactively, eliminates a large share of published results. Harvey later stresses that even this threshold may be insufficient when effects are rare, treating the problem as one of research culture rather than a simple threshold correction.

**Publication bias** compounds the problem: the publication process creates strong incentives toward significant results and selective reporting. The resulting distribution of published t-statistics shows truncation below 2.0 and unusual clustering in the ranges just above it, consistent with selective reporting of marginally significant results.

A growing body of evidence, however, suggests the data-mining problem may be less severe than these headline numbers imply:

- McLean and Pontiff (2016) find that predictor returns decline by about 26% beyond the original sample period (consistent with modest overfitting), while a further 58% post-publication decline suggests that arbitrage erodes returns after discovery, a separate mechanism from statistical bias.
- Jensen et al. (2022) mount the most comprehensive challenge, developing a hierarchical Bayesian replication framework across 153 factors in 93 countries. Testing CAPM alpha rather than raw returns (the theoretically correct benchmark), they find an 82% replication rate, show that the factors cluster into roughly 13 themes that hold up globally out-of-sample, and that the evidence is *strengthened*, not weakened, by the large number of observed factors.
- Chen (2024) arrives at a consistent conclusion from a different direction, developing false-discovery bounds showing that at least 75–91% of published predictors are statistically valid and demonstrating that Harvey, Liu, and Zhu’s high false-discovery estimates stem from misinterpreting statistical insignificance as falsity.

The emerging picture is that most published factors reflect genuine statistical regularities, but whether they survive trading costs and market learning is a distinct and often more demanding question.

#### Axis 2 – Identification and measurement

Hou, Xue, and Zhang (2020) show why construction choices matter so much. They define **microcaps** using the 20th percentile of NYSE market equity, and when that breakpoint is applied to the full stock universe, these firms represent a majority of listed names but only a very small share of total market capitalization. Therefore, equal-weighted anomaly portfolios give microcaps disproportionate influeven the conventional |ݐ| > 1.96 hurdle, with especially severe attrition in trading-frictions variables. ence. Under NYSE breakpoints and value-weighted returns, they report that 65% of 452 anomalies fail Anomaly evidence is therefore highly sensitive to the definition of the universe and portfolio construction, so any comparison between named and latent factors must keep those choices fixed.

#### Axis 3 – Risk versus mispricing

Even among factors that survive statistical and methodological filters, the field remains divided on *why* they work: whether the return premium compensates for systematic risk or reflects persistent behavioral mispricing sustained by limits to arbitrage. Cochrane (2011) frames this as modern finance’s “dark matter” problem: we observe the gravitational pull of risk premia but cannot directly identify the specific macroeconomic shocks that cause them. This risk-versus-mispricing distinction has direct consequences for model design and evaluation, a theme that recurs throughout *Sections 14.5–14.6*.

### The recurring core

Recent replication and model-selection work repeatedly returns to a compact core of factors (Hou, Xue, and Zhang, 2020; Fama and French, 2015; Barillas and Shanken, 2018):

- **Market**: The equity risk premium
- **Value**: Buying cheap assets, potentially proxying for distress or duration risk
- **Momentum**: Trend-following, reflecting behavioral underreaction or liquidity cascades
- **Profitability**: Buying productive assets, central to both the Fama-French five-factor model and the q-factor model
- **Investment**: Buying conservative assets, consistent with low discount rates in production models

This compact set is not a definitive list, but it is striking that independent lines of research converge on it. Fama and French’s five-factor model derives its factors from the dividend discount model; Hou, Xue, and Zhang’s (2015) q-factor model derives its factors from the firm’s investment optimization (the Investment CAPM), later augmented with an expected-growth factor (Hou et al., 2021). Both arrive at overlapping empirical predictions centered on profitability, investment, and valuation.

Model-comparison work by Barillas and Shanken (2018) finds that both factor families are dominated by specifications that include momentum, further reinforcing the recurrence of this core. Jensen et al. (2022) find that their 153-factor universe clusters into roughly 13 themes, and Swade et al. (2023) show that about 15 factors suffice to span the alpha of the full set, consistent with a true dimensionality much smaller than the zoo but somewhat larger than the five highlighted above.

Machine learning reinforces this picture. Gu, Kelly, and Xiu (2020) found that neural networks’ dominant predictive inputs are familiar (price trends, liquidity, accounting quality), and that gains come from nonlinear interactions among core factors rather than esoteric new signals. Feng et al. (2020) use a double-selection LASSO (*Chapter 15*) to show that most proposed new factors are redundant relative to the existing menu, with profitability and investment retaining the clearest incremental content, a finding about disciplined testing rather than a definitive reduction to a handful of labels.

### From the zoo to latent factors

A key conceptual distinction clarifies what latent factor models can and cannot resolve. Following Idzorek, Kaplan, and Ibbotson (2024):

- **Attribution** **factors** explain co-movement among assets but may carry zero expected return
- **Priced** **factors** emanate from asset pricing models and carry genuine risk premia

PCA is designed to extract common variation in returns, which may or may not line up with priced expected-return factors. The three objectives introduced previously organize methods that progressively tighten this link: variance maximization (*Sections 14.2–14.4*), variance plus pricing errors (*Section 14.5*), and direct pricing-error minimization (*Section 14.6*).

Latent factor models do not make the factor-zoo debate disappear. They change the object of selection. Instead of proposing and testing a long list of named characteristics one at a time, they infer a low-dimensional structure directly from returns, sometimes augmented by pricing restrictions or observable characteristics. This reduces one source of researcher discretion, but it does not, by itself, guarantee that the extracted factors are priced, economically interpretable, or stable out-of-sample.

The methods that follow are complementary responses to factor proliferation: powerful tools for discovering structure, not clean escape hatches from the data-mining problem.

## 14.2 Extracting latent factors with PCA

PCA solves the first of the three latent factor objectives introduced in *Section 14.1*: maximizing explained variance. It transforms a set of correlated asset returns into uncorrelated components that capture maximum variance (Goyal, 2012).

The transformation relies on eigendecomposition of the return covariance matrix, a standard linear algebra operation with direct financial interpretation. As a pure variance-maximization method, PCA finds the dominant directions of co-movement but is agnostic to whether those dimensions carry risk premia, a limitation that *Section 14.5* addresses.

### The mechanics of eigendecomposition

The PCA procedure begins with a matrix of asset returns (time periods as rows, assets as columns). Returns are demeaned; standardizing to unit variance is optional but standard for cross-sectional work (see *Assumptions and design choices*, the following subsection).

From the standardized return matrix, we compute an N×N covariance matrix Σ. When inputs are standardized to unit variance, this yields the correlation matrix, a distinction that matters for interpretation, taken up in the next subsection. PCA then solves the eigenvalue problem: finding all pairs (λ, v) such that Σv = λv.

The solution yields two key outputs. **Eigenvalues** (λi) are scalars representing the magnitude of variance explained by each corresponding principal component. They are ranked in descending order, so λ₁ ≥ λ₂ ≥ … ≥ λn. The proportion of total variance explained by the i-th component equals λi / Σλ. A “scree plot” shows eigenvalues in descending order; the “elbow,” where explained variance levels off, suggests how many components to retain (*Figure 14.2*). `01_pca_equity_sectors` constructs this scree plot for sector ETFs and adds bootstrap confidence intervals on the loadings to assess which sector exposures are statistically reliable.

![Figure 14.2](assets/figure_14_2.png)

*Figure 14.2: Variance explained by the top 5 principal components*

**Eigenvectors** (vi) are vectors of “loadings” that define each principal component as a linear combination of the original asset returns. These vectors are mutually orthogonal, meaning the resulting principal components are uncorrelated with one another. In finance, each eigenvector can be interpreted as portfolio weights, a concept we develop fully in the next section on eigenportfolios. `02_eigenportfolios` demonstrates this interpretation on a broad US equity universe.

### Assumptions and design choices

PCA is a useful tool, but its interpretation in finance depends on several important assumptions.

- **Linearity.** PCA captures only linear patterns of co-movement. In financial markets, however, relationships often change across regimes, especially during stress, and many interactions among characteristics are non-linear. For example, the effect of momentum may depend on volatility or liquidity. Rolling-window PCA can partially adapt to temporal variation, but it does not address the underlying linearity constraint. *Section 14.6* returns to this limitation when we introduce autoencoders. The notebook `01_pca_equity_sectors` illustrates rolling PCA for sector ETFs.
- **Variance is not the same as pricing relevance.** PCA is designed to explain as much return variation as possible, not to identify factors with high expected returns. A factor can explain a large share of covariance yet earn little or no risk premium; conversely, a factor with modest variance can be economically important if it commands a large premium. This gap between covariance structure and pricing relevance motivates RP-PCA in *Section 14.5*.
- **Focus on second moments.** PCA works entirely through the covariance matrix of centered returns, so it uses only second-moment information. Under Gaussian assumptions, this aligns with the maximum-likelihood estimation of a linear factor model. Financial returns, however, are rarely Gaussian: they are often skewed, heavy-tailed, and prone to jumps. In those settings, PCA still finds directions of maximum variance, but it may miss structure that appears in higher moments and may still matter economically.

Two key design choices deserve a closer look.

#### Covariance or correlation?

A basic implementation choice is whether to apply PCA to the covariance matrix or the correlation matrix.

- Using the **covariance matrix** preserves the natural scale of returns, so more volatile assets receive more weight in the decomposition
- Using the **correlation matrix** first standardizes each asset to unit variance, preventing high-volatility assets from dominating the results

For cross-sectional equity analysis, correlation-based PCA is usually preferred because it emphasizes common structure rather than differences in raw volatility. For multi-asset portfolios with markedly different volatilities, the choice should be deliberate rather than by default. The notebook `01_pca_equity_sectors` uses correlation-PCA for the sector ETF example.

#### Idiosyncratic volatility normalization

A useful third option goes beyond the simple covariance-versus-correlation choice: normalize each asset’s returns by its lagged idiosyncratic volatility before running PCA.

This adjustment targets a specific problem. In raw covariance PCA, high-volatility assets can dominate the decomposition. Correlation PCA solves that problem by forcing all assets to have equal total variance, but that is often too strong: it removes not only idiosyncratic volatility differences but also meaningful differences in common-factor exposure. Idiosyncratic volatility normalization is more selective. It scales returns only by the diversifiable component of volatility, leaving common variation more intact.

In practice, this often leads to cleaner separation between signal and noise, more stable eigenvectors over time, and more parsimonious factor structures, meaning that fewer components explain the same share of variation. To implement this correctly, idiosyncratic volatility must be estimated from residuals in a prior estimation window, and current returns must be normalized using only that lagged information. This preserves the point-in-time structure and avoids lookahead bias.

### Covariance estimation quality

of assets 𝑁, approaches or exceeds the number of time periods, 𝑇, the sample covariance matrix be-A practical warning: PCA is only as reliable as the covariance matrix it decomposes. When the number

comes difficult to estimate. In that regime, sample eigenvalues are distorted upward, weak factors become hard to distinguish from noise, and estimated eigenvectors can become unstable across samples. **Random matrix theory** provides a useful benchmark for separating signal from noise in large covarithat can arise from noise alone, given the ܰȀܶ ratio. Eigenvalues that fall within this range should be ance matrices. In particular, the **Marchenko-Pastur distribution** characterizes the range of eigenvalues

treated cautiously, since they may not reflect genuine common structure.

A common remedy is to use **shrinkage estimators**, with the **Ledoit-Wolf** family being the best-known example. These methods pull extreme sample eigenvalues toward a more structured target, reducing estimation error and improving stability. For large equity cross-sections, applying shrinkage before PCA is often advisable; otherwise, the leading eigenportfolios may reflect sampling noise rather than persistent market structure. See the Primer for a fuller discussion of the Marchenko-Pastur benchmark and shrinkage-based covariance estimation.

When the number of assets, 𝑁, is large relative to the number of time periods, 𝑇, PCA starts to pick How many PCA factors are real?

up noise as well as signal. In this high-dimensional setting, sample eigenvalues are biased upward, and weak factors can become indistinguishable from random variation (noise). tifies this: a factor is recoverable only if its population eigenvalue exceeds 1 + √ߛ, where ߛൌܰ Ȁܶ The **Baik-Ben Arous-Péché** (**BBP**) phase transition (Baik et al., 2005; see Paleologo, 2025, Ch. 7) quan-

and eigenvalues are normalized to unit idiosyncratic variance. Below this threshold, no estimation The practical message is simple: the larger ݊Ȁܶ becomes, the harder it is to recover anything except technique can separate signal from noise.

where 𝛾, the threshold is relatively low, so several components can be retained. For a 500-stock the strongest factors. In our case studies, this works out quite differently across datasets. For ETFs, equity panel with only one year of daily data, 𝛾, and only the market plus a few dominant sector factors are likely to survive. For CME futures, where 𝛾06, most factors remain identifiable.

A good rule of thumb is therefore to keep 𝑛 modest relative to 𝑇 whenever possible. If that is not feasible, expect PCA to recover only the dominant part of the factor structure. When 𝛾 becomes moderately

large, eigenvalue shrinkage is usually more reliable than a visual scree-plot rule for deciding how many components to keep. See the Primer for details on the BBP threshold, the Marchenko-Pastur benchmark, and shrinkage-based alternatives.

**Implementation**: See `01_pca_equity_sectors` for rolling PCA and bootstrap stability analysis.

Next, we discuss how to use the eigendecomposition techniques to construct equity portfolios.

## 14.3 Eigenportfolios for equity strategies

Reading eigenvectors as portfolio weights yields uncorrelated “eigenportfolios,” building blocks for risk management and systematic trading (Avellaneda and Lee, 2010). See `02_eigenportfolios` for the full PCA pipeline, including universe selection, variance decomposition, and market factor validation.

### Constructing and interpreting eigenportfolios

The first eigenportfolio, corresponding to the largest eigenvalue, captures the dominant source of variance. Empirically, nearly all weights are positive: it acts as a data-driven proxy for the broad market portfolio.

In our analysis of the 500 most liquid stocks by dollar volume from the US Equities Panel dataset, the first principal component (PC1) correlates with the equal-weighted market return by 0.99, confirming that it serves as a data-driven market proxy. Each stock’s loading on PC1 measures its exposure to this statistically extracted market mode. This is analogous to the **CAPM beta** in the sense that both quantify sensitivity to a common factor. Unlike CAPM beta, however, the PC1 loading is not estimated relative to a prespecified market portfolio and does not carry an asset-pricing interpretation; it follows mechanically from the sample covariance matrix and the PCA normalization.

![Figure 14.3](assets/figure_14_3.png)

Whereas the first eigenportfolio 𝑃1 loads positively on all sectors, PC2 reveals a growth-versus-value Figure 14.3: Eigenportfolio loadings across sectors

rotation, loading positively on Technology and Communication Services while negatively on Financials and Energy. PC3 captures defensive-versus-cyclical dynamics (see *Figure 14.3*). These **interpretable patterns** emerge purely from the return covariance structure, without any imposed economic labels.

Higher-order eigenportfolios are, by mathematical construction, orthogonal to each other. **Orthogonality** ensures that the extracted component returns are uncorrelated in the sample. Higher-order eigenportfolios are often approximately market-neutral in practice, but they are not automatically dollar- or beta-neutral. These portfolios are typically long-short in nature (positive weights in some stocks, negative in others) and often capture intuitive economic exposures such as sector rotations or style tilts. For example, the second eigenportfolio might represent a long position in growth stocks against a short in value stocks, or a long in technology against a short in utilities. Stationarity is one caveat to consider. Eigenportfolio loadings can rotate significantly across estimation windows. The second component might capture growth-versus-value in one decade and defensive-versus-cyclical in the next, as sector correlations shift with the macroeconomic regime. This instability limits the use of eigenportfolios in live trading without periodic re-estimation, and it means that economic interpretations of higher-order components should be treated as descriptive rather than structural.

Statistical factors can also be interpreted via regression. Statistical PCA factors are abstract by construction: they maximize variance explained but carry no economic label. A practical interpretation method: regress each eigenvector’s loadings cross-sectionally on observable firm characteristics such as 2009). The regression 𝑅2 indicates how much of the variation in each statistical factor is explained by sector indicators, market capitalization, momentum, and book-to-market ratio (Connor and Korajczyk,

factor under a different name. If PC4 shows low 𝑅2 against all available characteristics, it may capture known fundamentals. If PC1 loadings correlate strongly with sector dummies, it is effectively a sector

a genuinely novel latent risk dimension, precisely the type of structure that justifies data-driven factor discovery over pre-specified models.

### Applications in quantitative finance

Annualized return, volatility, and Sharpe ratio statistics for each eigenportfolio quantify the risk-return profile of each latent factor: the first component typically shows the highest volatility while higher-order components may offer modest risk-adjusted returns.

Any portfolio’s risk decomposes into exposures to these orthogonal factors, revealing concentrations that pre-specified factor models miss: a manager might discover that apparent sector diversification masks exposure to a single statistical factor. This decomposition feeds directly into *Chapter 17*’s portfolio construction: eigenportfolio betas serve as the risk dimensions over which position sizing is optimized.

Regressing stock returns on the top K eigenportfolios isolates residuals that are hypothesized to mean-revert; the eigenportfolio return series can also serve as inputs for factor-timing models. Both applications face practical headwinds: mean-reversion parameters are unstable out-of-sample, and evidence for reliable short-horizon factor timing remains mixed.

### Improving interpretability with hierarchical PCA

A common critique of standard PCA is that higher-order factors lack a clear economic interpretation and can be statistically unstable across different time periods. Marco Avellaneda’s **Hierarchical PCA** (**HPCA**) addresses this by injecting known economic structure into the analysis (Avellaneda, 2019).

HPCA proceeds in two steps. First, PCA is performed independently *within* each known economic grouping, such as GICS sectors. This identifies the dominant factor for technology stocks, another for financials, and so forth. Second, another PCA is applied to the correlation matrix of these sector-level factors to capture cross-sector dynamics. The resulting factors are much easier to interpret: they correspond to clear inter-sector bets (long financials versus short industrials) or intra-sector momentum effects, resolving much of the ambiguity of standard PCA. The notebook `02_eigenportfolios` demonstrates both standard eigenportfolio construction and the HPCA two-step procedure on GICS sectors.

### Fixing eigenvector instability

A persistent practical challenge with rolling PCA is that factor loadings can change abruptly between adjacent estimation windows, not because the underlying factor structure has shifted, but because of a mathematical pathology. Understanding and correcting this instability is essential before using PCA loadings for portfolio allocation or performance attribution.

Consider the problem of near-degenerate eigenvalues. When two eigenvalues are close in magnitude, the corresponding eigenvectors become poorly identified. A small perturbation (adding or removing a single day of returns, or changing one stock in the estimation universe) can cause the eigenvectors to rotate within their shared subspace or swap entirely. In rolling PCA on equity returns, this manifests as factor loadings that “flip sign” between adjacent estimation windows: sector exposures suddenly invert, creating phantom turnover in downstream portfolio weights. The first principal component is typically immune (its eigenvalue is well-separated), but components 3 and beyond frequently exhibit this instability (Paleologo, 2025). diagnostic. For eigenvectors 𝑣(ݐ) and 𝑣(ݐ൅ͳ), cosine similarity equals |ݒ(ݐ)்ݒ(ݐ൅ͳ)| (taking the Measuring instability is vital. Cosine similarity between consecutive eigenvectors provides a simple

absolute value because eigenvectors are identified only up to sign). A value of 1 indicates stable loadings; values near 0 indicate rotation into a different subspace. Track this metric for each principal component across estimation windows. Components with frequent drops below 0.8 are unreliable for attribution or allocation without correction.

A practical fix is to align loading matrices across adjacent windows by an **orthogonal Procrustes rotation**: Let 𝐵௧ and 𝐵௧ାଵ denote the ൈܭ loading matrices from two consecutive estimations Compute the singular value decomposition of 𝐵௧ ⊤𝐵௧ାଵ=

- ⊤ The orthogonal matrix that best aligns the two loading spaces is 𝑅∗ൌܸܷ
- ⊤, and the aligned next-window loadings are 𝐵̃௧ାଵ= 𝐵௧ାଵ
- ∗

This transformation preserves orthogonality and factor span, but it removes arbitrary rotations within nearly degenerate eigenspaces. The result is not a different covariance model; it is the same model written in a temporally smoother coordinate system.

Always check eigenvector stability before using PCA loadings for allocation or attribution:

- If the top 3–5 principal components have well-separated eigenvalues (ratios exceeding 2:1 between consecutive eigenvalues), Procrustes rotation is unnecessary: the loadings are naturally stable
- If eigenvalues are clustered, which is common for equity returns beyond the third component, always apply Procrustes

The stability diagnostic connects directly to *Chapter 17*’s portfolio construction: unstable loadings produce an unstable covariance matrix, which generates unnecessary turnover, and that turnover incurs real transaction costs.

### Practitioner recipe for two-stage production PCA

The notebooks in this chapter demonstrate PCA on a single estimation window with uniform time weighting. Risk models used by institutional investors apply a more sophisticated procedure that separates two empirical facts about financial markets: volatility changes quickly (days to weeks), while the correlation structure changes slowly (months to quarters). Mixing both dynamics in a single estimation window either overreacts to volatility spikes or underreacts to shifts in correlation.

The following two-stage procedure, formalized by Paleologo (2025), addresses this separation: to emphasize recent volatility. Run a preliminary PCA (𝑝 components), compute residuals,

1. **Capture volatility dynamics.** Apply exponentially-weighted time weighting (half-life ~20 days)

and estimate per-asset idiosyncratic volatility.

1. **Estimate the correlation factor structure.** Normalize returns by Stage 1’s idiosyncratic volatility, apply slow exponential weighting (half-life ~120 days), and perform the production PCA. loadings. Reconstruct the full covariance matrix: 𝛺ൌܦఙ൫ܤ𝛺௙ܤ்+ 𝛺ఌ൯ܦఙ, where 𝐷ఙ restores Apply BBP-informed eigenvalue shrinkage (*Section 14.2*) and Procrustes rotation to stabilize

the original volatility scale.

This procedure matters for the ML4T pipeline because it produces the covariance matrix that feeds directly into the portfolio construction methods of *Chapter 17*. Mean-variance optimization, hierarchical risk parity, and risk budgeting all consume a covariance estimate as their primary input. The separation of volatility and correlation estimation also connects to *Chapter 18*’s short-term volatility updating for transaction cost models. The two-stage structure ensures that a sudden volatility spike (for example, a VIX event) immediately updates risk estimates without destabilizing the slower-moving factor structure that determines diversification benefits. Next, we apply Principal Component Analysis to the yield curve to identify fixed income factors.

**Implementation**: See `02_eigenportfolios` for eigenportfolio construction, including sector loading heatmaps, HPCA, and cumulative return visualization.

## 14.4 Decoding the yield curve

PCA’s clearest success is in fixed income. Litterman and Scheinkman (1991) showed that three factors explain 95–99% of variation in the Treasury yield curve, a result that has been replicated across decades and markets.

### Level, slope, and curvature

When PCA is applied to daily or monthly changes in Treasury yields across maturities (from 1-month bills through 30-year bonds), studies consistently find that 95–99% of variation can be attributed to three orthogonal principal components, each with a widely-accepted economic interpretation (*Figure 14.4*):

- **Level**: Explains over 90% of the total variance. Loadings are nearly uniform across maturities, so a level shock corresponds to a parallel shift, all rates moving up or down together. Economically, this reflects broad changes in inflation expectations and real growth prospects.
- **Slope**: Explains 5–8% of variance. Loadings have opposite signs at short and long maturities, so a slope shock steepens or flattens the curve. This factor is closely tied to monetary policy: rate hikes compress the short-long spread, flattening the curve.
- **Curvature**: Captures 1–2% of variance. Mid-curve maturities move opposite to both ends, creating a “butterfly” twist. This factor is linked to rate volatility and path uncertainty. Observing it clearly requires at least 5–7 maturities spanning the curve; with fewer, the third component captures residual variation rather than a clean butterfly.

Three principal components are shown in *Figure 14.4*:

![Figure 14.4](assets/figure_14_4.png)

*Figure 14.4: Principal component analysis on yield curve components*

**Implementation**: `03_yield_curve_decomposition` applies PCA to daily changes in eight Treasury constant-maturity yields (1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y), which recovers the classical Level/Slope/Curvature decomposition with variance shares of 82.3%, 12.3%, and 3.1% (97.8% cumulative). Each maturity is reconstructed from the three factors, and the generalized-duration hedging framework is illustrated on the resulting exposures.

### Why PCA works for yield curves

The yield curve is the rare case where the variance objective and the pricing objective largely coincide: no-arbitrage constraints in fixed income aim to ensure that variance-explaining factors are also priced. The contrast with equities highlights a structural difference:

- Interest rate drivers are low-dimensional and persistent: inflation expectations, real growth, and monetary policy affect all bonds simultaneously, with sensitivity varying by maturity. At the same time, PCA on yield changes is a descriptive decomposition of curve movements, not a proof that the same components are the uniquely priced term-premium factors. No-arbitrage term-structure models and return-predictive term-premium models add further structure that PCA alone does not impose.
- Equities are the opposite: thousands of stocks driven by idiosyncratic news, sector effects, macro shocks, and behavioral factors produce a covariance structure that is complex, time-varying, and regime-dependent, explaining why PCA on equities yields less interpretable and less stable factors.

### Practical application for efficient hedging

This low-dimensional structure enables efficient hedging. Rather than managing exposure to dozens of individual bonds, a manager neutralizes three factor exposures, namely level, slope, and curvature, using liquid instruments (Treasury futures, interest rate swaps). This **generalized duration** approach is more precise, cheaper in transaction terms, and grounded in the yield curve’s empirical low-rank structure.

| Factor | Variance Explained | Movement Pattern | Standard Interpretation |
| --- | --- | --- | --- |
| Level (PC1) | >90% | Parallel shift | Infaltion, growth expectations |
| Slope (PC2) | ~5-8% | Steepening/falttening | Monetary policy stance |
| Curvature (PC3) | ~1-2% | Butterfyl twist | Rate volatility, uncertainty |

*Table 14.1: Principal components of the yield curve*

Economic interpretations are widely used heuristics, not definitive causal mappings, as shown in *Table 14.1*. Exact percentages depend on the maturity set, sample period, and whether PCA is applied to the correlation or covariance matrix of yield changes.

Next, we explore how to enrich the statistical techniques introduced so far with economic data to make them more useful in practice.

## 14.5 Bridging economics and statistics with advanced models

The yield curve’s low-dimensional success highlights PCA’s limitations for equities: high dimensionality, weak structural constraints, and time-varying relationships produce factors that explain variance but not necessarily returns. Two innovations address this:

- **IPCA** (Kelly, Pruitt, and Su, 2019) makes factor loadings dynamic through observable characteristics
- **RP-PCA** (Lettau and Pelger, 2020) explicitly targets pricing-relevant factors

Before deriving either model, we explain the predictive structure they share.

### The three-stage latent-factor forecasting adapter

Several factor-representation models in this chapter can be used for prediction by adding a forecasting layer. PCA, RP-PCA, IPCA, and CAE all estimate latent factors, but they do so for different purposes: PCA explains covariance, RP-PCA tilts factor extraction toward priced variation, IPCA estimates linear characteristic-conditioned loadings, and CAE generalizes that loading map nonlinearly. To turn these representations into ex-ante asset-level forecasts, the **three-stage latent-factor forecasting adapter** wraps each estimator in the same workflow (*Figure 14.5*): ൈܰ excess-return panel 𝑅௘ into a ൈܭ factor history 𝐹 and a per-asset loading map. PCA solves this by SVD on 𝑅௘; RP-PCA modifies the

1. **Stage 1: Factor estimation.** Compress the criterion by adding a pricing-error penalty; IPCA and CAE parameterize the loading map 𝛽௜ǡ௧ as

a function of lagged characteristics. Stage 1’s training objective (variance, variance-plus-pricing-error, or cross-sectional reconstruction) uses contemporaneous returns; this is what the **Stage 2: Factor-premium forecaster.** Predict 𝜆̂௧ାଵ∈ℝ௄ from the training-window factor history structural model is for. 𝐹ଵǣ௧. The simplest choice, used implicitly by KPS (2019) and GKX (2020) and labeled “IID-BS” by 2.

Engel et al. (2025), is the training-sample mean. The library’s `ExpandingMeanFactorForecaster` implements this baseline. AR(1), EWMA, gradient-boosted regression (LightGBM, an analog to Engel et al.’s Q-Boost), and pretrained foundation models (their ZS-Chronos) are drop-in alternatives that target the same object with progressively richer representations of factor **Stage 3: Asset map.** Combine today’s exposures with the forecast premium: 𝜇௜ǡ௧ାଵൌߚ௜ǡ௧ڄ ߣ̂௧ାଵ dynamics. *Figure 14.6* catalogs the forecasters used in this chapter. 3.

If Stages 2 and 3 are grouped as a single forecasting layer, the same workflow reads as a two-block structure: factor estimation followed by forecasting and mapping. The numbered three-stage formulation is the one used consistently throughout this chapter.

![Figure 14.5](assets/figure_14_5.png)

*Figure 14.5: The three-stage latent-factor forecasting adapter. Stage 1 compresses the excess-return panel into a factor history and a per-asset loading map; Stage 2 forecasts the next-period factor premium from that history; Stage 3 combines current loadings with the forecast premium to produce asset-level expected returns. RP-PCA, IPCA, and the CAE differ only in Stage 1*

Two design properties follow. First, Stage 1 and Stage 2 are *separable*: the same structural model can pair with any forecaster, and the same forecaster works across structural models. The chapter notebooks exploit this by sharing the Stage 2 catalog across IPCA, RP-PCA, and CAE. Second, forecast quality enters at Stage 2: better forecasts of the factor premium translate directly into better asset-level signals through Stage 3. The adversarial SDF and the supervised autoencoder do not pass through this adapter. The SDF estimates a pricing kernel directly from no-arbitrage moment restrictions and produces pricing-error diagnostics rather than a factor-return history; *Section 14.7* shows why a separate Stage 2 forecast is not the natural output. The supervised autoencoder predicts returns end-to-end and has no factor-return intermediate to forecast; *Section 14.7* presents it as a direct-prediction contrast.

Engel et al. (2025) extend the adapter by ranking factors by forecast *uncertainty* and using only the most predictable ones in tangency portfolios; that extension lives at Stage 2 and is also a natural place to plug in modern probabilistic forecasters.

The `ml4t-models` library packages the adapter as a single composable object, `LatentFactorForeca` `stPipeline(model, forecaster, mapper)`. The teaching notebooks are built from scratch at each stage; the library is the production format used by the case-study runners.

### Dynamic betas with instrumented PCA

Standard PCA estimates a single, static beta per asset, an unrealistic assumption when risk profiles evolve. Instrumented PCA (IPCA; Kelly, Pruitt, and Su, 2019) makes betas time-varying by modeling them as functions of observable characteristics: size, book-to-market, momentum, and others.

The IPCA model specifies excess returns as: 𝑟௜ǡ௧ାଵ ൌݖ௜ǡ௧ ⊤߁݂௧ାଵ൅ߝ௜ǡ௧ାଵ ௘ where 𝑧௜ǡ௧ is a 𝑃 vector of characteristics observed at time 𝑡, 𝛤 is a ൈܭ matrix, and 𝑓௧ାଵ is the 𝐾 vector of latent factor returns. Equivalently, 𝛽௜ǡ௧ൌ߁⊤ݖ௜ǡ௧, so characteristics govern time-varying

loadings rather than expected returns directly. This timing matters. Characteristics must be lagged relative to returns, and the interpretation is conditional risk exposure rather than a reduced-form anomaly regression.

The central insight is that *characteristics are covariances*: firm characteristics predict returns not because they represent standalone anomalies but because they proxy for time-varying exposures to latent risk factors. A small-cap value stock loads on different factors than a large-cap growth stock, and loadings update as characteristics change.

This interpretation enables a direct empirical test of risk versus mispricing:

- If characteristics primarily explain returns through their role in shaping risk exposures, IPCA identifies strong latent factors and insignificant pricing errors (alphas)
- If characteristics represent pure mispricing independent of risk, the model attributes their effect to a significant alpha term

Kelly, Pruitt, and Su (2019) report that a five-factor IPCA model explains cross-sectional returns far more accurately than existing models, with characteristic effects almost entirely absorbed by their role as instruments for risk exposures.

The original Kelly, Pruitt, and Su (KPS) study uses 94 characteristics; our `us_firm_characteristics` dataset provides 57 after filtering. Missing characteristics should be imputed (using the cross-sectional median is standard) rather than dropping observations, since missingness is informative: smaller firms tend to have fewer available characteristics. from characteristics to loadings) and updating 𝑓௧ (the latent factors). The number of characteristics IPCA estimation uses **Alternating Least Squares** (**ALS**), iterating between updating Γ (the mapping

P matters: too few limit the model’s ability to capture time-varying loadings; too many introduce estimation noise.

Three sensitivities regarding identification recur across post-2019 replications: Results depend on the number of latent factors 𝐾; report a range (typically 3–8) rather than

• selecting a single optimum, since the standard out-of-sample pricing-error rule is sensitive to the test-asset set (Didisheim et al., 2023, show that more complex factor models can outperform out-of-sample).

- Missing characteristics are absorbed into residuals rather than systematic risk. The 57 characteristics in our `us_firm_characteristics` implementation (down from 94 in the original The linear map ߚൌ߁⊤ݖ cannot capture interactions (momentum’s behavior conditional on KPS study) are a floor, not a ceiling. • volatility, size effects conditional on liquidity); *Section 14.6*’s non-linear extensions trade identification clarity for expressive power.

confirms ALS estimation recovers the true 𝛤 before applying the model to real data, and **Implementation**: `04_ipca` includes a parameter-recovery exercise on synthetic data that directly demonstrates 𝐾-sensitivity.

### Finding priced factors with risk-premium PCA

The second PCA flaw (focusing on variance rather than expected returns) is addressed by RP-PCA (Lettau and Pelger, 2020). The defining feature of RP-PCA is not a different loading map but a different objective: standard PCA is augmented with a pricing-error penalty so that factors that explain small amounts of variance yet carry meaningful risk premia are not overlooked.

RP-PCA minimizes a weighted combination of unexplained variance and cross-sectional pricing errors: ௸ǡி 1ܰܶ∑൫ܺ̃ ൅ ߢԜ 1ܰ∑(ܺ‾௜െܨ‾⊤߉௜)2 min ௜ǡ௧െܨ̃௧ ⊤߉௜) 2

௜ǡ௧ ௜

where the first term is the usual reconstruction loss on time-demeaned returns and the second term The parameter 𝜅 controls the trade-off. At 𝜅, RP-PCA reduces to standard PCA, which maximizes penalizes pricing errors in mean returns. pure variance with no regard for pricing. As 𝜅 increases, the objective shifts toward factors that explain cross-sectional return differences, even if they explain little of the total variance. In the limit 𝜅,

The objective is solved by applying standard PCA to a modified matrix, ((ͳȀܶ)ܺ the method focuses entirely on pricing and ignores variance altogether. െܺ‾ܺ‾⊤൯൅ߢԜܺ‾ܺ‾⊤, ⊤ the sample covariance plus a penalty that overweights the mean-return information (so 𝜅 recov-

ers ordinary covariance PCA). The result is factors that are both statistically significant (explaining co-movement) and economically significant (carrying high Sharpe ratios). A factor capturing credit risk might explain only 2% of the return variance yet have a Sharpe ratio 𝜅 elevates it to the 3rd. Lettau and Pelger (2020) report out-of-sample Sharpe ratios more than of 0.8, remaining invisible to standard PCA, which ranks it below the 15th component. RP-PCA with

double those of standard PCA on US equities, 1963–2017; results are sample-dependent and require out-of-sample validation.

Note that **double-selection LASSO** (Feng, Giglio, and Xiu, 2020), the modern standard for testing new candidate factors against an existing zoo, is a special case of the debiased ML framework developed in *Chapter 15*, applied to the factor returns from this chapter’s case studies.

multiple penalty values. The notebook’s `gamma` parameter is this 𝜅, and `gamma=0` reproduces **Implementation**: `05_rp_pca` for a comparison of standard PCA against RP-PCA across standard PCA. Lettau and Pelger write the penalty weight as ͳ ൅ߛ with PCA recovered at 𝛾; the chapter’s ߢൌͳ ൅ߛ shifts this so that 𝜅 is the no-penalty case.

### Test assets as a design choice

Test assets are a design choice, not a neutral backdrop. Recent work shows that factor strength depends on the span of the test assets used in estimation and evaluation:

- Giglio, Xiu, and Zhang (2021) emphasize this point and propose **supervised PCA** to select informative assets when weak factors are present
- Bryzgalova, Pelger, and Zhu (2025) take a complementary route, building test assets endogenously with trees so that the resulting basis assets better span the stochastic discount factor

The practical implication is straightforward: compare methods on a common benchmark set of test assets, but also report how results change when the test-asset span is enriched or redesigned.

All methods in this section share a limitation: factor loadings are linear functions of characteristics. Momentum behaves differently at high versus low volatility, size effects vary with liquidity, and linear models cannot capture these interactions without manual feature engineering. *Section 14.6* addresses this with deep learning models that learn non-linear mappings while retaining the economic structure developed here.

| Model | Estimation objective | Loading structure | Forecast adapter | Best use |
| --- | --- | --- | --- | --- |
| PCA | Maximize covariance explained | Static loadings | Optional | Risk decomposition |
| RP-PCA | Variance plus pricing- error penalty | Static latent loadings | Yes, as factor extractor | Priced-factor discovery |
| IPCA | Conditional latent factor model | Linear characteristic- conditioned betas | Yes | Interpretable conditional betas |
| CAE | Nonlinear conditional latent factor model | Neural characteristic- conditioned betas | Yes | Nonlinear conditional betas |
| Adversarial SDF | No-arbitrage moment restrictions | Pricing-kernel weights | No | Direct pricing-error minimization |
| SAE | Supervised return prediction plus reconstruction regularizer | Bottleneck representation | No | Predictive benchmark |

*Table 14.2: Latent-factor and related models: estimation objective, loading structure, and relationship to the three-stage forecasting adapter*

RP-PCA, IPCA, and CAE pass through the adapter as a matter of routine; PCA can be wrapped in it for factor-timing, but is used here for risk decomposition. The adversarial SDF and the SAE sit outside the adapter. The CAE is introduced in *Section 14.6*; the SDF and the SAE in *Section 14.7*.

The shared adapter, therefore, covers PCA (optionally), RP-PCA, IPCA, and CAE; the differences between these four models live in Stage 1. *Figure 14.6* catalogs the three-tier Stage 2 forecaster ladder (Tier 1: Constant, AR(1), EWMA; Tier 2: Ridge, LightGBM; Tier 3: LSTM, TCN) that the teaching notebooks make swappable; *Section 14.8* reports how the choice interacts with the structural estimator. The next section introduces non-linear techniques for dimensionality reduction.

![Figure 14.6](assets/figure_14_6.png)

*Figure 14.6: The Stage 2 factor-premium forecaster catalog. Tier 1 holds the constant, AR(1), and EWMA baselines; Tier 2 adds ridge and gradient-boosted regressions; Tier 3 adds LSTM and TCN sequence models. Any forecaster pairs with any Stage 1 estimator without retraining the factor model*

**Implementation**: `04_ipca` demonstrates ALS estimation with recovery on synthetic against RP-PCA across 𝜅 values within the same framework. For real-data applications data plus the swappable Stage 2 forecaster catalog; `05_rp_pca` compares standard PCA

with walk-forward validation, see applicable case study notebooks.

## 14.6 The conditional autoencoder

*Section 14.5* operationalized the three-stage forecasting adapter with linear loading maps: IPCA ties conditional betas to characteristics through a single matrix, and RP-PCA tilts the factor-extraction criterion toward priced variation. Both keep the map between characteristics and loadings linear. The **conditional autoencoder** (CAE; Gu, Kelly, and Xiu, 2019) is the nonlinear member of the same family: it keeps the conditional-factor structure and replaces only the linear loading map with a neural network.

The CAE is a strict nonlinear generalization of IPCA. Both start from the same conditional factor representation: 𝑟௜ǡ௧ାଵൌߚ௜ǡ௧ ௧ାଵ൅ߝ௜ǡ௧ାଵ ⊤

istics, 𝛽௜ǡ௧ൌ߁⊤ݖ௜ǡ௧, whereas the CAE replaces this map with a neural network, 𝛽௜ǡ௧ൌ݃ ൫ݖ௜ǡ௧Ǣܹ ൯. With and differ only in the loading map. IPCA restricts conditional betas to be linear in lagged character-

a single linear layer and no nonlinear activation, the CAE collapses to IPCA, so it is best understood as the same conditional-factor idea with a more expressive loading map, rather than a separate direct-prediction architecture.

That placement is what keeps the CAE inside the adapter. Its training objective reconstructs realized returns from conditional betas and latent factor returns; it does not directly optimize next-period expected returns. The fitted model is nevertheless a conditional asset-pricing representation: covariates guide the dimension reduction, and nonlinear interactions among characteristics shape the factor exposures. Once estimated, the CAE supplies the same two objects as IPCA, namely current conditional betas and a history of latent-factor returns, so the Stage 2 forecaster catalog from *Section 14.5* carries over unchanged. The training-sample mean of factor returns gives the baseline forecast, while AR(1), EWMA, Q-Boost, and ZS-Chronos replace it without retraining the CAE. The CAE therefore changes only Stage 1: the beta network learns nonlinear characteristic-to-loading relationships, and the factor side extracts latent factor returns from characteristic-managed portfolios.

This also locates the CAE relative to the time-series models of *Chapter 13*. Those models capture temporal patterns in realized return histories; the CAE captures cross-sectional structure at each point in time. Temporal features derived from *Chapter 13* models can enter the CAE as additional characteristics, but the CAE itself is not primarily a sequence model.

![Figure 14.7](assets/figure_14_7.jpeg)

*Figure 14.7: The conditional autoencoder. A beta network maps each asset’s lagged characteristics to conditional factor loadings; a factor side extracts latent factor returns from characteristic-managed portfolios; their inner product reconstructs the contemporaneous cross-section*

### Preparing the characteristic panel

The `us_firm_characteristics` case study (*Section 14.1*) provides the main dataset. Custom panels should apply the same survivorship-bias-free, point-in-time requirements introduced in *Chapter 4*: characteristics must be observable before the return they explain or forecast, and universe membership must be defined without future information. The investable universe should be fixed before estimation, because microcap and illiquidity filters determine which firms contribute to the estimated factor structure and which premia the model can learn. Since characteristic-based predictability is often concentrated among smaller and less liquid firms, report results both with and without the smallest 20% of firms by market capitalization to separate tradable signal from microcap and liquidity artifacts.

The dataset used here contains 63 precomputed features: the 57 fundamental characteristics from *Section 14.5* plus six temporal features. The original CAE study uses a larger set, but the implementation logic is unchanged. Neural networks are sensitive to input scale, and firm characteristics often [−1,1]. This cross-sectional transformation reduces the influence of outliers, makes characteristics carry extreme outliers, so we rank stocks by each characteristic each month and rescale the ranks to

comparable across units, and preserves the relative information available at each date.

### Specifying the dual-network architecture

The **beta network** maps each stock’s 𝑃 lagged characteristics into The CAE contains two linked components: a beta network and a factor side.

conditional factor loadings. In the

![Figure 14.8](assets/figure_14_8.png)

typically pyramidal (for example, 64 → 32 → 16 → 8). For each date 𝑡, it produces an 𝑁௧ൈܭ matrix of PyTorch implementation, it is a feed-forward network with ReLU activations and batch normalization,

conditional exposures: 𝐵௧ൌ݃ (ܼ ௧ିଵǢܹ ) where 𝑍௧ିଵ∈ℝே೟ൈ௉ holds lagged characteristics for the 𝑁௧ firms observed at date 𝑡.

The **factor side** estimates the latent factor returns used to reconstruct the cross-section. Let 𝑟௧∈ℝே೟

collect contemporaneous excess returns. The first step forms characteristic-managed portfolio returns by projecting realized returns on the lagged characteristic matrix: 𝑥௧= ( ௧−1)−1 ⊤ݎ௧ ⊤ ௧−1 ௧−1 where 𝑥௧∈ℝ௉ summarizes how the cross-section of returns loads on each characteristic at date 𝑡; a pseudo-inverse or regularized inverse replaces the inverse when 𝑍௧ିଵ ⊤𝑍௧ିଵ is ill-conditioned. The

resulting vector reads as returns on characteristic-managed basis portfolios. This is a cross-sectional object is the projection coefficient vector 𝑥௧. A simple factor network, often a single linear layer, maps projection rather than a univariate sort: the sorting interpretation is useful intuition, but the formal these managed-portfolio returns into 𝐾 latent factor returns:

𝑓௧= ℎ(ݔ௧Ǣ ߆)

The beta network and factor side are estimated jointly so that the model reconstructs realized returns as: 𝑟௜ǡ௧ൌߚ௜൫ݖ௜ǡ௧ିଵǢܹ ൯ ⊤ ௧

This is the central difference between the CAE and a direct prediction network. The CAE does not map characteristics to next-period returns; it estimates a conditional factor representation and then uses it as input to a separate forecasting step.

### Estimating the CAE

For each training window, the CAE minimizes the cross-sectional reconstruction loss: ෍ቀݎ௜ǡ௧െߚ௜൫ݖ௜ǡ௧ିଵǢܹ ൯ ⊤ ௧) 2

௧ǡ௜

with regularization and early stopping. This objective estimates the latent factor structure and should not be confused with the final forecasting objective. The implementation uses L1 regularization to encourage sparse weights and early stopping to limit overfitting. Deep networks are sensitive to random initialization, so the notebook trains an ensemble across seeds and averages the predictions; ensemble averaging reduces the variance of the fitted loading map and yields more stable downstream forecasts than a single initialization. layers, widths from 32 to 128 in a pyramidal structure, learning rates from 10−4 to 10−2, L1 regulariza-From the `us_firm_characteristics` case study, effective beta-network ranges are two to three hidden tion from 10−5 to 10−3, and batch sizes of 1,000–5,000 observations, typically close to a full monthly

cross-section. Dropout above 0.3 tends to underfit; below 0.05, overfitting becomes more likely. The here too: the number of latent factors 𝐾 should be large enough to capture persistent cross-sectional factor network is simpler, usually a single linear layer. The identification caveats from *Section 14.5* apply structure but small enough to avoid redundant or unstable factors. Start with 𝐾–10 for equity

that 𝐾 is too large or that regularization is too weak. models; after applying the model’s normalization conventions, highly correlated learned factors signal

### Turning latent factors into forecasts

After Stage 1 estimation, the CAE provides current conditional betas and a historical series of latent factor returns. Forward prediction requires the Stage 2 factor-premium forecast. The simplest rule uses the training-sample mean of each latent factor return, recovering the baseline GKX-style setup; 𝜆̂௧ାଵ, the asset-level expected return is: AR(1), EWMA, Q-Boost, or ZS-Chronos replace only this factor-premium forecaster. Given a forecast 𝑟௜ǡ௧ାଵൌߚ௜൫ݖ௜ǡ௧Ǣܹ ൯ ⊤ߣ̂௧ାଵ

The decomposition is useful because the CAE estimates the cross-sectional structure, while the forecasting rule specifies how much compensation each latent factor is expected to earn in the next period. A CAE can reconstruct realized returns well and still forecast weakly if the extracted factors carry little persistent premium, which is why model selection rests on downstream forecast quality rather than reconstruction loss alone.

### Tuning, validation, and failure modes

Reconstruction loss is an optimization diagnostic: it shows whether the model can fit the training cross-section, not that the learned factors earn stable out-of-sample premia. The validation protocol therefore evaluates the complete forecasting workflow. At each walk-forward origin, fit the characteristic transformation, CAE parameters, regularization choices, and factor-premium forecaster on the training window only; then freeze those components, generate forecasts for the next validation period, and evaluate the asset-level predictions. Compare configurations on validation IC, IC information ratio, quintile spreads, turnover, and portfolio-level performance, reserving reconstruction loss for diagnosing optimization failure or underfitting. Optuna’s Bayesian optimization (*Section 12.4*) can explore the architecture and regularization space, using downstream validation IC or IC information ratio as the objective, with at least 50 trials to ensure a stable search.

The common failure modes are visible in the estimated factors, betas, and validation behavior. Diverging reconstruction loss usually indicates either too high a learning rate or insufficient gradient an oversized 𝐾 model or weak regularization; near-identical betas across assets indicate beta-network control; strong seed sensitivity argues for a larger ensemble; highly correlated latent factors point to

collapse, diagnosed by cross-sectional beta dispersion. Statistical performance must also withstand economic checks: a high-IC signal with high turnover may be untradeable after costs are deducted. Quintile spreads, turnover, and factor decay across horizons indicate whether predictability is ecois net return ≈ gross return −2 × turnover × one-way cost, with cost assumptions matched to the nomically meaningful, and gross and net performance should both be stated. A simple approximation

universe: 5–10 basis points one-way for large-cap equities, 50 or more for microcaps.

### Empirical evidence and replication caveats

In the original GKX study of US equities, long-short portfolios built on CAE predictions outperformed linear IPCA benchmarks, with the gains driven by nonlinear interactions among firm characteristics. The economics are intuitive: the effect of momentum, value, size, volatility, or liquidity need not be additive or constant across the cross-section, and a neural loading map can represent interactions such as momentum behaving differently between high- and low-volatility stocks without the researcher pre-specifying those terms. The gains are not mechanical, and our `us_firm_characteristics` implementation illustrates the diffithe supervised autoencoder treated in *Section 14.7*, a result not directly comparable to the predictive 𝑅2 culty of replication. On the primary one-month label, the CAE flips negative at −0.030 in IC, well below

metrics reported by GKX. It still points to the same conclusion as the literature: latent-factor signals are modest in absolute terms and sensitive to sample period, universe definition, characteristic coverage, and validation design. The case study uses a shorter and more recent sample, a smaller universe, and fewer characteristics than the original, which likely accounts for much of the gap.

Neural factor loadings lack the eigenvector interpretation available for PCA. SHAP (*Section 11.4*) provides a partial diagnostic by attributing each asset’s learned loadings to its input characteristics: a high loading on a learned factor may be associated with momentum and low volatility, while size contributes negatively; aggregating SHAP values across assets yields factor-level importance rankings. Two caveats are essential. SHAP explains how the fitted beta network uses characteristics; it does not prove those characteristics causally drive returns, and with correlated inputs the attribution can be unstable across samples. It is an interpretability diagnostic, not an independent validation of the economic mechanism.

Recent extensions modify Stage 1 while preserving the broader adapter. Self-attention lets the beta network condition each asset’s loading on the broader cross-section, capturing sector, peer, and supply-chain relationships that per-asset feed-forward networks miss (Kelly et al., 2025), and text embeddings from earnings calls, filings, and news (*Chapter 10*) can enter as additional characteristics. In each case, the Stage 2 forecaster and Stage 3 asset map stay conceptually unchanged. Engel et al. (2025) push the idea further, using a CAE to extract latent factor portfolios and then training separate time-series models on each factor, with predictive uncertainty used to select the most forecastable factors for tangency-portfolio construction.

**Implementation**: `06_conditional_autoencoder` builds the full workflow in PyTorch: characteristic preprocessing, dual-network estimation, ensemble training, three Stage 2 forecasters (Constant, AR(1), EWMA), the Stage 3 asset map, SHAP interpretability, and beta-dispersion diagnostics.

## 14.7 The stochastic discount factor and the supervised autoencoder models

The adapter of *Section 14.5* separates factor estimation from factor-premium forecasting, and PCA, RP-PCA, IPCA, and the CAE all fit it: each estimates a factor representation that a Stage 2 model then forecasts. Two deep models break that pattern for opposite reasons. The **stochastic discount factor** (SDF; Chen, Pelger, and Zhu, 2021) collapses the stages by learning the pricing object directly from no-arbitrage conditions, leaving no factor-return history to hand to a separate forecaster. The **supervised autoencoder** (SAE) skips the structure altogether, mapping characteristics to forward returns end-to-end with no factor intermediate. They are grouped here by that shared trait: each bypasses the three-stage adapter, one because it prices directly and one because it predicts directly. This architecture is shown in *Figure 14.8*:

![Figure 14.9](assets/figure_14_9.png)

*Figure 14.8: Where the SDF and the supervised autoencoder sit relative to the three-stage adapter*

The SDF learns the pricing kernel directly from no-arbitrage moment conditions and emits pricing-error diagnostics rather than a factor-return history; the supervised autoencoder maps characteristics to forward returns end-to-end. Neither exposes a factor premium for a separate Stage 2 forecaster.

### The stochastic discount factor

The Chen-Pelger-Zhu (CPZ, 2021) framework operationalizes the third objective introduced in *Section* naturally written one step ahead. For gross returns 𝑟௜ǡ௧ାଵ the stochastic discount factor 𝑀௧ାଵ satisfies *14.1*: minimizing pricing errors directly through conditional moment restrictions. No-arbitrage is 𝐸௧ൣܯ௧ାଵݎ௜ǡ௧ାଵ] = 1, or equivalently 𝐸௧ൣܯ௧ାଵݎ௜ǡ௧ାଵ ] = 0 for excess returns. CPZ parameterize 𝑀௧ାଵ with ௘

violations of these conditional moment restrictions. The output is a learned pricing kernel 𝑀̂௧ and the a flexible network of firm characteristics and macro-state variables, and estimate it by minimizing

associated pricing-error diagnostics, not a factor-return history designed for a downstream premium forecaster, which is why the model has no natural Stage 2 hook.

### Building the SDF with adversarial deep learning

The conditional moment set is effectively infinite, so evaluating every restriction is infeasible; the adversarial mechanism is how CPZ make the problem tractable. A second network learns to construct the worst-case test portfolio, the linear combination of assets that the current SDF estimate would most misprice. Training then alternates between two objectives. The SDF network updates to minimize pricing errors on the adversary’s portfolio, and the adversary updates to find new portfolios the improved SDF still misprices. This minimax structure disciplines the SDF against the hardest cases rather than the average ones, the crucial difference from the CAE’s average reconstruction loss. The `07_stochastic_discount_factor` notebook implements this as a three-phase training loop with configurable adversarial rounds. The state variables enter through a recurrent macro encoder. A MacroLSTM learns a low-dimensional state from FRED indicators (yield spreads, inflation, volatility) so the kernel adapts to the regime without the researcher pre-specifying which macro variables matter, the time-series analog of letting the beta network discover characteristic interactions in the CAE.

Evaluation differs from the CAE because the object being estimated is different. Report the SDF-implied maximum Sharpe ratio, which the Hansen-Jagannathan bound ties to the magnitude of pricing errors; the pricing errors against benchmark factor models; and the worst-case adversarial portfolio at each step. One caveat warrants close attention: cross-sectional GMM can yield spuriously high fit through weighting-matrix choices when the model is misspecified (Gospodinov, Kan, and Robotti, 2014), so rerun under alternative weightings and across both fixed and adaptive test-asset sets. The teaching and case-study notebooks run these diagnostics by default. For comparability with the other model families, the case study selects the reported SDF checkpoint on the validation IC rather than the validation-Sharpe criterion used in the original CPZ protocol.

### The supervised autoencoder

The supervised autoencoder is the other model outside the adapter, but it sits there as a practical benchmark rather than a structural asset-pricing object. It is the useful direct-prediction alternative when the goal is a strong tabular signal rather than a factor decomposition. An encoder compresses characteristics to a bottleneck, a decoder reconstructs the inputs as a regularizer, and a supervised prediction head maps the bottleneck to forward returns. The architecture is best known as the winning entry in the Jane Street-sponsored Kaggle market-prediction competition (2020–2021), which motivates its inclusion as a strong baseline; the chapter README links the original write-up.

There is no Stage 1/Stage 2/Stage 3 separation. The supervised loss shapes the bottleneck, and the prediction head is the asset mapper, so evaluation follows direct-prediction conventions, cross-sectional IC and AUC for directional labels, rather than the pricing-error or factor-premium diagnostics that suit the SDF. The contrast with the CAE is the point worth carrying forward: both compress characteristics through a network, but the CAE’s bottleneck is a conditional factor structure disciplined by reconstruction, while the SAE’s bottleneck is whatever representation best predicts the label. The first is interpretable as risk; the second is not, and is not meant to be.

**Implementation**:

- `07_stochastic_discount_factor` builds the adversarial SDF with MacroLSTM and three-phase minimax training on a top-200 US equities universe with eight characteristics
- `08_supervised_autoencoder` implements the supervised variant with purged cross-validation, performing direction classification (AUC scoring) with characteristic-reconstruction regularization; `ml4t.models.SAEModel` is the production form

## 14.8 Case study insights

*Chapters 11* through *13* trained models directly on the forward-return label, from regularized linear regressions to gradient boosting and deep sequence networks. The latent-factor estimators reach the same cross-sectional ranking from the other side: they recover structure from the joint behavior of returns and characteristics with little or no return supervision, then score assets on it. The question here is whether that structure ranks assets as well as the supervised families already do.

Five of the nine case studies have cross-sections sufficiently balanced to fit the full latent menu: ETFs, the S&P 500 equity-and-options study, US Equities, US Firm Characteristics, and CME Futures. The five estimators differ in what they optimize: PCA takes the leading variance directions, IPCA fits conditional linear betas from characteristics, the conditional autoencoder (CAE) minimizes reconstruction error, the stochastic discount factor (SDF) targets a no-arbitrage pricing error, and the supervised autoencoder (SAE) adds a return-prediction task to the reconstruction.

Coverage is uneven: ETFs and the equity-and-options study run all five estimators; US Firm Characteristics drops PCA, because anonymized firm identifiers are not stable from month to month and break the balanced panel PCA requires; CME Futures runs PCA and the SDF, and US Equities runs PCA and IPCA. Each estimator is summarized by its average daily information coefficient (IC) and a 95% confidence interval computed with a HAC correction for the autocorrelation in the daily series; an interval that excludes zero is the bar for significant signal.

### Where latent factors add ranking content

The strongest latent estimator clears zero on four of the five case studies (ETFs, US Firm Characteristics, CME Futures, and US Equities) and overlaps zero on the equity-and-options study. *Table 14.3* places each one beside the strongest model from *Chapters 11* through *13*. Neural objectives carry the latent result almost everywhere: the SDF leads on ETFs and CME Futures and the SAE on US Firm Characteristics, while IPCA, the conditional linear estimator, carries US Equities at a small but credibly positive value. PCA, the unconditioned baseline, reaches the highest latent IC on none of the five.

| Case study | Horizon | Best latent model | Latent IC (t) | Strongest prior model | Prior IC | Δ IC |
| --- | --- | --- | --- | --- | --- | --- |
| ETFs | 21 days | SDF | +0.085 (4.9) | NLinear | +0.062 | +0.023 |
| US Firm Characteristics | 1 month | SAE | +0.062 (5.5) | GBM | +0.080 | −0.018 |
| CME Futures | 5 days | SDF | +0.037 (2.8) | GBM | +0.032 | +0.005 |
| S&P 500 Eq+Opt | 5 days | SDF | +0.012 (0.7) | TabM | +0.011 | +0.001 |
| US Equities | 1 day | IPCA | +0.005 (2.2) | GBM | +0.032 | −0.027 |

*Table 14.3: Highest-IC latent estimator at each case study’s primary forward-return label, with its HAC t-statistic, beside the strongest supervised model from Chapters 11 through 13 (a regularized linear model, gradient boosting, TabM, or a deep sequence network). Δ IC is the latent point estimate minus that of the strongest prior model*

Clearing zero is not the same as improving on the best model already in hand, and the Δ column shows the two apart. To compare each latent estimator directly against the strongest prior model, we compute the daily IC series difference on the dates they share and place a HAC interval around the mean difference, the paired counterpart of the single-model intervals in *Table 14.3*.

On this measure, the latent estimator credibly leads the strongest prior model on ETFs, where the SDF sits 0.023 above NLinear, and credibly trails on two case studies: US Equities, where IPCA sits 0.027 below gradient boosting, and US Firm Characteristics, where the SAE sits 0.019 below it. On CME Futures and the equity-and-options study, the latent point leads in expectation, but the interval spans zero. Latent factors therefore match or beat the strongest supervised family on three of the five case studies and credibly trail on two, a substantial showing for estimators that lean on structure rather than direct return supervision.

The equity-and-options study is the one case where the primary label hides the signal rather than lacking it. Every estimator overlaps zero at the 5-day return, but IPCA on the risk-adjusted 5-day label reaches +0.041 with an interval clear of zero, the only credibly nonzero latent point on that study.

### Which objective extracts the signal

The estimator that does best is the one whose objective matches what the cross-section rewards, and the case studies do not agree on which that is. *Figure 14.9* sets the five estimators against the five case studies; conditioning and supervision help where unconditioned variance does not. PCA recovers the directions of largest return variance, which need not price the cross-section, and it ranks no case study best. The SDF and SAE, which add a pricing constraint or a prediction task on top of the factor structure, reach the highest latent IC on four of the five; IPCA’s characteristic-conditioned betas carry the fifth.

![Figure 14.10](assets/figure_14_10.png)

*Figure 14.9: IC by latent estimator across the five case studies at the primary label. Shade encodes IC magnitude*

US Firm Characteristics shows the spread most cleanly because it runs four objectives on a single monthly cross-section. At the one-month label, the SAE reaches +0.062 while the CAE lands at −0.030, with the SDF and IPCA bunched near zero between them (*Figure 14.10*). Both extremes clear their intervals and carry opposite signs: a multi-task objective that uses the return label yields a positive ranking, while a pure reconstruction objective, which fits contemporaneous factors with no forward target, points in the wrong direction. This is the prediction-protocol caveat from *Section 14.5*: contemporaneous-factor reconstruction is a poor proxy for one-step-ahead forecasting on monthly characteristics.

![Figure 14.11](assets/figure_14_11.png)

*Figure 14.10: Objective ladder on US Firm Characteristics: SAE highest, CAE negative, SDF and IPCA near zero between them*

Because the objectives capture different structure, their predictions disagree. Across the three neural estimators for US Firm Characteristics, the pairwise rank correlations have different signs, so the models are not near-duplicates of a single latent signal. That disagreement is the empirical opening for the multi-objective averaging *Chapter 20* builds on.

### Dimensionality and stability

The latent estimators face markedly different estimation problems across the five case studies, and panel dimensionality accounts for much of the spread. CME Futures and ETFs sit in a favorable regime, with far more time periods than assets, so the signal eigenvalues cleanly separate from the noise floor predicted by random-matrix theory. US Equities and US Firm Characteristics are severely high-dimensional (US Firm Characteristics carries roughly nine times as many assets as time periods), where the sample return covariance is too noisy to invert directly. The estimators that still resolve a credible IC there, IPCA on US Equities and the SAE on US Firm Characteristics, both sidestep that covariance by mapping characteristics to factor loadings rather than estimating loadings from returns.

Stability over the walk-forward folds tracks the resolution of the label. *Figure 14.11* plots the per-fold ICs against their means; even the case studies whose intervals exclude zero with margin show wide fold-tofold variation, which is why the HAC interval, not the point estimate, is the threshold throughout. The monthly US Firm Characteristics result, built on a structural cross-sectional edge, is positive on every fold; the daily US Equities result, built on a thin per-day edge, is positive on about three folds in five. The first is a small effect measured cleanly; the second a smaller effect measured against more noise.

![Figure 14.12](assets/figure_14_12.png)

*Figure 14.11: Per-fold IC distribution against fold means across the five case studies*

### Regression versus classification

Only US Firm Characteristics carries a classification label through the latent stack, the one-month direction label. Under the symmetric metric introduced in *Section 11.6* (IC for a regression score against the continuous return, AUC for a classifier against the binary direction), IPCA’s continuous score reaches an IC of +0.010 against the realized return, a low-margin result whose interval spans zero. It neither confirms nor contradicts the ordering established by the regression labels, which remains the more informative reading for this monthly cross-section.

### Choosing a method

These estimators are complements rather than substitutes. PCA focuses on risk decomposition, with its eigenportfolios supporting position sizing in *Chapter 17* and risk management in *Chapter 19*; IPCA, the CAE, the SDF, and the SAE add conditioning, non-linearity, a no-arbitrage discipline, and a supervised task. Estimating several under one walk-forward protocol, then reading agreement as a likely signal and divergence as model risk, is what carries forward.

The case studies where a latent estimator clears zero (ETFs, US Firm Characteristics, and CME Futures most clearly) take that ranking into *Chapters 16* through *19*, which weigh it against turnover and cost, while the disagreement measured here is what *Chapter 20* turns into a multi-objective ensemble.

## 14.9 Summary

The four latent-factor methods of this chapter share a common predictive workflow, the three-stage forecasting adapter. Stage 1 estimates a factor representation, Stage 2 forecasts the factor premium, and Stage 3 combines today’s exposures with that forecast. RP-PCA, IPCA, and the CAE pass through the adapter as a matter of routine, and PCA can be wrapped in it for factor-timing. The four differ in Stage 1 alone: PCA maximizes covariance explained, RP-PCA adds a pricing-error penalty, IPCA estimates linear characteristic-conditioned betas, and the CAE generalizes that map nonlinearly.

The adversarial SDF and the supervised autoencoder sit outside the adapter. The SDF estimates a pricing kernel directly under no-arbitrage moment restrictions; the SAE predicts returns end-to-end and earns its place as a neural-network benchmark rather than a structural factor model. The `ml4t-` `models` library packages the adapter as a composable pipeline, while the chapter notebooks derive every stage from scratch.

Across the five case studies, the objective that matches what the cross-section rewards reaches the highest information coefficient, and the case studies disagree on which objective that is. The SDF reaches the highest latent IC on three of them, the supervised autoencoder on US Firm Characteristics, and IPCA on US Equities; unconditioned PCA leads on none. The clearest case is ETFs, where the SDF clears the HAC threshold with an IC of +0.085. The strongest latent estimator meets that threshold in four of the five case studies, overlapping zero only in the equity-and-options study.

Clearing zero is not the same as improving on the supervised families of *Chapters 11* through *13*. Compared with the strongest prior model on shared dates, the latent estimator matches or beats it in three case studies (credibly leading on ETFs and statistically indistinguishable on CME Futures and the equity-and-options study) and credibly trails in two, US Equities and US Firm Characteristics, where gradient boosting ranks clearly higher. That is a substantial showing for estimators that lean on structure rather than direct return supervision, though not a wholesale improvement over models that learn the label directly.

The practical lesson is that latent-factor models compete where the panel carries more periods than assets and genuine cross-sectional structure, fall behind where richly featured supervised models dominate, and stay useful as risk-decomposition tools regardless of predictive ranking (*Chapters 17* and *19*). *Chapter 15* moves from prediction to causal estimation; *Chapter 16* turns the predictions from *Chapters 11* through *14* into trading signals.
