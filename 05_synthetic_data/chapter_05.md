# Chapter 5: Synthetic Financial Data

Every quantitative strategy depends on historical data, but history provides only one realized path through a large space of market outcomes. A backtest on a 2010–2025 sample can indicate a stable relationship, but it can also look strong simply because the realized path was favorable to the strategy’s assumptions and parameter choices. One response is to augment the single realized history with additional, statistically consistent market paths sampled from a fitted generative model.

Those sampled paths are **synthetic data**, drawn from a model fitted to the empirical joint distribution of observed data, designed to preserve marginal behavior and dependence structure across time, assets, and regimes. Related terms include **simulated data** (typically parametric Monte Carlo under an assumed process) and **generated scenarios** (samples conditioned on a specified regime or state). All three supplement a single realized history with additional samples to assess robustness.

The objective is *robustness assessment*, not prediction. If generated paths reproduce key characteristics of financial time series, then strategy performance can be evaluated as a distribution under the learned model rather than as a single point estimate. Synthetic data can also support *model development* by augmenting real training samples. In these settings, the performance must still be assessed on real out-of-sample data, and generators must be treated as potential sources of bias.

Throughout this chapter, synthetic data is treated as a *decision-support tool* rather than as a substitute for historical evidence. Its value depends on whether it improves a downstream task. In finance, this standard is demanding: general distributional similarity is not enough if the generated data fails to preserve tail behavior, dependence, volatility clustering, or regime dynamics. By the end of this chapter, you will be able to:

- Diagnose when a historical backtest is underpowered and when synthetic generation is a reasonable supplement.
- Select a generative architecture based on data type and sampling frequency.
- Apply stylized-fact diagnostics to generated returns.
- Evaluate synthetic data using **Fidelity–Utility–Privacy criteria** and Train-on-Synthetic-Teston-Real validation.
- Identify generator-specific risks, including bias amplification, generator overfitting, and missing novel scenarios.

The chapter first motivates the use of synthetic financial data, describes the stylized facts it needs to reproduce, and explains a practical evaluation framework. We then turn to classical simulation methods, before proceeding to learned generators, including GANs, diffusion models, and LLM-based generators for tabular financial data. The chapter closes by applying the evaluation framework across these methods.

## 5.1 The quant’s dilemma

A quantitative strategy is developed and validated on a single realized history. This is true for discretionary rule-based systems and for machine-learning pipelines that fit predictive models and then backtest the implied trading rules. The historical record contains only a limited number of crises, regime shifts, and correlation breakdowns, so inference can be fragile because evidence is **path-limited**: apparent performance may be dominated by episodes that need not repeat.

### The multiple testing challenge

This path limitation manifests as inflated performance when strategy research adapts as results emerge: strategy development is an iterative search over many coupled choices, including the universe and filters; signal and feature definitions; label horizons and sampling schemes; model class and regularization; cross-validation design; hyperparameters; risk controls; execution assumptions; and portfolio construction details. Each additional decision expands the search space and increases the chance of selecting an in-sample winner that reflects noise, data quirks, or favorable market episodes rather than a persistent edge.

Adaptive research creates a selection problem. As the number of tested variants grows, selection inflation becomes material: the best observed in-sample performance improves mechanically with the number of trials, even when true performance is zero. Bailey et al. (2015) formalize this risk and argue that the probability of backtest overfitting can exceed 50%: under independence and normal-error approximations, after testing 10 configurations, the expected maximum in-sample Sharpe, a measure of risk-adjusted performance, is 1.57, even when all configurations have a true Sharpe of 0; with 100 trials, the expected maximum exceeds 2.5.

Interpreted broadly, this is a statement about the end-to-end research process: the reported result is the output of a selection procedure operating on limited data. The implication is not that backtests, or predictive models, are useless, but that **performance estimates should be treated as conditional on the research path taken**. Bailey and Lopez de Prado (2014) propose the **Deflated Sharpe Ratio**, which adjusts performance inference for non-normality and the number of trials (see *Chapter 17*). We will encounter additional multiple-testing corrections throughout the book. However, selection-aware statistical corrections can only adjust inference to account for this effect; they do not create additional market histories to assess robustness.

### Synthetic data as simulation infrastructure

Synthetic generation addresses path-limited evidence by turning a single realized history into a distribution of plausible histories. Cetingoz and Lehalle (2025) describe this as sampling additional trajectories from an estimated distribution to support robustness checks beyond a single realized path. In practice, the workflow fits a generative model to observed data, aiming to capture key **stylized facts of financial time series**, such as:

- **Heavy tails:** Extreme returns occur more frequently than under a Gaussian model.
- **Volatility clustering:** Large absolute returns cluster in time; the autocorrelation of squared or absolute returns decays slowly.
- **Leverage effect:** Negative returns are associated with higher subsequent volatility, especially in equity markets.
- **Weak return autocorrelation:** Linear autocorrelation in returns is typically small at daily horizons.

At a minimum, the resulting model should generate data that capture marginal distributions, temporal dependence, and cross-asset relationships, enabling it to produce alternative trajectories consistent with the learned structure.

However, a synthetic return generator is useful only if it reproduces empirical regularities that matter for downstream decisions (Takahashi and Mizuno, 2024). This limitation is especially acute in finance because the features that drive decisions are concentrated in the extremes. Risk management, leverage, and portfolio construction depend on tail outcomes, correlation breakdowns, and regime transitions that are rare and unstable. A method that produces samples that “look realistic” in the bulk of the distribution can still be useless if it understates drawdowns, misses volatility clustering, or fails to reproduce dependence in stress.

In this chapter, we therefore treat the generator as part of the modeling stack and evaluate it primarily by whether it supports downstream decisions, not by whether its samples are visually or marginally similar to historical returns.

Used carefully, synthetic data supports three practical capabilities:

- **Robust parameter selection:** Evaluate candidate parameters across many sampled trajectories rather than relying on a single realized history.
- **Regime stress testing:** Generate additional stress-like trajectories consistent with observed volatility and dependence to probe drawdowns, correlation breakdowns, and risk-control stability.
- **Privacy-preserving development:** When transaction-level data cannot be shared, synthetic data can support development and collaboration, provided it is subject to explicit privacy evaluation (*Section 5.8*).

Synthetic data does not remove selection bias. If strategies are tuned on synthetic trajectories, they can overfit to the generator’s inductive biases and failure modes. The generator must therefore be treated as part of the modeling stack: fix it before strategy selection when possible, validate it on held-out real data, and use an outer validation loop when synthetic data affects model selection. Before introducing specific generators, we first define the validation framework used throughout the chapter. The key question is not whether generated samples look realistic in isolation, but whether they preserve the structure required for the intended use case while controlling privacy and leakage risk.

## 5.2 Evaluating synthetic financial data

Synthetic data can appear plausible yet still be unsuitable for its intended use. In finance, this risk is acute because the economically relevant structure often lies in rare events, shifts in dependence, and conditional dynamics rather than at the center of the distribution. A generator that matches marginal returns but understates drawdowns, weakens volatility clustering, or misses stress-period correlations can produce misleading conclusions.

For this reason, the validation protocol often matters more than the generative architecture. Throughout this chapter, we evaluate synthetic data using three criteria: **fidelity**, **utility**, and **privacy**. These objectives typically trade off against one another. Improving privacy can reduce fidelity; improving distributional fidelity does not necessarily improve downstream task performance; and a generator that performs well for one use case may fail for another.

### Does it preserve the relevant data structure?

**Fidelity** asks whether synthetic samples reproduce the empirical structure of the real data at the resolution required by the application. This includes marginal behavior, dependence, and time-series dynamics.

For tabular or cross-sectional data, start with feature-level distributions. Compare real and synthetic samples using statistics such as the Kolmogorov–Smirnov statistic or Wasserstein distance, and complement these with histograms, empirical CDFs, and QQ plots. Summary statistics alone can hide artifacts, especially in the tails.

Dependence must be tested separately. A generator can match each feature’s marginal distribution while failing to preserve cross-feature relationships. For multivariate financial data, compare correlation or covariance matrices, rank correlations, sector- or regime-level relationships, and other dependence measures appropriate to the use case.

For financial time series, fidelity also requires stylized-fact diagnostics. At a minimum, test whether the generated series preserves heavy tails, weak raw-return autocorrelation, volatility clustering, leverage effects where relevant, and cross-asset dependence. The autocorrelation function of squared or absolute returns is a useful compact diagnostic because it summarizes volatility persistence, but it should be complemented with tail and dependence checks.

### Does it support the intended use case?

**Utility** evaluates whether synthetic data improves or preserves performance on the downstream task. Fidelity is usually necessary but not sufficient: synthetic data can match broad distributional properties while missing the specific signal needed for forecasting, classification, risk estimation, or strategy evaluation. A common benchmark is **train-on-synthetic, test-on-real** (**TSTR**):

1. Train a model using synthetic data only.
2. Evaluate it on held-out real data.
3. Compare the result with **train-on-real, test-on-real** (**TRTR**) using the same model class, features, target, and evaluation period.

For error metrics such as MSE or MAE, a useful summary is: ݁ݎݎ݋ݎݎܽݐ݅݋ൌܧݎݎ݋ݎ்ௌ்ோ ܧݎݎ݋ݎ்ோ்ோ 

Values near 1 indicate that synthetic data preserves task-relevant information. Values above 1 indicate utility loss. Values below 1 can occur when synthetic data suppresses idiosyncratic noise. Still, they require careful interpretation, as they may also signal leakage, target simplification, or an evaluation design that is too narrow.

For score metrics where higher is better, such as AUC or accuracy, report the TSTR and TRTR scores directly, or use a score ratio with the direction clearly stated. Do not mix error ratios and score ratios without explaining the convention.

**Utility is always use-case-specific**. Tail-GAN should be evaluated on tail-risk metrics such as VaR and Expected Shortfall; Sig-CWGAN should be evaluated on path-wise criteria; diffusion models require temporal and dependence diagnostics; and tabular LLM generators require schema validity, distributional checks, and downstream predictive tests.

### Does it leak training information?

**Privacy** validation tests whether synthetic samples reveal information about the records used to train the generator. This is especially important for transaction-level, customer-level, and proprietary institutional data.

At a minimum, check for exact or near duplicates between synthetic and training records. More informative diagnostics compare nearest-neighbor distances between synthetic records and well-trained and held-out real records. Membership-inference tests provide a stronger adversarial check by asking whether an attacker can infer whether a specific record was included in the training set.

When formal privacy guarantees are required, train the generator with a **differentially private mechanism**. Methods such as **Differentially Private Stochastic Gradient Descent** (DP-SGD, Abadi et al., 2016) by a budget (ߝǡ ߜ). Smaller values of 𝜀 imply stronger privacy but usually reduce fidelity and utility. clip per-sample gradients and add calibrated noise during training, with privacy loss summarized

Privacy should therefore be treated as a design constraint, not as an after-the-fact label attached to generated data.

### Synthetic-specific failure modes

The fidelity–utility–privacy framework is necessary but incomplete. Synthetic data introduces failure modes that are specific to generated samples.

1. Generators can **amplify biases in the training data**. If the historical sample overrepresents one regime, one sector, or one market state, the generator may reproduce that imbalance more strongly.
2. Strategies can **overfit to the generator**. Selecting a strategy because it performs well across many synthetic paths can produce a solution tailored to the generator’s assumptions rather than to the market. Finalists should always be evaluated on held-out real data that was not used to train the generator.
3. Synthetic data has **limited novelty**. Most generators are better at interpolating within the support of the training distribution than at producing genuinely unprecedented scenarios.

Synthetic data is therefore useful for robustness checks, stress testing, and model development. Still, it should not be treated as evidence about events outside the training regime unless the generator was explicitly designed and validated for that purpose.

### Practical validation checklist

At a minimum, every synthetic-data experiment in finance should report:

- A **fidelity diagnostic**, such as marginal distribution error, correlation-matrix distance, tail statistics, or volatility-persistence error
- A **utility benchmark**, such as TSTR versus TRTR or a task-specific risk metric
- A **privacy** **check**, such as duplicate detection, nearest-neighbor analysis, membership inference, or a formal DP budget
- A **real-data holdout result**, especially when synthetic data is used for model selection or strategy development

Thresholds should be calibrated to the data frequency, sample size, asset class, and downstream objective. There is no universal synthetic-data score. The relevant question is whether the generated data preserves the structure on which the decision depends.

## 5.3 Classical simulation baselines

Classical simulation methods provide the first test of the evaluation framework. Before using learned generators, practitioners should compare them with transparent baselines that are easier to calibrate, diagnose, and govern.

These baseline models can be grouped into **bootstrap models**, which resample historical data, and **parametric models of stochastic processes** that generate new data based on model assumptions. Both families are limited, but their limitations are visible, which makes them valuable reference points for evaluating more flexible learned generators.

### Bootstrap methods

Bootstrap methods generate new paths by resampling observed returns:

- Their main advantage is fidelity to the empirical marginal distribution: heavy tails and other non-Gaussian features present in the data are preserved by construction.
- The key limitation is equally direct: resampling cannot create fundamentally new events outside the historical record. If no crisis-scale move or correlation breakdown appears in the sample, bootstrapping cannot invent it.

Key bootstrap techniques include:

- **IID bootstrap** (IID means *independently and identically distributed*) draws individual returns with replacement. It preserves the marginal distribution but destroys temporal dependence, including volatility clustering. As a result, it is typically unsuitable for risk or sizing studies that require time-varying volatility.
- **Block bootstrap** resamples contiguous blocks of fixed length, preserving dependence within blocks. This recovers the short-horizon autocorrelation structure but can introduce artificial boundaries between blocks and tends to weaken long-range persistence when blocks are too short. Block length is therefore a bias–variance tradeoff: longer blocks preserve dependence better but reduce sample diversity. A common starting point is one month (about 22 trading days), then adjust based on diagnostics.
- The **stationary bootstrap** (Politis and Romano, 1994) uses random block lengths (geometric), thereby reducing boundary artifacts while preserving dependence in expectation. In practice, it is often a better default than fixed blocks when you want volatility clustering without imposing a parametric volatility model.

A high-value diagnostic for bootstrap methods is the **autocorrelation function** (**ACF**) of squared (or absolute) returns: the IID bootstrap collapses this structure toward zero, whereas block and stationary bootstrap retain a decaying pattern that is at least directionally consistent with volatility clustering.

### Parametric price and volatility models

Parametric models simulate market data by specifying a stochastic process for prices or returns **differential equations** (**SDEs**), where 𝑑 denotes a small time step, 𝑑 denotes a Brownian (Wiener) and sampling paths from that process. In continuous time, these models are written as **stochastic** increment (random noise that scales with √݀ݐ), and 𝑑 denotes a Poisson jump-count increment.

Parametric models can generate paths “beyond history,” which is useful for stress testing and scenario expansion, but the results are only as good as the assumptions; misspecification tends to show up in the tails and during regime transitions.

![Figure 5.1](assets/figure_5_1.jpeg)

*Figure 5.1: Parametric price and volatility models*

*Figure 5.1* compares stylized simulations from five stochastic time-series models over two years, with all prices normalized to 100 at the start. The top panel shows different price trajectories produced by each model; the bottom panel characterizes the distinct volatility patterns. **Geometric Brownian Motion** (**GBM**) defines a foundational baseline. Price 𝑆 follows a trajectory de-

scribed as follows: 𝑑= ߤ𝑑, 𝑑ݐ+ ߪ𝑑, 𝑑ܹ

 where 𝜇 is drift and 𝜎 is volatility. Log-price increments are Gaussian, and volatility is constant. GBM

is analytically tractable (Black-Scholes) but fails core stylized facts: it produces no volatility clustering and tails that are too thin (no excess kurtosis).

**Jump-Diffusion** (**Merton model**) adds discontinuous moves via a compound Poisson process: 𝑑= ߤ𝑑, 𝑑ݐ+ ߪ𝑑, 𝑑ܹ + 𝑆( ௒−1), 𝑑ܰ

where 𝑑 is a Poisson increment with intensity 𝜆 and ׽ܰ ൫ߤ௃ǡ ߪ௃ 2) is the log jump size. In practice,

the drift is typically adjusted to account for the expected jump contribution; otherwise, the unconditional mean can be misstated. Jump diffusion generates fat tails and explicit crash-like moves, but in its basic form, jump arrivals are independent of the volatility state, unlike real markets, where jumps cluster during episodes of market stress.

**Mean-Reversion** (**Ornstein-Uhlenbeck**) models log-prices (or spreads) gravitating toward equilibrium: 𝑑(݈݋݃ܵ ) ൌߢ(ߠെ݈݋݃ܵ ), 𝑑ݐ൅ߪ, 𝑑ܹ

 where 𝜅 is the mean-reversion speed and 𝜃 is the long-run level. The half-life is 𝑙(2)Ȁߢ. This is ap-

propriate for spreads and some rates/commodity settings, but it is not a general return model for trend-capable assets. **Stochastic volatility** (**Heston**) introduces a separate variance process 𝑣: 𝑑= ߤ𝑑, 𝑑ݐ+ √ݒ, 𝑑, 𝑑 ௌ 𝑑ൌߢ(ߠെ𝑑), 𝑑ݐ൅ߦ√𝑣, 𝑑 ௩ ܥ݋ݎݎ(ܹ݀ ௌǡܹ݀ ௩) ൌߩ

Negative 𝜌 produces a leverage effect (price drops increase volatility), and the model can generate fat

tails and volatility clustering. Practical use depends on careful discretization (for example, full-truncation Euler or Andersen’s QE) to maintain non-negative variance. returns in discrete time, with parameters 𝑝 and 𝑞 denoting the numbers of lagged squared shocks **Generalized Autoregressive Conditional Heteroskedasticity**, GARCH(p,q), models the variance of

and lagged conditional variances, respectively (see also *Chapter 9*). The most common specification is GARCH(1,1), which uses one lag of each: 𝑟௧ൌߤ൅ߪ௧ߝ௧, ߝ௧׽ܰ (0,1)

2 ൌ߱ ൅ߙ(ݎ௧ିଵെߤ)2 ൅ߚ𝜎௧ିଵ 𝜎௧ 2 where 𝛼 captures the reaction to recent return shocks and 𝛽 captures volatility persistence; a common stationarity condition is ߙ൅ߚ൏ͳ. GARCH is typically calibrated by maximum likelihood, making

it a pragmatic bridge between theory and data. Basic GARCH often reproduces volatility clustering well in practice.

### Limitations and how to use these baselines

Classical baselines make strong tradeoffs explicit. Bootstrap methods preserve the empirical distribution but are constrained to the historical record: they cannot generate truly novel extremes or structural breaks. Parametric models can generate paths “beyond history,” but only along the dimensions implied by their dynamics; misspecification often shows up exactly where finance cares most (tails, stress dependence, regime transitions).

In this chapter, we use these methods as reference points. They provide sanity checks for learned generators (a deep model should at least outperform an appropriate baseline on the diagnostics that matter), and they remain useful when interpretability, small-sample robustness, or governance requirements dominate. The next section provides an overview of different generative models.

Implementation: `00_classical_simulation.py` implements all methods with calibration diagnostics.

## 5.4 Generative model taxonomy

This section covers **learned generators**: models that approximate the joint distribution of the variables of interest (for example, financial returns of multiple assets) by optimizing a training objective rather than by specifying a parametric process or resampling scheme. The goal is not to produce “realistic-looking” samples, but to preserve the temporal structure, dependencies, and tail behavior required for downstream robustness analysis or model development. A useful distinction is between **discriminative** models, which learn 𝑝(ݕפܺ ), and **generative** models, which learn 𝑝(ܺ) or 𝑝(ܺǡ ݕ) and can therefore produce new samples. In finance, learned generators

are attractive because they can represent complex dependence structures that are difficult to write down parametrically. Still, they can also fail silently by smoothing tails, collapsing modes, or learning spurious regime structure.

![Figure 5.2](assets/figure_5_2.png)

*Figure 5.2: Comparing various deep generative models*

The learned generators used in this chapter fall into four families:

- **Variational Autoencoders** (**VAE**) learn a latent-variable generative model that aims to both reconstruct the input and conform to certain distributional assumptions. VAEs often train more stably than adversarial models and can serve as strong baselines for mixed-type structured data. Still, they may oversmooth and require calibration or more expressive likelihoods to avoid underrepresenting rare categories and tail behavior.
- **Generative Adversarial Networks** (**GANs**) learn through adversarial training. They can produce sharp samples and capture complex dependence, but training can be unstable. *Section 5.5* focuses on GAN variants designed to reduce collapse and improve path-wise and tail fidelity.
- **Diffusion models** learn to generate by iteratively denoising. Training is typically stable, and the framework supports conditional generation. *Section 5.6* presents a diffusion model for regime-conditioned time-series generation.
- **LLM-based generators for tabular data** serialize records as text and generate rows autoregressively. This can work well for heterogeneous, mixed-type datasets, especially when conditional generation is important. *Section 5.7* covers this approach, along with faster specialized alternatives where appropriate.

The following sections focus on specific, finance-oriented instantiations of these families. These learned generators should be treated as part of the modeling stack: they can expand the space of plausible trajectories, but they can also introduce their own inductive biases. Each model family below should be evaluated against the framework introduced in *Section 5.2*. GAN variants are useful when their objectives align with the desired notion of fidelity, such as temporal structure, tail risk, or path-wise similarity. Diffusion models often provide more stable training and stronger conditional generation, at the cost of slower sampling. LLM-based tabular generators can handle heterogeneous structured data, but they require strict schema validation and privacy checks.

The practical question is therefore not which generator is universally best, but which generator preserves the structure required by the downstream decision, performs competitively against classical baselines, and satisfies the applicable privacy constraints.

## 5.5 GANs for financial time series

GANs generate synthetic samples via an adversarial game between a **generator** and a **discriminator** (*Figure 5.3*). The generator maps random input to candidate samples, while the discriminator scores both real and generated samples as real or fake.

Training alternates between strengthening the discriminator and updating the generator to fool it. This setup provides a useful learning signal even when the data likelihood is hard to specify.

![Figure 5.3](assets/figure_5_3.png)

*Figure 5.3: The GAN training workflow*

*Figure 5.3* illustrates the core loop without committing to any particular sequence model, loss function, or stabilization technique. In finance, however, vanilla GANs tend to prioritize the high-density “center” of the distribution and underrepresent rare but economically important tail events. The models that follow keep the adversarial loop but adapt it to what vanilla GANs miss: temporal dependence (**Time- GAN**), tail behavior and risk measures (**Tail-GAN**), path-wise similarity (**Sig-CWGAN**), and irregularly sampled continuous-time dynamics (**GT-GAN**).

### Using TimeGAN for sequential data

**TimeGAN** (Yoon, Jarrett, and van der Schaar, 2019) adapts adversarial training to time series by adding supervised temporal objectives and by learning in an embedding space rather than directly on raw sequences.

#### Architecture

TimeGAN comprises five networks trained in stages: **Embedding network** (݁): maps a raw sequence 𝑥 to a latent representation ℎ. **Recovery network** (ݎ): reconstructs 𝑥 from ℎ; together, (݁ǡ ݎ) form an autoencoder trained to 1. 2. **Supervisor network** (ݏ): predicts the next latent state ℎ௧ାଵ from ℎ௧, adding a supervised loss minimize reconstruction error. 3. **Generator** (݃): produces synthetic latent paths from noise. that encourages temporal coherence in latent space. **Discriminator** (݀): distinguishes real from synthetic latent trajectories. 4. 5.

#### Training

Training usually proceeds in three phases:

1. Pretrain the embedding and recovery networks as an autoencoder.
2. Train the supervisor on real embeddings.
3. Jointly optimize the full system under reconstruction, supervised, and adversarial losses.

#### Data requirements

Many reference implementations use sigmoid activations in the recovery network, constraining outputs to ([0,1]). This is convenient for normalized OHLCV inputs (see *Chapter 3*) but incompatible with unbounded return series unless returns are mapped to a bounded range or the output layer is modified. For returns, a linear output layer preserves the full real-valued support of the target.

#### When to use TimeGAN

TimeGAN is a strong baseline for multivariate sequence generation: it is well documented, has reference implementations, and provides a benchmark for comparing newer architectures. It does not explicitly target tail fidelity and assumes a regular sampling grid.

Implementation: The notebook `01_timegan.py` trains TimeGAN on six stocks (BA, CAT, DIS, GE, IBM, KO) using 24-step windows. Discriminative accuracy is 68%, with ROC AUC 0.87, indicating residual separability between real and synthetic sequences.

The **TSTR** ratio for one-step forecasting is 1.76: training on TimeGAN-generated multi-stock adjusted-close data produces 76% higher one-step forecast error than training on real data, suggesting that synthetic data is informative but imperfect for downstream prediction.

Under the validation framework from *Section 5.2*, these results show partial utility but incomplete fidelity: the generated data contains enough structure to support a weak downstream forecasting task, yet the discriminator and dependence diagnostics indicate that real and synthetic sequences remain materially separable.

![Figure 5.4](assets/figure_5_4.png)

*Figure 5.4: TimeGAN fidelity, showing how PCA and t-SNE overlap*

In our analysis, synthetic sequences underrepresent temporal and cross-asset dependence relative to real data, consistent with TimeGAN’s focus on overall distribution rather than fine-grained dynamics. On similar inputs, Diffusion-TS (*Section 5.6*) performs better in our experiments (TSTR ratio: 1.00 compared to 1.76).

### Using Tail-GAN for risk-constrained generation

Tail-GAN (Cont et al., 2025) variants address a limitation of standard GAN objectives: optimizing for average distributional fidelity can underweight rare but consequential tail events. For risk management, matching tail risk measures such as **Value-at-Risk** (**VaR**) and **Expected Shortfall** (**ES**) is often more important than matching the distribution center.

Tail-GAN augments the generator loss with penalties that discourage discrepancies between real and synthetic tail risk measures:

![Figure 5.5](assets/figure_5_5.png)

where 𝛼 is the tail probability (often 1% or 5%), and (ߣ1ǡ ߣ2) trade off tail matching against overall fit.

#### Portfolio focus

Tail constraints are typically computed on portfolio returns rather than on individual assets. This aligns the objective with risk management practice, where joint dependence drives drawdowns and limits are set at the portfolio level.

#### When to use Tail-GAN

Use Tail-GAN when the downstream task is risk measurement and tail accuracy dominates other fidelity notions. The method may sacrifice fit in the distribution center to improve VaR/ES matching. long-short portfolio strategies. At 𝛼, VaR relative error is 13% and ES error is 11%, Implementation: `02_tailgan_tail_risk.py` trains Tail-GAN on 5 ETFs using 32 random

configuration, synthetic VaR is slightly more negative than real (synthetic ≈−10.33 versus illustrating that explicit tail penalties produce usable risk estimates. In this 32-strategy real ≈−9.13), so the trained Tail-GAN modestly overstates loss magnitude rather than

understating it.

Tail-GAN improves selected tail statistics, but the constraints are usually defined on specific benchmark portfolios and risk levels. Matching VaR and ES for those portfolios does not guarantee accurate tail dependence for other portfolios, individual assets, or path-dependent risk measures.

More generally, VaR and ES capture only part of the tail: two models can match these quantities yet on the choice of tail probability 𝛼, the benchmark portfolios used during training, and the regimes differ in tail shape, clustering, and temporal dynamics. Performance, therefore, depends materially

represented in the data.

### Using Sig-CWGAN for path signatures for temporal fidelity

Sig-CWGAN (Ni et al., 2020) replaces the learned discriminator with an analytic criterion based on **path signatures** from rough path theory. The objective reduces reliance on an adaptive discriminator and provides a structured notion of distance between path distributions.

A path signature is a **sequence of iterated integrals** that summarizes the shape of a path: not just where it starts and ends, but how it moves through its coordinates, in what order, and with what directional interactions.

Intuitively, the first level records the path’s net increments, the second level records pairwise interaction effects between coordinates and the order in which moves occur, and higher levels capture progressively richer geometric structure.

A key mathematical property is that the signature depends on the path’s traced-out shape rather than the speed at which the path is traversed, so it is **invariant to time reparameterization**. In financial apcan themselves contain information. Therefore, implementation often augments 𝑋 with time as an plications, this is often useful but not always desirable, because elapsed time and observation spacing additional channel. For a path 𝑋[Ͳǡܶ] ՜ܴ ௗ, a level-𝑘 term has the form:

௜1 ڮ݀ܺ 𝑆(ܺ)௜1ǡǥǡ௜ೖ= ∫ ௜ೖ ௧1 ௧ೖ ଴ழ௧1ழڮழ௧ೖழ் and the depth-𝑛 signature collects all terms up to order (n). In practice, one can think of the truncated

signature as a finite feature map that converts a sequential path into a set of statistics describing its geometry at increasing levels of detail. Implementations often augment (X) with time as an additional channel and use truncated signatures at modest depth.

Sig-CWGAN minimizes a signature-kernel **maximum mean discrepancy** (**MMD**) between real and synthetic path distributions, yielding a stable training signal without adversarial classification. The number of signature terms grows as 𝑂( ௡) in the path dimension 𝑑 and truncation depth 𝑛. With two assets, time 𝑑, and depth 3, the truncated signature has 1 + 3 + 9 + 27 = 40 terms. By contrast,

50 assets at depth 3 would require over 125,000 terms, making straightforward scaling impractical without dimensionality reduction.

Use Sig-CWGAN when **path-wise fidelity** matters (for example, in path-dependent pricing or sequential decision problems) and the dimension is small enough to compute signatures reliably. The analytic objective can reduce failure modes driven by discriminator overfitting, but scalability is the binding constraint.

**Implementation**: `03_sigcwgan_signatures.py` reproduces the paper-exact setup: a single asset (S&P 500 index log returns, 2005–2020) at signature depth 4 over 16-day windows, trained for 2,500 generator steps.

- The Sig-W1 distance (RMSE of expected signatures) descends to 0.054 in training and 0.42 on the held-out window, and the train-on-synthetic, test-on-real ratio reaches 0.954 - close to parity with the real data.
- Stylized facts are only partially captured: synthetic skewness and excess kurtosis fall far short of the empirical values, and the squared-return autocorrelation collapses to 0.05 (versus 0.49 in the data), so volatility clustering is not preserved.

Sig-CWGAN is most attractive in low-dimensional settings. Truncating the signature discards higher-order path information, while increasing depth quickly becomes computationally expensive and statistically harder to estimate. In practice, **the method is best viewed as a structured low-dimensional generator** rather than a general-purpose solution for broad multivariate market panels.

### Using GT-GAN for irregular time series

GT-GAN (Jeon et al., 2022) extends adversarial generation to **irregularly sampled** time series, in which observations occur at non-uniform intervals (for example, tick data, event-driven bars, consolidated feeds).

GT-GAN models continuous-time dynamics with a neural **ordinary differential equation** (**ODE**) that parameterizes: 𝑑 𝑑ݐൌ݂ ఏ(𝑑ǡ ݐ)

and integrates the dynamics to generate values at arbitrary timestamps.

The discriminator evaluates both sampled values and their associated timestamps, and a supervised component encourages temporal coherence, analogous to TimeGAN’s supervisor.

**GT-GAN is designed for naturally irregular data**, where observation times carry information (for example, trade arrivals or information events). Applying it to regularly sampled series with artificial irregularity (random subsampling) is unlikely to outperform simpler discrete-time generators. Use GT-GAN when observation times carry information. For regularly sampled daily or minute bars, discrete-time generators are usually simpler and more efficient.

**Implementation**: `04_gtgan_irregular.py` trains GT-GAN on NVDA dollar bars from *Chapter 3*. Reconstruction MSE is 0.026, and the interpolation/real smoothness ratio is 0.02 - the ODE-driven paths are considerably smoother than real bars. Because the experiment uses only 446 bars from a single trading day, these results should be read primarily as an illustration of the continuous-time training procedure rather than as a competitive benchmark of distributional fidelity.

GT-GAN introduces substantially more modeling and optimization complexity than discrete-time generators. With limited data, a continuous-time generator may produce visually smooth paths without accurately matching the true event-time distribution, dependence structure, or regime behavior. The method is most compelling when irregular timing is economically meaningful; if irregularity mainly reflects sampling choices or microstructure noise, the additional continuous-time machinery may add complexity without improving fidelity.

### GAN training challenges

GAN variants share recurring implementation risks:

- **Mode collapse:** The generator covers only a subset of regimes, underrepresenting rare states that matter for risk.
- **Training instability:** Optimization can oscillate or diverge; stabilization methods include spectral normalization, gradient penalties (WGAN-GP), and phased training (TimeGAN).
- **Hyperparameter sensitivity:** Learning rates, architectures, and regularization weights often require dataset-specific tuning.
- **Evaluation ambiguity:** There is no ground-truth label; evaluation relies on held-out distributional tests, downstream utility (TSTR), and diagnostics for temporal structure and cross-sectional dependence.

Kwon and Lee (2024) report that GANs can approximate marginal return distributions but often struggle with finer temporal structure and multivariate dependence; reported performance varies across architectures, and multivariate generation can collapse to a low-dimensional dependence structure. Passing marginal tests is therefore necessary but not sufficient; temporal dynamics and cross-sectional relationships require separate validation.*Section 5.6* turns to diffusion models.

## 5.6 Diffusion models for financial time series

Diffusion models have shown strong empirical performance on synthetic data, from images to financial time series.

![Figure 5.6](assets/figure_5_6.png)

*Figure 5.5: DDPM forward/reverse schematic*

In *Figure 5.5*, the forward process progressively corrupts the real data into near-pure noise. During training, a timestep *t* is sampled, noise is added to the real data, and a single denoiser network is optimized to predict the added noise. At generation time, sampling starts from near-pure noise and iteratively applies the same denoiser to produce real data; the reverse chain is stochastic, injecting fresh noise at intermediate steps (except the final step), so each run can yield a different sample.

Their iterative denoising mechanism can reproduce non-Gaussian features of market data that simpler generators often miss, which makes them useful when sample fidelity is the primary objective (Takahashi and Mizuno, 2024). The main trade-off is computational cost: training and generation are typically slower than with one-shot generators.

### The denoising diffusion framework

Diffusion models define a **forward process** that gradually corrupts an observed sequence with noise, and a **reverse process** that learns to remove it. Sampling starts from noise and applies learned reverse The forward process is a Markov chain that adds Gaussian noise over diffusion steps ݐൌͳǡ ǥ ǡܶ. Given transitions to generate a synthetic sequence. original data 𝑥0, define:

𝑞(ݔ௧פ ݔ௧ିଵ) ൌܰ ൫ݔ௧; √ߙ௧ݔ௧ିଵ, (ͳ െߙ௧)ܫ൯

where 𝛼௧ൌͳ െߚ௧ and 𝛽௧ is a noise schedule. With sufficiently many steps, 𝑥் approaches an isotropic

Gaussian. The schedule determines the signal-to-noise ratio across timesteps and shapes the difficulty The reverse process learns a conditional transition from 𝑥௧ to 𝑥௧ିଵ: of denoising.

𝑝ఏ(ݔ௧ିଵפ ݔ௧) ൌܰ ൫ݔ௧ିଵǢ ߤఏ(ݔ௧ǡ ݐ)ǡ ߑఏ(ݔ௧ǡ ݐ)) The denoiser takes 𝑥௧ and an embedding of 𝑡 as inputs and outputs either parameters 𝜇ఏǡ ߑఏ or an equivalent quantity such as the noise 𝜀. Training samples diffusion steps and minimizes an expectation

over timesteps, so the model learns to invert the corruption process at multiple noise levels. error loss between the realized forward-process noise 𝜀 and the network prediction 𝜀ఏ(ݔ௧ǡ ݐ), avoiding The **Denoising Diffusion Probabilistic Model** (**DDPM**) (Ho et al., 2020) popularized a mean squared direct reconstruction of 𝑥0 during training. This objective is widely used because it is stable and maps

directly to an iterative sampling procedure.

### Why diffusion models fit financial returns

The stylized facts in *Section 5.1* - heavy tails, volatility clustering, leverage effects, and weak return autocorrelation - constrain generators for returns or return-like series. A useful generator must match both the marginal distribution and conditional dynamics, such as persistent volatility and time-varying tail risk. Diffusion models can capture such structure because the denoiser is trained across noise levels rather than being optimized to match only low-order moments.

The three families have distinct failure profiles. VAEs can produce overly smooth samples during decoding, as they average over uncertainty. GAN training can be unstable and may underrepresent tails or minority modes. Diffusion training is typically more stable, but sampling can be slow because it performs many reverse steps per generated sequence.

### Conditional generation for regime stress testing

Conditional generation produces samples consistent with specified attributes. Two common approaches are:

- **Classifier guidance**, which trains a separate classifier on noised data and uses its gradient to steer sampling
- **Classifier-free guidance**, which embeds conditioning during training

Both can direct generation toward states such as “low volatility” or “crisis,” enabling regime-specific scenario generation and stress testing.

A workflow that integrates with regime detection methods (discussed in more detail in *Chapter 9*) is:

1. Fit a regime model (for example, an HMM) to label historical periods. At generation time, compute 𝛻௫೟݈݋݃݌(ݎ݁݃݅݉݁ פ ݔ௧), the gradient of the log-probability of the tar-
2. Train a classifier to predict regime labels from noisy sequences across diffusion timesteps. get regime with respect to the current noisy sample 𝑥௧, and use it to adjust the denoising update. 3.

Classifier guidance (Dhariwal and Nichol, 2021) requires tuning. Strong guidance can reduce diversity sampling (for example, Denoising Diffusion Implicit Models with 𝜂) can help balance targeting and exaggerate rare regimes, producing unrealistic extremes. Temperature scaling and stochastic

and diversity. *Figure 5.6* shows the result: low- and high-volatility regime samples produce visibly separated paths, with a 2.1× regime-volatility ratio.

![Figure 5.7](assets/figure_5_7.png)

*Figure 5.6: Diffusion-TS regime-conditional sample paths*

Because regimes are model-derived, the conditional generator inherits errors from the regime detector. Misclassified periods become mislabeled training data, which can distort conditional samples. Use regime-conditioned sequences as model-based scenarios and validate them with regime-specific diagnostics before downstream use.

### Applying Diffusion-TS with interpretable decomposition

Time-series generators must represent dependencies across the sequence, not just marginal distributions. Diffusion-TS (Yuan and Qiao, 2024) adapts the diffusion denoiser to sequential data by using an **encoder-decoder transformer** and explicitly decomposing the reconstructed sequence into low- and The denoiser takes a noised sequence 𝑥௧ and a timestep embedding 𝑡. An encoder produces a context mid-frequency components. representation of 𝑥௧, and a decoder uses cross-attention to predict the clean sequence 𝑥0.

Rather than predicting 𝑥0 directly, Diffusion-TS constrains the output to be a sum of two components:

𝑥0 ൌݐݎ݁݊݀ ൅ݏ݁ܽ ݏ݋݊

The **trend** component is parameterized as a low-order polynomial fit, intended to capture slow-moving basis (top-𝑘 modes), which targets periodic or quasi-periodic structure. drift over the generation horizon. The **seasonal** component is represented using a truncated Fourier A **Fourier-domain loss** complements the time-domain loss by penalizing mismatches in the frequency representation, encouraging spectral fidelity, and helping preserve autocorrelation-related patterns.

This decomposition is not a guarantee of **interpretability** in the economic sense, but it provides an operational handle: when samples fail diagnostics, the error can often be localized to the slow component (trend), the periodic component (seasonality), or the residual high-frequency variation that neither component represents well.

In practice, Diffusion-TS shares training and evaluation challenges with other DDPM models:

- **Variance calibration:** A trend-plus-seasonal output can understate high-frequency variation and reduce realized return variance. In the reference implementation, raw unconditional samples understate variance by approximately one-third in the normalized space before post hoc rescaling. Downstream tasks that depend on volatility matching may require explicit variance-calibration terms during training.
- **Compute and latency:** DDPM sampling can be slow because it requires many reverse steps per sample. DDIM and distillation-style methods reduce the number of steps, typically at the expense of speed.
- **Temporal coherence:** Alternative approaches impose sequential structure via path summaries such as signatures, which are stable under reparameterization. For example, Sig-CWGAN uses a signature-based discrepancy in its training objective (*Section 5.5*). For diffusion models, include temporal diagnostics (ACF, volatility persistence, leverage proxies) in addition to marginal distribution tests.
- **Validation:** Matching stylized facts is necessary but not sufficient: a generator can satisfy distributional diagnostics yet fail to preserve task-relevant structure. *Section 5.8* presents validation aligned with the intended application.

**Implementation**: `05_diffusion_ts.py` implements Diffusion-TS end-to-end with interpretable decomposition, Fourier loss, and classifier-guided regime-conditional generation.

For the 20 ETF daily return series (2018–2025), the implementation reports a KS statistic of 0.06, a correlation error of 0.04, and an ACF error of 0.05. Regime-conditional generation produces low-volatility samples at 0.93× historical volatility and high-volatility samples at 1.08× historical volatility.

### Choosing between GANs and diffusion models

Diffusion models trade generation speed for training stability. GANs produce samples in a single forward pass; diffusion models require iterative denoising, though DDIMs reduce this cost to 50–100 steps. The GAN variants in *Section 5.5* each address a specific limitation: Tail-GAN targets risk measures, Sig-CWGAN targets path fidelity, and GT-GAN handles irregular timestamps. Diffusion-TS offers a more general-purpose alternative with built-in conditional generation, but specialized GANs remain appropriate when their design objective aligns with the evaluation target. Direct comparison of these toy implementations would be misleading - each generator was designed for a different task with different data requirements. TimeGAN requires bounded data; Tail-GAN optimizes for portfolio-level tail metrics; Sig-CWGAN works only at low dimensionality; GT-GAN needs naturally irregular timestamps. The practical question is not “which generator is best?” but “which generator matches my use case?” For tabular data, large language models can be a good candidate, as we will see next.

## 5.7 LLMs for structured financial data

**Large language models** (**LLMs**) can generate realistic synthetic samples from tabular datasets when the table is represented as text. Although LLMs are trained on natural language, they can learn and reproduce complex cross-column dependencies in structured data, which is useful for financial tables that mix numerical, categorical, and occasional free-text fields.

**Serialization** converts each table row into a text record that explicitly encodes column–value pairs. A serialized table becomes a corpus of short records that an autoregressive model can learn to reproduce. Consider a credit application record:

| income | debt_ratio | emp_length | home_status | approved |
| --- | --- | --- | --- | --- |
| 85000 | 0.32 | 7 | MORTGAGE | 1 |

*Table 5.1: A table showing a record for a credit application*

One serialized representation is:

`income is 85000, debt_ratio is 0.32, emp_length is 7, home_status is MORTGAGE,` `approved is 1`

Training on many such records encourages the model to capture conditional structure across fields (for example, how income and debt ratio relate to approval, conditional on employment and housing status).

### The GReaT framework

**Generate Realistic Tabular Data** (**GReaT**) formalizes a simple workflow (Borisov et al., 2023):

1. **Serialize the training data**. Convert each row into a text record of column–value pairs. To reduce sensitivity to a fixed presentation order, training often randomizes the column order.
2. **Fine-tune an autoregressive LLM**. Fine-tune a pre-trained language model (for example, GPT-2scale) on the serialized corpus so it learns to predict the next token given the preceding tokens, thereby implicitly modeling a joint distribution over fields.
3. **Generate new samples**. Sample new text records and parse them back into rows. Filter invalid outputs (malformed records, missing fields, out-of-range values).

The framework can be useful for financial applications subject to some limitations and risks.

### Financial applications

LLM-based tabular generation is well-suited to datasets with mixed field types and complex conditional dependencies:

- **Credit and loan data:** Application records with demographic variables, financial ratios, and categorical fields are natural candidates. Synthetic datasets can support model development while reducing exposure of sensitive customer data.
- **Customer profiles:** Retail banking and wealth datasets often combine transaction aggregates, account attributes, and occasional text fields. Serialization provides a uniform interface to this heterogeneity.
- **Corporate fundamentals:** Financial statements and derived ratios, together with industry codes and accounting flags, can be augmented or anonymized using LLM-based generators, subject to strict point-in-time and accounting-convention checks.

### Limitations and risks

LLM-based generation introduces failure modes that require explicit mitigation:

- **Invalid records:** Autoregressive generation can produce internally inconsistent combinations (for example, an employment history that is incompatible with age). Post-generation validation is required.
- **Numerical fidelity:** LLMs optimize for token likelihood, not distributional accuracy. High-precision numeric strings are difficult to represent reliably; practitioners often discretize, bin, scale, or normalize numeric features before serialization. Even with preprocessing, marginal distributions and tail behavior can drift relative to the training data.
- **Compute cost:** Fine-tuning and inference add computational overhead. In many tabular settings, moderate-sized models can be sufficient, but cost should be evaluated relative to simpler tabular generators and classical baselines.
- **Privacy leakage:** Synthetic samples can memorize and reproduce rare training examples. Privacy evaluation (*Section 5.8*) is required before using generated data outside a controlled environment.

Serialized generation often violates domain constraints. To manage this risk, implement a **constraint layer** that validates each parsed row:

- **Type constraints:** Age is an integer; ratios are non-negative
- **Range constraints:** Employment years `< age − 16`
- **Logical constraints:** If `home_status = "RENT"`, then `mortgage_balance = 0`

Monitor rejection rates as a diagnostic. Sustained rejection rates above 10–15% suggest that the serialization format, preprocessing, or training protocol needs adjustment.

**Implementation**: `06_llm_tabular_great.py` fine-tunes `distilgpt2` on ETF-derived tabular features (returns, volatility, volume ratios, and categorical regime labels). Fifty epochs of fine-tuning on an RTX 3090 take roughly thirteen minutes - modest compared with training a dedicated tabular generator from scratch. On a held-out test set of 600 samples (extreme-move classification, |݂ݓ݀ ௥݁ݐହௗ| above the 90th

• percentile), TSTR achieves an AUC-ROC of 0.70, close to the 0.74 TRTR baseline; the TSTR ratio is 92%, indicating that synthetic training data largely preserves downstream model utility for this task.

- Distributional fidelity is weaker than utility. Among numerical columns, only `volume_ratio` matches well (KS = 0.10); `volatility` and all four return features (1-, 5-, 20-day, and forward 5-day) sit in the 0.30–0.59 range with p-values below 0.001, so synthetic and real marginals are statistically distinguishable. The categorical distribution is heavily mode-collapsed: the model overgenerates the negative-return regime (`direction = down` 94% synthetic versus 45% real) and the no-trend regime (`momentum = flat` 69% versus 36%), and undergenerates strong-trend states (`momentum = strong` 10% versus 28%; `vol_regime = high` 1% versus 7%). This gap between high task utility and modest distributional fidelity is the practical limit of short LLM fine-tuning on small tabular datasets - longer fine-tuning, larger base models, or rejection sampling on schema constraints close it at the cost of compute, which is the tradeoff *Section 5.2* formalized.

This result illustrates why utility and fidelity must be evaluated separately. The synthetic data preserves enough task-relevant structure to support the extreme-move classifier, but the marginal and categorical diagnostics reveal substantial distributional distortion. The next section presents the comprehensive evaluation framework recommended for practical use cases.

## 5.8 Applying the Fidelity–Utility–Privacy framework

The preceding sections show that the quality of synthetic data depends on the intended use case. No generator dominates across fidelity, utility, privacy, dimensionality, and computational cost. The Fidelity–Utility–Privacy framework from *Section 5.2* provides a common language for comparing methods without reducing them to a single score.

### Reading the Diffusion-TS results

The Diffusion-TS reference run on 20 daily return series of ETFs illustrates how to interpret the framework in practice. The model reports a KS statistic of 0.06 for marginals, a correlation error of 0.04 for cross-asset dependence, and an ACF error of 0.05 for volatility persistence. These are **fidelity** diagnostics: they suggest that the generated samples preserve broad distributional and temporal structure in this configuration.

**Utility** is evaluated separately. For the extreme-move classification task, the TSTR result is approximately at parity with the TRTR benchmark, indicating that synthetic training data preserves the task-relevant signal for this specific prediction problem.

**Privacy** is not established by these results. Unless the generator is trained with a formal privacy mechanism or subjected to leakage tests, strong fidelity and utility do not imply privacy. A model can reproduce useful structure while still memorizing rare records or leaking proprietary information. The interpretation is therefore conditional: Diffusion-TS is a strong, general-purpose generator in this experiment, especially in terms of broad fidelity and task utility. Tail-focused risk applications may still require explicit tail penalties, as in Tail-GAN. Low-dimensional path-dependent applications may benefit from signature-based objectives, as in Sig-CWGAN. Privacy-sensitive applications require a separate privacy mechanism and leakage evaluation.

### Privacy–utility trade-off with DP-GAN

When formal privacy guarantees are required, differential privacy provides a mathematical bound on the influence of individual training records.

**Implementation**: The notebook `07_dp_gan.py` demonstrates this trade-off by training a GAN with DP-SGD using Opacus and sweeping across privacy budgets.

| Privacy Budget | Mean Abs Diff | Correlation Distance | Interpretation |
| --- | --- | --- | --- |
| 1.0 | 8.02 | 0.12 | Strong privacy, poor utility |
| 5.0 | 2.40 | 0.15 | Good balance |
| 10.0 | 2.67 | 0.06 | Good utility |
| 50.0 | 1.68 | 0.11 | Best utility, weak privacy |

*Table 5.2: Privacy-utility tradeoff example*


At 𝜀, the noise required for stronger privacy substantially degrades distributional fidelity. As the privacy budget increases, utility improves, but the formal privacy guarantee weakens. In this example, the middle range around 𝜀 ∈ [5,10] offers a more practical trade-off, but the appropriate threshold depends on the data sensitivity, regulatory context, and intended use.

depends on the data sensitivity, regulatory context, and intended use.

The important lesson is that **privacy cannot be inferred from realism**. Synthetic data may look different from the training data and still leak information; it may also satisfy privacy constraints while becoming too distorted for downstream modeling. Privacy-sensitive synthetic-data workflows therefore require explicit leakage tests or formal DP accounting in addition to fidelity and utility evaluation.

## 5.9 Summary

Synthetic data generation addresses a core limitation in quantitative finance: we only observe one realized market path, yet strategies must remain robust across many plausible alternatives. Modern generative models can produce “alternative histories” that preserve key statistical structure, enabling more rigorous robustness checks, stress testing, and parameter selection than a single backtest trajectory can support.

Effective use hinges on disciplined benchmarking and validation. Generators should reproduce core stylized facts, and classical models remain strong, interpretable baselines when transparency or data constraints matter. Deep generative approaches (GAN variants, diffusion models, and LLMs for serialized tabular data) can target objectives such as tail risk, path-wise fidelity, irregular time sampling, and heterogeneous schema generation. Each must be evaluated against the Fidelity–Utility–Privacy triad and stress-tested for synthetic-specific failure modes such as bias amplification, limited novelty, and overfitting to the generator, with conclusions confirmed on held-out real data.

In *Chapter 6*, we present the strategy research framework that guides the ML4T workflow and our nine case studies.
