# Chapter 11: The ML Pipeline

Previous chapters constructed features across the case studies and evaluated their predictive power through information coefficients, factor spreads, and the triage framework of *Chapter 7*. Now we will take the next step: transforming those features into trading signals using machine learning.

This chapter introduces regularized linear models (Ridge, LASSO, and Elastic Net) as interpretable baselines that introduce the ML signal-generation workflow within the trading setup defined in *Chapter 6*. We will build the walk-forward validation pipeline to prevent look-ahead bias, the interpretability tools to verify economic sensibility, and a conformal prediction framework to quantify uncertainty, foundations that carry forward to every model in *Chapters 12–15* and translate into positions in *Chapters 16–19*.

By the end of this chapter, you will be able to:

- Choose between *regression and classification* formulations based on how predictions will be translated into trading decisions.
- Fit regularized linear models, including Ridge, LASSO, Elastic Net, and logistic regression, using point-in-time preprocessing and standardization.
- Tune and evaluate linear models with walk-forward validation, temporal buffers, and, when needed, nested cross-validation to reduce selection bias.
- Interpret model behavior with SHAP-based diagnostics to assess feature importance, economic plausibility, and stability across refits.
- Construct and evaluate conformal prediction intervals or prediction sets, and monitor where coverage degrades under non-stationary market conditions.
- Use cross-case-study evidence to judge when linear models provide a strong baseline and when a weak linear signal motivates more flexible models.

We will begin by motivating the use of regularization for prediction, then introduce linear regression and classification, show how to use SHAP values for explainability, and present conformal predictions to quantify uncertainty. We conclude by presenting insights from the case studies. *Chapters 12–14* extend the models introduced here. Every model must beat the regularized linear baseline from this chapter. A gradient-boosting model that underperforms Ridge has added complexity without improving predictive power. Each modeling chapter loads the linear baseline automatically and reports the improvement (or lack thereof) alongside the candidate model.

## 11.1 From inference to prediction

Classical **econometrics** (the application of statistics to economic data) and machine learning approach the same data with different objectives. Econometrics asks: “What is the relationship between these variables, and is it statistically reliable?” Machine learning asks: “Given inputs, what is the best forecast of the outcome?” This chapter develops linear models for this exact predictive task. The shift from inference to prediction changes which properties of an estimator matter and motivates the regularization methods that follow.

### Two modeling cultures

Leo Breiman’s 2001 paper “Statistical Modeling: The Two Cultures” formalized a distinction that had long been implicit:

- The *data modeling culture*, dominant in econometrics, posits a stochastic model for the data-generating process (DGP) and estimates its parameters
- The *algorithmic modeling culture*, dominant in machine learning, treats the DGP as unknown and evaluates models solely by their predictive accuracy on data not used to develop the model

The distinction is not about complexity. A linear regression can serve either culture: used for inference, it recovers coefficient estimates and tests hypotheses about them; used for prediction, it generates forecasts evaluated by out-of-sample error. The distinction is about the *success criterion*. In the data modeling culture, a model succeeds when it recovers interpretable, statistically significant parameters under a correctly specified model. In the algorithmic culture, a model succeeds when it predicts well. As Breiman argued, predictive accuracy should serve as the primary check on whether it has captured meaningful structure in the data.

For algorithmic trading, the more relevant criterion is predictive accuracy (generating stable, profitable forecasts), not unbiased parameter recovery (Simonian, 2024).

### What inference requires

estimator behind models from the CAPM to Fama-French-Carhart) recovers coefficient estimates 𝛽̂௝ The inferential framework is powerful but demanding. **Ordinary Least Squares** (**OLS**) regression (the

𝑥௝ on the outcome. Those conditions are the **Gauss-Markov assumptions** that require: that, under certain conditions, are unbiased estimates of the true *ceteris paribus* effects of each feature **Linearity** in the parameter vector 𝛽, implying the functional formݕൌܺ ߚ൅ߝ **Strict exogeneity**: 𝐸[ߝפܺ ] = 0, which implies that regressors contain no information about • • the error term (no omitted variables or lagged dependent variables) others, so 𝛽 is identifiable

- **No perfect multicollinearity**, meaning no regressor is an exact linear combination of the **Spherical errors**, Var(ߝפܺ ) ൌߪ2ܫ, which assumes errors have constant variance and are

• uncorrelated across observations

When all four hold, the Gauss-Markov theorem guarantees that OLS is the **Best Linear Unbiased Estimator** (**BLUE**): among all unbiased linear estimators, OLS has the smallest variance.

The linearity and exogeneity assumptions require that the model include the right variables in the right functional form. This is a strong requirement when the true DGP is unknown. In financial markets, the DGP almost certainly involves nonlinearities, regime changes, and time-varying parameters. When the model is misspecified, the unbiasedness guarantee fails, and the coefficient estimates become difficult to interpret as causal or structural parameters.

Given a correctly specified model, hypothesis testing evaluates whether a coefficient is distinguishable puted, assuming the null hypothesis (𝛽௝= 0) is true. This is useful for inference (determining whether from zero. A p-value measures the probability of observing a test statistic as extreme as the one com-

a variable has a statistically reliable relationship with the outcome), but it does not measure predictive power. A coefficient can be highly significant yet economically negligible (for example, a large sample size with a tiny effect), or insignificant yet genuinely predictive when combined with other features.

The Gauss-Markov optimality of OLS is defined within the class of unbiased estimators. But for prediction, unbiasedness is not inherently desirable. What matters is total prediction error, which de-MSE = Bias2 + Variance + ߪఌ composes as: 2

An estimator that introduces some bias but substantially reduces variance can achieve lower MSE than an unbiased estimator, particularly when the number of features is large relative to the number of observations. The Gauss-Markov theorem does not claim that OLS minimizes MSE; it claims that OLS minimizes variance **subject to the constraint of unbiasedness**. Relaxing that constraint opens a path to lower total error.

### Why prediction demands a different approach

Consider a strategy research problem: predicting next-month returns using 100 features from momentum, valuation, and sentiment feature families across 240 monthly observations. OLS will fit this model. Under the Gauss-Markov assumptions, the resulting coefficient estimates are unbiased. But **unbiasedness alone does not ensure good predictions**. With 100 parameters estimated from 240 obin 𝛽̂, and therefore in predictions. The model fits training data well (it has enough free parameters to servations, estimation variance is large: small changes in the training sample produce large swings

absorb idiosyncratic patterns), but those patterns do not generalize.

This is not a failure of OLS as an inferential tool. It is a consequence of using an estimator optimized for unbiased parameter recovery in a setting where predictive stability matters more. **OLS has no mechanism to trade bias for variance**, because doing so would compromise the inferential guarantees that justify its use in econometrics. Three aspects make the problem particularly acute in finance: **High dimensionality relative to observations**. As ݌Ȁ݊ grows, OLS estimates become unstable; when ݌൐݊ • , they are not uniquely defined.

- **Multicollinearity.** Financial features are pervasively correlated: momentum overlaps with trend, valuation ratios share accounting inputs. OLS assigns arbitrary weights to correlated **Low signal-to-noise ratio**. Cross-sectional return regressions typically achieve 𝑅2 of 1–3% features, producing offsetting coefficients that amplify noise. • in-sample and lower out-of-sample. Variance dominates MSE: an unconstrained model fits noise, while a constrained model sacrifices modest signal to suppress far more spurious patterns.

**Regularization** addresses this tradeoff directly. It adds a penalty to the OLS loss function that increases ized estimates no longer center on the true 𝛽), but it reduces variance by preventing the model from with the magnitude of the coefficients, shrinking estimates toward zero. This introduces bias (penal-

assigning extreme weights to any feature or distributing weight erratically across correlated features.

The form of the penalty encodes a *prior about the structure of the signal*: **Ridge** (𝐿2) shrinks all coefficients proportionally, which is suited to diffuse signals with per-

• **LASSO** (𝐿1) drives coefficients to zero, performing automatic selection when the signal convasive multicollinearity • centrates in a sparse subset

- **Elastic Net** combines both, retaining correlated feature groups while still shrinking toward sparsity, a common requirement when momentum, trend, and mean-reversion indicators overlap rameter 𝜆 controls the tradeoff, and cross-validation selects the degree of shrinkage that minimizes These are principled responses to the **bias-variance tradeoff**, not ad hoc corrections. The tuning pa-

out-of-sample error. *Section 11.2* develops each method formally.

The label families defined in *Chapter 7* (continuous returns, thresholded direction, quantile membership) map to regression (predicting magnitudes) and classification (predicting categories). The choice depends on alignment with the trading setup and the data. We first discuss regularized regression, then classification in more practical detail.

**Implementation**: `01_ols_inference` shows OLS summary tables, the Gauss-Markov diagnostic battery, variance inflation factors, and robust standard errors on the ETFs case study.

## 11.2 Regularized regression

Ridge, LASSO, and Elastic Net each encode a different assumption about how the signal is distributed across features. This section presents their formal objectives, the geometry that gives each method its distinctive behavior, and the practical machinery (standardization, hyperparameter optimization, and loss function choice) required to deploy them within the walk-forward protocol of *Chapter 6*.

### Ridge regression for distributed signal **Ridge regression** augments the OLS objective with an 𝐿2 penalty on the coefficient vector:

௡ min ఉ∑(ݕ௜െܺ ௜ߚ)2 ൅ߣ|ߚ|2 2 ௜ୀଵ 2 = ∑ߚ௝ where 𝜆 controls regularization strength and |ߚ|2 2 ௝ . The penalty increases with the sum ⊤ܺ)−1 The closed-form solution reveals the mechanism. OLS solves 𝛽̂OLS = ( of squared coefficients, discouraging large values but never forcing any coefficient exactly to zero. ⊤ݕ; Ridge solves 𝛽̂Ridge = ( ൅ߣܫ)−1 ⊤ݕ. ⊤ Adding 𝜆 to 𝑋⊤𝑋 stabilizes the inversion, shrinking coefficient estimates, and bounding their variance. When 𝜆, the solution reduces to OLS; as 𝜆, all coefficients shrink toward zero.

The geometry is instructive. In the **singular value decomposition** (**SVD**) of the feature matrix, each singular value corresponds to a direction in the feature space and measures the amount of variance the data exhibit along that direction:

- OLS estimates are least stable along directions of low variance: small denominators in the Ridge adds 𝜆 to every singular value before inverting, which disproportionately shrinks coinversion produce large, noisy coefficients • efficients along low-variance directions while leaving well-identified directions largely intact (Hastie, Tibshirani, and Friedman, 2009)

The practical consequence: strong signals supported by substantial variation in the data survive with modest shrinkage; weak signals in poorly identified directions (precisely the ones most likely to reflect noise) are heavily attenuated.

This makes Ridge well-suited to settings where many features contribute modestly to the signal and multicollinearity is pervasive. If momentum, trend-following, and relative-strength indicators capture overlapping information, OLS assigns weights in an erratic manner; Ridge retains all of them with reduced, stabilized weights. `02_regularization_paths` traces Ridge coefficient paths across regularization strengths and compares IC against the OLS baseline.

![Figure 11.1](assets/figure_11_1.png)

*Figure 11.1: Ridge etfs performance for increasing regularization*

*Figure 11.1* traces Ridge prediction quality across ten orders of magnitude in penalty strength α, using the `etfs` case study with 21-day return labels (n≈394,000):

- Below α≈100, the penalty is negligible, and Ridge is indistinguishable from OLS (IC≈0.028, ICIR≈0.54, right-hand scale).
- Between 10³ and 10⁶, both IC and ICIR improve as shrinkage suppresses noisy coefficient directions; the peak ICIR of 1.30 at α≈10⁶ represents a 2.4× improvement over the OLS baseline, driven by a 60% increase in mean IC (0.046 versus 0.028) combined with a 33% reduction in the IC standard deviation across folds.
- Mean IC peaks earlier at α≈3×10⁴ (IC=0.047) and declines slightly thereafter, but IC variance keeps shrinking faster, so ICIR continues improving until α≈10⁶, where it plateaus.

### LASSO regression for sparse signal **LASSO** replaces the 𝐿2 penalty with an 𝐿1 penalty:

௡ min ఉ∑(ݕ௜െܺ ௜ߚ)2 ൅ߣ|ߚ|1

௜ୀ1 where |ߚ|1 = ∑|ߚ௝| . Unlike Ridge, the 𝐿1 penalty produces *sparse* solutions: some coefficients are ௝

exactly zeroed, performing automatic feature selection. This encodes a prior that the signal concentrates in a small subset of features: only a handful of the candidate indicators truly predict returns, The **sparsity** arises from the geometry of the 𝐿1 constraint set. The absolute-value penalty creates a and the rest are noise.

typically intersect these corners, setting some coefficients exactly to zero (Tibshirani 1996). The 𝐿2 diamond-shaped feasible region with corners on the coordinate axes. The OLS objective’s contours

penalty’s circular constraint set has no corners, which is why Ridge shrinks but never zeros. Sparsity has practical appeal: fewer active features mean fewer data pipelines to maintain and clearer PnL attribution. However, LASSO is **unstable when features are correlated**. Among a set of similar indicators, LASSO selects one at random and sets the others to zero. Which indicator survives can change with minor perturbations in training data, making the active feature set unstable across walk-forward windows. This instability does not necessarily harm predictive accuracy because the selected feature may serve as a reasonable proxy for the group. However, it complicates interpretation and can increase turnover if the trading signal depends on which specific features are active. as 𝜆 decreases and how the active set shifts across walk-forward folds (see *Figure 11.2*). The LASSO coefficient path in `02_regularization_paths` shows which features enter the model first

![Figure 11.2](assets/figure_11_2.png)

*Figure 11.2: Ridge (a) shrinks all 57 coefficients smoothly toward zero as λ increases, retaining every feature; LASSO (b) eliminates features sequentially, snapping coefficients to exactly zero at different λ thresholds. The 10 largest-magnitude features are highlighted, the remaining 47 in gray. Panel (c) compares mean out-of-sample IC (±1σ) for OLS, Ridge, LASSO, and Elastic Net. All panels use standardized features from the etfs case study (21-day returns, single training fold).*

For the `etfs` case study, LASSO tells a different story than Ridge: even mild sparsity (α≈0.0001) zeros out roughly a quarter of features, and stronger penalties *reduce* IC below OLS, while α≥0.01 eliminates virtually all coefficients.

With 57 correlated momentum and volatility features, the ETF signal is diffusely distributed: Ridge’s uniform shrinkage preserves it, while LASSO’s variable selection discards informative features along with noise.

### Combining both priors with elastic net **Elastic Net** blends the 𝐿1 and 𝐿2 penalties:

௡ + ߣ൤ߙ|ߚ|1 + ͳ െߙ min ఉ∑(ݕ௜െܺ ௜ߚ)2 |ߚ|2 2] 2 ௜ୀ1 where 𝜆 controls overall regularization strength and 𝛼[0,1] controls the mixing ratio: 𝛼 recovers LASSO, 𝛼 recovers Ridge.

The combination produces a distinctive grouping behavior (Zou and Hastie, 2005). The 𝐿2 component encourages correlated features to receive similar coefficients; the 𝐿1 component then performs

group-level selection, keeping or discarding entire clusters. For financial feature sets with many correlated indicators (momentum variants, overlapping sentiment measures, related valuation ratios), this grouping is more stable than LASSO’s arbitrary within-cluster selection and more selective than Ridge’s retention of everything. ߚ|2 + ߙ[ℓ1_ratio ⋅|ߚ|1 + 2௡|ݕെܺ |ߚ|2 2] , 1 1−ℓ1_ratio 2 Scikit-learn’s `ElasticNet` parameterizes the objective as `0.0` = pure Ridge). Note the ͳȀʹ݊ normalization: `ElasticNet` averages the loss, unlike `Ridge`, which where `alpha` controls overall penalty strength, and `l1_ratio` controls the L1/L2 mix (`1.0` = pure LASSO,

uses the unnormalized **sum-of-squares error** (**SSE**), so the `alpha` scales are not directly comparable To begin, cross-validate over a grid of `l1_ratio` values (for example, 0.1,0.3,0.5,0.7,0.9) and `alpha` between the two. values log-spaced over the range appropriate to 𝑛 (see the scaling heuristic in the Optuna section below, adjusting for the ͳȀʹ݊ normalization). If the optimal `l1_ratio` consistently lands near 1.0 across walk-forward windows, the data favor sparsity; near 0, the signal is diffuse across features. Unstable

`l1_ratio` across windows reflects genuine variation in which feature relationships dominate. Consider averaging predictions across several mixing ratios rather than selecting a single point estimate. The case study pipeline runs this grid search (see `06_linear` in each case study directory).

### Standardization

Regularization penalties penalize all coefficients equally, whereas raw coefficients depend on the units of the features. A momentum signal, measured in percentage points, and a volume indicator, measured in millions of shares, have vastly different magnitudes. Without standardization, the penalty shrinks large-scale features more aggressively, a result driven by unit choices rather than predictive importance. (std) = 𝑥௝െߤ௝ Standardize each feature to zero mean and unit variance: 𝑥௝ ߪ௝ 

After standardization, coefficient magnitudes are directly comparable: a coefficient of 0.3 on standardized momentum and 0.1 on standardized volume means the model assigns three times as much weight to momentum, regardless of the original scales.

Two implementation details are critical: **Fit on training data only.** Compute 𝜇௝ and 𝜎௝ from the training fold, then apply those parameters

• to transform validation and test data. Using full-sample moments introduces look-ahead bias. In a walk-forward protocol, recompute standardization parameters at every refit. **Winsorize before standardizing.** A single extreme observation can inflate 𝜎௝, compressing all

• other observations toward zero. Clip at the 1st and 99th percentiles (computed from training data) before standardizing. Both errors (leaking moments and skipping winsorization) are silent: no error is raised, but the model benefits from information unavailable at prediction time.

The walk-forward pipeline in `02_regularization_paths` applies `StandardScaler`, fitted only to the training data, at every fold, implementing the leakage-safe workflow described here.

### Hyperparameter optimization with Optuna

Ridge, LASSO, and Elastic Net each introduce hyperparameters (regularization strength and, for Elastic Net, the mixing ratio) that control the bias-variance tradeoff. Grid search over a pre-specified schedule works for one or two parameters but scales poorly to joint optimization and does not adapt to the loss surface.

Optuna (Akiba et al. 2019) provides **sequential, model-based hyperparameter optimization** through a define-by-run API: the objective function samples parameters from distributions, evaluates walk-forward cross-validation performance, and Optuna’s **Tree-structured Parzen Estimator** (TPE) sampler steers subsequent trials toward promising regions of the search space.

The critical risk is **validation overfitting**: running many trials on the same validation folds can select hyperparameters that exploit idiosyncrasies of the validation period rather than a durable signal. Nested cross-validation, with an inner loop for hyperparameter optimization and an outer loop for evaluation, is the cleanest correction, although it increases computation. For time series, the same principle requires nested walk-forward validation, with each outer walk-forward split evaluated only after tuning inside the corresponding training window (see *Chapter 6*).

A useful diagnostic is to examine the full distribution of validation ICs, the stability of trial rankings across folds, and the uncertainty around the selected trial’s IC. A best trial that materially exceeds the rest of the search but owes its advantage to outliers in a single fold, regime, or random seed is a warning sign. Report the number of trials, the search space, the selected hyperparameters, and the fold-level dispersion of the selected configuration. *Chapter 12* develops Optuna’s advanced features (pruning, multi-objective optimization, and time-series-aware tuning) for the larger hyperparameter space of gradient boosting.

Before launching an adaptive search, the search space must be calibrated to span substantively different model regimes.

This is especially important for the regularization strength because the penalty’s numerical scale depends on the estimator’s loss convention and feature scaling. With standardized features, `sklearn.` tion often scale with the sample size through the eigenvalues of 𝑋⊤𝑋. By contrast, `sklearn.ElasticNet` `Ridge` uses an unnormalized sum-of-squares objective, so values of alpha that materially affect the soluaverages the loss over observations, absorbing a factor of 𝑛; comparable penalties therefore differ by

In the ETF case study, with approximately 400,000 observations, a Ridge search that stopped at 𝜆102 roughly this factor when translating between conventions.

would mostly explore the near-OLS region and could falsely suggest that regularization has little effect. The notebook therefore maps a wider range, from 10−2 to 109, to include both the under-regularized

plateau and the strongly regularized regime. **Implementation**: `04_nested_cv_hpo` sweeps 𝜆 from 10−2 to 109 across walk-forward folds. Mean IC stays mildly negative across most of the range, troughs near 𝜆103 at −0.039, recovers monotonically as regularization tightens, and only crosses zero at 𝜆107 to reach a noisy peak of ≈+0.007 at the top of the search range. The fold-level standard deviation ranges from roughly 0.04 to 0.10 across the grid and dwarfs the cross-𝜆 mean, flagging

the landscape as noise-dominated. The notebook then implements both single-loop and nested cross-validation with Optuna, quantifying the selection-bias inflation (Cawley and Talbot, 2010).

When the underlying signal is weak, the protocol distinction is exactly when selection bias matters most. Here, it is large enough to flip the reported sign: single-loop CV reports a positive mean IC of single hyperparameter (Ridge 𝜆); *Chapter 12* extends to the joint tuning of multiple hyperparameters. +0.028; nested CV returns -0.032 on identical splits. We illustrate these principles by first optimizing a

**Box 11.1: Incremental updates**

The walk-forward protocol above retrains from scratch at every refit. For high-frequency case studies, such as Crypto (8-hour) and NASDAQ-100 (minute bars), full retraining is computationally expensive.

Scikit-learn’s `SGDRegressor` and `SGDClassifier` support `partial_fit` for incremental updates: present a single mini-batch of new observations and update the coefficients via one gradient step, rather than re-solving the full optimization. This is the fastest for online deployment but introduces sensitivity to the learning rate schedule and can drift without periodic full retraining. *Chapter 25* develops the full online learning framework for live trading systems.

### Loss function choice

The regularizers above are usually paired with **mean squared error** (**MSE**) by default, especially in standard Ridge and Lasso regression. But the penalty does not require MSE: you can combine the same regularization idea with other loss functions when a different target (such as a median, a robust central tendency, or a tail quantile) better matches the trading objective.

MSE is a natural choice when the goal is to estimate the conditional mean of the return distribution, but the conditional mean may not be the right target:

- **MSE** penalizes errors quadratically, forcing the model to accommodate extreme observations that may reflect idiosyncratic events rather than systematic patterns.
- **Mean absolute error** (MAE) penalizes linearly and estimates the conditional *median* rather than the mean. Because outliers shift the median far less than the mean, MAE produces more stable coefficients when a few large moves dominate. (like MAE) at a threshold 𝛿, chosen as a tuning parameter or scaled to a robust estimate of
- **Huber loss** transitions from quadratic for small errors (like MSE) to linear for large errors

the residual dispersion, preserving MSE’s efficiency for typical observations while limiting outlier influence (Huber, 1964).

All three losses estimate a location parameter of the conditional return distribution. Scikit-learn’s `SGDRegressor` provides a unified interface: `loss='squared_error'`, `loss='epsilon_insensitive'` (MAE proxy), and `loss='huber'` all accept the same regularization penalties and `sample_weight` parameter. `02_regularization_paths` compares all three at the best Ridge alpha, isolating the loss function’s effect on IC.

**Quantile regression** changes the estimand entirely, predicting specified percentiles rather than a central tendency. If a long-short strategy trades only the top and bottom deciles, the conditional mean is the wrong target: most of the model’s capacity is spent distinguishing among assets that will never be traded. *Section 11.5* develops quantile regression as the foundation for conformalized prediction intervals.

### Evaluating regression models

Training losses determine what the model optimizes; evaluation metrics determine how we judge the result. The appropriate metric depends on how predictions map to trades:

- **Information Coefficient (IC).** Spearman’s rank correlation between predicted and realized returns is the primary metric for cross-sectional models that generate ranked signals. IC captures monotonic association without assuming linearity. In general, the coefficient of variation is the ratio of the mean to the standard deviation for a given metric; for the IC, this measure is also called the **IC Information Ratio** (**ICIR**; see *Chapter 7*).
- **RMSE and MAE.** RMSE, the square root of the MSE, measures the standard deviation of prediction errors; MAE measures the average absolute error and is less sensitive to outliers. Both are useful for relative model comparison within the same dataset and label definition, but neither has a standalone threshold for “good” performance: the relevant benchmark is always a naive model that, for example, predicts the cross-sectional mean.
- **Turnover.** No statistical metric measures tradability: a model that improves IC by 20% but triples turnover may degrade net performance after transaction costs. Report turnover alongside every metric. *Section 11.6* examines this tradeoff empirically across all nine case studies; *Chapter 17* formalizes the turnover-adjusted evaluation framework.

Since IC measures global ranking quality, strategies that trade only the extremes should also track quintile or decile spreads to verify that ranking accuracy translates into separation in the traded tails.

**Implementation**: `02_regularization_paths` reports IC, RMSE, and R² across all four model types for the `etfs` case study. The stability of predicted rankings for the same test data, measured as the correlation between cross-sectional rankings from models trained on consecutive windows, diagnoses signal robustness to retraining; `02_regularization_paths` computes this for Ridge across all fold pairs.

### Sample weighting

Most scikit-learn estimators accept a `sample_weight` parameter, allowing observations to contribute unequally to the loss function. Two weighting schemes from *Section 7.2* apply directly: **Uniqueness weighting** (𝑤௧ǡ௔ = average inverse concurrency) corrects for label overlap when using the 𝐻‾-bar labels defined in *Chapter 7*. Overlapping labels violate the independence as-•

sumption implicit in the sum-of-squared-errors objective; down-weighting observations with recency ൌ݁ **Recency weighting** (𝑤௧ high concurrency partially mitigates this. ఒ(்ି ௧)) gives more influence to recent observations, adapting the model to regime evolution. The decay rate 𝜆 controls the effective lookback horizon. • combined = 𝑤௧ǡ௔ uniqueness ⋅𝑤௧ The two compose multiplicatively: 𝑤௧ǡ௔ 𝑁eff ൌσݓ௧ǡ௔ combined alongside performance metrics. A small 𝑁eff relative to the nominal sample size sigrecency. Report the effective sample size

nals that a few observations dominate the fit, increasing the risk of overfitting to those observations.

**Implementation**: `02_regularization_paths` shows recency weighting with exponential decay in the final walk-forward fold and reports the effective sample size.

### Training loss versus trading objective

A model can reduce MSE while degrading PnL if the improvement comes from fitting the middle of the cross-section (assets that are never traded) or from increasing turnover. We therefore maintain a strict separation between *model selection* (validation loss and rank metrics within the leakage-safe walk-forward protocol) and *strategy evaluation* (turnover- and cost-adjusted returns, *Chapter 18*). Without this separation, in-sample statistical improvements masquerade as trading value. We now turn to a different prediction task that focuses on direction, or extreme moves.

## 11.3 Predicting direction with logistic regression

The previous section treated the label as a continuous return. Many trading setups, however, operate on discrete decisions (go long, go short, or stay flat), and the label families defined in *Chapter 7* include binary up/down flags, three-class directional labels, and quantile memberships. For these setups, classification models that directly predict discrete outcomes are the natural choice, and logistic regression is the regularized baseline.

Classification is not universally superior to regression. Direction prediction can misalign with PnL when payoffs are asymmetric: a model that correctly predicts direction 55% of the time but systematically misses large moves may underperform a regression model that captures magnitude in the tails.

The choice should follow from the *signal-to-trade mapping* defined in *Chapter 7*. If the mapping applies a threshold or quantile rule to convert predictions into positions, classification fits naturally because the model directly optimizes the decision that matters. If the mapping scales position size in proportion to the predicted return, regression preserves the magnitude information that classification discards. Ultimately, however, what works better in a given case is an empirical matter.

### Binary logistic regression

Logistic regression models the probability that the outcome belongs to the positive class: 1 𝑃(ݕ= 1 פܺ ) = −௑ఉ= ߪ(ܺߚ) 1 ൅݁

where 𝜎(⋅) is the sigmoid function, mapping any real-valued linear combination to the (0,1) interval. The model learns 𝛽 by maximizing the **log-likelihood** of the observed labels, or equivalently by min-

imizing the **cross-entropy** (log-loss) between predicted probabilities and observed outcomes. rather than a point forecast of the return. A prediction of 0.65 means that, under the fitted logistic The key difference from linear regression is that logistic regression produces a *probability estimate*

specification, the model assigns a 65% probability to an up move. This interpretation is justified by 𝑃(ݕൌͳ פܺ ), provided the model class is rich enough and the data-generating process is stable. the log-loss objective: in population, cross-entropy is minimized by the true conditional probability

In practice, however, the fitted probabilities *may not be well calibrated* due to misspecification, finite sample size, regularization, class weighting, resampling, noisy labels, or distribution shift. The output should therefore be treated as a probability estimate, not automatically as a calibrated empirical frequency. Calibration diagnostics or post-hoc calibration may be needed before using these probabilities Each coefficient 𝛽௝ represents the change in log-odds per unit change in 𝑥௝. After standardization, a for risk-adjusted position sizing (discussed later in this section). coefficient of 0.5 on momentum means that a one-standard-deviation increase in momentum raises the log-odds of an up move by 0.5. At a baseline probability of 50, this changes the predicted probability to 𝜎(0.5) ≈62. At a baseline probability of 90, the same log-odds increase changes the predicted probability to approximately 94. The probability-point effect is therefore not constant: **coefficients**

**are linear on the log-odds scale**, not on the probability scale.

Classification labels collapse return magnitudes into categories: a 50% return and a 2% return both count as “up.” This is simultaneously a strength and a limitation. It prevents extreme observations from dominating the loss function, as they would with MSE in regression, producing more stable coefficient estimates in the presence of fat tails. But it discards valuable magnitude information. The trade-off may favor classification when the trading decision is binary (long or short) and regression when position size varies with the predicted magnitude.

**Implementation**: `03_logistic_classification` traces the full walk-forward binary logistic regression pipeline on ETF data, including the label construction and the L2/L1 regularization sweeps described below. The binary model extends to 𝐾 classes via **multinomial logistic regression** (**softmax**). For the three- Multi-class extension

class label, with classes long (+1), neutral (0), and short (−1), the model estimates: 𝑃(ݕ=݇ פܺ ) =݁ ௑ఉೖ ∑ ௄ ௑ఉೕ ௝ୀଵ  Each class receives its own coefficient vector 𝛽௞, and probabilities sum to one across classes ( `multi_`

`class='multinomial'` in scikit-learn). The neutral zone filters observations near the zero-return parameter count by roughly 𝐾. boundary, reducing label noise, but each additional class adds a full coefficient vector, increasing

For quantile strategies trading only the extremes, a binary framing (top decile versus bottom decile, discarding the middle) is typically more efficient than a multi-class model that must also learn middle categories.

The 𝐿1, 𝐿2, and Elastic Net penalties from *Section 11.2* apply identically in form, added to the negative Regularization

log-likelihood rather than the sum of squared residuals: ௡ min െ෍ൣݕ௜ߪ(ܺ ௜ߚ) + (ͳ െݕ௜)൫ͳ െߪ(ܺ ௜ߚ))] + ߣ|ߚ|2 2 ఉ ௜ୀଵ Scikit-learn’s `LogisticRegression` uses ܥൌͳȀߣ (higher 𝐶 means less regularization), with `penalty`

and `l1_ratio` controlling the penalty type. All guidance from *Section 11.2* carries over. when classes are perfectly or nearly separable. In high-dimensional feature spaces where 𝑝 approaches Regularization is more critical here than in regression: the maximum likelihood objective diverges 𝑛, a separating hyperplane almost always exists, driving coefficients toward ±∞ and producing arbi-

solutions, a concern absent from OLS, which has a closed-form solution whenever 𝑋⊤𝑋 is invertible. trarily confident predictions that fail out of sample. Regularization bounds coefficients, ensuring finite

### Probability calibration

𝑃(ݕ= ͳ פܺ ) = 0.7 for a set of observations, approximately 70% of them will have 𝑦. In practice, Under correct specification, logistic regression is calibrated by construction: if the model predicts

two forces distort this correspondence:

- **Regularization** systematically compresses predicted probabilities toward the base rate. Shrinking coefficients toward zero flattens the sigmoid, pulling extreme predictions toward 0.5. This is a predictable, directional distortion: the model becomes systematically underconfident in its predictions.
- **Misspecification** introduces less predictable distortions. If the true decision boundary is nonlinear but the model is linear, calibration breaks in ways that depend on the specific form of the misspecification.

Calibration matters when position sizes depend on predicted probability levels: if a probability of 0.7 triggers a larger position than 0.6, miscalibrated probabilities translate directly into misallocated capital:

- **Platt scaling** (fitting a logistic function to the model’s raw outputs on a held-out set) corrects for the systematic compression caused by regularization.
- **Isotonic regression** (fitting a non-parametric monotone function) handles more complex miscalibration patterns but requires more data to estimate reliably (Niculescu-Mizil and Caruana, 2005).

Both must use data not used for training: the validation fold in the walk-forward protocol serves this purpose. Scikit-learn’s `CalibratedClassifierCV` automates the procedure.

Bin predictions by predicted probability (for example, [0.5, 0.55), [0.55, 0.6), …) and compute accuracy within each bin. Monotonically increasing hit rates as predicted probability moves away from 0.5 indicate well-calibrated confidence; non-monotonic patterns (high-confidence predictions performing worse than moderate-confidence ones) suggest overfitting or misspecification.

For threshold-based and rank-based signal generation, calibration is less critical because only the *ordering* of predictions matters. Assess whether the signal-to-trade mapping depends on probability levels or only on ranks before investing effort in calibration.

**Implementation**: `03_logistic_classification` produces calibration curves for the walk-forward predictions and compares them against perfect calibration.

### From probabilities to trading signals

The conversion from predicted probabilities to positions should match the trading setup defined in *Chapter 7*. Common approaches include: **Threshold-based**: Go long if 𝑃(ݕൌͳ פܺ ) ൐߬ upper, short if 𝑃(ݕൌͳ פܺ ) ൏߬ The thresholds control selectivity: 𝜏upper = 0.6 and 𝜏lower = 0.4 is a common starting point, but

- lower, flat otherwise.

optimal values depend on transaction costs and the base rate. Asymmetric thresholds accommodate asymmetric costs of long versus short positions.

- **Probability-weighted**: Position size is proportional to the distance from 0.5, expressing graduated confidence. This requires calibrated probabilities: if the model systematically underestimates confidence (as regularization tends to do), positions will be systematically undersized relative to the true signal.
- **Rank-based**: Within a cross-section, rank assets by predicted probability and go long the top decile, short the bottom decile. This ensures a fixed number of positions regardless of the probability distribution and depends only on ordering, making it robust to miscalibration.

Thresholds and quantile cutoffs must use only information available at decision time: the walk-forward protocol (*Chapter 6*) enforces this by recomputing them within each training window. **Implementation**: `03_logistic_classification` compares all three conversion methods on the same fold, showing that identical predictions yield substantially different portfolios. *Chapter 16* extends this comparison across full backtests.

### Handling class imbalance

When predicting rare events (the 5% of stocks exceeding two standard deviations, or the minority class in an asymmetrically thresholded label), the minority class contributes little to the log-likelihood. The model minimizes loss by predicting the majority class for all observations, achieving high accuracy but generating no trading signal.

**Class weighting** scales each class’s loss contribution inversely to its frequency, ensuring that minority-class errors carry proportionally more weight. Scikit-learn’s `class_weight='balanced'` computes these weights automatically. This does not change the decision boundary’s location in feature space but adjusts how aggressively the optimizer penalizes errors on each side. `03_logistic_classification` compares default and balanced class weighting on the last fold, confirming that the effect is modest when classes are roughly balanced. 𝑁eff: a sharp drop below the nominal sample size signals that a few observations dominate the fit. This When combining class weights with the uniqueness and recency weights from *Section 11.2*, monitor

risk is acute when minority-class observations cluster in crisis periods. Verify that reweighted training sets span multiple market conditions rather than concentrating on a single regime.

### Evaluating classification models

Classification evaluation requires metrics suited to the probabilistic and discrete nature of the outputs. As with regression (*Section 11.2*), the appropriate metric depends on how predictions map to trades:

- **AUC (area under the ROC curve)** measures the quality of ranking across all possible classification thresholds. An AUC of 0.5 indicates random ranking; values of 0.55–0.60 represent meaningful predictive power for financial applications, where signal-to-noise ratios are low. Values above 0.65 on out-of-sample financial data warrant scrutiny: they may indicate data leakage or an evaluation window that happens to align with a strong trend.
- **Precision and recall.** Precision (fraction of predicted positives that are true positives) guards against wasted trades; recall (fraction of true positives that are predicted) guards against missed opportunities. The trade-off depends on costs: high transaction costs favor precision (fewer, more accurate trades); strategies with capacity constraints favor recall (capturing all available opportunities). The F1 score balances both but assigns equal weight to each, which may not reflect the trade-offs in economics.
- **Log-loss.** If the signal-to-trade mapping uses probability levels rather than ranks, log-loss directly evaluates the quality of the probability estimates. It penalizes confident wrong predictions heavily, making it a useful complement to AUC in settings where calibration matters. The Brier score (the mean squared difference between predicted probabilities and binary outcomes) provides a complementary calibration-sensitive metric. **Turnover**

As with regression, report turnover alongside every classification metric. A model that improves AUC by shifting the decision boundary (changing which assets are traded each period) may increase turnover beyond what the strategy can absorb.

Now that we have familiarized ourselves with the inner workings of the most popular linear models, let us take a closer look at model interpretability.

**Implementation**: `03_logistic_classification` shows the full workflow on the ETF case study. The per-case-study notebooks apply both regression and classification pipelines across all nine datasets.

## 11.4 Interpreting models with SHAP

A model that predicts returns but cannot explain why offers limited value. Understanding which features drive predictions (and whether those relationships make economic sense) separates robust signals from spurious curve-fitting. **SHapley Additive exPlanations** (SHAP) is the primary interpretability framework, embedded in the walk-forward pipeline as a continuous diagnostic.

The case for interpretability rests on three pillars:

- **Diagnosis**: When a model fails in production, you need to know why. A Ridge model that suddenly underperforms might be responding to a vendor data migration that changed how momentum is computed, or to genuine decay in the momentum premium. SHAP analysis of recent versus historical predictions can distinguish these scenarios.
- **Economic sanity-checking**: A momentum feature should positively contribute to high-momentum stocks; a volatility feature should reflect the expected risk-return relationship. When attributions contradict established theory, the model is more likely to fit noise than capture signal.
- **Governance**: Model risk frameworks such as SR 11-7 (Federal Reserve) and SS1/23 (Bank of England) require that firms understand and justify algorithmic outputs. Even without regulatory obligations, “the model said so” does not explain a drawdown.

For standardized linear models, coefficient magnitudes offer a tempting shortcut, but shrinkage entangles importance with the penalty strength, correlated features split or steal credit depending on the penalty type, and coefficient interpretation does not generalize beyond linear models.

SHAP supplies a unified attribution language: it disentangles regularization distortion, allocates credit fairly among correlated features, and carries forward unchanged to trees (*Chapter 12*) and deep learning (*Chapter 13*).

### Game-theoretic attribution with SHAP

SHAP builds on **Shapley values** from cooperative game theory, which determine each player’s contribution to the outcome. Here, the “players” are features, the “payoff” is the gap between a specific prediction and the model’s average prediction (the baseline), and the question is: how much did each The *Shapley value* for feature 𝑗 averages its marginal contribution across all orderings in which features feature contribute to moving this prediction away from baseline?

could enter the model: |ܵ|! (|ܨ| −|ܵ| −1)! 𝜙௝= ∑ [݂(ܵ׫ ሼ݆ሽ) −݂ (ܵ)] |ܨ|! ௌكி̳ ሼ௝ሽ where 𝐹 is the full feature set, 𝑆 ranges over subsets excluding 𝑗, and 𝑓(ܵ) is the expected prediction when only features in 𝑆 are known.

Shapley (1953) proved that this is the **unique allocation satisfying four axioms**:

- Efficiency: attributions sum exactly to the prediction minus baseline
- Symmetry: identically contributing features receive equal credit
- Null player: irrelevant features receive zero credit
- Linearity: attributions for a sum of models equal the sum of attributions

Lundberg and Lee (2017) unified **LIME**, **DeepLIFT**, and classic Shapley regression within this framework, showing that SHAP is the unique additive attribution method that satisfies all four axioms.

Exact computation requires evaluating 2|ி| subsets, which is intractable for realistic feature sets. Mod- Efficient computation

el-specific algorithms make it practical: **LinearSHAP** exploits the closed form for linear models: 𝜙௝ൌߚ௝ڄ ൫ݔ௝െܧൣݔ௝]), computed over the training set. This is exact and 𝑂(ܨ) per observation, the right choice for this chapter. •

- **TreeSHAP** (*Chapter 12*) and **KernelSHAP** (*Chapter 13*) extend the framework to nonlinear models.

**Implementation**: `05_shap_analysis` shows LinearSHAP end-to-end on the ETF case study, including verification that SHAP values match the closed-form coefficient attribution.

### Practical application in global and local analysis

SHAP supports both global views (feature importance summarized across all predictions) and local views, decomposing the contribution of each feature to a single model score.

#### Global feature importance SHAP summary plots rank features by mean |߶௝| across all test observations. Each point represents the

SHAP value for one observation, colored by its feature value. The plot reveals which features matter most, the direction of effects, and whether importance is consistent or conditional.

In the `etfs` case study, volatility horizons dominate the top of the ranking: six-month volatility (`vol_126d`) contributes most, with a positive feature-to-SHAP relationship, while three-month volatility (`vol_63d`) is the second-most important and pulls in the opposite direction. Risk-adjusted return appears next (`sharpe_63d`, positive), followed by `vol_ratio_medium` and `yield_curve_slope`. variance at informative observations can have a higher mean |߶௝| than a feature with a larger coefficient SHAP importance can differ from coefficient rankings: a feature with a moderate coefficient but high

but lower variance. *Figure 11.3* shows the beeswarm summary for the ETF case study.

![Figure 11.3](assets/figure_11_3.png)

*Figure 11.3: Importance of the top 20 etfs features*

#### Local explanations

**Waterfall plots** decompose a single prediction from baseline to final value, feature by feature. For a high-conviction stock: baseline of 0.5% → momentum adds +0.8% → mean reversion subtracts −0.3% → volatility adds +0.2% → final prediction of 1.4%. This enables two checks:

1. An *attribution audit*: is outperformance driven by trusted features or suspicious ones (an illiquidity proxy reflecting data staleness)?
2. A *concentration-risk flag*: a prediction dominated by a single feature is fragile. of the total |߶௝| could provide an automated alert for manual review or position-size reduction. *Figure* In practice, flagging predictions in which a single feature accounts for more than a certain threshold

*11.4* contrasts a correct high-conviction prediction whose attribution spreads across ten features with an incorrect one in which a single liquidity-proxy feature carries most of the model’s conviction; the right-panel pattern is exactly what the concentration-risk flag is designed to catch.

![Figure 11.4](assets/figure_11_4.png)

*Figure 11.4: SHAP waterfall decomposition of two high-conviction predictions from the `etfs` case study (Ridge, 21-day forward returns). Both predictions sit in the top 5% by predicted magnitude. Panel (a): a correct call where ten features contribute roughly evenly, and the prediction reflects diversified evidence. Panel (b): an incorrect call where `vol_63d` (volume ratio) alone contributes +0.018, yet the realized return was −11.7%*

**Implementation**: `05_shap_analysis` includes waterfall decompositions for both correct and incorrect high-conviction predictions.

#### Feature dependence

SHAP dependence plots show how a feature’s SHAP value varies with its own value, colored by an interacting feature. For linear models, the relationship is linear by construction, but the plot still serves as a sanity check for sign consistency and outlier behavior.

For tree-based and neural network models (*Chapters 12* and *13*), dependence plots reveal threshold effects, saturation, and interaction structures that are invisible in coefficient tables.

### Building an economic narrative

SHAP’s primary value for systematic investing is epistemic: it provides a structured protocol for distinguishing genuine signal from overfitting, organized in four layers:

- **Sign consistency**: does each feature’s contribution have the expected sign? A model that has learned the opposite of every documented factor relationship is almost certainly fitting noise.
- **Magnitude plausibility**: Does a single feature imply a 5% monthly return contribution in a universe where median monthly returns are 0.8%? If so, the model is extrapolating.
- **Stability across walk-forward windows**: Compute SHAP values per fold and track whether Plotting mean |߶௝| over time produces a “SHAP stability chart,” one of the most informative the same features rank in the top five, maintain consistent signs, and show stable magnitudes.

validation diagnostics (*Figure 11.5*).

- **Regime-conditional analysis**: Partition test data by volatility tercile or regime label (*Chapter 7*) and compute SHAP summaries within each partition. A model that shifts from momentum to mean reversion across regimes may be economically sensible, but only if regime identification is itself robust.

A refinement: fold-to-fold variation in SHAP rankings conflates temporal regime shifts (interesting) with finite-sample estimation noise (uninteresting). Bootstrapping SHAP values within a single fold provides confidence intervals on feature importance, separating the two sources.

![Figure 11.5](assets/figure_11_5.png)

*Figure 11.5: SHAP stability across folds*

A complementary diagnostic compares SHAP **profiles for** **high-conviction predictions** that were direcpredictions (measured by the ratio of mean |߶௝| for incorrect to correct cases, or by large directional tionally correct versus those that were incorrect. Features where the model relied heavily on wrong

differences in mean signed SHAP) are candidates for re-engineering or nonlinear modeling. This right-versus-wrong decomposition targets the predictions that matter most for PnL.

### Limitations and failure modes

*Causal misinterpretation* is a major risk. A high SHAP value for feature *j* means it contributed to *this model’s prediction*, not that it caused the return. A feature such as `obv_zscore_63d` with high SHAP importance could reflect genuine predictive power, correlation with an omitted predictor, or a pipeline confound. Kumar et al. (2020) formalize how Shapley-based explanations can mislead when features are correlated or when they are interpreted causally. Use SHAP for descriptive attribution (“what is the model responding to?”), not causal claims. *Feature dependence* and *impossible coalitions* are also significant issues. Standard SHAP marginalizes out absent features independently, creating impossible feature combinations when features are correlated. For linear models, this is manageable; for nonlinear models, Aas et al. (2021) propose **conditional SHAP**, which requires estimating conditional distributions. **Permutation importance** and LIME lack SHAP’s axiomatic guarantees: permutation importance inflates scores for correlated features; LIME attributions need not sum to the prediction.

### Integration with the walk-forward pipeline

The SHAP stability chart across folds serves as a model-selection tool alongside IC and coverage metrics, The SHAP stability chart in `05_shap_analysis` tracks mean |߶௝| across all walk-forward folds for the providing qualitative evidence that the model has learned stable, economically sensible relationships.

top features. Models that pass quantitative hurdles yet exhibit unstable or economically implausible SHAP patterns warrant skepticism, regardless of their reported performance.

**Implementation**: `05_shap_analysis` implements SHAP computation within the walk-forward framework, including summary plots, waterfall decompositions, and right-versuswrong diagnostic comparisons for high-conviction predictions. *Section 11.5* turns from explaining individual predictions to bounding them: how to attach calibrated uncertainty to each forecast.

## 11.5 Quantifying predictive uncertainty

A point forecast is incomplete because it suppresses information about the reliability of the estimate. In trading, this matters directly: the same predicted return can imply very different decisions depending on how uncertain the forecast is.

Consider a Ridge regression model that predicts a 2% return for two ETFs. For the first ETF, the current feature values lie near the center of the training distribution. For the second, volatility has spiked, liquidity has deteriorated, and several features sit near their historical extremes. The point forecast is the same, but the second prediction is much less reliable. Treating both forecasts as equivalent inputs to portfolio construction would overstate the precision of the second signal. intervals of the form ݕොേݖఈȀଶߪො assume a useful residual scale estimate and often rely, explicitly or Classical **prediction intervals** often rely on assumptions that are fragile in financial data. Gaussian

implicitly, on approximately normal, homoskedastic errors. Asset returns instead exhibit heavy tails, volatility clustering, nonlinear residual dispersion, and regime dependence. The uncertainty around a forecast is rarely constant across assets, horizons, or market states.

**Conformal prediction** offers a model-agnostic alternative. It acts as a calibration layer placed on top of an existing forecasting model. After fitting the base model, we evaluate its forecast errors on a held-out calibration set and use the empirical distribution of those errors to construct prediction intervals or prediction sets for future observations. Because the procedure uses observed forecast errors rather than a parametric error model, it does not require normal residuals, homoskedasticity, or a correctly specified likelihood. The key assumption behind the formal guarantee is **exchangeability**. A sequence of observations is exchangeable if its joint distribution is unchanged by reordering the observations. Independent and identically distributed observations are exchangeable, but exchangeability is weaker than independence. This assumption matters because conformal prediction relies on the rank of the next observation’s calibration score among the scores already observed. If the calibration scores and the next test score are exchangeable, the future score is equally likely to occupy any rank, which yields finite-sample marginal coverage.

Under exchangeability, conformal prediction delivers a finite-sample **marginal coverage guarantee**. For example, a 90% conformal interval will contain the realized outcome at least 90% of the time over repeated samples. This guarantee is distribution-free: it does not depend on the correctness of the base model, the shape of the return distribution, or the sample size. It is, however, a marginal guarantee. It does not imply that coverage is exactly 90% within every asset, fold, volatility regime, or predicted-return bucket.

Financial time series are not exchangeable in the strict sense because order matters. Autocorrelation, volatility clustering, changing liquidity, and structural breaks mean that calibration residuals from one period may not represent those in the next. Conformal prediction remains useful in this setting, but the theorem no longer mechanically justifies coverage. In financial applications, conformal intervals should therefore be treated as empirically calibrated uncertainty estimates whose coverage must be monitored, stress-tested, and adapted over time.

### Split-conformal prediction

The simplest practical variant is split-conformal prediction (Papadopoulos et al., 2002; Lei et al., 2018). It separates model fitting from uncertainty calibration:

1. **Partition the data.** Divide the available training observations into a proper training set and a calibration set. The proper training set fits the model and selects any hyperparameters. The calibration set must remain untouched until the fitted model is evaluated for uncertainty **Compute calibration scores.** For each calibration observation 𝑖, compute a nonconformity score. calibration. For a symmetric regression interval, the usual score is the absolute residual:ݏ௜ൌ|ݕ௜െ݂̂ (ݔ௜)|. 2.

These scores estimate the empirical distribution of the model’s out-of-sample forecast errors. **Compute the conformal quantile.** Let 𝑛cal be the number of calibration observations, and let:

1. 𝑠(1) ≤𝑠(2) ≤⋯≤𝑠(௡cal)

denote the sorted calibration scores. For target miscoverage rate 𝛼, define:

𝑘( cal + 1)(1 െߙ)⌉

The conformal correction is the conservative order statistic ݍොൌݏ(௞). This corresponds to using ൐݊

the higher empirical quantile rather than an interpolated quantile. If cal, the calibration set is too small to provide a finite conformal quantile at the desired coverage level. **Construct prediction intervals.** For a new observation 𝑥, the split-conformal interval is:

4.

![Figure 11.6](assets/figure_11_6.png)

This interval has fixed width ʹݍො for every observation. That simplicity is useful: split conformal is

easy to implement, easy to audit, and robust to misspecification of the base model. Its limitation is equally clear: it cannot vary the interval width across market regimes, assets, or feature configurations. A forecast made during a volatility spike receives the same residual correction as one made during a quiet regime.

Under exchangeability of the calibration observations and the next test observation, split-conformal prediction provides **finite-sample marginal coverage**:

![Figure 11.7](assets/figure_11_7.png)

This guarantee holds for any fitted base model and any outcome distribution, provided the calibration set was not used to fit or tune the model whose scores are being calibrated. The guarantee is marginal: it averages over the joint distribution of future observations. It does not ensure correct coverage conditional on a particular date, asset, volatility state, sector, or signal-strength bucket.

**Implementation**: `06_conformal_prediction` implements split-conformal prediction with Ridge regression on the ETF case study.

#### The exchangeability problem

The exchangeability assumption is fragile in financial time series. The order of observations carries information: volatility clusters, correlations change, liquidity varies, and regimes shift. Randomly permuting daily returns or cross-sectional residuals would destroy part of the data-generating structure. As a result, the rank argument behind the conformal guarantee no longer applies exactly. Yesterday’s calibration residuals may not be representative of tomorrow’s forecast errors.

This does not make conformal prediction unusable. It changes how the method should be interpreted. In finance, conformal prediction is best viewed as a disciplined empirical calibration procedure whose formal guarantee is exact under exchangeability and approximate under sufficiently stable nonexchangeable conditions. Coverage should therefore be measured out of sample, monitored by regime, and updated when the calibration distribution becomes stale.

Several adaptations are useful in practice: **Rolling calibration windows.** Instead of using all historical calibration scores, compute 𝑞 from

• a recent window, such as the most recent 250 trading days. This makes the score distribution more responsive to current volatility and liquidity conditions. The trade-off is higher estimation noise because fewer calibration observations are used.

- **Weighted conformal calibration.** Assign larger weights to more recent or more relevant calibration observations. Exponential time decay is a practical heuristic for nonstationary time series. More formal weighted conformal methods are justified under specific distributional shift assumptions, such as covariate shift, when the relative likelihoods of calibration and test covariates can be estimated.
- **Regime-conditional calibration.** Partition the calibration scores by pre-specified market states, such as realized-volatility quantiles, VIX terciles, liquidity buckets, or macro regimes, and compute separate conformal corrections within each state. This can improve conditional coverage when residual dispersion differs sharply across regimes. The partition should be defined ex ante, and each regime must contain enough calibration observations to estimate a stable quantile.
- **Online coverage feedback.** Adaptive conformal methods monitor realized coverage as outcomes arrive and adjust the effective miscoverage rate or calibration threshold. Recent misses widen subsequent intervals; persistent overcoverage narrows them. This provides a practical feedback mechanism when the historical calibration distribution no longer matches current conditions.

These adaptations do not fully restore the original exchangeability theorem. They make the calibration layer more responsive to the types of drift and dependence that matter in financial forecasting.

### Adaptive conformal inference

for online, nonstationary environments. Rather than fixing the miscoverage level 𝛼 once, ACI updates **Adaptive Conformal Inference** (ACI) (Gibbs and Candès, 2021, 2023) modifies conformal calibration an effective miscoverage parameter 𝛼௧ after each realized outcome.

At time 𝑡, the model constructs a conformal set 𝐶௧( ௧) using the current value 𝛼௧. After observing 𝑌௧, the

update is:

![Figure 11.8](assets/figure_11_8.png)

where 𝛼 is the target miscoverage rate and 𝛾 is the learning rate.

The sign of the update is intuitive. If the interval misses, then ૚ሼܻ௧ב ܥ௧( ௧)} = 1, so 𝛼௧ାଵ decreases. A

zero, so 𝛼௧ାଵ increases and the next interval narrows. Undercoverage triggers expansion; persistent lower miscoverage level produces a wider next interval. If the interval covers, then the indicator equals

overcoverage triggers contraction.

**ACI provides a long-run coverage control mechanism for nonstationary sequences**. Its guarantee is weaker than the finite-sample marginal guarantee of split conformal under exchangeability: it targets average empirical coverage over time rather than exact coverage for each time point, subgroup, or regime. This distinction is important in trading. ACI can help the interval sequence recover after drift, but it does not eliminate the need to inspect coverage by fold, volatility regime, asset group, and The *learning rate* 𝛾 controls the speed-stability trade-off. Larger values adapt faster after misses but signal bucket.

slowly to abrupt changes in volatility. TheT notebook tunes values such as 𝛾[0.005,0.05] within the can make interval widths oscillate. Smaller values produce smoother intervals but may respond too walk-forward framework. In implementation, the adaptive miscoverage level 𝛼௧ should be clipped to a bounded range, such as [0.001,0.999], to avoid degenerate intervals. 𝛼௧ trajectory, and the effect of adaptive feedback on coverage and interval width. **Implementation**: `06_conformal_prediction` shows the ACI online update, the resulting

### Conformalized quantile regression

Split-conformal intervals have constant width, which is often unrealistic in financial data. **Conformalized Quantile Regression** (CQR; Romano et al., 2019) addresses this limitation by combining conditional quantile models with conformal calibration. For target coverage ͳ െߙ, CQR proceeds as follows:

1. Train lower and upper quantile models. On the proper training set, estimate conditional quantiles:

![Figure 11.9](assets/figure_11_9.png)

For 90% coverage, these correspond to the 5th and 95th conditional quantiles.

1. Compute CQR calibration scores. On the calibration set, compute:

![Figure 11.10](assets/figure_11_10.png)

A positive score indicates that the realized outcome fell outside the estimated quantile band. A negative score indicates that the outcome fell inside the band, with room to spare. vative conformal order statistic 𝑞CQR used in split conformal.

1. Compute the conformal correction. Sort the calibration scores and compute the same conser-Construct calibrated intervals. For a new observation 𝑥, define:

4.

![Figure 11.11](assets/figure_11_11.png)

CQR retains the adaptive shape of the quantile model: intervals widen where the model predicts greater conditional dispersion and narrow where the model predicts lower uncertainty. The conformal correction then calibrates the empirical coverage of these intervals. Unlike split conformal, the interval width can vary across assets, dates, regimes, and feature configurations.

**CQR is often preferable when residual dispersion is visibly heterogeneous**, and enough calibration data is available. For cross-sectional financial models with many assets and regime-dependent uncertainty, it is usually more informative than a fixed-width split conformal. However, CQR depends on the quality of the underlying quantile model. If the quantile estimates are unstable, the calibration sample is small, or the regime shifts sharply, rolling split conformal or ACI may be more robust. This uncertainty estimate also has a direct **interpretation in terms of allocation**. A simple preview is to scale the position size inversely with the interval width: 1 𝑤௜∝ ൫ܥ(ݔ௜))

so that forecasts with tighter calibrated intervals receive larger allocations, all else equal. A stock with a 50 basis point interval receives twice the uncertainty-adjusted weight of one with a 100 basis point interval before applying volatility, correlation, turnover, and portfolio-level constraints. *Chapter 17* develops this idea within a full-allocation framework.

**Implementation**: `06_conformal_prediction` implements CQR with native quantile models and compares its interval-width distribution with split conformal. Ridge-based results should be interpreted as a baseline comparison unless the model explicitly estimates lower and upper conditional quantiles.

### Conformal prediction sets for classification

Conformal prediction also applies to classification. Instead of producing a prediction interval for a probability at least ͳ െߙ under exchangeability. continuous return, it produces a prediction set: a subset of labels that contains the true class with For a classifier with predicted class probabilities 𝑝(ݕפ ݔ), a simple nonconformity score is:

𝑠௜ൌͳ െ݌Ƹ(ݕ௜פ ݔ௜)

where 𝑦௜ is the true class for calibration observation 𝑖. Large scores indicate that the model assigned low probability to the realized class. After computing the conformal quantile 𝑞 from the calibration

scores, the prediction set for a new observation is: 𝐶(ݔ) ൌሼݕǣ ͳ െ݌Ƹ(ݕפ ݔ) ൑ݍොሽൌሼݕǣ ݌Ƹ(ݕפ ݔ) ൒ͳ െݍොሽ

tion labels such as long, neutral, and short, a singleton set such as {long} conveys greater directional The result is a set of plausible labels rather than a single forced classification. For three-class direcconfidence than a wider set such as {long,neutral}. A full set containing all classes indicates that the

classifier cannot distinguish the alternatives at the desired coverage level.

This maps naturally to trading decisions. One conservative rule is to trade only singleton prediction sets and abstain when the conformal set contains multiple labels. This can reduce exposure when the model is uncertain about direction. However, the conformal guarantee applies to the prediction sets over all evaluated observations, not automatically to the subset of observations selected for trading. Coverage, hit rate, abstention rate, turnover, and realized performance should therefore be reported separately for singleton and non-singleton cases.

Conformal classification calibrates set coverage, not the probability estimates themselves. Poorly calibrated probabilities can still produce valid conformal sets under the required assumptions, but inefficient probability estimates may lead to unnecessarily large sets. Probability calibration and conformal set calibration are related diagnostics, but they are not the same object.

### Evaluating calibration quality

A conformal method should be evaluated by whether its realized coverage matches its claimed coverage and whether the resulting intervals are useful for decision-making. A nominal 90% interval that covers only 70% of realized outcomes is overconfident: it claims a 10% failure rate but delivers a 30% failure rate. Conversely, a method that covers 99% of outcomes at a 90% target may be too conservative, producing intervals too wide to support meaningful allocation decisions.

A calibration plot summarizes this relationship. For target coverage levels: 1 െߙא ሼ0.10,0.20, … ,0.90}

plot nominal coverage against empirical coverage. Points below the diagonal indicate overconfidence; points above the diagonal indicate conservatism.

![Figure 11.12](assets/figure_11_12.png)

*Figure 11.6: Conformal prediction calibration*

Marginal coverage can hide failures that matter for trading. Conditional coverage should therefore be evaluated by stratifying observations by:

- **Walk-forward fold**, to test whether coverage is stable through time
- **Predicted return magnitude**, to test whether intervals fail in the signal tails
- **Market regime**, to test whether coverage deteriorates during high volatility or low liquidity
- **Asset characteristics**, to test whether coverage varies systematically by sector, liquidity, volatility, or instrument type

Coverage estimates should be reported with uncertainty. A realized coverage rate of 89.9% may be statistically indistinguishable from a 90% target, while 88.4% may or may not be material depending on the effective sample size, serial dependence, and cross-sectional clustering. Fold-level dispersion, block-bootstrap confidence intervals, or cluster-robust standard errors make the comparison more informative than a single pooled number. On the `etfs` case study, conformal prediction is evaluated using daily data for 99 ETFs over 2006–2025, with out-of-sample evaluation across eight walk-forward folds spanning 2015–2023. Within each fold, the model uses an 80/20 proper-training/calibration split. The base model is Ridge regression. **Width** 𝜎**(%)**

| Method | Target | Marginal Coverage | Mean Width (%) | Width 𝜎 (%) |
| --- | --- | --- | --- | --- |
| Split-Conformal | 90% | 88.4% | 16.9% | 3.3 |
| CQR | 90% | 89.9% | 15.7% | 7.5 |
| ACI on SC base (𝛾01) | 90% | 89.5% | 16.3% | 6.8 |

*Table 11.1: Conformal evaluation results*

The results illustrate the trade-offs among the three methods:

- Split conformal is simple and auditable but produces fixed-width intervals and undercovers relative to the 90% target
- CQR comes closest to the target and produces the narrowest average interval, while allowing interval width to vary substantially across observations
- ACI improves the static split-conformal baseline by adapting interval width over time, though its realized coverage remains slightly below target in this run

The width dispersion is also informative. Split conformal has relatively low width variation because all observations in a fold receive the same conformal correction. CQR has higher width dispersion because its quantile models adapt to the conditional uncertainty of each observation. ACI also produces more variable widths because recent coverage errors feed back into the effective miscoverage level.

These results are consistent with the challenges posed by nonexchangeable financial data, but they should not be mechanically attributed solely to exchangeability violations. Undercoverage can also arise from calibration leakage, overly small calibration windows, unstable quantile estimates, inappropriate score definitions, or implementation choices such as interpolated quantiles. The correct diagnostic response is to inspect coverage by fold, regime, and asset group before changing the method.

Several warning signs deserve attention. Coverage below 80% during volatility spikes, for a 90% target, indicates that the calibration distribution is stale or poorly stratified. Monotonically deteriorating coverage over the backtest suggests drift in the calibration scheme’s tracking. Asymmetric violations, with many more outcomes above or below the interval, indicate a directional bias or residual skewness. Symmetric split conformal can widen intervals but cannot recenter a biased forecast. CQR can represent asymmetric conditional tails if the quantile models learn them, but the conformal correction itself calibrates coverage rather than correcting the predictive center.

**Implementation**: `06_conformal_prediction` shows the full calibration comparison, including marginal coverage, per-fold coverage, interval-width distributions, conditional coverage diagnostics, and ACI adaptation dynamics.

## 11.6 Case study insights

One regularized linear pipeline, run unchanged across all case studies, lets us ask a single question nine times: when does a linear model turn off-the-shelf features into a useful ranking of forward returns, and when does it not? Most of the time it does not. The linear model is the baseline the more flexible families in *Chapters 12* through *14* have to beat, so it matters as much where it fails as where it succeeds. Throughout, the measure of success is the average daily information coefficient (IC) between the predicted score and the realized return, read together with its 95% confidence interval (CI); all *t*-statistics use the Newey–West (HAC) correction. The full evidence is in `07_case_study_insights`.

The signal is real in three of the nine case studies and absent in the other six. *Table 11.4* sorts the case studies by whether the confidence interval around their IC clears zero. Three intervals exclude zero, three are positive but wide enough to contain it, and three sit below it. Reliable cross-sectional ranking from a linear model on engineered features is therefore the exception rather than the rule, even before any of the frictions of trading enter the picture.

| Category | Count | Case studies |
| --- | --- | --- |
| CI excludes zero | 3 | ETFs, US Equities, NASDAQ-100 |
| CI overlaps zero (positive) | 3 | Crypto, SP 500 Options, FX |
| CI overlaps zero (negative) | 3 | CME Futures, SP 500 Eq+Opt, US Firms |

*Table 11.4: Case studies by whether the regression IC confidence interval excludes zero*

All nine case studies are scored this way on a regression label. Four of them also carry a direction label, which is a classification problem scored differently; those results are held back to the dedicated comparison near the end of the section, so that the regression picture stays clean first.

### Where the signal is

ETFs, US equities, and intraday NASDAQ-100 are the three case studies whose linear models rank returns with an IC that is reliably positive. *Table 11.5* lists the best configuration, and its IC for each case study at its primary label, and *Figure 11.7* plots the same estimates so that the intervals clearing zero are visible at a glance. The prose below reads the pattern; the numbers live in the table.

| Case study | Horizon | IC | t | 95% CI | n |
| --- | --- | --- | --- | --- | --- |
| ETFs | 21 days | +0.054 | 2.4 | [+0.009, +0.098] | 2,016 |
| US Equities | 1 day | +0.016 | 9.1 | [+0.012, +0.019] | 4,018 |
| NASDAQ-100 | 15 min | +0.005 | 4.3 | [+0.003, +0.007] | 64,580 |
| Crypto | 8 hours | +0.009 | 1.4 | [−0.003, +0.020] | 1,956 |
| SP 500 Options | ~30 days | +0.007 | 0.6 | [−0.015, +0.029] | 502 |
| FX | 1 day | +0.005 | 0.6 | [−0.012, +0.022] | 2,064 |
| CME Futures | 5 days | −0.000 | −0.0 | [−0.034, +0.034] | 1,290 |
| SP 500 Eq+Opt | 5 days | −0.006 | −0.4 | [−0.033, +0.022] | 502 |
| US Firms | 1 month | −0.005 | −0.6 | [−0.022, +0.011] | 120 |

*Table 11.5: Best linear configuration for the primary regression label measured on the validation set*

The interval is the HAC 95% CI; n is the number of trading periods over which the daily IC is averaged. The labels correspond to the total forward return over the given horizon, except for S&P 500 options, where the label is the per-position profit and loss of an at-the-money short straddle held from formation to expiry (25–35 days), following the hold-to-maturity construction of O’Donovan and Yu (2024). See case study `README` for more details.

![Figure 11.13](assets/figure_11_13.png)

*Figure 11.7: Best daily IC per case study at the primary regression label, with HAC 95% CI. Filled markers mark intervals that exclude zero; open markers mark intervals that overlap it*

Two things separate these three from the rest. The first is that their features genuinely align with the cross-section of returns: momentum and value carry ranking information in equities and ETFs in ways that engineered features do not for currencies or futures. The second is statistical, not economic. Whether an IC clears zero depends as much on how many independent observations support it as on its size and variance. NASDAQ-100 reaches significance on the smallest IC of the three because its intraday history is large. Sample size, not effect size, is doing much of the work, which is worth remembering before reading a small but significant IC as a strong signal; both statistical and economic significance matter.

### Choice of regularization

Each linear model is fit over a grid of regularization strengths, with the strength chosen by cross-validation, under three penalties: Ridge, which shrinks every coefficient toward zero; LASSO, which drives some coefficients to exactly zero; and Elastic Net, which blends the two. All three are fit on seven of the nine case studies. Ridge gives the highest IC on every case study where the linear model has measurable signal. LASSO and Elastic Net match it elsewhere, and edge it on CME Futures, but the IC there is indistinguishable from zero under any penalty, so the gap is noise rather than a finding. Ridge holds up because of how the features are built: momentum at several horizons, volatility at several scales, and cross-sectional ranks are correlated by construction, so a penalty that keeps all of them and shrinks them together discards less information than one that zeroes some out. The gain from regularizing at all, relative to an unpenalized fit, is small, and it is largest exactly where the unpenalized fit lands far from the best strength. Regularization here buys numerical stability; it does not manufacture signal the features do not carry.

Ridge, LASSO, and Elastic Net are all fit on seven of the nine case studies. NASDAQ-100 and US Equities are represented by their best Ridge fit in *Table 11.5*; LASSO and Elastic Net add nothing to the comparison on those two and are not reported.

### Whether the signal persists

A positive IC averaged over the whole validation period can still come from a few good stretches, so the more demanding question is whether the signal holds up over time. The clearest way to see this is a rolling three-month average of the daily IC, which traces how ranking quality evolves rather than collapsing it to a single number. Because each case study covers a different validation window, the view is most informative for case studies with overlapping periods. *Figure 11.8* shows ETFs and FX over the same 2016–2023 span: the ETF signal spends most of the period above zero, with sharp but temporary drawdowns, while the FX signal is split almost evenly across zero with no persistent sign. The ETF edge is a property of the whole period; the FX entry in *Table 11.5* averages a series that is positive and negative in roughly equal measure, which is why its interval includes zero.

![Figure 11.14](assets/figure_11_14.png)

*Figure 11.8: Three-month rolling average of the daily IC for the ETF and FX case studies over their common 2016–2023 validation window. The ETF series holds a positive level; the FX series crosses zero repeatedly*

A second, simpler check guards against the opposite error: reading a near-zero IC as the symptom of an unstable fit rather than a genuine absence of signal. Across folds, the model keeps the sign of its coefficients in the vast majority of cases for each case study, even when the IC overlaps zero. The fits are stable; what the flat case studies lack is ranking content in the features, not a settled model.

### Horizon effects

The horizon at which a label is defined affects how much signal a linear model can extract, and the effect varies by case study (*Figure 11.9*). For ETFs, US equities, and SP 500 Eq+Opt the IC rises with horizon, as the day-to-day noise averages out and a slower cross-sectional signal shows through. NAS-DAQ-100 is flat across its intraday horizons: a linear read of microstructure features carries the same ranking content at five, fifteen, and sixty minutes. Crypto behaves the same way across its eight- and twenty-four-hour labels, consistent with a funding-rate signal that does not decay over a day.

![Figure 11.15](assets/figure_11_15.png)

*Figure 11.9: Best daily IC per case study and regression label, with HAC 95% CI bands, on a log-horizon axis. Only case studies with two or more regression labels appear*

The practical reading is to match the horizon to the signal: where information accumulates slowly, a longer holding period yields a higher IC; where it lives inside the window, a shorter horizon captures as much. A single model trained jointly on several horizons can pool this structure, an idea *Chapter 14* develops in latent-factor form.

### Direction versus magnitude

Predicting the sign of a return and ranking its size are different problems, and a model can be good at one while indifferent to the other. Three case studies (Crypto, SP 500 Eq+Opt, and US Firms) train both a regression model on continuous returns and a classifier on binary direction over the same horizon, allowing us to score each model on both targets. The classifier is read by its AUC against the direction; the regression model by its IC against the return; and each is also cross-evaluated on the other’s target (*Table 11.6*).

| Case study | Horizon | Native AUC | Cross-IC (t) | Cross-AUC (regression) |
| --- | --- | --- | --- | --- |
| Crypto | 8 hours | 0.509 [0.498, 0.521] | +0.033 (5.3) | 0.498 |
| US Firms | 1 month | 0.539 [0.528, 0.550] | +0.074 (6.8) | 0.494 |
| SP 500 Eq+Opt | 5 days | 0.510 | overlaps zero | 0.496 |
| SP 500 Eq+Opt | 10 days | 0.507 | overlaps zero | 0.523 |

*Table 11.6: Classifier and regression scores at matched horizons. Native AUC scores the classifier on direction; cross-IC scores the classifier on the continuous return; cross-AUC scores the regression model on direction*

The two targets do not track each other. On Crypto and US Firms the classifier’s score ranks continuous returns well, yet only on US Firms does its AUC separate direction from chance with any confidence. Running the comparison the other way, the regression score barely moves AUC away from one-half even where it ranks returns. The lesson is not that one target is better, but that direction and magnitude encode different information, so choosing which to predict is a modeling decision rather than a formality.

### From IC to profitability

A higher IC does not guarantee a more profitable strategy because realized returns also depend on turnover, transaction costs, and position sizing. `08_ml_backtest_intro` makes the gap concrete on the ETF case study, where a simple 126-day momentum signal carries a lower IC than Ridge yet matches or beats it on net Sharpe ratio, because the Ridge portfolio trades far more to act on its scores. Turning a ranking into a return is the subject of *Chapters 16* through *19*: strategy simulation, portfolio construction under turnover constraints, transaction-cost modeling, and risk management.

## 11.7 Summary

This chapter established the end-to-end ML pipeline for translating alpha factors into trading signals. Regularized regression (Ridge, LASSO, and Elastic Net) provides the first rung of the baseline ladder: interpretable models whose coefficients reveal which features the model relies on and how strongly. Logistic regression extends the framework to direction prediction, offering an alternative when continuous return signals are weak. SHAP attributions verify that the model’s reasoning aligns with economic intuition and remains stable across walk-forward folds. Conformal prediction wraps any base model in calibrated uncertainty bands, enabling position sizing that reflects prediction confidence rather than treating all forecasts as equally reliable. The cross-dataset evaluation in *Section 11.6* reveals where linear models work and where they do not. These results define the baseline for *Chapter 12* (gradient boosting), *Chapter 13* (deep learning), and *Chapter 14* (latent factors), which test whether nonlinear models can extract signal on the datasets where the linear baseline runs out.
