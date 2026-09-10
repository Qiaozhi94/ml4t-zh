# Chapter 12: Advanced Models for Tabular Data

**Gradient Boosting Machines** (**GBMs**) dominate tabular prediction in both academic research and production trading systems, combining predictive power with computational efficiency. While *Chapter 11*’s linear models assume additive feature relationships, GBMs capture the interaction effects and threshold dynamics that financial data exhibit, without manual feature engineering.

This chapter covers the three industry-standard GBM libraries (XGBoost, LightGBM, and CatBoost) and the techniques that make them production-ready. By the end of this chapter, you will be able to:

- Explain how boosting differs from bagging and why sequential error correction makes GBMs effective for financial tabular prediction.
- Select among XGBoost, LightGBM, and CatBoost based on categorical structure, compute environment, latency needs, and leakage risk.
- Choose appropriate GBM objectives and constraints for financial tasks, including pointwise regression, learning to rank, and monotonic constraints.
- Tune GBMs efficiently with Optuna using pruning, multi-objective search, and time-seriesaware validation.
- Use TreeSHAP to analyze feature effects, interactions, instability, and drift in deployed treebased models.
- Evaluate when tabular deep learning alternatives such as TabPFN, TabM, and TabR are worth considering relative to GBMs.
- Interpret cross-case-study evidence to decide when nonlinear tree models earn their added complexity relative to linear baselines

We progress from the shared boosting framework through Bayesian hyperparameter optimization with Optuna, TreeSHAP-based interpretation and drift monitoring, and learning-to-rank objectives for quantile strategies. We then evaluate deep learning alternatives (TabPFN and TabM) using decision criteria for when added complexity pays off, and conclude with a cross-case study comparison that benchmarks GBMs against the linear baselines from *Chapter 11*.

## 12.1 From decision trees to ensembles

Gradient boosting rests on three building blocks: how decision trees partition feature space, how Random Forests reduce variance by averaging independent trees, and why averaging alone cannot correct systematic bias.

A **decision tree** makes predictions by recursively partitioning the feature space into regions. At each node, the algorithm selects the feature and threshold that best separate the data, typically using information gain for classification or variance reduction for regression.

Decision trees handle mixed feature types (continuous returns, categorical sectors) without encoding, capture non-linear relationships through hierarchical splits, and need no feature scaling. Their critical weakness is high variance: a small change in training data can produce a dramatically different tree structure, making individual predictions unstable. Yet individual trees can learn that momentum is predictive only when volatility is below a threshold, an interaction that linear models require explicit engineering to capture. The question is how to harvest this flexibility without the instability.

### Random forests – Reducing variance with bagging

**Random forests** (Breiman, 2001) apply the bagging principle to decision trees with an additional twist: at each split, only a random subset of features is considered. This feature randomization decorrelates the trees, making the variance reduction from averaging more effective.

For a regression task, a Random Forest prediction is simply the average of all individual tree predictions: 𝑦1 ெ ܯ෍ܶ (ݔ) ௠ ௠ୀଵ where 𝑀 is the number of trees and 𝑇௠(ݔ) is the prediction from tree 𝑚.

Random forests require minimal hyperparameter tuning and rarely overfit, even with many trees. In the model comparison notebooks for this chapter, Random Forests serve as a baseline whose accuracy GBMs must materially improve upon to justify their additional complexity. *Figure 12.1* illustrates the contrast between bagging (averaging independent trees) and boosting (sequential error correction).

![Figure 12.1](assets/figure_12_1.png)

*Figure 12.1: Bagging versus boosting*

However, because trees are trained independently, they cannot correct each other’s systematic errors. If all trees share a consistent bias (for example, underestimating extreme returns), averaging does not help. This limitation motivates the boosting paradigm covered next.

### The path to boosting

Boosting builds models sequentially, so each new model targets the errors of the current ensemble. If the current prediction is 2% but the true return is 5%, the next tree learns to predict the 3% residual. Adding that correction to the ensemble gradually eliminates systematic under-prediction, the bias that averaging independent trees cannot address.

This sequential error correction, combined with shrinkage and regularization, forms the foundation of gradient boosting machines, which we detail in the next section. The notebook `01_ensemble_foundations` benchmarks Random Forests against XGBoost, LightGBM, and CatBoost on the Chen–Pelger–Zhu firm characteristics dataset (414K train/307K valid/497K test observations across 1967–2016) and finds a consistent ordering on this benchmark: each GBM achieves higher out-of-sample test IC than the Random Forest baseline (0.058–0.060 versus 0.056), with library-to-library differences (~0.002 IC) smaller than the boosting-versus-bagging gap.

## 12.2 Gradient boosting machines

The three dominant GBM libraries (XGBoost, LightGBM, and CatBoost) share the core boosting algorithm but differ in tree construction, categorical handling, and regularization. This section covers the shared framework, then examines each library’s distinctive innovations.

### High performance with boosting

Gradient boosting, formalized by Friedman (2001), frames model training as iterative optimization in function space. Rather than fitting a single complex model, we construct an additive ensemble in which each new component explicitly targets the errors in the current prediction.

The algorithm proceeds as follows: ൌͳǡʹǡ ǥ ǡ ܯ:

1. Initialize with a constant prediction (typically the target mean).
2. For each boosting iteration

a. Compute the negative gradient of the loss function with respect to the current predictions. Add the new tree’s predictions to the ensemble, scaled by a learning rate 𝜂. b. Fit a weak learner (typically a shallow tree) to these “pseudo-residuals.” c.

The update at each iteration is: 𝐹௠(ݔ) = 𝐹௠ିଵ(ݔ) ൅ߟڄ݄ ௠(ݔ)

where ℎ௠ is the tree fitted in iteration 𝑚. The final GBM prediction after 𝑀 boosting rounds can be

written as: ெ ݕොൌܨெ(ݔ) ൌܨ0(ݔ) ൅ߟ෍݄ (ݔ) ௠ ௠ୀଵ where 𝐹0(ݔ) is the initial constant prediction, often the training-set mean for squared-error regression, and each ℎ௠(ݔ) is a shallow tree fitted to the loss gradient implied by the current ensemble. This

expression highlights the key difference from the Random Forest prediction above. A Random Forest adds targeted corrections. Each tree is therefore not a standalone forecast of 𝑦, but an incremental averages independently trained trees, whereas a GBM starts with a baseline prediction and iteratively

adjustment to the current prediction. For classification, the same additive structure applies to the model’s raw score scale, such as log-odds for binary classification, before applying the appropriate The **learning rate** 𝜂 (typically 0.01–0.3) controls how aggressively each tree’s contribution is added. link function to obtain probabilities.

Smaller values require more trees but often yield better generalization: a GBM with 100 trees is not averaging 100 predictions but accumulating 100 incremental corrections.

The **choice of loss function** determines what “errors” the algorithm targets. In *Chapter 11*, we discussed mean squared error, mean absolute error, and Huber loss:

- **Quantile loss** predicts specific quantiles rather than the conditional mean, enabling direct estimation of prediction intervals.
- All three GBM libraries support **custom objectives**, enabling asymmetric losses (for example, penalizing under-prediction more than over-prediction) as long as you can provide stable first and second derivatives with respect to the model’s predictions (or use a smooth approximation when the loss is non-smooth).

The next section explains why this blend of flexibility and regularization gives GBMs such a strong track record on tabular data.

### Why GBMs excel on tabular data

Gradient boosted decision trees have an **inductive bias** well-matched to financial tabular data:

- Financial datasets mix continuous variables (returns, volatility), categorical features (sector, exchange), and derived indicators. Tree-based methods accommodate **heterogeneous tabular inputs** well: continuous variables can be split by thresholds, while categorical variables are handled through library-specific encodings or native categorical split mechanisms. The details differ materially across libraries.
- Financial relationships often exhibit **discontinuities** (a P/E ratio above a threshold may trigger fundamentally different investor behavior than just below it), and trees capture these non-smooth boundaries directly. Unlike neural networks that impose smoothness through continuous activation functions, trees partition the feature space into discrete regions. Trees also discover feature interactions through hierarchical splits: a sufficiently deep tree learns that “high momentum is predictive only when volatility is moderate” without manual interaction engineering.

Several practical advantages compound this expressive power:

- Boosted trees rely on **threshold splits**, so they are typically more robust to extreme feature values than models that depend directly on magnitudes or distances
- Modern GBM implementations **handle missing values** by learning the optimal split direction for missing features, rather than requiring imputation
- Trees can also operate on raw feature values without normalization, eliminating a class of preprocessing bugs

The next section examines the most widely used libraries that implement these ideas: XGBoost, LightGBM, and CatBoost.

### Inside the engines – XGBoost, LightGBM, and CatBoost

XGBoost, LightGBM, and CatBoost share the gradient boosting framework but differ in key algorithmic innovations that affect their performance characteristics.

#### XGBoost – Regularization and sparsity

XGBoost (eXtreme Gradient Boosting), introduced by Chen and Guestrin (2016), brought several innovations that shaped current gradient boosting practice.

**Regularized objective**: Unlike earlier implementations that relied primarily on early stopping and shrinkage, XGBoost explicitly adds regularization terms to the objective function: ൌ෍݈ (ݕ௜ǡ ݕො௜) ൅෍ߗ (݂௞)

௜ ௞ where: + 1 𝛺(݂௞) = ߛܶ 2 ߣ෍ݓ௝ 2் ௝ୀଵ penalizes both tree complexity and the magnitude of the model’s leaf scores. Here, 𝑇 is the number of leaves in tree 𝑘, 𝑤௝ is the prediction assigned to leaf 𝑗, 𝛾 controls the penalty for adding additional leaves, and 𝜆 controls the L2 penalty on leaf weights. Larger values of 𝛾 encourage simpler trees, while larger values of 𝜆 shrink leaf predictions toward zero. This regularization directly limits overfitting

rather than relying only on heuristic stopping rules:

- **Sparsity-aware split finding**: Real-world data often contains extensive missing values or zero entries. XGBoost learns a default direction for each tree node: when a sample has a missing value for the split feature, it follows this learned default path. This eliminates the need for imputation while often improving predictive performance.
- **Second-order approximation**: XGBoost uses a second-order Taylor expansion of the loss function, incorporating both the gradients and the Hessian, which contains the second derivatives. This provides more accurate updates than first-order methods, particularly for loss functions with significant curvature.
- **Cache-aware access patterns**: XGBoost optimizes how data is laid out and accessed in memory to make repeated computations better use of the CPU cache. This improves training speed, especially on large tabular datasets where inefficient memory access can become a bottleneck.

#### LightGBM – Speed through sampling and bundling

LightGBM (Ke et al., 2017), developed by Microsoft, focuses on dramatically improving training speed while maintaining or improving accuracy. On large, sparse datasets, LightGBM used to train 5–10× faster than pre-histogram XGBoost, but the CPU speed advantage has narrowed as XGBoost has adopted the histogram mode introduced by LightGBM, which added several innovations:

- **Histogram-based learning**: Instead of sorting feature values for exact split finding, LightGBM discretizes continuous features into histograms. Split finding then examines only histogram bin boundaries. Histogram construction still requires scanning the data, but once the histogram is built, split gain can be evaluated over bin boundaries rather than all distinct feature values. The split-evaluation step, therefore, scales with the number of bins, which is typically far smaller than the number of observations.
- **Leaf-wise tree growth**: LightGBM is best known for leaf-wise, best-first tree growth. At each step, it splits the leaf with the largest loss reduction. This can reduce training loss faster with a fixed number of leaves than depth-wise growth does, but it also increases the risk of overfitting on small or noisy datasets unless constrained by `num_leaves`, `max_depth`, `min_data_in_leaf`, or related regularization parameters.
- **Gradient-based One-Side Sampling (GOSS)**: The key insight is that samples with large gradients (those that the model is currently getting wrong) contribute more to information gain than samples with small gradients (those that are already well-predicted). GOSS keeps all large-gradient samples but randomly samples only a fraction of small-gradient samples (and reweights them to preserve a good estimate of split quality), significantly reducing computational cost without sacrificing split quality.
- **Exclusive Feature Bundling (EFB)**: In high-dimensional sparse data (common after one-hot encoding categorical variables), many features are mutually exclusive: they never take non-zero values simultaneously. EFB identifies and bundles such features into combined features, reducing the effective number of features to evaluate when finding splits.

#### CatBoost – State-of-the-art categorical handling

CatBoost (Prokhorenkova et al., 2018), developed by Yandex, addresses a critical challenge: **how to handle categorical features without introducing target leakage**.

Standard target encoding (replacing a category with the mean target value of samples in that category) creates a subtle form of data leakage. When computing the encoded value for a training sample, that sample’s target value influences its feature value, inflating the model’s apparent performance.

CatBoost’s solution is “**ordered target statistics**.” For each training sample, target statistics are computed using only samples that appear earlier in a random permutation of the data. This ensures no sample’s target value influences its own features. Multiple permutations are used and averaged to reduce variance.

Unlike XGBoost and LightGBM, where each node can split on a different feature, CatBoost uses symmetric, or **“oblivious” decision trees,** by default, where all nodes at the same depth use the same split condition. This simplicity enables extremely fast prediction (often the fastest inference among the three libraries) because the prediction path can be computed as a bitwise operation rather than a sequential tree traversal. This makes CatBoost particularly well-suited for latency-sensitive applications where inference speed matters.

While one-hot encoding becomes impractical for high-cardinality categoricals with thousands of unique values, CatBoost’s ordered encoding naturally handles any cardinality. Features such as individual stock identifiers, analyst IDs, and detailed sector codes can be used directly.

### Practical comparison – Which GBM to choose?

Scikit-learn’s `HistGradientBoostingRegressor` provides a no-dependency baseline. It uses the same histogram-based splitting as the specialized libraries and is often the fastest option on CPU for small-to-medium datasets. However, it lacks GPU support and the tuning ecosystem (Optuna callbacks and early stopping flexibility) that specialized libraries provide. We include it in the `02_gbm_comparison` benchmark alongside the three specialized libraries. *Table 12.1* summarizes the library selection decision points. The specialized libraries still differ in what they optimize for. XGBoost is the most general-purpose option of the three: it supports both CPU and CUDA-based GPU training, includes native categorical support, and remains a strong default when you want one library that covers a wide range of production and research settings. The choice between XGBoost, LightGBM, and CatBoost depends on your specific constraints and data characteristics:

| Criterion | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| CPU training (4.9M rows, 8 threads) | ~100s | ~50s — fastest (~2× XGBoost) | ~135s — slowest |
| GPU training (4.9M rows, CUDA) | ~14s (~7× its CPU) | ~14s (~3.5×; FP64-only, see below) | ~10s (~14×) — fastest |
| Prediction speed (heavy preset, 24K test rows) | 0.5M rows/s CPU; 1.7M rows/s GPU | 0.07M rows/s CPU (slowest) | 6.5M rows/s CPU; symmetric trees → bitwise path, ~12× faster than XGBoost on CPU |
| High-cardinality categoricals | Native support | Native support | Ordered target statistics avoid leakage |
| Default performance | Good | Good | Often strong, especially with many categorical variables |
| Tuning sensitivity | Moderate | Moderate-high (leaf-wise growth requires depth/ leaf constraints) | Often lower, but data- dependent |
| Memory efficiency | Moderate | Often best (histogram bins reduce memory) | Moderate |

*Table 12.1: GBM library characteristics*

The `02_gbm_comparison` notebook benchmarks all four libraries on CPU and GPU with light, medium, and heavy presets, measuring accuracy, training time, and memory usage.

LightGBM remains especially attractive in CPU-first tabular workflows. Its leaf-wise tree growth can reduce training loss quickly, but it also makes the library somewhat more sensitive to parameters such as `num_leaves`, `min_data_in_leaf`, and sometimes `max_depth`

All three libraries support **GPU-accelerated training**, but the speedup depends on dataset size and numerical precision:

- On the **US equities panel** (4.92M training rows, 72 features, 8 threads), CatBoost achieves the largest GPU speedup at ~14× with `task_type='GPU'` (~135s CPU → ~10s GPU), and XGBoost ~7× using `device='cuda'` (~100s → ~14s). LightGBM trains in ~50s on CPU (the fastest CPU configuration in the scale benchmark) and ~14s on GPU with `device='cuda'`, a more modest ~3.5× speedup; its GPU wall time matches XGBoost’s but not CatBoost’s, for the FP64 reason discussed below.
- On the smaller **ETF dataset** (227K rows), launch overhead compresses the speedups: for XGBoost and CatBoost the gain ranges from ~2.4× (XGBoost light) to ~5.2× (CatBoost heavy). LightGBM is the exception: its FP64 CUDA path is counterproductive at this scale, training *slower* on GPU than on CPU (about ~9.5s CPU versus ~48s GPU on the heavy preset), so LightGBM should stay on CPU for datasets this size. For XGBoost and CatBoost the GPU still helps, but the relative advantage is roughly an order of magnitude smaller than at the 4.9M-row scale.

LightGBM’s CUDA speedup is modest even though its GPU wall time (~14s at the 4.9M-row scale) matches XGBoost’s: its **CUDA backend uses double precision (FP64) exclusively**: the `gpu_use_dp` toggle that enables single-precision training “can be used only in OpenCL implementation (`device_type='gpu'`), in CUDA implementation only double precision is currently supported.” Note that:

- **Consumer GPUs** have an FP32:FP64 throughput ratio of roughly 64:1 (the RTX 3090 delivers 35.6 TFLOPS FP32 but only ~0.6 TFLOPS FP64), so LightGBM’s GPU histogram computation runs at a fraction of the available throughput.
- **Data center GPUs** with better FP64 ratios (A100: 2:1) would show a different result.

LightGBM also offers a separate OpenCL backend (`device='gpu'`) that defaults to single precision, but it requires an OpenCL SDK and produces slightly different results due to the precision difference. The CUDA build requires compilation from source with `USE_CUDA=ON`, which is the build that produces the GPU times reported here; CUDA-enabled conda packages are now available on supported systems.

The practical picture at scale is therefore:

- **CatBoost GPU delivers the largest training speedup** (~14× at the 4.9M-row scale), making it the clear choice when GPU-accelerated training time matters. At that scale CatBoost finishes GPU training in ~10s, with XGBoost and LightGBM together at ~14s. On CPU, LightGBM is the fastest at ~50s, the fastest CPU configuration in the scale benchmark.
- CPU times at the 4.9M-row scale (8 threads) spread from LightGBM (~50s) to XGBoost (~100s) to CatBoost (~135s); LightGBM’s histogram plus leaf-wise growth has the best parallel efficiency at this thread count and dataset size.

In practice, the accuracy gap between libraries is smaller than the gap between good and bad hyperparameter configurations of any single library. The four libraries fall within a narrow IC band after tuning, and their fold-level rankings vary over time.

The real selection criteria are therefore operational: CatBoost’s GPU backend delivers the fastest training at scale, CatBoost’s ordered encoding eliminates subtle leakage bugs when working with sector codes or analyst identifiers, and XGBoost’s mature ecosystem provides the best integration with SHAP, Optuna callbacks, and deployment tooling. On CPU at the 4.9M-row scale, LightGBM is the fastest of the three (roughly 2× XGBoost and 2.7× CatBoost at 8 threads). For readers with CUDA-capable GPUs and large datasets, CatBoost’s GPU acceleration offers the most substantial training-time savings.

This book’s nine **case study pipelines use LightGBM** as the default library because its CPU speed is particularly advantageous when running walk-forward cross-validation across multiple hyperparameter configurations. Readers with access to **CUDA-capable GPUs** should consider switching to **XGBoost** for the larger case studies (US equities, S&P 500 equity-option analytics) where GPU acceleration provides meaningful time savings.

### Native feature importance and its limitations

Every GBM library reports feature importance after training: typically either the total gain from splits on each feature or the number of times each feature is used for splitting. These metrics are fast to compute and provide a quick first look at which features the model relies on. However, they have well-documented failure modes:

- **Gain-based importance** is biased toward high-cardinality features because more unique values create more candidate splits.
- **Split-count importance** is biased toward continuous features over categoricals for the same reason.

Both metrics are *unstable*: retraining with a different random seed can substantially change importance rankings, especially for correlated features where the model can substitute one for another.

These limitations motivate the SHAP-based analysis in *Section 12.5*, which provides consistent, theoretically grounded attributions. Use native importance for quick screening during development and SHAP for any analysis that informs decisions: feature selection, model monitoring, or risk committee reporting.

### Learning to rank

Standard GBM training minimizes pointwise losses (MSE or log-loss), treating each prediction independently. But a long-short portfolio that buys the top decile and sells the bottom profits from correct *ranking*, not accurate magnitude. When MSE is dominated by large absolute errors in the middle of the distribution, the model sacrifices ranking accuracy in the tails where trading actually occurs.

**Learning-to-Rank** (**LTR**) objectives directly optimize ranking quality. LightGBM supports the `lambdarank` objective, which implements LambdaMART, a gradient boosting algorithm where the loss function is defined implicitly through the gradients (the “lambdas”) that would improve pairwise ordering. Each gradient accounts for the change in a ranking metric when two samples swap positions, weighted by the metric’s sensitivity at that position. The result is a model trained to produce well-ordered predictions rather than accurate point estimates.

The standard target metric is **Normalized Discounted Cumulative Gain (NDCG)**, which assigns exponentially higher value to correctly ranking items near the top of the list. This aligns naturally with quantile strategies: correctly identifying the best and worst stocks matters far more than distinguishing between the 40th and 50th percentiles.

#### When LTR outperforms pointwise regression

LTR provides the largest gains in high-breadth cross-sections where only the tails are traded. Consider a universe of 500 stocks ranked monthly: a long-short decile strategy trades only the top and bottom 50 names. MSE-trained models optimize prediction accuracy across all 500 stocks equally, including the 400 in the middle that generate no trading signal. LambdaMART concentrates its fitting capacity on correctly separating the extremes. The advantage diminishes for strategies that trade the full cross-section (for example, rank-weighted portfolios) or for low-breadth universes where most assets enter positions. It also diminishes when the signal-to-noise ratio is so low that ranking and pointwise objectives converge to the same solution.

A practical consideration: LTR models produce scores that are well-ordered but not calibrated to return magnitudes. If downstream portfolio construction requires return forecasts (for example, mean-variance optimization in *Chapter 17*), either calibrate the LTR scores post-hoc or use pointwise predictions. For strategies that simply go long the top quintile and short the bottom quintile, uncalibrated ranking scores suffice.

#### A worked example – LambdaMART versus Pointwise MSE

In the notebook `02_gbm_comparison`, we compare LambdaMART against standard MSE regression on the ETF dataset using the same feature set and train/test fold. The key configuration difference is minimal:

- **Pointwise**: `objective='regression'`, optimizing MSE
- **LTR**: `objective='lambdarank'`, with `label_gain` mapping within-date return quintiles to relevance grades and `ndcg_eval_at=[10]` targeting top-of-rank quality

Both models produce ranked predictions, but they differ in where they spend fitting capacity. The notebook evaluates mean group-wise NDCG@10 (ranking quality by date) alongside IC against raw returns. This makes the objective–metric alignment explicit: LambdaMART is optimized for ordering, while pointwise regression is optimized for magnitude fit. For tail-trading strategies, ranking quality is usually the more relevant diagnostic.

LightGBM’s `lambdarank` objective requires grouping samples by cross-section (date), specified via the `group` parameter. Each group defines the set of assets ranked against each other. XGBoost provides equivalent functionality through its `rank:ndcg` objective. CatBoost supports ranking via `YetiRank` and `YetiRankPairwise` objectives, which use their own gradient formulations optimized for CatBoost’s symmetric tree architecture.

The worked example is implemented in `02_gbm_comparison`, including date-group construction, relevance labeling, LambdaMART training, and NDCG@10 comparison against a regression baseline.

### Enforcing economic logic with monotonic constraints

While GBMs excel at discovering non-linearities, they are blind to economic theory. A purely data-driven model might learn that higher P/E ratios generally predict lower returns, but due to noise in specific training regions, it might predict a spike in returns for extremely high P/E values, almost certainly an artifact rather than a structural signal.

**Monotonic constraints** enforce directional relationships: a positive constraint (+1) requires that the prediction not decrease as the feature increases; a negative constraint (−1) requires the opposite. Monotonic constraints serve as theory-driven regularization: in low-signal-to-noise environments, enforcing a sign (for example, higher-value scores predict higher returns) prevents the model from fitting non-monotonic artifacts in noisy regions. The constraint also improves deployability (risk committees accept strategies with directional logic they can audit) and robustness to regime shifts, because the model cannot learn complex, non-monotonic patterns that are unlikely to repeat.

All three libraries support monotonic constraints via a parameter-mapping feature that maps names to their intended directions.

**SHAP dependence plots** (*Section 11.4*) make the effect of constraints immediately visible. An unconstrained model’s dependence plot for a value feature might show a generally downward slope with local reversals, regions where higher value scores paradoxically predict lower returns. The same model with a negative monotonic constraint produces a strictly monotone dependence curve, eliminating these artifacts while preserving the overall relationship. When the constrained model’s IC is comparable to the unconstrained version (as it typically is for economically motivated constraints), the constraint is acting as regularization rather than information loss. This provides a concrete implementation of the economic sanity-checking framework in *Section 11.4*.

See `02_gbm_comparison` for constrained versus unconstrained SHAP dependence plots on the ETF case study.

We now turn to a more recent addition: deep learning models for tabular data.

## 12.3 Deep learning alternatives for tabular data

GBMs remain the defensible default for tabular financial data, but the landscape shifted materially between 2024 and 2026. Tabular foundation models, led by TabPFN’s Nature publication (Hollmann et al., 2025), moved from research curiosity to frontier baseline, matching or exceeding tuned GBMs on small-to-medium datasets without task-specific training. Parameter-efficient neural ensembles like TabM have shown that well-engineered MLPs, rather than exotic architectures, can compete with GBMs when given equivalent tuning budgets. Retrieval-augmented models like TabR showed that hybridizing deep learning with nearest-neighbor lookups improves predictions when local similarity matters.

The **TabArena** living benchmark (Erickson et al., 2025), evaluating 16 model families across 51 curated datasets, formalized a key finding: with sufficient tuning and ensembling, deep learning can match or exceed GBMs, but the practical winner depends on dataset size, feature composition, and compute budget.

For finance, a critical caveat applies: most major tabular benchmarks assume IID splits and exclude temporal dependence. The *TabReD* benchmark (Rubachev et al., 2024), which tested models under temporal distribution shift, found that attention-heavy architectures degraded faster than GBMs and simple MLPs, directly relevant to non-stationary financial data. Benchmark wins do not imply trading profitability without walk-forward validation.

### Tabular foundation models

The most conceptually striking development is the rise of **tabular foundation models** (**TFMs**) that use **in-context learning** (**ICL**). Unlike conventional models that train on a specific dataset via gradient descent, a TFM is pre-trained once on millions of synthetic datasets generated from structural causal models. At inference time, it uses the training data as context to produce predictions for test samples in a single forward pass: no gradient updates, no hyperparameter tuning on the target dataset. The model has already learned *how to learn* from tabular structure during pre-training.

The interface is deceptively simple: given `(X_train, y_train, X_test)`, the model returns predictions immediately. Training is replaced by in-context computation, where the transformer’s attention mechanism identifies patterns in the provided context that generalize to query samples.

**TabPFN** pioneered this paradigm for tabular data. The original version (Hollmann et al., *Nature*, 2025) validated the approach on curated benchmarks: datasets of up to 10,000 samples and 500 features, where the default TabPFN outperformed tuned baselines without any task-specific optimization. **TabPFN v2.5** (Grinsztajn et al., 2025) substantially expanded the practical envelope (to approximately 50,000 rows and 2,000 features) through sketching and feature selection. A separately released distillation tooling path converts the TFM’s predictions into compact MLPs or tree ensembles for production deployment, addressing the inference latency bottleneck that limited earlier versions to batch-only workflows.

There are several known **limitations for financial applications**. TabPFN performs best on numeric-only data and degrades on purely categorical datasets (Ye et al., 2024). IID pretraining on synthetic data does not capture temporal dependence or regime shifts. Per-sample inference latency exceeds that of optimized tree libraries at high throughput, though TabPFN 2.5’s distillation path mitigates this in batch workflows. Calibration depends on similarity between the target data structure and the synthetic pretraining distribution; structural breaks absent from the synthetic priors may degrade zero-shot predictions.

There is a practical role that TabPFN can play. For financial ML, TabPFN’s strongest use case is **rapid prototyping**: test whether a feature set contains predictive signal before investing in GBM tuning. If TabPFN achieves non-trivial IC on a subsample, the features warrant deeper investigation with production-grade models. The distillation engine now makes deployment feasible by converting the TFM’s predictions into a tree ensemble that inherits the foundation model’s accuracy while achieving GBM-class inference speed. TabPFN 2.5 weights are released under a non-commercial license by default; readers working in production trading systems should treat them as a research prototyping tool unless a commercial license is secured.

The notebook `03_dl_vs_gbm` includes TabPFN as an optional model in the walk-forward comparison; if installed, it runs alongside GBMs and TabM on the ETF case study with per-fold IC reporting.

### Modern neural baselines – TabM and strong defaults

A quieter but arguably more impactful development than foundation models is the rehabilitation of simple neural architectures through better training recipes and engineering. Holzmüller et al. (2024) showed that **meta-tuned MLP defaults** (training recipes optimized across hundreds of datasets) can make neural networks competitive with GBMs without any architectural novelty. Their **RealMLP** achieves a strong time-accuracy trade-off on medium-to-large datasets by improving defaults rather than inventing new layers. The implication is that many older “DL loses to trees” conclusions reflected weak neural baselines rather than fundamental architectural limits. training 𝑀 independent MLPs (which is computationally prohibitive for large financial datasets), **TabM** (Gorishniy et al., 2025) takes this further with parameter-efficient ensembling. Rather than

𝑀 diverse ensemble members, all trained simultaneously in a single forward pass. Each individual TabM maintains a shared weight backbone and applies rank-1 adapters (scaling parameters) to create

member’s predictions are noisy: the rank-1 perturbations introduce enough diversity that individual members overfit in different directions. Averaging cancels these uncorrelated errors while preserving the shared signal, achieving the generalization benefits of deep ensembles at the computational cost of a single network.

In evaluations across 46 benchmarks, TabM achieved the strongest performance among deep learning models, including on domain-aware temporal splits, which are historically challenging for neural networks. This temporal robustness is directly relevant to financial applications in which the training distribution shifts across walk-forward folds.

TabM represents the most practical case for “deep learning a practitioner can actually use on tabular financial data.” It is architecturally simple, well-documented, and competitive without transformer-scale complexity. Interpretability uses standard model-agnostic methods (permutation importance and KernelSHAP) at a moderate computational cost. Training is slower than GBMs but does not require the multi-GPU infrastructure of large transformer models.

The notebook `03_dl_vs_gbm` compares LightGBM, a vanilla MLP, and a TabM rank-1 adapter ensemble on the ETF case study across walk-forward folds, reporting per-fold IC and training time.

### Retrieval-augmented models – TabR **TabR** (Gorishniy et al., 2024) hybridizes a deep encoder with 𝑘-nearest-neighbor retrieval. At predic-

tion time, the model retrieves similar training instances from a learned embedding space and uses an attention-like mechanism to extract signals from their features and labels. The retrieved neighbors effectively provide local context that the base model can exploit, a principled way to help neural networks with the kind of local pattern recognition that trees perform naturally through hierarchical splitting.

The conceptual appeal for finance is immediate: “these neighbors drove the prediction” aligns with intuitions about regime similarity and peer-relative valuation. If the current market environment resembles a historical period, retrieval surfaces that analogy explicitly rather than requiring the model to encode it implicitly in fixed weights.

**Benchmark evaluations** show that TabR consistently outperforms non-retrieval deep baselines, and it is included as a first-class method in modern tabular toolkits (Ye et al., 2024). The trade-off is operational complexity: retrieval adds memory overhead for the neighbor index and increases inference latency in proportion to the database size. **Box 12.1: Lookahead bias in retrieval-augmented models**

Standard TabR implementations build a retrieval index over the entire training set without temporal constraints. In a walk-forward setting, the model can retrieve neighbors from time periods that overlap with or follow the prediction target, a form of lookahead bias that standard implementations do not prevent. Any deployment must enforce temporal isolation: neighbors must come strictly from periods before the prediction date, with the same purging and embargo logic applied to the walk-forward splits. Implementing this remains an open engineering challenge; treat TabR results without temporal isolation as upper bounds, not production estimates.

### When to look beyond GBMs – A decision framework

The choice between GBMs and neural alternatives for tabular (cross-sectional) prediction is regime-dependent. *Table 12.2* summarizes the decision framework for this class of problems, synthesizing the benchmark evidence with the operational constraints that matter in production financial ML. Sequential models (LSTMs, transformers, temporal convolutional networks) add a temporal dimension to this decision; *Chapter 13* covers that extension.

| Data regime | Recommended | Rationale |
| --- | --- | --- |
| Fewer than 10K efefctive samples, rapid signal testing | TabPFN (v2.5; distilled) | Strong “no-tuning” baseline for small data; fast iteration. Distillation engine enables deployment. |
| 10K–100K samples, production tabular | LightGBM (or XGBoost) | Strong default accuracy/robustness; efficient training; TreeSHAP explanations are practical for tree ensembles. |
| 10K–100K samples, DL warranted by complexity | TabM (rank-1 adapter ensemble) | Strongest tabular DL baseline in recent evaluations; validate under walk-forward temporal splits. |
| More than 100K samples, heavy categoricals | CatBoost (or LightGBM) | CatBoost’s ordered scheme prevents leakage with categoricals; LightGBM supports categoricals via integer codes. |
| More than 1M samples, GPU available | LightGBM and TabM (or TabR) | No reliable crossover point. Trees often remain competitive; validate both families under your constraints. |
| Multi-modal inputs (text and tabular) | End-to-end DL (transformers) | Joint representation learning across modalities requires DL; GBMs don’t natively fuse modalities. |
| Local similarity/regime matching matters | TabR | Nearest-neighbor retrieval improves on some benchmarks; enforce temporal isolation to avoid leakage. |

*Table 12.2: Heuristics for model selection*

For readers who want a performance ceiling rather than a single production model, **AutoGluon Tabular** (Erickson et al., 2020) provides a useful reference. Its “best quality” preset blends GBMs, neural networks, and TabPFN into a multi-layer stacking ensemble with automated hyperparameter selection. These stacks frequently achieve the best benchmark results by exploiting complementary model strengths. The cost (multi-hour training budgets and operational overhead) makes direct deployment burdensome.

In our workflow, AutoGluon serves as a “ceiling sanity check”: if your tuned LightGBM or TabM is within striking distance, you have extracted most of the available signal. If AutoGluon dramatically outperforms your single-model baseline, inspect which model family in the stack contributes the lift. This diagnoses whether you are leaving signal on the table that a different architecture could capture.

For the medium-sized datasets typical of financial ML (where walk-forward splits reduce effective training size, signal-to-noise ratios are low, and features exhibit heavy tails), well-tuned GBMs remain a strong default. The gap is narrowing, and the regime map above identifies where neural alternatives are most likely to pay off.

The notebook `03_dl_vs_gbm` provides a systematic comparison of LightGBM, TabM, TabPFN, and a vanilla MLP on the ETF case study under walk-forward splits. LightGBM achieves competitive IC at a fraction of neural network training time, while TabM’s rank-1 adapter ensemble provides the strongest neural baseline, consistent with its showing across the broader case study comparison in *Section 12.6*, where TabM beats GBM on 3 of the 8 case studies with TabM coverage.

**Box 12.2: Reading tabular benchmarks critically**

The tabular ML literature now includes several large-scale benchmarks. Before transferring their conclusions to financial applications, four methodological issues deserve scrutiny:

- *IID assumptions.* TabArena (Erickson et al., 2025) and TALENT (Ye et al., 2024) focus on IID classification and regression. Both explicitly exclude temporal dependence from their current scope. Conclusions about model rankings under IID splits may not hold under walk-forward evaluation.
- *Tuning budget inequity.* Results depend heavily on whether models receive equal time for hyperparameter optimization. A tuned GBM versus a default neural network (or vice versa) proves nothing about the model families. Fair comparisons must specify and equalize the compute budget.
- *Post-hoc ensembling.* TabArena reports that cross-model ensembles (stacking GBMs with neural methods) often produce the best results. This changes the definition of “baseline” and complicates single-model comparisons. If your production constraint is a single model family, ensemble results are aspirational rather than actionable.
- *Dataset leakage.* Multiple benchmark studies have identified leakage in popular tabular datasets: features that encode the target, mislabeled tasks, or timestamp columns that enable lookahead. Curated subsets exist for a reason: TabArena retained 51 datasets from an initial pool of 1,053.

Regardless of the model family, performance depends heavily on hyperparameter configuration. The next section introduces Optuna’s Bayesian approach to efficiently navigating that search space.

## 12.4 Advanced hyperparameter tuning with Optuna

GBMs introduce tree depth, learning rate, regularization strength, sampling ratios, and leaf counts into the hyperparameter search space, making Bayesian optimization essential for cost-effective parameter optimization rather than just convenient.

### The tree-structured Parzen Estimator

Optuna implements hyperparameter optimization (HPO) using sampling methods to learn the rela-**Estimator** (**TPE**, Bergstra et al., 2011) maintains two density estimators: 𝑙(ݔ) for hyperparameters that tionship between parameter configurations and objective values. The default **Tree-structured Parzen** yielded good objective values, and 𝑔(ݔ) for those that yielded poor objective values. The acquisition function maximizes 𝑙(ݔ)Ȁ݃(ݔ), concentrating evaluations on promising regions. As trials accumulate,

the density estimators sharpen, concentrating proposals on regions that have historically produced strong validation metrics.

TPE is particularly effective for the discrete and conditional parameter spaces common in GBM tuning. It handles integer parameters, categorical choices, and conditional dependencies naturally, for example, sampling `num_leaves` only when tree growth is leaf-wise. The notebook `04_optuna_tuning` shows TPE-based optimization on the ETF case study, with emphasis on search-space design and early-stopping integration.

### The define-by-run API

Optuna’s define-by-run API defines the search space dynamically within the objective function. During each trial, calls to `trial.suggest_int()` or `trial.suggest_float()` request parameter values based on Optuna’s current model of the objective landscape. This enables conditional parameters to sample regularization strength only for certain model types, hierarchical search to select a library, then tune its specific parameters, and dynamic stopping via pruning.

For iterative models like GBMs, we can evaluate intermediate performance after each boosting iteration. Optuna’s **pruners automatically terminate trials** that appear unpromising:

- **MedianPruner** prunes trials worse than the median at each step
- **HyperbandPruner** uses adaptive resource allocation for more aggressive early pruning

Pruning can reduce total computation by 50% or more without sacrificing optimization quality.

For production-scale tuning, studies can be backed by a shared database (PostgreSQL, MySQL) so that multiple workers evaluate trials in parallel across machines.

### GBM-specific tuning strategy

GBM hyperparameters fall into three families with distinct effects:

- **Tree structure** parameters (max depth, number of leaves, min samples per leaf) control model complexity: deeper trees capture more interactions but carry a higher risk of overfitting.
- **Boosting parameters** (learning rate, number of iterations, subsampling ratios) control training dynamics: lower learning rates with more iterations generally produce better results but increase training time.
- **Regularization parameters** (L1/L2 penalties, min child weight) directly penalize complexity.

A common pitfall is tuning the tree structure extensively while neglecting regularization. Regularization parameters (`reg_alpha`, `reg_lambda`) often have the greatest impact on out-of-sample performance because they directly address overfitting in low-signal-to-noise regimes. Set the learning rate at a low value (0.01–0.05) and let Optuna determine the optimal number of rounds via early stopping. This is generally more efficient than tuning both jointly. Start with 100–200 trials; beyond this range, marginal improvement tends to diminish while the risk of validation overfitting increases (*Box 12.3*).

**Box 12.3: Validation overfitting in hyperparameter search**

Validation overfitting is the primary risk of automated tuning. In low-signal-to-noise environments like daily return prediction, 500 TPE trials will find hyperparameters that exploit validation-set noise rather than genuinely better configurations: the 0.02 IC improvement vanishes on truly held-out data.

Defenses: limit trials to 50–100, use coarser parameter grids, and always evaluate final candidates on data untouched during optimization. This is not a theoretical concern; it is the most common source of backtest-to-live performance degradation.

### Multi-objective optimization

Financial models often face competing objectives that cannot be collapsed into a single metric. Optuna supports multi-objective optimization through its `NSGAIISampler` (Deb et al., 2002), which finds the **Pareto frontier** of non-dominated solutions, configurations where improving one objective necessarily harms another.

#### Optimizing IC versus turnover in practice

The notebook `06_optuna_multi_asset` tunes a LightGBM model on the ETF case study with two objectives: maximize rank IC and minimize portfolio turnover. Aggressive hyperparameters (deeper trees, lower regularization, more boosting rounds) tend to increase both IC and turnover simultaneously, because the model fits finer-grained patterns that change more frequently across rebalancing periods.

**The Pareto frontier** on this dataset is tight. NSGA-II finds two non-dominated solutions, one at IC≈0.029 with normalized turnover ≈0.004 and one at IC≈0.030 with turnover ≈0.005, both at similarly conservative parameter ranges with modest depth and sample weights. Single-objective optimization on IC alone, run with TPE on the same search space, converges to IC≈0.010, substantially below either Pareto solution; the multi-objective NSGA-II search reached a region of the parameter space that the single-objective search did not. The multi-objective approach also reveals when objectives are less conflicted than assumed. If the Pareto frontier is nearly flat in one region, for example, high IC with only modestly higher turnover, you can claim both objectives without significant compromise. Conversely, a steeply curved frontier signals that marginal IC improvements come at rapidly increasing turnover costs, warranting careful analysis of transaction-cost sensitivity.

Other common multi-objective pairs include return versus drawdown (aggressive models achieve higher returns while experiencing larger peak-to-trough declines) and Sharpe ratio versus capacity (strategies with higher Sharpe often trade in smaller, less liquid names).

**Implementation**:

- `04_optuna_tuning` demonstrates HPO on the ETF case study.
- `05_cross_library_hpo` compares the three GBM libraries XGBoost, LightGBM, and CatBoost with loss type as a tunable hyperparameter on the firm characteristics dataset.
- `06_optuna_multi_asset` compares single-objective and multi-objective approaches across asset classes.
- `07_hpo_comparison` compares the efficiency of grid search and Optuna on the same tuning problem, including empirical evidence of validation overfitting as trial budgets increase.

Well-tuned models need interpretation before deployment. The next section covers TreeSHAP as the primary tool for explaining GBM predictions and detecting feature drift.

## 12.5 Model explainability with SHAP

*Chapter 11* introduced SHAP values (Lundberg and Lee, 2017) as sanity checks for economic models. values in 𝑂(ܶܮܦ2) time per prediction, where 𝑇, 𝐿, and 𝐷 are the number of trees, leaves per tree, and TreeSHAP (Lundberg et al., 2020) makes this framework routine for GBMs by computing exact Shapley

average tree depth. For a 500-tree LightGBM model on 100K samples, computation takes seconds, fast enough to run on every walk-forward fold as standard diagnostic infrastructure. By contrast, neural networks require approximate methods (for example, KernelSHAP, sampling) that are orders of magnitude slower and introduce approximation errors.

### Dependence and interaction effects

*Section 11.4* covered SHAP dependence plots and their use for identifying feature interactions. For GBMs, TreeSHAP additionally provides exact *interaction values* that decompose each prediction into main effects and pairwise interactions: 𝑓(ݔ) ൌܧ[𝑓(ܺ)] ൅෍߶௝ ൅෍߶௝௞

௝ main ௝ழ௞ interaction  This decomposition quantifies how much of each prediction comes from features acting independently versus in combination, a capability unique to tree-based SHAP that approximate methods do not provide efficiently. In our ETF case study, the strongest interaction (𝐻43) is between the yield-curve 𝑧-score and

the short-horizon volatility ratio, the top pair among 29 statistically strong interactions identified by the H-statistic. The SHAP interaction matrix shows that the yield-curve signal carries most of its predictive content when short-term volatility deviates meaningfully from its longer-horizon baseline; the interaction contribution can shift sign across volatility regimes. This conditional structure is invisible in standard feature-importance rankings, which rank the yield-curve z-score as a top feature regardless of regime, one reason the same model can look stable on average while its mechanism shifts beneath the surface. `08_shap_analysis` shows TreeSHAP computation, dependence plots, and the full interaction decomposition on the ETF case study.

### SHAP-based drift monitoring

The SHAP stability chart from *Section 11.4* (tracking feature importance across walk-forward folds) extends naturally to production drift monitoring, as illustrated in *Figure 12.2*, where the comparison shifts from training folds to rolling prediction windows.

![Figure 12.2](assets/figure_12_2.png)

*Figure 12.2: SHAP-based drift detection*

Comparing SHAP distributions between a baseline period and recent predictions detects changes in mechanisms before they manifest in performance metrics. When features that were predictive lose their SHAP importance (or when previously irrelevant features gain prominence), investigate before performance deteriorates. This provides an earlier warning than outcome-based monitoring (rolling IC, Sharpe ratio) because it detects changes in the model’s *reasoning*: by the time returns degrade, the damage is done. `08_shap_analysis` implements this comparison by tracking the mean |SHAP| across walk-forward folds and flagging features whose importance changes exceed 50%. Critically, SHAP drift is a *diagnostic signal* warranting investigation, not proof of concept drift. Feature distributions can shift without affecting predictions if the model does not rely heavily on those features. Validate with downstream metrics (rolling IC, residual patterns) before acting. *Chapter 19* covers risk monitoring infrastructure; *Chapter 26* covers production alert thresholds and automated response protocols.

### The Rashomon Effect – When equally good models disagree

Models with similar predictive performance can produce strikingly different explanations, a phenomenon known as the Rashomon Effect, after *Rashomon*, the 1950 film directed by Akira Kurosawa, in which witnesses give contradictory accounts of the same event. If a Random Forest and a gradient boosting ensemble both achieve 0.05 IC but attribute predictions to different features, each explanation reveals model-specific logic, not ground truth about the data-generating process. This multiplicity is inherent and should temper confidence in any single model’s explanation. When SHAP attributions matter for downstream decisions such as feature pruning, risk allocation, or regulatory reporting, validate findings across multiple model specifications.

In practice, when a GBM and a TabM disagree on which features drive predictions, this disagreement is a signal to investigate the underlying relationship in the data rather than declare one correct. Features that both model families rank as important are more likely to reflect genuine structure; features that only one family highlights may reflect architecture-specific fitting patterns. `09_xai_limitations` shows explanation instability: predictions that differ by roughly 1.3% can yield entirely different topthree SHAP contributors.

SHAP also applies to token-level attributions for NLP models. The notebook `10_shap_nlp_sentiment` shows this for FinBERT sentiment predictions; *Chapter 10* covers the text feature-engineering pipeline in detail.

#### Using SHAP for feature selection and pruning

TreeSHAP’s efficiency enables SHAP-based feature selection as a practical workflow step. Compute top-𝑘 features. This approach can produce models with comparable or improved out-of-sample permean absolute SHAP values across the training set, rank features by importance, and retrain on the

formance when the removed features primarily contribute noise, though the improvement depends on the original feature set’s redundancy and the signal-to-noise ratio. Unlike wrapper methods that require retraining for each feature subset, SHAP importance from a single trained model provides a strong initialization for the feature selection search.

For financial models with dozens of features, SHAP-based pruning improves interpretability and reduces the risk of spurious correlations. Features with consistently near-zero SHAP values across walk-forward folds are strong candidates for removal. Features with high SHAP variance across folds (important in some periods but not others) may indicate regime-dependent signals worth investigating rather than discarding.

Comparing SHAP rankings against permutation importance (PFI) and mean decrease in impurity (MDI), as shown in `08_shap_analysis`, identifies features that are robustly important across methods, and helps filter out artifacts of any single technique.

### From explanation to uncertainty and robustness

Conformal prediction (*Chapter 11*) applies directly to GBM residual workflows. The notebook `11_conformal_gbm` implements four variants side by side: split conformal, quantile regression, **conformalized quantile regression** (**CQR**), and an **adaptive conformal** (**ACI**-style) online update loop. Split conformal provides the finite-sample coverage guarantee under exchangeability; CQR preserves asymmetric intervals while restoring calibration; ACI targets stable coverage under non-stationary residual dynamics through online alpha updates. When drift monitoring flags a regime change, these interval diagnostics typically widen, creating a feedback loop between explainability and uncertainty quantification. *Chapter 19* covers the position-sizing implications.

With the model-building and explanation toolkit in place, we now benchmark GBMs against the linear baselines from *Chapter 11* across all nine case studies to assess where nonlinear models earn their additional complexity.

## 12.6 Case study insights

*Chapter 11* fit a linear baseline to nine case studies and asked each one question: does the model turn engineered features into a cross-sectional ranking that survives out-of-sample? The question and the case studies stay fixed; the two more flexible families introduced in this chapter (gradient-boosted trees and the TabM tabular network) now sit beside that baseline. The `12_case_study_insights` notebook fits all three families to every case study at its primary regression label and reads the results from the locked registry. Each model is summarized by its average daily information coefficient (IC), the mean of the daily cross-sectional rank correlations between prediction and realized return. The uncertainty around it is the 95% confidence interval computed with a HAC correction for the autocorrelation in that daily series. An interval that excludes zero is the bar for distinguishable cross-sectional signal.

Added flexibility does not win by default. Gradient boosting’s interval clears zero on five of the nine case studies, against three for the linear baseline; TabM clears it on three of the eight it covers. The flexible families widen the set of case studies with measurable signal, but only modestly, and on no case study do they manufacture a ranking where the linear model found none.

### Where flexibility adds ranking content

*Figure 12.3* places the three families side by side: for each case study, the highest-IC configuration of each family at the primary label, with its HAC interval. *Table 12.3* holds the underlying numbers. Read together, they separate three situations, beginning with the numbers in the table.

| Case study | Horizon | Target | Linear | GBM | TabM |
| --- | --- | --- | --- | --- | --- |
| US Firms | 1 month | forward return | −0.005 | +0.080 | +0.031 |
| ETFs | 21 days | forward return | +0.054 | +0.037 | +0.041 |
| CME Futures | 5 days | forward return | −0.000 | +0.032 | +0.004 |
| US Equities | 1 day | forward return | +0.016 | +0.032 | +0.017 |
| SP500 Options | to expiry | straddle return | +0.007 | +0.018 | +0.002 |
| Crypto | 8 hours | forward return | +0.009 | +0.011 | +0.003 |
| NASDAQ-100 | 15 minutes | forward return | +0.005 | +0.006 | — |
| SP500 Eq+Opt | 5 days | forward return | −0.006 | +0.006 | +0.011 |
| FX | 1 day | forward return | +0.005 | +0.003 | +0.007 |

*Table 12.3: Average daily IC at each case study’s primary regression label, for the highest-IC configuration of each family. Bold marks a HAC 95% confidence interval that excludes zero. TabM has no NASDAQ-100 coverage. The SP500 Options target is the return of a hold-to-maturity at-the-money straddle; shorter-horizon and delta-hedged alternatives post higher in-sample IC but, as O’Donovan and Yu (2024) document, do not survive realistic option transaction costs, so each case study is held at its primary label*

*Figure 12.3* then presents the same comparison visually:

![Figure 12.3](assets/figure_12_3.png)

*Figure 12.3: Highest-IC configuration of each family per case study at the primary regression label, with HAC 95% confidence intervals. Filled markers indicate an interval that excludes zero; open markers indicate one that overlaps zero*

On the case studies where the linear model already found signal, the flexible families match or extend it. US Equities and NASDAQ-100 microstructure carry signal under all three families; on US Equities gradient boosting raises the daily IC from +0.016 to +0.032 with a tight interval, and on the NASDAQ-100 all three agree at a small but well-resolved value near +0.005. Gradient boosting then crosses the significance line on three case studies whose linear interval overlapped zero: US Firms (+0.080), CME Futures (+0.032), and the SP500 Options straddle (+0.018). This is the clearest sign that nonlinear structure carries ranking content the linear projection cannot reach. The reverse also happens. On ETFs the linear model posts the highest IC of the three (+0.054) and is the only family whose interval clears zero; gradient boosting (+0.037) and TabM (+0.041) order the cross-section slightly worse and less certainly. Cross-asset momentum is close to linear in these features, and tree splits spend capacity on interactions that do not survive out of sample. Where the linear baseline was flat (FX and the combined SP500 equity-and-options case study), all three families stay within noise of zero. Added capacity moves the point estimate without moving the interval off zero.

TabM follows gradient boosting in sign more than in magnitude: when boosting adds content, TabM usually does too, but the two disagree on which case study each serves best. TabM edges boosting on ETFs and FX; boosting is far ahead on US Firms and the SP500 Options straddle. The two encode nonlinearity through different inductive biases (axis-aligned tree splits against learned dense embeddings), and neither dominates across the case studies.

### What the lift rests on

The interval comparison in *Figure 12.3* already marks where boosting genuinely improves: on US Equities its interval clears the linear one outright, and on US Firms, CME Futures, and the SP500 Options straddle it crosses zero over a linear baseline whose own interval overlapped it. Where the two families’ intervals overlap (the NASDAQ-100, where both sit near +0.005), there is nothing to separate. Where boosting does add content, the mechanism is nonlinear conditioning: splits encode regime indicators, volatility-rank crossings, and cross-asset correlations whose predictive value depends on conditional structure a linear model cannot express. On the case studies with both saved linear coefficients and a stored booster, the most important features reorder substantially between the linear coefficient ranking and the boosting gain ranking, and the per-fold feature ranking is far more stable on US Firms than on US Equities even though US Equities carries the tighter IC. Credible cross-sectional signal does not require a stable top-feature set.

### Choosing the configuration

The registry grid crosses three loss functions (squared error, absolute error, and Huber) with leaf budgets from 7 to 127. This is a deliberately small slice of the hyperparameter space: it varies the loss and the leaf budget but leaves at their defaults the regularization controls that usually accompany tree complexity, namely maximum depth, the L1 and L2 leaf penalties, and feature and row subsampling (LightGBM documentation, *Parameters Tuning*). A production model would search those jointly; the goal here is to compare the families on equal footing, not to wring out every last basis point, and the book returns at the end to how lightly these demonstrations are tuned relative to a deployment workflow. On most case studies the loss changes the IC by less than one interval half-width: absolute error (MAE) posts the highest IC on eight of the nine primary labels and squared error (MSE) on the remaining one, ETFs, with Huber competitive but never highest. US Firms is the exception that shows why the default matters. *Figure 12.4* contrasts it with US Equities on shared axes. On US Firms the MAE curve sits near +0.080 across every leaf budget while Huber and MSE bunch around +0.030 to +0.035, a gap of two to three interval half-widths that absolute error opens by refusing to chase the heavy-tailed extreme returns squared error spends capacity on. On US Equities the same three losses fall within a single half-width at every depth; the configuration barely resolves and the operative choice lies elsewhere.

![Figure 12.4](assets/figure_12_4.png)

*Figure 12.4: Daily IC by loss function and leaf budget for the GBM grid, on US Firms and US Equities, with HAC 95% confidence intervals and shared axes. The configurations spread widely on US Firms but fall within a single interval half-width on US Equities*

Tree depth is the least consequential of these hyperparameters. The highest-IC leaf budget scatters across the interior of the grid (seven leaves on three case studies, 63 on three others), and within any case study the depth profiles differ by less than one interval half-width, so depth rarely changes whether the interval clears zero. The hyperparameter that pays operationally is early stopping. The boosting trajectories peak across a wide range: SP500 Options and ETFs by 50 trees, CME Futures and Crypto by 100, FX by 150, while US Firms peaks near 350 and US Equities, the NASDAQ-100, and the SP500 equity-and-options case study are still climbing at the 500-tree budget cap. Early stopping should be a tuning outcome rather than a fixed prior: a coarse checkpoint grid retires a large fraction of the Ranking each case study’s grid by HAC-corrected significance (𝑡ு஺஼) rather than by IC selects the trees on the early-peaking case studies and is simply inactive where IC keeps rising.

NASDAQ-100) both choices sit far inside the region where the interval excludes zero (𝑡ு஺஼≈17 and same configuration on seven of the nine case studies, and on the two exceptions (US Equities and the ≈4), so the substitution never changes whether a case study’s interval clears zero. The two rankings

nearly coincide because the HAC standard error varies little across configurations within a case study and family; scoring by IC and by significance reorders only configurations whose ICs are already statistically indistinguishable.

### Whether the lift persists

A single-period IC can hide a signal that lived in a few windows. *Figure 12.5* tracks the rolling threemonth daily IC of gradient boosting relative to the linear baseline across two contrasting case studies, with walk-forward fold boundaries marked. On US Equities, across 2000–2015, the boosting curve sits above the linear curve through most of the fifteen-year span, including the 2008–2009 dislocation. The improvement is a standing feature, not a one-window artifact. On ETFs, across 2015–2023, the two curves cross repeatedly and neither holds a durable margin, consistent with the linear model being the better-resolved choice there. Persistence, not the point estimate, is what separates a deployable edge from a lucky window.

![Figure 12.5](assets/figure_12_5.png)

*Figure 12.5: Rolling three-month daily IC for gradient boosting and the linear baseline on US Equities (2000–2015) and ETFs (2015–2023). Vertical rules mark walk-forward fold boundaries*

### Regression versus classification

Three case studies (Crypto, the SP500 equity-and-options case study, and US Firms) train gradient boosting on both the continuous return and the binary direction over the same horizon, allowing the two targets to be compared directly. Each model emits a continuous score, but they are trained and evaluated against different targets: the regression score is read by IC against the continuous return, and the classifier score by AUC against the realized direction (*Chapter 7*). *Table 12.4* cross-evaluates them, asking whether the scores are interchangeable: whether the classifier also ranks magnitudes, and whether the regression score also separates signs.

| Case study | Horizon | Classifier score → IC | Classifier → AUC | Regression score → AUC |
| --- | --- | --- | --- | --- |
| Crypto | 8 hours | +0.020 | 0.513 | 0.502 |
| US Firms | 1 month | +0.084 | 0.543 | 0.540 |
| SP500 Eq+Opt | 5 days | +0.004 | 0.507 | 0.489 |
| SP500 Eq+Opt | 10 days | −0.010 | 0.505 | 0.498 |

*Table 12.4: GBM regression-versus-classification cross-evaluation. “Classifier score → IC” reads the binary-direction model’s score by IC against the continuous return; “Classifier → AUC” scores that model against its own direction label; “Regression score → AUC” reads the continuous-return model’s score against direction. Bold marks an interval that excludes the null (zero for IC, one-half for AUC)*

They are not interchangeable. On Crypto the classifier’s eight-hour score reaches an IC of +0.020 against the continuous return, with an interval excluding zero, and its direction AUC of 0.513 clears one-half: the gradient-boosted classifier recovers directional content on Crypto that the linear classifier in *Chapter 11* did not. On US Firms the classifier score reaches IC +0.084 with a direction AUC of 0.543. On the SP500 equity-and-options case study neither target separates at either horizon: the classifier IC overlaps zero and its AUC sits at one-half. Read the other way, the regression score’s AUC against direction stays within ±0.04 of one-half on every pairing, so a model tuned to rank magnitudes is close to uninformative about sign. The same lesson held in *Chapter 11*; the loss function, logistic against squared, decides which discriminative content the score carries. Both views feed downstream: *Chapter 16*’s signal stage ranks on the IC view, and the direction-aware classifiers continue to drive long–short construction in the case-study chapters.

### Reading the model

Native gain and split-count importances rank features differently (gain rewards large conditional effects, split count rewards features used for fine partitioning), so neither alone explains a booster. TreeSHAP resolves the disagreement by decomposing each prediction into exact per-feature contributions. *Figure 12.6* shows the SHAP beeswarm for the ETFs and SP500 Options boosters: on ETFs the yield-curve slope and its z-score dominate the spread of contributions, while on the SP500 straddle the GARCH variance-risk-premium and the realized-volatility ratios lead. The fold-level view in the notebook shows some features contributing consistently and others only in particular regimes; monitoring the regime-dependent features for drift, as *Section 12.5* develops, is the operational implication.

![Figure 12.6](assets/figure_12_6.png)

*Figure 12.6: SHAP beeswarm of per-feature contributions for the GBM boosters on ETFs and SP500 Options. Each dot is one prediction; shade encodes the feature value; horizontal position is the SHAP value*

### From IC to profitability

An interval that clears zero certifies a ranking, not a return. On five case studies (US Firms, US Equities, CME Futures, the SP500 Options straddle, and the NASDAQ-100) gradient boosting supplies the cross-sectional ordering that the portfolio-construction and simulation chapters turn into positions. Where boosting does not clear zero, another family may still carry the ranking (the linear model does so on ETFs), so a null for boosting is not a null for the case study. *Chapters 16* through *19* take these ranked predictions through execution, cost, and risk, to ask which of the statistically distinguishable signals survives as economic profit.

## 12.7 Summary

Gradient boosting machines are the defensible default for tabular financial prediction. XGBoost, LightGBM, and CatBoost achieve similar accuracy after tuning. The real selection criteria are operational: training speed (LightGBM), categorical integrity (CatBoost), or ecosystem maturity (XGBoost). The cross-case study evaluation confirms that shallow trees with MAE loss and early stopping constitute a strong starting configuration, while monotonic constraints encode economic logic as regularization without sacrificing predictive power. TreeSHAP provides the interpretability infrastructure (dependence analysis, interaction decomposition, and drift monitoring) that production deployment demands.

Deep learning alternatives are narrowing the gap. TabPFN enables rapid signal testing without tuning; TabM provides a practical neural baseline competitive with GBMs on medium datasets. TabPFN degrades faster than GBMs under temporal distribution shift, and all benchmark claims must be validated under walk-forward protocols with purging and embargo before they translate to trading decisions. The regime-dependent decision framework in *Section 12.3* identifies where neural alternatives are most likely to pay off.

*Chapter 13* extends to sequential data, covering deep learning architectures (transformers, state-space models, and temporal convolutional networks) for time-series prediction, in which the temporal structure that GBMs treat as features becomes the model’s native input format.
