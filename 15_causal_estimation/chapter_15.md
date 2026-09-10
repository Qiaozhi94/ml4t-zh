# Chapter 15: Causal Machine Learning

We introduced causal thinking as a falsification filter in *Chapter 7*: directed acyclic graphs (DAGs) to encode mechanism assumptions, the three structural roles of confounders, mediators, and colliders to guide conditioning decisions, and four plausibility checks to triage features before committing to heavier modeling. Those diagnostics are bivariate – they test one feature at a time and cannot detect multivariate confounding. This chapter provides the multivariate estimation machinery that the survivors require.

Feature-level triage asked whether a single feature’s association with the label is consistent with a proposed mechanism. The methods here ask harder questions:

- What is the magnitude of a treatment effect after orthogonalizing against high-dimensional confounders?
- Did a discrete event actually move prices relative to a data-driven counterfactual?
- Which variables in a multivariate system cause which, and at what lags?

Each question demands a different estimator, and each estimator rests on assumptions that must be stated and tested. After completing this chapter, you will be able to:

- Define a **causal research question** in terms of treatment, outcome, estimand, and counterfactual, and use DAGs to justify an admissible adjustment set.
- Apply validation and refutation tools, including placebo tests, sensitivity analysis, and subset-stability checks, to assess whether a causal claim is robust enough to inform trading research.
- Use Double Machine Learning (DML) to **estimate causal effects of continuous treatments** in the presence of high-dimensional confounders, while respecting temporal cross-fitting and pre-treatment timing discipline.
- Use Bayesian Structural Time-Series (BSTS) to **estimate the** **impact of discrete events** by constructing data-driven counterfactuals and evaluating spillover risk in control series.
- **Use causal discovery methods** such as PCMCI, NOTEARS, and VAR-LiNGAM to generate candidate structures and interpret them as hypotheses requiring further validation rather than as definitive causal truth.
- Distinguish predictive signal from causal effect, and interpret cross-dataset evidence with attention to confounding bias, multiple testing, and the gap between statistical significance and refutation survival.

The chapter begins with the conceptual bridge from predictive features to causal estimands, then develops the identification framework in *Section 15.2* and the validation and refutation toolkit in *Section 15.3* that every causal claim must survive. *Section* *15.4* applies Double Machine Learning to continuous treatment-effect estimation across the case studies, while *Section* *15.5* shifts to discrete-event impact measurement using BSTS. *Section* *15.6* reverses the direction of inquiry, using causal discovery algorithms to infer structure from data rather than imposing it. *Section 15.7* synthesizes the cross-dataset evidence, and *Section 15.8* distills the practical guidelines.

## 15.1 From theory to estimation

The conceptual toolkit from *Section 7.5* (DAGs, structural roles, mechanism plausibility checks) establishes *whether* a causal story is worth pursuing. This section bridges from that qualitative assessment to quantitative estimation: given a DAG and an identification strategy, how do we measure causal effects with valid uncertainty quantification?

### The causal estimation challenge

Classical causal designs begin with a research design, not with an algorithm. They ask **what source of variation could identify the effect of interest**:

- Backdoor adjustment with observed controls
- An instrument that shifts treatment without directly affecting the outcome
- A before-and-after comparison with a credible control group
- A threshold that approximates random assignment
- A counterfactual path constructed from an untreated series

In each case, the central question is the same: *why should the observed data isolate the causal effect rather than reflect confounding, selection, anticipation, or spillover*?

In finance, these designs remain essential, but they are often difficult to apply cleanly. Confounders are numerous, nuisance relationships are nonlinear, effects vary across regimes, events propagate across assets, and treated units are often few. Machine learning helps estimate these **nuisance components** more flexibly, but it does not create identification where none exists. The methods in this chapter, therefore, combine causal design with modern predictive tools:

- **Double Machine Learning** (DML) for continuous treatments under backdoor-style adjustment
- **Bayesian Structural Time-Series** (BSTS) for event-level counterfactual construction
- **Causal discovery** methods for generating structural hypotheses when the graph is not known in advance

Before we make these design families concrete, we need to distinguish identification from estimation, and clarify where the chapter’s three main methods fit.

### Three estimation problems

The chapter is organized around **three distinct causal questions** within that broader design landscape, each requiring a different estimator:

1. Does momentum *cause* forward returns after controlling for volatility and regime? **DML** orthogonalizes the treatment against confounders using flexible ML models, yielding debiased effect estimates with valid confidence intervals.
2. How much did a Fed rate decision move bond ETF prices relative to what would have happened absent the announcement? **BSTS** constructs a synthetic counterfactual from the control series and measures the gap.
3. Which cross-asset return series actually drive others, and which merely co-move? **PCMCI** tests conditional independence at multiple lags, recovering a time-lagged causal graph; neural methods (NOTEARS, VAR-LiNGAM) scale discovery to higher dimensions.

*Table 15.1* maps research questions to estimators and Python libraries.

| DAG known? | Treatment type | Method | Library |
| --- | --- | --- | --- |
| Yes | Binary/Cont. | Backdoor adj. | DoWhy |
| Yes | Continuous | DML | EconML |
| Yes | Binary (event) | BSTS | tfp-causalimpact |
| No | N/A (discovery) | PCMCI | Tigramite |
| No | N/A (discovery) | NOTEARS | causal-learn |
| No | N/A (discovery) | VAR-LiNGAM | causal-learn |

*Table 15.1: Method selection guide, from research question to Python library*

A sophisticated estimator cannot rescue an identification failure. The pipeline runs: DAG specification → adjustment-set identification → estimand derivation → ML-based estimation → refutation testing. *Section 15.2* covers the shared identification and validation framework that precedes all three estimation strategies; the method-specific sections that follow assume valid identification is in place.

The goal is not to prove causation from observational data but to distinguish robust claims from those that collapse under scrutiny. A key step in this regard is identifying the causal mechanism, to which we turn next.

**Implementation**: `01_library_overview` compares the libraries in *Table 15.1*.

## 15.2 Identification and validation

Causal estimation is only as credible as the design that supports it. Estimation asks how to compute an effect once the causal question has been defined. Identification asks the prior question: whether the effect can be learned from the available data under explicit assumptions. This distinction matters in financial markets because prediction, correlation, and causation often point in different directions. A variable may predict returns because it captures risk, proxies for a hidden market state, is downstream of the outcome, or reflects a genuine causal mechanism. Causal analysis begins by separating these possibilities before fitting a model.

### From outcome and intervention to estimand

Every causal question requires specifying what is being varied, what is being measured, and what comparison defines the effect. These choices should be made before fitting a model.

The **intervention**, or treatment, is the exposure whose effect we want to estimate. In markets, treatments are rarely interventions that a researcher can assign. They are usually events, rules, or shocks that approximate interventions: a change in index membership, a regulatory change, an earnings surprise, or a revision to a portfolio allocation rule. The *treatment definition* should specify the treated units, the timing of treatment, and the relevant treatment horizon.

The **outcome** is the variable whose response defines the question. It should match both the proposed mechanism and the decision problem. Depending on the application, the outcome may be a forward return, volatility, drawdown, liquidity measure, or risk exposure. Outcome choice is part of the research design because it defines the estimand and affects the plausibility of the identifying assumptions. A mechanism closely related to the outcome may support a more credible causal claim than a broad trading outcome that is influenced by many other forces. This does not make the broader outcome irrelevant; it means the causal claim is likely harder to defend.

The **estimand** is the causal quantity to be estimated. Common examples include:

- **Average Treatment Effect** (**ATE**): the average effect across the target population
- **Average Treatment Effect on the Treated** (**ATT)**: the average effect for the units that actually received treatment, often the relevant estimand in event studies
- **Conditional Average Treatment Effect** (**CATE):** the effect conditional on observed state variables, such as volatility regime, liquidity, or valuation conditions

The estimand also fixes the target population, treatment timing, outcome horizon, and comparison being made. For example, the effect of a funding-rate shock on next-period premium reversion is not the same estimand as the effect of the same shock on multi-day forward returns. The first asks whether the funding premium itself mean-reverts after the shock. The second asks whether that shock translates into a broader trading outcome after other return drivers have also acted.

Several closely related terms recur throughout the chapter and are easy to conflate. The following definitions are used consistently in the sections and notebooks that follow.

| Term | Definition |
| --- | --- |
| ATE | Average marginal treatment efefct across the target population, under the specified controls |
| Subgroup ATE | The ATE re-estimated within a stratum, such as a volatility regime |
| CATE | A covariate-conditional efefct from a single heterogeneity model (causal forest, X-learner) |
| Efefct modifier | A variable along which the treatment efefct is allowed to vary |
| Confounder (control) | A pre-treatment variable used for adjustment to block a backdoor path |
| Refutation | A diagnostic that can weaken a causal claim but cannot prove it |

*Table 15.2: Definitions for key terms*

The distinction between a subgroup ATE and a CATE matters in practice. Splitting the sample by regime and re-estimating the ATE within each stratum yields subgroup ATEs; it does not fit a model of how the effect varies with covariates. A CATE requires a heterogeneity estimator that directly learns the effect surface. Likewise, a confounder and an effect modifier play different roles: a confounder is adjusted for to remove bias, whereas an effect modifier defines how the effect changes. Many estimators, including the double machine learning interface used later in this chapter, expose these two roles through separate arguments, and using the wrong one changes the quantity being estimated.

Several assumptions are required before an estimand can be interpreted causally:

- **Consistency** means that the observed outcome under the observed treatment corresponds to the relevant potential outcome
- **Overlap** means that the data support the comparison, so both treated and untreated observations are available for the relevant covariates
- **No interference**, or **Stable Unit Treatment Value Assumption (SUTVA)**, means that one unit’s treatment does not change another unit’s outcome
- **Conditional exchangeability** means that, after conditioning on an admissible set of pre-treatment variables, treatment assignment is as good as random with respect to the potential outcomes

These assumptions are **demanding in financial markets**. Prices aggregate information quickly, assets interact through common factors and portfolio flows, and treatment timing is often ambiguous. The point is not to pretend these assumptions are automatically satisfied. The point is to state the causal question precisely enough that the assumptions can be inspected.

Define the treatment, outcome, estimand, target population, and timing before fitting any model. form of the selection bias it aims to eliminate. We will use 𝑇 to denote the treatment, 𝑌 the outcome, Without this discipline, causal analysis degenerates into post-hoc rationalization, which is another and 𝑊 a set of pre-treatment covariates.

### Directed acyclic graphs and adjustment sets

A directed acyclic graph, or DAG, makes the identification assumptions explicit. *Section 7.5* introduced confounders, mediators, and colliders as structural roles in a causal graph. Here, those roles become rules for deciding which variables belong in the adjustment set, a specific group of covariates conditioned on to estimate the causal effect of an intervention. The central problem is confounding. A **confounder** is a common cause of treatment and outcome. If market stress affects both funding rates and future returns, then a raw association between funding rates and returns may partly reflect a common driver rather than a causal effect of funding rates. **Adjustment** aims to compare treated and untreated observations that are similar with respect to such common causes. ables 𝑊 is an admissible adjustment set for the effect of 𝑇 on 𝑌 if it blocks every noncausal path from The **backdoor criterion** formalizes the conditions under which this comparison is valid. A set of vari-𝑇 to 𝑌 that begins with an arrow into 𝑇, and if it contains no descendants of 𝑇. Under consistency, overlap, no interference, and conditional exchangeability given 𝑊, the interventional distribution for

a discrete treatment can be written as: 𝑃൫ܻ = ݕפ݀ ݋(ܶ = ݐ)) = ∑𝑃 ( = ݕפܶ = ݐǡܹ = ݓ) 𝑃(ܹ = ݓ)

௪

For a continuous outcome, the same idea is often more useful in expectation form: 𝔼[ פ݀ ݋(ܶ = ݐ)] = ∑𝔼 [ פܶ = ݐǡܹ = ݓ]Ԝܲ(ܹ = ݓ)

௪ When 𝑊 is continuous, the corresponding expression is:

𝔼[ פ݀ ݋(ܶ = ݐ)] = ∫𝔼[ܻ פܶ = ݐǡܹ = ݓ]Ԝ݀ܨௐ(ݓ)

𝑊, where the relevant backdoor paths have been blocked, and then averages those conditional com-Adjustment compares treated and untreated observations within strata of admissible covariates parisons over the population distribution of 𝑊. In words, it asks what the outcome would look like if everyone were set to treatment level 𝑡, while the distribution of pre-treatment covariates remained

unchanged. how they approximate the conditional expectation 𝔼[ פܶ ൌݐǡܹ ൌݓ]. Linear regression imposes a This expression defines the estimand implied by the DAG assumptions. Estimation methods differ in

parametric form. Matching and weighting more directly approximate the comparison. Double machine all cases: is 𝑊 an admissible adjustment set? learning uses flexible models for the nuisance functions. The identification question comes first in

because conditioning on them blocks spurious association. Strong pre-treatment predictors of 𝑌 may The answer depends on **causal structure**, not predictive power. True confounders are good controls

also improve precision, provided they are not colliders, mediators, treatment descendants, or selection variables. By contrast, poor controls can worsen a causal estimate. Conditioning on a mediator removes part of the pathway through which treatment affects the outcome, thereby changing the estimand. Conditioning on a collider opens a noncausal path that was previously closed. Conditioning on a treatment descendant introduces post-treatment bias.

This is why **kitchen-sink regression is dangerous** in causal work. Adding more variables does not automatically make an estimate safer. It may do the opposite if some of those variables sit downstream of treatment or are common effects of treatment and outcome. The DAG determines which variables belong in the adjustment set. Statistical significance, feature importance, and predictive accuracy do not. Timing discipline is the most practical screen in trading applications. Every control in 𝑊 should be

known before treatment is realized. Event studies should avoid variables that may react contemporaneously to the event, especially when macro announcements move broad markets immediately. Factor studies should avoid realized performance, portfolio outcomes, or variables mechanically constructed from future returns. This timing rule implements the requirement that the adjustment set contain no treatment descendants.

Timing discipline is necessary, but not sufficient. A variable observed before treatment can still be a collider, a selection variable, or a proxy for a conditioning event that changes the population being analyzed. For example, filtering on liquidity, analyst coverage, or fund flows may appear harmless because these variables are observed before the outcome horizon. Yet they can still create bias if they are common effects of prior performance, investor attention, and treatment exposure. The DAG remains the governing object.

This also clarifies the relationship to the simpler checks from *Section 7.5*. The timing placebo and shared-driver checks introduced there were informal, single-feature versions of the same logic. A feature that survives those checks has passed an initial plausibility screen. The adjustment-set problem asks the harder multivariate question: whether the apparent effect survives conditioning on a causally admissible set of pre-treatment confounders.

In practice, the DAG should be written down before the estimator is chosen. Libraries such as `DoWhy` make this discipline operational by requiring the researcher to specify the graph, declare the treatment and outcome, and identify the estimand implied by the assumptions. The software does not remove judgment; it exposes where judgment enters. If identification fails, the problem is not that the estimator is weak. The problem is that the research design has not justified a causal interpretation.

Once the DAG has been translated into an admissible adjustment set and an explicit estimand, the next question is whether adjustment on observed controls is credible. When important confounders may be unobserved, poorly measured, or impossible to condition on without introducing new bias, identification must come from a different source of variation.

### Alternative identification designs

Backdoor adjustment identifies a causal effect by conditioning on observed confounders. When that strategy is not credible, the design problem changes. The question is no longer only which controls to include, but which source of variation can plausibly separate treatment from the factors that would otherwise confound the comparison.

Each alternative design uses a different comparison. Instrumental variables use variation in treatment induced by an external instrument. Difference-in-differences uses changes over time between treated and untreated groups. Regression discontinuity uses assignment rules near a threshold. Event-study counterfactual methods use pre-event dynamics and control series to estimate a no-event path. These designs differ in implementation, but they serve the same purpose: they replace broad unconfoundedness with a narrower assumption about the source of identifying variation.

![Figure 15.1](assets/figure_15_1.png)

*Figure 15.1: Alternative identification designs isolate causal effects using different comparisons*

Instrumental variables use a variable 𝑍 that shifts treatment 𝑇 but affects the outcome 𝑌 only through Instrumental variables

that treatment. The instrument must satisfy three conditions: **Relevance** requires that 𝑍 meaningfully moves 𝑇. **Independence** requires that 𝑍 is unrelated to the unobserved causes of 𝑌. 1. **Exclusion** requires that 𝑍 has no direct path to 𝑌 except through 𝑇. 2. 3.

The exclusion restriction is usually the hardest condition to defend. A policy announcement, an index assignment rule, or a change in market structure may shift the treatment of interest, but it may also affect attention, liquidity, risk appetite, funding conditions, or other channels that directly influence causal effect of 𝑇 alone. the outcome. If those channels cannot be ruled out, the IV estimate should not be interpreted as the

IV estimates also require careful interpretation. When treatment effects vary across units, the design often identifies a local effect for the units whose treatment status is shifted by the instrument, rather than an average effect for the entire population. This may still be the relevant estimand, but it should be stated explicitly. We do not implement IV estimation in this chapter; for applied work, the `linearmodels` library provides standard estimators.

#### Difference in differences

**Difference in differences** (**DiD**) identifies effects from differential changes over time. It compares treated and control units before and after an event, asking whether treated units changed more than comparable untreated units. The identifying assumption is that, absent treatment, the treated and control groups would have followed parallel trends. A typical finance example is a short-selling ban that applies to one group of stocks but not to another. The relevant comparison is not simply whether treated stocks moved after the ban. The question is whether their returns, spreads, or volatility differed from those of a comparable control group exposed to the same broad market conditions.

The main threat is that treated and control assets may not share the same counterfactual trend. Sector shocks, time-varying betas, liquidity regimes, or macro news can all generate differential movements unrelated to the treatment. Pre-trend checks, narrow event windows, and economically justified controls can make the design more credible, but they do not prove parallel trends. In panel settings with staggered treatment timing, simple fixed-effects implementations also require care, as heterogeneous treatment effects can distort the resulting estimate.

#### Regression discontinuity

Regression discontinuity exploits treatment assignment rules with sharp thresholds. If units just above and just below a cutoff are otherwise comparable, the discontinuous change in treatment at the threshold can approximate local randomization.

In finance, thresholds arise in index inclusion rules, credit ratings, eligibility constraints, regulatory categories, and portfolio mandates. The identifying comparison is local: units close to the cutoff are compared with nearby units on the other side of the cutoff. The resulting estimate applies to the margin around the threshold, not necessarily to the full population.

The key assumption is continuity. In the absence of treatment, potential outcomes should evolve smoothly through the cutoff. Credibility is weakened if units can manipulate their position around the threshold, if other rules change at the same cutoff, or if too few observations lie near the boundary. RD can be highly persuasive when the assignment rule is transparent and difficult to manipulate, but its strength comes from its locality.

#### Event study counterfactuals

Event-study counterfactual methods identify effects by comparing the observed path after an event with an estimated no-event path. The source of the identified variation is event timing, combined with the assumption that pre-event dynamics and unaffected control series can predict what would have happened without the event.

BSTS, discussed in *Section 15.5*, implements this logic with a Bayesian time-series model. It estimates a counterfactual path from pre-event behavior and contemporaneous controls, then attributes the gap between the observed and counterfactual paths to the event. The design is synthetic-control-like because it uses control series to model the untreated trajectory, but it remains a model-based counterfactual rather than a randomized experiment.

The main threat is contamination. Controls must help predict the treated series while remaining unaffected by the event. This is difficult for macro events because the same announcement may move nearly all plausible control assets. For example, an unexpected Fed decision may affect a bond ETF, related rates instruments, credit spreads, equity indices, and risk sentiment at the same time. In such settings, the estimate should be interpreted as model-dependent and supported with placebo dates, alternative control sets, and sensitivity checks.

#### Choosing among designs

The right design is the one whose identifying variation is most credible for the question. Backdoor adjustment is appropriate when the main confounders are observed and can be conditioned on without introducing post-treatment or selection bias. IV is useful when an external source shifts treatment but does not otherwise affect the outcome. Difference-in-differences is appropriate when treated and control units plausibly share a counterfactual trend. Regression discontinuity is strongest when a threshold creates locally comparable treated and untreated units. Event-study counterfactuals are useful when the no-event path can be modeled from stable pre-event relationships and uncontaminated controls.

No design is automatically superior. Each exchanges one set of assumptions for another. The research task is to choose the comparison whose assumptions are most defensible, state the corresponding estimand, and then test whether the resulting claim is fragile under plausible perturbations.

## 15.3 Validation and refutation

Identification follows from assumptions. Validation asks whether the resulting claim is fragile, contradicted by placebo evidence, or dependent on implausibly strong relationships with omitted confounders. These checks do not prove causality. They test whether the design behaves as it should if the identifying assumptions are credible.

A useful validation workflow separates three questions:

- First, does the method detect effects where none should exist? This is the role of placebo and negative-control tests.
- Second, how much omitted confounding would be required to overturn the result? This is the role of sensitivity analysis.
- Third, does the estimated effect persist across reasonable samples, specifications, and outcome definitions? This is the role of stability checks.

### Placebo and negative control tests

Placebo tests ask whether the pipeline produces an effect when the treatment, timing, or outcome should not generate one. A credible design should estimate near-zero effects in these settings. If placebo effects are large or frequent, the model is likely capturing confounding, leakage, overfitting, or contaminated controls.

A **placebo treatment** replaces the actual treatment with a randomized, shifted, or otherwise irrelevant treatment assignment. For example, an event-study pipeline can be rerun by moving event dates to periods when no event occurred. A backdoor adjustment design can be tested using a treatment variable that should not affect the outcome once the same controls are applied. The goal is not to obtain exactly zero in finite samples, but to verify that the method does not systematically discover effects where the design says none should exist.

A **placebo date** is especially useful for event studies. The same counterfactual model is applied to dates without the event. If the model repeatedly finds large effects on non-event dates, the estimated event effect may reflect model instability or contaminated controls rather than the event itself. A **negative control** is an outcome that should not respond to the treatment. If the treatment appears to affect such an outcome, the result points to residual confounding, timing leakage, or misspecified adjustment. Negative-control outcomes are especially useful when the proposed mechanism is specific. They test whether the estimated effect appears only where the mechanism predicts it should appear.

### Sensitivity to omitted confounding

Adjustment only addresses confounders that are measured and included in an admissible way. Even after careful DAG design, unobserved confounding may remain. Sensitivity analysis asks how strong an omitted confounder would need to be to materially change the conclusion.

A practical implementation perturbs the design by adding synthetic confounding, varying assumed confounder strength, or measuring how the estimate changes under alternative assumptions about hidden common causes. `DoWhy` supports refutation tests and sensitivity checks that probe this kind of robustness. These diagnostics do not prove that omitted confounding is absent. They quantify how dependent the conclusion is on the assumption that remaining confounding is limited.

The interpretation should be conservative. If a small omitted-confounder perturbation changes the sign, magnitude, or significance of the estimate, the result is not separable from the assumed confounder structure at the strengths tested. It should be treated as a hypothesis rather than as evidence for a causal effect. If the estimate remains stable under large and economically plausible perturbations, the claim becomes more credible, but still conditional on the design assumptions.

### Stability across samples, specifications, and outcomes

Stability checks assess whether the estimated effect is specific to a single sample, model, or outcome definition. They are not substitutes for identification, but they reveal whether the claim depends on narrow modeling choices:

- **Subsample stability** checks whether the effect persists across time periods, regimes, assets, or market conditions. A funding-rate effect that appears only in one short volatility regime, for example, may still be real, but the estimand should then be described as regime-specific rather than general.
- **Specification stability** checks whether the effect survives reasonable alternative adjustment sets, nuisance-function models, or estimator choices. In DML, this includes varying the models used for the treatment and outcome nuisance functions. In event-study counterfactuals, it includes using alternative control series and model windows. A claim that depends on one narrow specification is weaker than a claim that survives a defensible range of specifications.
- **Outcome triangulation** assesses whether the pattern of effects aligns with the proposed mechanism. The same treatment need not affect every outcome in the same direction or with the same strength. A mechanism-near outcome may show a clearer effect than a broad trading outcome influenced by many other forces. The relevant question is whether the estimated effects appear where the mechanism predicts they should appear, weaken where the mechanism is indirect, and disappear for outcomes that should not respond.

Taken together, placebo tests, sensitivity analysis, and stability checks define the evidentiary status of the result. A claim that passes these checks is not proven causal. It is credible under the stated assumptions and tested perturbations. A claim that fails them should not be discarded automatically; rather, it should be downgraded from a causal conclusion to a research hypothesis that requires a stronger design.

**Implementation**: `02_dowhy_causal_graph` walks through the full validation workflow (DAG specification, identification, estimation with temporal splits, placebo and refutation checks) for the crypto funding-rate two-outcome contrast (premium reversion against forward returns). We develop the quantitative diagnostics in the next section.

## 15.4 Isolating factor effects with DML

**Double/Debiased Machine Learning** (**DML**) bridges modern machine learning and causal inference (Chernozhukov et al., 2018). It estimates the causal effect of a factor while controlling for many potential confounders, using flexible ML models for the nuisance components and a debiased score for the target parameter.

The *identification checklist* for DML, based on *Section 15.2*, reads as follows: Treatment 𝑇 timing is fixed; covariates 𝑋 are strictly pre-treatment

• Unconfoundedness argued: 𝑋 blocks all backdoor paths from 𝑇 to 𝑌

- Estimand declared (ATE, ATT, or CATE) •
- Cross-fitting respects time ordering (expanding or rolling windows, not random folds)
- Sensitivity analysis probes robustness to omitted confounders

### The intuition behind double machine learning

DML estimates a *low-dimensional set of causal parameters* while using flexible machine learning models to capture the high-dimensional nuisance relationships among covariates, treatment, and outcome (Chernozhukov et al., 2018). In this chapter, the relevant setting is a *backdoor-style design*: after specifying treatment 𝑇, outcome 𝑌, and an admissible pre-treatment adjustment set 𝑋 (the covariate set 𝑊 from *Section 15.2*), DML estimates the effect of 𝑇 on 𝑌 while allowing the relationships involving 𝑋 to be nonlinear and high-di-

mensional.

In its most intuitive form, DML proceeds in three steps (see *Figure 15.2*): **Predict the outcome from the controls.** Estimate the nuisance function 𝑔(ܺ) ൌॱ[ܻ פܺ ]. This captures the part of the outcome predictable from the controls. The residual 𝑌̃ ൌ𝑌െ݃ො(ܺ) is 1. the part of the outcome not explained by 𝑋. **Predict the treatment from the controls.** Estimate the nuisance function 𝑚(ܺ) ൌॱ[ܶ פܺ ]. The residual 𝑇̃ ൌ𝑇െ݉ෝ(ܺ) is the part of the treatment not explained by 𝑋. 2. **Estimate the causal effect from the residualized variables.** Regress 𝑌̃ on 𝑇̃. In the partially

3. of 𝑇 on 𝑌. linear case, the slope from this final-stage regression is the DML estimate of the causal effect

This three-step workflow is the machine-learning analog of the **Frisch-Waugh-Lovell** logic from econometrics: remove the variation explained by admissible controls, then estimate the relationship in the remaining variation. It also generalizes the partial-IC idea from *Section 7.5*, in which known drivers were partialled out before examining the residual signal.

![Figure 15.2](assets/figure_15_2.png)

*Figure 15.2: The Double ML workflow: orthogonalize both treatment and outcome against confounders, then regress the residuals*

What makes DML statistically useful, however, is not residualization alone. The key idea is that DML uses an **orthogonal score**, meaning an estimating equation constructed so that small errors in the nuisance models have only a second-order effect on the target causal parameter. This makes it possible to use flexible machine learning models for the nuisance functions without letting regularization bias dominate inference on the treatment effect (Chernozhukov et al., 2018).

DML also uses **cross-fitting**. Nuisance models are trained on one fold and evaluated on another, so the same observations are not used for both learning the nuisance structure and scoring the causal effect. This reduces overfitting bias in the final estimate. For time-series data, cross-fitting must respect temporal ordering: use expanding or rolling windows rather than random folds, and include a temporal gap when needed to avoid leakage from overlapping forward returns.

This point is important for interpretation. The three-step residualization picture is the clearest intuition for the *partially linear ATE-style setup* used in this section. More general DML-style estimators extend the same logic to treatment effects on the treated and to heterogeneous treatment effects, but they do so using different orthogonal scores and specialized learners rather than a single generic regression recipe.

DML, therefore, depends on two ingredients:

1. **Credible identification**: the covariates (X) must form an admissible adjustment set under the causal design described in *Section 15.2*.
2. **Adequate nuisance estimation**: if the nuisance models (the ML models used in steps 1 and 2) are too weak, unstable, or poorly tuned, the resulting causal estimate may be noisy or fragile even when the design itself is sound.

For this reason, nuisance-model sensitivity is part of the empirical diagnosis: poorly specified nuisance models produce biased causal estimates. Mitigate nuisance-model dependence by using flexible learners (gradient boosting, random forests), tuning via cross-validation, and comparing estimates across at least two different model classes.

Large changes in the estimated effect across reasonable nuisance learners should be treated as evidence `03_econml_dml` illustrates this: the DML ATE ranges from −0.055 (linear nuisance) to −0.060 (shallow that the causal conclusion is unstable (Chernozhukov et al., 2018). The ETF momentum analysis in gradient boosting) to −0.088 (deep gradient boosting), a roughly 1.6 × spread that reflects genuine

estimation uncertainty.

Beyond average effects, frameworks such as `EconML` support **heterogeneous treatment effects**: estimating how causal impacts vary across conditions (for example, whether momentum works differently in high- and low-volatility regimes). `EconML` provides metalearners and causal forests for this purpose.

A closely related application is *post-double-selection LASSO* for factor-zoo validation. Adapting Belloni, Chernozhukov, and Hansen (2014) following Feng, Giglio, and Xiu (2020), one tests whether candidate factors retain marginal pricing power after controlling for the existing factor zoo. Among the naive significance bar at 𝑡; that signal collapses to insignificance once the first ten principal four managed-portfolio factors examined in `11_factor_zoo_validation`, only MeanRev clears the

components of the asset universe are included as controls: the BCH/FGX correction substantively changes the inferential conclusion.

**Implementation**:

- `03_econml_dml`: DML for ETF momentum, manual and EconML implementations; nuisance-model sensitivity sweep across linear and gradient-boosting specifications; block-permutation refutation.
- `11_factor_zoo_validation`: post-double-selection LASSO applied to four managed-portfolio factors, with principal-component controls.

### Measuring the causal impact of momentum

We apply DML to the ETF Rotational Momentum strategy to estimate the causal effect of momentum on forward returns, controlling for volatility and market regime, two confounders that affect both the treatment and the outcome.

The causal framework is:

- **Treatment (T)**: Momentum score (risk-adjusted 6-month return)
- **Outcome (Y)**: 21-day forward return
- **Confounders (X)**: Realized volatility, yield curve regime

The naive approach (regressing returns on momentum) produces a biased estimate because volatility and regime affect both treatment and outcome. OLS slope is −0.0385 with HAC-corrected 95% confidence interval [−0.0484, −0.0285], a negative Applied to ETF momentum data in `03_econml_dml` under a strict temporal train/test split, the naive

relationship consistent with reversal at the 21-day horizon. The HAC bandwidth matches the 21-day understate. After orthogonalization, the EconML DML estimate is −0.0537 with 95% confidence forward outcome, since overlapping returns induce autocorrelation that a shorter bandwidth would interval [−0.0613, −0.0462], which excludes zero; a manual DML implementation with walk-forward cross-fitting agrees at −0.0494. The causal effect of momentum on forward returns becomes *more* neg-

magnitude by roughly 28%, so naive factor research misses part of the effect rather than overstating ative once volatility and regime confounders are controlled for. The naive slope understates the DML

it. The confounders enter the estimator as controls used for residualization, not as effect modifiers, the distinction that separates a single average effect from a model of how the effect varies.

On crypto perpetual funding rates, the picture is reversed: in `04_dml_crypto_regime`, the naive and DML estimates keep the same sign, but confounders suppress roughly 90 percent of the effect magnitest that shuffles treatment in seven-day blocks passes (𝑧), confirming the effect is distinguishtude, so the naive estimate captures only a fraction of the DML-estimated signal. A block-permutation able from placebo; shortening the blocks to one day inflates the statistic to 𝑧, an illustration

however (naive 𝑡44, DML 𝑡23): even DML cannot guarantee robustness when the underthat too-short blocks understate the autocorrelation null. Neither estimate is statistically significant,

lying signal is weak.

Confounders distort the naive estimate through two channels: high volatility weakens momentum signals and increases return dispersion, and regime transitions affect both the treatment and the outcome. In the crypto configuration, regime-stratified estimates remain directionally similar, and (𝑡63, 𝑝53), so the dominant lesson is that the effect magnitude is confounded rather than a single-model interaction test finds the high-vol versus low-vol difference statistically insignificant

that regime heterogeneity is strong.

When evaluating any factor, the workflow proceeds as follows:

1. Identify potential confounders based on domain knowledge.
2. Apply DML to estimate the causal effect after controlling for confounders.
3. Compare to the naive estimate to quantify confounding bias.
4. Examine confidence intervals to assess statistical certainty.
5. Test robustness using refutation tests (placebo treatments, random common causes, data subsets) and block-permutation tests.

A narrow confidence interval that excludes zero, combined with stable estimates in sensitivity analyses, provides stronger evidence of factor robustness than correlation-based approaches. Robust estimates change minimally when synthetic confounders are added or when the estimator is run on data subsets; fragile estimates flip sign under small perturbations. This qualitative difference determines which causal claims can support trading decisions.

### Outcome choice shapes causal credibility

A key lesson from the crypto funding-rate analysis is that causal credibility depends not only on the treatment and estimator, but also on the outcome being analyzed. The same treatment can support a more or less plausible causal interpretation depending on whether the outcome is tightly connected to the proposed mechanism or exposed to many broader market forces.

We define the treatment as episodes of extremely high funding-rate premiums: observations where the funding-rate z-score exceeds 2. We then compare two outcomes: subsequent forward returns and subsequent premium reversion. These are related but distinct causal questions. Forward returns are a broad market outcome. Extreme funding rates may contain information about future returns, but the relationship is exposed to common drivers such as speculative demand, leverage, volatility, liquidity conditions, and market-wide risk appetite. These forces may influence both the treatment and the outcome, making the causal interpretation fragile.

Premium reversion is more closely tied to the institutional mechanism of perpetual futures. Funding is designed to help anchor the perpetual contract price to the spot price. When the perpetual trades at a large premium, positive funding makes long positions more expensive and short positions more attractive. This creates an incentive, subject to trading costs, liquidity, and limits to arbitrage, to sell the perpetual contract and buy the spot asset. The proposed treatment-outcome link is therefore narrower and easier to connect to an explicit market mechanism.

This does not mean that premium reversion is automatically causal. Because the treatment is defined by an extreme value of the same funding-rate process, subsequent reversion may partly reflect ordinary mean reversion after an extreme observation. The stronger claim is therefore not that premium reversion proves an arbitrage effect, but that it is a more mechanism-consistent outcome than broad forward returns.

`DoWhy`'s *refutation checks* help summarize this distinction. Here, we report two diagnostics derived from the refutation results:

1. **Out-of-sample** **drift** (OOS) measures how much the estimated effect changes when the model is refitted or evaluated across alternative sample splits; larger drift indicates lower stability.
2. **Placebo ratio** reports the share of placebo refutations in which the placebo estimate is large enough to challenge the original effect; a higher ratio suggests that comparable effects are easier to reproduce under falsified conditions.

These diagnostics do not establish identification and do not rule out omitted confounding. They provide narrower evidence: whether the estimate remains stable across resamples and whether similar effects appear under placebo treatments or outcomes.

| Outcome | Mechanism | OOS Drift | Placebo Ratio |
| --- | --- | --- | --- |
| Forward returns | Broad market outcome exposed to sentiment, positioning, and risk appetite | 85% | 12.7% |
| Premium reversion | Mechanism-adjacent outcome linked to funding incentives and arbitrage pressure | 51% | 5.7% |

*Table 15.3: Refutation diagnostics for two outcome choices*

The results are consistent with this interpretation. The forward-return estimate is substantially more fragile: its estimated effect changes more across out-of-sample configurations, and placebo tests more often produce effects resembling the original estimate. This is unsurprising for a broad market outcome. Extreme funding rates may predict returns, but the same latent forces that produce extreme funding may also drive subsequent returns. Unless these forces are adequately measured and adjusted for, the estimated effect remains vulnerable to confounding.

The premium-reversion estimate is less fragile, but not conclusive. Its lower OOS drift and placebo ratio suggest that the result is more stable and less easily reproduced under falsified conditions. This is consistent with the proposed mechanism: high funding premiums alter the relative attractiveness of long and short positions in the perpetual market and can create pressure for the premium to compress. At the same time, the remaining 51% drift is not small, and the outcome is mechanically close to the treatment definition. The evidence, therefore, supports a *cautious conclusion*: premium reversion is a more credible outcome for this causal question than forward returns, but the refutation checks do not by themselves prove a causal arbitrage effect.

The broader methodological lesson is that causal estimation should begin with the treatment-outcome pair, not the estimator. Before applying DML or any other causal method, ask whether the outcome is connected to a plausible, testable mechanism. A narrow, mechanism-consistent outcome is not automatically valid, and a broad market outcome is not automatically invalid. But broad outcomes usually require stronger assumptions, richer controls, and more careful sensitivity analysis. A simple estimator applied to a well-chosen outcome can support a more credible claim than a sophisticated estimator applied to an outcome only weakly connected to the treatment.

**Implementation**: `02_dowhy_causal_graph` shows the full two-outcome comparison, including sensitivity and placebo-refutation tests.

### Regime-conditional position sizing – A case study

The preceding sections establish DML methodology. The case study tests whether the estimated causal effects translate into improved position sizing on out-of-sample data; see `05_momentum_causal_trading` for implementation details.

ETF rotational momentum (buying recent winners and selling losers) has historically generated positive returns. Standard implementations apply uniform position sizing regardless of regime. The CATE estimates from DML offer a candidate alternative: scale exposure by the regime-conditional causal strength of the signal.

To avoid in-sample overfitting, we implement a strict *temporal split*:

- Training period (2007-01 to 2022-11): Estimate all causal effects, learn regime scaling
- Test period (2022-12 to 2025-12): Apply learned scaling to held-out data
- No leakage: Regime thresholds, CATE estimates, and position rules are fixed before the test period begins

`CausalForestDML` from `EconML` is fit on training data with market volatility and the regime indicators as effect modifiers (EconML’s `X` argument) and the volatility-horizon panel plus yield-curve slope as controls (EconML’s `W` argument, which together with the effect modifiers form the adjustment set), +0.019 in low volatility, +0.043 in mid volatility, and −0.066 in high volatility. The ATE is −0.0097 with using the same walk-forward folds as the LinearDML stage. The CATE varies in *sign* across regimes: 95% CI [−0.0138, −0.0055], against a naive OLS coefficient of −0.0037, a 61% bias once the controls

The sizing rule applies a regime multiplier to a gross-normalized portfolio weight. Let 𝑤௜ǡ௧ are absorbed. base denote the long-short quintile weight for asset 𝑖 at date 𝑡, rescaled so ∑|ݓ௜ǡ௧ base| = 1. The regime-scaled weight is: ௜ 𝑤௜ǡ௧= 𝑤௜ǡ௧ baseԜݏ(ݎ௧)

where 𝑠(ݎ௧) uses the signal-to-noise ratio CATE/std within each regime, clipped to [0.5,1.5] so that imprecise estimates shrink toward neutral exposure. The resulting factors are 0.50 in high volatility, 1.09 in low volatility, and 1.28 in mid volatility: exposure is cut by half where the effect reverses sign

and rises where the signal is both strong and precisely estimated.

The companion notebook compares three strategies during the test period:

1. **Naive**: Uniform momentum, no regime adjustment.
2. **Causal-Informed**: Scale by training-period CATE estimates, shrunk by their estimation uncertainty.
3. **Simple Heuristic**: Reduce exposure in high-vol (no causal machinery). ratios are positive in training (Naive +0.12, Causal +0.23, Heuristic +0.20) and negative in holdout Results include transaction costs (10 basis points), with a sweep over the 5–20 basis-point range. Sharpe (Naive −0.18, Causal −0.05, Heuristic −0.11). Causal is the best of the three in holdout, but does not

deliver positive risk-adjusted returns:

- Causal scaling reduces exposure in high volatility, where the training-period CATE was negative, and increases it in mid volatility, where the effect was largest and most precisely estimated. Momentum IC drops from +0.023 in training to −0.053 in holdout: the base signal inverts post-That heterogeneity partly persists, which is why the causal strategy degrades least. • 2022. No sizing rule rescues a treatment whose relationship to forward returns flips sign; the negative holdout Sharpes reflect that signal failure rather than a failure of the causal machinery.

Causal analysis reduces out-of-sample degradation when training-period heterogeneity carries information; it does not guarantee alpha when the underlying signal breaks down.

All estimates use only training data; the test period remains untouched. Cross-fitting uses walk-forward folds within training (no random folds). Regime classification uses pre-treatment volatility. Transaction costs are included. Results are sample-dependent and may not generalize.

The workflow differs from correlation-based analysis in two ways. First, CATE estimates measure the causal strength of momentum after controlling for the volatility and regime confounders, rather than the unconditional statistical correlation. Second, the test period is held out before any estimation choice is made, so the reported Sharpe ratios reflect post-decision behavior rather than in-sample fit. DML handles continuous treatment effects: how a *dose* of factor exposure causally affects returns. Many financial questions instead involve discrete events: a specific Fed announcement, a regulatory change, a CEO departure. These require explicit counterfactual construction rather than averaging across treatment intensities, which we will discuss next.

## 15.5 Measuring event impact with Bayesian structural time-series

While DML addresses continuous treatment effects, many financial questions involve discrete events: Did the Fed’s surprise rate hike affect bond prices? What was the impact of a CEO’s departure on the stock? Bayesian Structural Time-Series (BSTS) models provide a rigorous framework for answering such questions by constructing data-driven counterfactuals (Brodersen et al., 2015).

The identification checklist for BSTS, applying *Section 15.2*, is:

- Estimand is the post-period counterfactual difference (ATT on the treated series)
- Control series selection rule defined; spillover risk assessed
- Pre-period relationship stability verified via fit diagnostics
- Placebo dates tested to validate model specification
- Robustness checked across alternative control sets

### The intuition behind BSTS

BSTS constructs a counterfactual: a model of what the target series would have done in the absence of the event. The difference between the observed series and this “synthetic twin” is the estimated causal impact, illustrated by *Figure 15.3*.

![Figure 15.3](assets/figure_15_3.png)

*Figure 15.3: BSTS constructs a counterfactual from pre-event control relationships, then measures the gap between observed and projected series as the causal impact*

The workflow involves three steps:

1. Learn target-control relationships in the pre-event period.
2. Project the counterfactual “business as usual” path using post-event control movements.
3. Take the difference between actual and counterfactual as the causal impact.

Bayesian methods provide posterior distributions with credible intervals. Unlike traditional event studies, which assume constant expected returns, BSTS adapts to changing market conditions through control series.

Two assumptions underlie BSTS event studies:

- **Stable relationships**: The target-control relationship learned in the pre-period must hold in the post-period, absent the event.
- **No spillover**: The event must not affect the control series. If a Federal Reserve Open Market Committee (FOMC) monetary policy announcement moves both IEF and SPY, using SPY as a control corrupts the counterfactual.

Spillover is the most common failure mode in financial event studies. Mitigate it by choosing controls from asset classes with weak fundamental linkages to the event (for example, agricultural commodities for banking regulation). A useful diagnostic is to run BSTS on an asset the analyst believes is unaffected: if it shows a “significant” effect, the control group is contaminated.

`06_fed_announcement_bsts` automates this by running BSTS with each control as the target; any control showing significant impact is flagged for exclusion. For macro events where truly unaffected controls may not exist, rely on placebo-date validation and robustness across alternative control sets rather than assuming the problem away.

These assumptions cannot be tested directly: they must be justified by domain knowledge and validated indirectly through placebo tests.

The original Google R package (Brodersen et al., 2015) uses Bayesian structural time-series. We use `tfcausalimpact`, the actively maintained TensorFlow-Probability port that computes posterior credible intervals for the cumulative event effect.

### The impact of a Fed announcement on bond ETFs

We apply BSTS to measure the impact of FOMC announcements on the IEF Treasury bond ETF.

The event-study setup is:

- **Target**: IEF (7-10 Year Treasury Bond ETF)
- **Controls**: VEA (Developed Markets), EFA (EAFE), DBC (Commodities)
- **Event**: March 2023 FOMC announcement (25bp hike amid banking stress)
- **Pre-period**: 60 trading days before announcement
- **Post-period**: 20 trading days after announcement

The controls are correlated with bond prices through macro factors, but should not be directly affected by the FOMC decision.

#### Illustrative results

BSTS analysis of four FOMC announcements (March 2023, July 2023, November 2023, March 2024) in `06_fed_announcement_bsts` runs on log-price indexes anchored to the start of each window. Hence, the cumulative effect reads as a cumulative abnormal log return rather than a dollar quantity that depends on the ETF’s price level. The estimated effects on IEF carry signs consistent with the macro context: mulative effect of +0.59 over the post-window, consistent with flight-to-quality demand for

- The March 2023 announcement (a 25bp hike amid banking stress) produced a positive cu-

Treasuries. of −0.28: the same policy action, an opposite market response.

- The July 2023 announcement (a 25bp hike in calmer conditions) produced a cumulative effect

negative cumulative effects (−0.09 and −0.29).

- The November 2023 hold and the March 2024 hold-with-revised-projections both produced

All four events have 95% credible intervals on the cumulative effect that exclude zero. Taken at face value, that would mean every announcement had a significant impact on IEF. The validation tests below show why the model in its current form does not support that reading.

#### Placebo test validation

We re-run the same BSTS specification on twelve trading days with no Fed announcement. A well-specified model should flag very few of these as “significant.” In the companion notebook, seven of twelve 58%. With only twelve placebo dates, this is a coarse estimate, and these dates are not screened against placebo dates yield credible intervals that exclude zero, corresponding to a *placebo false-positive rate of*

other macro releases such as CPI prints, payrolls, or Treasury refunding announcements, so the rate is illustrative rather than precise. Even so, the magnitude is large enough to indicate that the model finds “significant” effects on arbitrary non-event dates at over half the rate it finds them on FOMC dates. Under that condition, the apparent significance of the four FOMC events cannot be attributed to the announcements themselves.

The control-as-target spillover check uses BSTS to ask whether each control series shows a significant response on each FOMC date. Running the check per event, rather than assuming the verdict from a single date transfer, shows the contamination is pervasive: all three control ETFs (VEA, EFA, DBC) respond on the March 2023, November 2023, and March 2024 dates, and DBC alone responds in July 2023. A control is flagged solely when its own post-period cumulative-effect credible interval excludes zero. For macro events such as FOMC announcements, this is not surprising (the same policy shock propagates across global equities, developed-market bonds, and commodities through the dollar and global risk-appetite channels). Still, it means the counterfactual constructed from these controls absorbs the announcement effect rather than isolating it. The control set is the plausible choice, not an unaffected one; a more conservative design would use duration-matched non-US sovereign bonds or a local-level model with no cross-asset controls.

Taken together, the validation tests indicate that BSTS on log-price indexes with global cross-asset controls is not a credible event-study design for FOMC announcements. The four reported event impacts should not be read as each Fed announcement moving IEF by a known cumulative amount; the placebo and spillover checks reject the design before any individual finding can be taken seriously.

#### Posterior probability versus credible intervals

A common source of confusion is how different libraries report uncertainty. A true **posterior probability of a causal effect** is the integral of the posterior distribution above or below zero, a Bayesian tail probability over the parameter, not over time. The fraction of point-effect rows whose mean is positive is not such a probability; it is a sample frequency over the post-period that can be high or low for reasons unrelated to whether the effect is real. The reliable diagnostic is whether the **credible interval for the cumulative effect** excludes zero, supported by placebo tests with a low false-positive rate. When the placebo rate is high, neither the credible interval nor any derived sign frequency is trustworthy.

Both DML and BSTS require specifying the causal structure. When the structure is unknown, we turn to *causal discovery*: learning the DAG from data.

**Implementation**: `06_fed_announcement_bsts` covers the spillover diagnostics, the four-FOMC main analysis, and twelve placebo dates.

## 15.6 Causal discovery from observational data

The methods discussed so far (DML and BSTS) begin with a specified causal question: treatment, outcome, and identification strategy are declared in advance. **Causal discovery** addresses a different problem: when the structure is not known, can observational data suggest a plausible causal graph? In finance, the answer is only provisional. Discovery methods can generate hypotheses about lead-lag structure, indirect pathways, and candidate controls, but they do not convert observational market data into causal truth.

All causal discovery methods rely on strong assumptions. Depending on the method, these include some combination of:

- **Causal sufficiency**: There are no unobserved common causes
- **Faithfulness**: Observed independences reflect the graph rather than the exact cancellation of effects
- **Stationarity**: Structural stability over time

These assumptions are demanding in financial markets, where omitted macro drivers, regime changes, simultaneous feedback, and evolving market structure are common. For that reason, discovered edges should be treated as **hypotheses for validation** rather than as deployable trading signals (Spirtes, Glymour, and Scheines, 2000; Runge et al., 2019b).

### Time-series discovery with Granger, PCMCI, and VAR-LiNGAM

A natural baseline is **Granger causality**. Granger asks whether past values of one series improve the prediction of another beyond that series’ own history. This makes it intuitive and useful for preliminary screening. Still, it remains limited: standard implementations are often pairwise, typically linear, and do not by themselves distinguish direct effects from common drivers or indirect paths (Shojaie and Fox, 2022). A more ambitious time-series method is **PCMCI**, implemented in the **Tigramite** library. PCMCI is designed for multivariate time series with autocorrelation and potentially nonlinear dependence. It proceeds in two stages, which also explain what the acronym stands for:

1. **PC (Peter and Clark) Phase:** This step uses a modified version of the PC algorithm to identify a “superset” of potential causal parents for each variable. It acts as a fast condition-selection step to reduce high dimensionality by filtering out irrelevant variables.
2. **MCI (Momentary Conditional Independence) Phase**: This second step performs more rigorous conditional independence tests to prune the remaining links. It specifically addresses the challenges of autocorrelation and time-delayed influences, which often cause standard methods to produce false positives.

This is more informative than pairwise Granger testing because it accounts for autocorrelation, indirect links, and common drivers, rather than testing each pair in isolation (Runge et al., 2019b). This distinction is important. PCMCI does not simply condition on all other variables. Its advantage is that it reduces the conditioning problem to relevant parent sets, which improves statistical power and tractability in sparse systems. The output is a **time-lagged graph of candidate dependencies** that can be interpreted causally only under the method’s assumptions.

A second time-series causal discovery method is **VAR-LiNGAM** (Hyvärinen et al., 2010). VAR-LiNGAM combines vector autoregression (VAR) with the Linear Non-Gaussian Acyclic Model (LiNGAM) framework to estimate the lagged and contemporaneous causal structure within a linear, non-Gaussian model. Identification rests on the *assumptions* that:

- The structural errors are non-Gaussian
- Acyclicity of the contemporaneous structure
- The absence of hidden common causes

These are strong assumptions, but the non-Gaussianity requirement is often more plausible in finance than Gaussianity because returns, flows, and volatility measures frequently exhibit skewness and fat tails.

VAR-LiNGAM is therefore best understood as a **structured alternative to PCMCI** rather than as a universally stronger method. PCMCI is built around conditional independence testing in multivariate time series; VAR-LiNGAM is more model-based and derives its identification from distributional structure. When the two disagree, the disagreement usually indicates assumptions rather than evidence that one graph is simply correct.

### Cross-sectional DAG learning with NOTEARS

NOTEARS addresses a different problem from PCMCI and VAR-LiNGAM. Its original contribution is to reformulate cross-sectional DAG learning as a continuous optimization problem subject to an exact, differentiable acyclicity constraint (Zheng et al., 2018). In its original form, NOTEARS is a linear structure-learning method that has since been extended. That distinction matters because the appeal of NOTEARS is conceptual as much as practical. Classical crete search with smooth constrained optimization. For an adjacency matrix 𝑊, the graph is acyclic DAG learning often involves combinatorial search over graph structures. NOTEARS replaces that dis-

if and only if: ℎ( ) = tr(݁ ௐלௐ) െ݀ = Ͳ

where 𝑑 is the number of variables and ∘ denotes the element-wise product. This constraint allows

DAG learning to be cast as an optimization problem rather than a graph-search problem.

NOTEARS is most naturally applied to *cross-sectional or contemporaneous* structure, not to lagged time-series discovery. That is why it should be distinguished from PCMCI and VAR-LiNGAM. Later work extends the basic idea to *nonlinear and temporal settings*, but those extensions are separate developments. The core NOTEARS formulation should therefore be presented as a method for sparse DAG learning in a linear, contemporaneous setting (Zheng et al., 2018).

### Empirical method comparison

These methods answer related but distinct questions:

- **Granger causality** is best viewed as a predictive screening tool
- **PCMCI** is a multivariate time-series discovery method that attempts to control for indirect paths and common drivers
- **VAR-LiNGAM** is a time-series method that exploits linear structure and non-Gaussian errors for identification
- **NOTEARS** is a cross-sectional or contemporaneous DAG-learning method based on differentiable optimization

This distinction helps explain why their outputs can diverge materially on the same financial universe.

All four methods are applied to a daily ETF panel spanning 2015–2024. The seven assets cover distinct market segments:

- **SPY** (S&P 500), **QQQ** (NASDAQ-100), **IWM** (Russell, 2000): US large-cap, tech-heavy, and smallcap equities
- **TLT**: Long-term US Treasuries
- **GLD**: Gold
- **EEM**: Emerging-market equities
- **XLF**: US financials sector

This mix includes assets with plausible lead-lag relationships (financials and broad equities) and macro-diversified pairs (bonds and gold) where temporal structure is less expected. *Table 15.4* compares the discovered structure:
| Method | Result | Key Characteristics |
| --- | --- | --- |
| Granger | 26 FDR-significant edges | Pairwise only, ignores multivariate confounding |
| PCMCI | 42 lagged links (𝑝05) | Multivariate; conditions on other variables |
| NOTEARS | 5 contemporaneous edges (all 5 stable under bootstrap) | Continuous optimization, cross-sectional sparsity |
| VAR-LiNGAM | 1 lagged edge (causal-learn); 11 (from scratch) | Time series native, requires non-Gaussian residuals |

*Table 15.4: Causal discovery method comparison on daily ETF returns (7 assets, 2015–2024)*

The spread from a single edge to 42 edges on identical data is the central lesson of this comparison. On daily financial returns, method outputs diverge materially because each method operationalizes different assumptions and responds differently to noise, dimensionality, and weak dependence. The four methods answer different questions and produce different graphs; the candidate causal structure is **assumption-sensitive**. lagged links at 𝑝05 out of 147 possible (7 variables × 7 targets × 3 lags). XLF (financials) emerges PCMCI discovers substantial temporal structure on the seven-asset panel, identifying 42 significant

as the dominant causal driver, with its one-day lag on SPY, QQQ, and EEM among the strongest links (absolute partial correlation greater than 0.10). Most of the 42 links, however, have small effect sizes (median absolute value ≈ 0.06), statistically distinguishable from zero but not necessarily economically meaningful.

This **contrast between statistical significance and effect size** is itself a useful lesson: practitioners should examine effect magnitudes, not just p-values, when deciding which edges warrant further investigation.

This result contrasts with the 4-asset analysis in `07_tigramite_time_series`, which finds zero significant lagged links among SPY, IEF (intermediate-term US Treasuries), GLD, and VIX. Block-bootstrap stability confirms the null: the most stable edge (IEF→VIX) appears in only 37% of resamples, below the 50% robustness threshold. The difference from the 7-asset VAR-LiNGAM result is not a contradiction: the 4-asset universe primarily comprises macro-diversified assets with little lead-lag structure at the daily frequency, whereas adding sector ETFs (especially XLF) introduces genuine temporal dependencies. The sensitivity to universe composition reinforces the need to interpret discovery results conditionally on the assets, frequency, and sample period used.

The VAR-LiNGAM row illustrates implementation sensitivity. VAR-LiNGAM is implemented twice on this panel: first from scratch (VAR via Ridge regression, then raw ICA on residuals) and then using the `causal-learn` library, which applies DirectLiNGAM (a more principled ordering algorithm) with statistical pruning. The from-scratch version finds 11 lagged edges; the library version retains only 1 (XLF→SPY). The same algorithm, under different estimation and pruning strategies, can produce radically different graphs. The shared edge (financials leading the broad market by one day) is the most robust signal precisely because it survives both implementations and is also the dominant driver in the PCMCI graph. NOTEARS, applied to contemporaneous standardized returns rather than lagged links, recovers a sparse same-day structure of five edges, all of which survive in at least half of the block-bootstrap resamples. The strongest is SPY→QQQ, present in every resample, followed by IWM→SPY and IWM→XLF at above 80%. The edge XLF→SPY, stable in 69% of resamples, corroborates the financials-leading-the-market signal that PCMCI and VAR-LiNGAM also surface, here as a within-day rather than lagged relationship. Because NOTEARS targets cross-sectional sparsity, its graph is best read as a hypothesis about which assets share contemporaneous structural dependence, not as evidence of temporal lead-lag.

Granger causality finds 26 FDR-significant edges out of 42 directed pairs, comparable in density to PCMCI despite testing only pairwise. While Granger cannot distinguish direct from indirect effects, its agreement with PCMCI on the densest links (especially XLF→equity) provides additional evidence for the strongest signals. The practical implication is that edges confirmed by multiple methods carry more weight than those that depend on a specific specification.

*Table 15.5* summarizes the typical use and the central limitation of each method.

| Method | Typical use | Main strength | Key limitation |
| --- | --- | --- | --- |
| Granger causality | Preliminary time-series screening | Simple and intuitive | Pairwise and predictive rather than fully causal |
| PCMCI | Multivariate lagged discovery | Controls for autocorrelation and indirect links | Strong assumptions; can be conservative |
| VAR-LiNGAM | Temporal discovery with structural assumptions | Uses non-Gaussianity for identification | Linear model; assumes no hidden common causes |
| NOTEARS | Cross-sectional or contemporaneous DAG learning | Converts DAG search into continuous optimization | Original formulation is linear and not time-series specific |

*Table 15.5: Causal discovery methods and their typical use cases*

**Box 15.1: What the ADIA Lab challenge does and does not show**

The ADIA Lab Causal Discovery Challenge is best read as a **benchmark for a special supervised setting**, not as evidence that causal discovery from observational data has been solved. Participants were given many synthetic datasets together with their true DAGs, and the organizers explicitly allowed either unsupervised discovery methods or supervised prediction models. The scoring rule then reduced each predicted graph to the local causal role of every node relative to a designated treatment X and outcome Y and evaluated multiclass balanced accuracy (Olivetti et al., 2026).

On this task, **supervised and hybrid methods dramatically outperformed** classical discovery baselines. That result is real, but its interpretation is narrow. The benchmark shows that when many labeled dataset–graph pairs are available from a stable simulator family, a supervised model can learn the mapping from statistical patterns to causal-role labels better than a stand-alone discovery algorithm. This is a legitimate result about amortized inference under a known synthetic regime, not a general result about causal discovery in realistic financial settings. That distinction matters in finance. In practice, **researchers rarely have a labeled training set of true causal graphs**, and market data are shaped by latent confounding, structural breaks, simultaneity, equilibrium feedback, and endogenous sampling. For that reason, the ADIA result should not be read as “supervised causal discovery is the new default.” Its real value is cautionary: benchmark success may reflect the structure of the benchmark itself more than transferable causal competence. Synthetic benchmarks can still be useful, especially as sandboxes for experimentation, but their outputs remain proposals to be validated rather than truths to be deployed (Reisach, Seiler, and Weichwald, 2021).

**Implementation**:

- `07_tigramite_time_series`: PCMCI on a four-asset macro panel (SPY, IEF, GLD, VIX); produces the null result discussed above.
- `08_neural_causal_discovery`: seven-asset daily ETF panel with PCMCI, NOTEARS, VAR-LiNGAM, and Granger causality side by side; produces the values in *Table 15.4*.
- `09_adia_causal_benchmark`: reproduces the ADIA Lab supervised-versus-unsupervised contrast in a simplified setting and shows how a supervised classifier exploits regularities in synthetic graph-generated data.

## 15.7 Case study causal evidence

Estimating one causal treatment effect per case study with Double Machine Learning applies a single identification standard to the cross-section of factor claims. Most primary-label effects turn out to be small once observed confounders are controlled for; the gap between the naive and orthogonalized coefficients is wide, and only a few estimates survive both credibility checks applied here. Each case study estimates one primary treatment (momentum on ETFs, US Equities, and FX; the funding-premium z-score on Crypto; an order-flow microstructure effect on the NASDAQ-100; the IV–RV spread on SP500 Eq+Opt; a monthly momentum factor on US Firms; carry on CME Futures; and the variance risk premium on SP500 Options) and reports the orthogonalized coefficient with heteroskedasticity- and autocorrelation-consistent (HAC) inference alongside a block-permutation refutation 𝑝-value. Two gates summarize each estimate at its primary label: HAC significance at |𝑡| > 1.96, and a refutation pass at 𝑝ref < 0.05. *Table 15.6* lists the results:

| Case | Treatment | DML effect | HAC 𝑡 | HAC 𝑝 | Bias | 𝑝 ref | Both gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETFs (21d) | momentum | -0.058 | -9.84 | <0.001 | +33.2% | 0.00 | yes |
| US Firms (1m) | factor | +0.007 | +1.98 | 0.048 | +42.8% | 0.03 | yes |
| SP500 Options | VRP, HTM | -0.123 | -4.69 | <0.001 | +49.8% | 0.03 | yes |
| US Equities (1d) | momentum | -0.001 | -2.16 | 0.031 | +68.8% | 0.00 | yes |
| NASDAQ-100 (15m) | microstructure | 0.00 | +2.52 | 0.012 | -9.9% | 0.11 | HAC only |
| Crypto (8h) | premium z | -0.001 | -1.03 | 0.303 | +68.3% | 0.00 | refutation only |
| SP500 Eq+Opt (5d) | IV–RV spread | -0.002 | -1.50 | 0.135 | +86.5% | 0.00 | refutation only |
| FX Pairs (1d) | momentum | +0.0004 | +0.42 | 0.675 | -213.5% | 0.30 | neither |
| CME Futures (5d) | carry | +0.0003 | +0.75 | 0.454 | -59.7% | 0.19 | neither |

*Table 15.6: Primary-label DML estimate with HAC significance (|𝑡| > 1.96) and a block-permutation refutation gate (𝑝ref < 0.05). Bias is (naive − DML)/|DML|, signed: positive when the naive coefficient exceeds the DML estimate in signed terms, negative when it falls below. Where the effect is negative, this runs counter to a magnitude reading: a positive signed bias means the DML estimate is more negative than the naive estimate, so the naive coefficient understates the magnitude*


These are conditional estimates under the stated causal models, not proof of causation: DML requires that all confounders be observed and that the nuisance models be well specified. A refutation pass strengthens the evidence but does not close the unconfoundedness gap.

### The two gates

The two gates rarely agree. Four case studies clear both (ETFs, US Firms, SP500 Options, and US Equities), one clears HAC significance alone, the NASDAQ-100; two clear refutation alone, Crypto and SP500 Eq+Opt; and two clear neither, FX and CME Futures. *Figure 15.4* places each case study on the two axes.

The disagreement is informative because the gates probe different properties. Refutation asks whether the point estimate is distinguishable from a placebo-generated artifact under the chosen null; HAC significance asks whether it is distinguishable from zero under inference robust to heteroskedasticity and serial correlation. A treatment can pass one and fail the other. The NASDAQ-100 microstructure effect is significant under HAC inference yet does not separate from a placebo-permuted treatment, the pattern expected when autocorrelation or specification structure carries part of the apparent signal. Crypto and SP500 Eq+Opt show the reverse: both reproduce above the placebo background, but their HAC standard errors are wide enough that the estimate is not distinguishable from zero. Neither gate alone suffices.

![Figure 15.4](assets/figure_15_4.png)

*Figure 15.4: Causal credibility dashboard: refutation 𝑝-value versus HAC |ݐ| across the nine case studies, with the two gates (𝑝ref < 0.05 and |ݐு஺஼| > 1.96) marked*

### Confounding bias

Orthogonalization moves the estimate substantially. The median absolute gap between the naive OLS only case study where naive and DML disagree on sign, though with 𝑡ு஺஼= +0.42 and 𝑝ref = 0.30 the coefficient and the DML estimate is 59.7% and exceeds 50% in five of the nine case studies. FX is the

reversed estimate is itself indistinguishable from noise. The implication reaches past these case studies: a factor evaluation that skips orthogonalization draws its conclusions from systematically distorted coefficients, which is how much of the backtesting literature reports factor premia.

### Variation across horizons

Horizon and label specification move the causal estimate as much as they move the predictive IC of the is positive and HAC-significant; at the 21-day horizon, it is negative and larger in magnitude (−0.058 same treatment. ETFs momentum is the clearest case. At the five-day horizon, the orthogonalized effect against +0.025), and only the longer horizon clears both gates. SP500 Options reports five horizons,

all negative, but only the hold-to-maturity return (the option held to expiry, abbreviated HTM in *Table 15.6*) clears both gates; the raw ten-day horizon is HAC-significant without refutation, and the shorter Firms, where winsorizing the monthly label raises the HAC 𝑡-statistic, but the estimate then fails to and delta-hedged variants carry intervals that overlap zero. The same sensitivity appears within US

be refuted. The lesson from the predictive chapters carries over: label engineering is a first-order decision for the causal estimate, not a preprocessing detail.

### Causal evidence and predictive signal

The orthogonalized causal estimate and the cross-sectional prediction signal are different objects, and a feature can carry one without the other. SP500 Eq+Opt is the clearest illustration: the predictive case study extracts a small cross-sectional IC from the IV–RV spread, while the causal estimate reproduces above placebo but its HAC interval overlaps zero. Crypto sits in the same cell: *Chapter 12*’s predictive case study reaches a small but credibly positive IC, while the causal premium-z estimate’s HAC interval includes zero. These cases support continued predictive use of the feature without a causal claim of the magnitude implied by the naive coefficient.

The reverse holds as well. A treatment that clears both causal gates is not, on its own, a tradable signal: alpha generation needs the cross-sectional ranking machinery developed in *Chapters 11* through *14*, DML analysis (the per-treatment 𝑡ு஺஼ and refutation distributions, and the bias decomposition) is asand *Chapter 16*’s strategy stack consumes predictions, not causal coefficients. The cross-case-study

sembled in `10_case_study_insights`. These findings distill into practical guidelines for incorporating causal analysis into the systematic trading research workflow.

## 15.8 Summary

The methods in this chapter (DML for continuous treatments, BSTS for discrete events, PCMCI and neural methods for structure discovery) share a discipline: make assumptions explicit, test them where possible, and know when conclusions are fragile. Method selection follows the assumption-fragility ordering: where a DAG and a valid adjustment set are defensible, DML estimates a continuous treatment effect with the strongest finite-sample guarantees; where the treatment is a discrete event, BSTS shifts the burden onto control-series stability and spillover testing; and where neither condition holds, PCMCI for time series or PC/NOTEARS/FCI for cross-sections generate candidate structures whose credibility scales with the strength of their stationarity and confounding assumptions.

The right method is the one whose assumptions best match the data and question, not the most sophisticated one available. DML with a well-specified DAG on a clean panel will produce more credible results than PCMCI applied to a short, noisy time series, even though PCMCI is technically more ambitious.

Causal effect estimates feed directly into downstream chapters: portfolio construction (*Chapter 17*) can weight factors by causal confidence, and risk management (*Chapter 19*) can account for effect uncertainty when sizing positions. The progression from *Section 7.5*’s single-feature plausibility checks through this chapter’s multivariate estimation provides a complete pipeline: screen features cheaply, then invest in formal causal analysis for the survivors.

In the next chapter, we will move from machine learning to strategy development and introduce the historical simulation of trading strategies to evaluate how model outputs translate into entry and exit signals.
