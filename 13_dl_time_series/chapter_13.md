# Chapter 13: Deep Learning for Time Series

Deep learning enters time-series prediction when the ordering of observations may carry information beyond the most recent datapoint. *Chapters 11* and *12* treated prediction mainly as a cross-sectional problem: given today’s features, forecast a future label. This chapter asks when the sequence history itself adds incremental signal, which architectures can extract it, and how the formulation of the forecast target changes the answer.

That formulation choice matters. Some models generate a future path one step at a time; others predict the entire future path in a single forward pass; many financial ML pipelines skip the path and predict only a horizon label, such as a 5-day return, 21-day return, direction, or cross-sectional rank. This chapter focuses on the architectures and evaluates them in terms of predictive performance, in line with the other modeling chapters. *Chapter 17* then returns to the trading objective itself, showing how deep learning can learn positions or portfolio weights directly under allocation-level losses, such as the Sharpe ratio, drawdown, and cost-aware objectives.

The answer to the architecture choice is not “use Transformers.” Modern time-series modeling is better understood as a competition among inductive biases: recurrence, explicit decomposition, attention, convolutions, state-space models, and strong linear baselines. Each makes different assumptions about temporal structure, covariates, and forecast horizon. The practical task is to decide when sequence-model complexity is warranted, when a direct horizon-label model is the cleaner formulation, and when the problem should be moved downstream into an allocation objective.

By the end of this chapter, you will be able to:

- Explain why recurrent networks became a bottleneck for long-context forecasting tasks.
- Compare the main temporal modeling philosophies and explain when each is most appropriate.
- Use strong baselines and diagnostics to judge whether sequence-model complexity is warranted.
- Distinguish the design logic of modern time-series Transformer variants and relate those choices to multivariate structure, covariates, and forecast horizon.
- Distinguish recursive one-step forecasting, direct multi-horizon path forecasting, and direct horizon-label prediction, and explain why the distinction matters.
- Evaluate time-series foundation model adaptation modes for financial applications, including the implications of transfer mismatch and pretraining contamination.
- Apply practical uncertainty estimation methods, including MC Dropout and deep ensembles, to support risk-aware trading decisions.

*Figure 13.1* lays out the timeline of the architecture evolution. We begin with Long Short-Term Memory (LSTM) networks and the limitations that created space for both N-BEATS and Transformer-based architectures. A pivotal 2022 critique by Zeng et al. showed that simple linear models often outperform complex Transformers, challenging assumptions about the need for architectural sophistication. Modern architectures like PatchTST and iTransformer respond through stronger temporal inductive biases. The chapter concludes with a practitioner’s framework for selecting architectures based on problem characteristics rather than research hype.

![Figure 13.1](assets/figure_13_1.png)

*Figure 13.1: Timeline of time-series architecture evolution*

Recurrent networks come first; their limitations motivated every architecture that follows.

## 13.1 Recurrent networks and their limits

**Recurrent neural networks** (RNNs) maintain a **hidden state** that evolves as the network processes each element of a sequence. This design made them the default architecture for time series forecasting through the mid-2010s.

**Long Short-Term Memory** (**LSTM**) networks (Hochreiter and Schmidhuber, 1997) addressed the problem that limited vanilla RNNs: **vanishing gradients** prevented the model from learning long-range dependencies (Hochreiter et al., 2001). The LSTM introduced a cell state that acts as a conveyor belt, allowing information to flow across many time steps with minimal degradation.

### The LSTM gating mechanism

The LSTM’s power derives from three gates (forget, input, and output) that regulate information flow (*Figure 13.2*). The cell state update combines their decisions: 𝐶௧ൌ݂ ௧⊙𝐶௧ିଵ൅݅௧⊙tanh( ஼⋅[ℎ௧ିଵǡ ݔ௧] ൅ܾ ஼) where 𝑓௧ and 𝑖௧ are sigmoid-gated functions of the previous hidden state and current input. This gat-

ing mechanism allows LSTMs to selectively remember or forget information, theoretically capturing dependencies spanning hundreds of time steps. In financial applications, LSTMs have been used for volatility forecasting, return prediction, and limit order book modeling (Zhang, Zohren, and Roberts, 2019).

![Figure 13.2](assets/figure_13_2.png)

*Figure 13.2: LSTM cell with forget, input, and output gates*

### Two limitations

LSTMs have two limitations that motivated the search for alternatives. **The sequential computation bottleneck.** The hidden state at time 𝑡 depends on the hidden state at time 𝑡, creating an inherent computational dependence on the sequence’s length. •

LSTMs still benefit from GPU acceleration, but their recurrence limits *temporal* parallelism compared with attention-based models discussed in *Section 13.3*. For high-frequency data with tens of thousands of observations, this bottleneck can become prohibitive. Transformers process all time steps simultaneously through matrix operations, often yielding substantially faster training at similar sequence lengths.

- **Gradient flow challenges.** While LSTMs were designed to address vanishing gradients, they do not eliminate the problem entirely. For very long sequences, gradients must still flow through many gating operations during backpropagation through time (Werbos, 1990). In practice, long nominal lookbacks do not guarantee that the model effectively uses distant information; they often increase noise, training instability, and sensitivity to hyperparameters. For financial series, where the signal is weak and non-stationary, that distinction matters more than the architecture’s theoretical capacity.

`01_core_architectures` illustrates the **computation-versus-signal trade-off** on a deliberately simple ETF task: pooled univariate 60-day return windows predict the same ETF’s next-day return. On this noisy objective the LSTM trains about twice as slowly as a comparable MLP (32 versus 17 seconds) yet ranks next-day returns no better: both models post a cross-sectional Spearman IC near zero. The **Gated Recurrent Unit** (**GRU**) simplifies the LSTM by merging the forget and input gates into a single update gate and eliminating the separate cell state, reducing parameters by roughly one-quarter in the `01_core_architectures` implementation. This streamlining can reduce model size and sometimes sequential computation remains 𝑂(ܶ), and gradient flow through the merged gating mechanism still improve generalization on smaller datasets. However, the GRU inherits both fundamental limitations:

degrades over long sequences. The GRU trains slightly faster, and its IC is also near zero: a lighter recurrent cell does not solve the return-prediction problem.

The comparison shows architecture, not strategy: recurrence adds sequential computation without improving next-day return ranking, which reflects the weak signal in return sequences.

Some financial processes exhibit **long-memory** properties, such as volatility clustering (*Chapter 9*), persistent order flow, and calendar effects, in which dependencies span multiple trading days or longer. These long-range dependencies can exceed what LSTMs reliably capture through gradient-based learning, especially when the useful signal is weak and non-stationary.

These limits motivated parallel architectures along two lines: embedding classical decomposition into neural networks (*Section 13.2*) and importing attention from NLP (*Section 13.3*).

## 13.2 N-BEATS and explicit decomposition

**N-BEATS** (Oreshkin et al., 2019) encodes a specific inductive bias: time series are compositions of interpretable components (trend and seasonality) that neural networks should decompose explicitly rather than discover implicitly.

### Blocks, stacks, and doubly residual learning

The N-BEATS architecture builds complexity from simple components through a hierarchical structure.

- **The** **basic** **block**. The fundamental unit is a fully-connected block that takes a lookback window of historical data as input. The input passes through several dense layers with ReLU activations. The block produces two outputs: a *forecast* for the future horizon and a *backcast* that reconstructs the historical input. The backcast represents what the block has “understood” about the input signal.
- **Residual** **stacking**. Multiple blocks connect sequentially within a stack. The first block receives the original input. Each subsequent block receives the *residual* from its predecessor: the original input minus the previous block’s backcast. This forces each block to focus on what earlier blocks failed to capture, progressively extracting finer patterns from the signal.
- **Doubly residual architecture**. The full model chains multiple stacks together, applying the same residual principle at a higher level. The second stack receives the residual from the entire first stack. This creates a deep, hierarchical decomposition in which early stacks capture dominant patterns, and later stacks refine the details. The final forecast is the sum of all block-level forecasts across all stacks. This progressive refinement acts as a strong regularizer, paralleling boosting in tree-based methods (*Chapter 12*), since each block must explain only what its predecessors missed.

*Figure 13.3* illustrates the complete block-stack-doubly-residual structure.

![Figure 13.3](assets/figure_13_3.png)

*Figure 13.3: N-BEATS block-stack doubly-residual architecture*

### Basis expansion

The mechanism by which N-BEATS generates its outputs distinguishes it from generic neural networks. Rather than directly outputting forecast values, each block’s final layers produce *expansion coefficients* For the forecast, the block outputs coefficients 𝜃௙ that combine with basis vectors 𝑔௙: that weight the *basis functions*.

|ఏ೑| ݕොൌ෍ߠ௙ǡ௜ ڄ݃ ௙ǡ௜ ௜ୀଵ

This approach generalizes classical decomposition. Fitting a trend is equivalent to finding coefficients for a polynomial basis; modeling seasonality is equivalent to finding coefficients for a Fourier basis. In the generic N-BEATS configuration, the model learns suitable basis functions from data. In the interpretable configuration, these bases are explicitly constrained.

### Making N-BEATS interpretable

The *Trend Stack* uses a polynomial basis. For a polynomial of degree 𝑝, the trend forecast becomes: The interpretable configuration, **N-BEATS-I**, constrains each stack to model a specific component.

௣ 𝑦௧௥௘௡ௗ(ݐ) ൌ෍ߠ௧௥௘௡ௗǡ௜ ڄ ݐ௜

௜ୀ଴

The neural network learns the optimal polynomial coefficients that capture the underlying trend component. The *Seasonality Stack* uses a Fourier basis of sine and cosine terms, with the network learning the coefficients that best represent periodic patterns.

The final forecast decomposes additively: Forecast = Trend + Seasonality. You can plot each component separately, providing direct insight into what drives predictions. This transparency is rare among deep learning models and valuable for building trust in production systems.

### Extensions – N-BEATSx and N-HiTS

N-BEATS gained prominence by matching the M4 competition winner’s accuracy using a purely deep-learning approach, showing that appropriate inductive biases could close the gap between neural and traditional methods across diverse forecasting domains.

The architecture has inspired two notable extensions:

- **N-BEATSx** incorporates exogenous variables, which are crucial in applications where external factors, such as weather or macroeconomic indicators, drive forecasts. For financial applications, this enables conditioning on macroeconomic releases, calendar effects, or cross-asset signals.
- **N-HiTS** (Challu et al., 2022) introduces hierarchical interpolation and multi-rate signal processing, improving long-horizon forecasting accuracy by capturing patterns at multiple temporal resolutions simultaneously. Its MaxPool downsampling reduces input dimensionality for longrange stacks while preserving detail in short-range ones.

Both extensions preserve the core philosophy of structured decomposition while expanding the model’s applicability.

### Practical caveats

N-BEATS is sensitive to the **lookback-to-horizon ratio**: too short a lookback starves the decomposition of context, while very long horizons can destabilize training. The interpretable configuration can underperform the generic variant when the true generating process does not decompose cleanly into polynomial trend and Fourier seasonality, a common situation in financial data where regime changes violate both assumptions. N-BEATS-I should therefore be benchmarked against N-BEATS-G on the target data before committing to the interpretable variant.

Decomposition is one escape from sequential processing; importing the attention mechanism from language models is another, and it changes how models relate distant time steps.

**Implementation**: `02_nbeats_interpretable` implements this architecture from scratch in PyTorch, building blocks, stacks, and the doubly-residual connections step by step. It also trains and evaluates both configurations on SPY data.

## 13.3 Attention for time series

While N-BEATS evolved classical decomposition principles, a more radical approach came from natural language processing. The Transformer architecture (Vaswani et al., 2017) dispensed with recurrence entirely in favor of self-attention, a mechanism that models relationships between all pairs of sequence elements simultaneously. Its adaptation for time series represents a bet that a sufficiently general mechanism can learn all temporal structure from data alone, without decomposition priors.

### Adapting Transformers for temporal data

Applying Transformers to time series requires addressing the architecture’s origin in discrete token sequences. Three key adaptations bridge this gap:

- **Tokenization via patching**. Unlike words, which are natural discrete units, time series values are continuous and densely sampled. A naive approach treats each time step as a token, but this creates prohibitively long sequences for the attention mechanism. The modern solution is *patching*: segmenting the input window into fixed-length subsequences that become input tokens. A lookback window of 512 time steps might yield 32 patches, each of length 16. Patching reduces computational cost while preserving local temporal patterns within each patch.

**Caution: Data leakage in patching**

Patching can introduce data leakage if normalization statistics are computed across the full window rather than causally. Practitioners must ensure that all preprocessing respects temporal boundaries: either by normalizing each patch independently or by using only backward-looking statistics.

it cannot distinguish the sequence (ݔ1ǡ ݔ2ǡ ݔ3) from (ݔ3ǡ ݔ1ǡ ݔ2). For time series, where temporal

- **Positional encoding**. Self-attention is *permutation-invariant*: without additional information,

order is essential, this is a critical deficiency. Positional encodings inject order information by adding position-dependent vectors to each token’s embedding. The original Transformer used fixed sinusoidal encodings; many time series variants learn these encodings during training.

- **Encoder-decoder structure.** The standard Transformer comprises an encoder that processes the historical lookback window to create contextual representations, and a decoder that generates forecasts. Time-series adaptations use several variants of that template, including encoder-only and direct multi-horizon heads.

Forecast formulation is a separate design choice from architecture. A model can be recurrent, attention-based, decomposition-based, or linear, but it still needs a target definition. Three formulations matter in this chapter (see *Figure 13.4*): predicts 𝑦௧ାଵ from the past window, appends that prediction to the input, predicts 𝑦௧ାଶ, and

1. Recursive one-step forecasting learns a one-period-ahead model and rolls it forward. The model repeats until 𝑦௧ାு. This is the natural extension of one-step statistical forecasting, including

the ARIMA-style models used in *Chapter 9* for model-based features, but forecast errors can compound across the horizon. (ݕ௧ାଵǡ ݕ௧ାଶǡ ǥ ǡ ݕ௧ାு). Many deep forecasting architectures, including sequence-to-sequence mod-

1. Direct multi-horizon path forecasting predicts the entire future path in a single forward pass

els and multi-horizon heads, are designed for this task. They avoid recursive error propagation, but they still model the future path and must learn the joint structure of all intermediate steps.

1. Direct horizon-label prediction skips the path and predicts only the final label used by the investment problem: a 5-day return, 21-day return, direction, volatility premium, or cross-sectional rank. This is the dominant formulation in *Chapters 11*, *12*, *13*, and *14*. The model may use sequential inputs, but the target is a supervised horizon-label problem rather than a time-series path forecast.

![Figure 13.4](assets/figure_13_4.png)

*Figure 13.4: Three forecast formulations for financial time series*

Recursive forecasting generates the path one step at a time, direct multi-horizon forecasting predicts the path in one pass, and direct horizon-label prediction skips the path and predicts only the final target.

All three formulations recur in this chapter, but the case-study evaluation focuses on how sequence architectures perform under direct horizon-label prediction.

### The self-attention mechanism

The **self-attention mechanism**, introduced in *Chapter 10* for language applications, operates identically when applied to time series tokens. Each position attends to all others via Query, Key, and Value projections (see *Figure 13.5*): ) = softmax (ܳܭ Attention(ܳǡ ܭǡܸ ቇܸ √ ௞ 

Unlike RNNs, which must pass information sequentially through hidden states, self-attention creates direct connections between any two positions, regardless of distance. Multiple **attention heads** learn different relationship types in parallel (local trends, seasonality, regime shifts), and their outputs combine through a learned projection.

![Figure 13.5](assets/figure_13_5.png)

The vanilla Transformer’s 𝑂(ܮ2) (quadratic) attention complexity motivated several *efficiency-focused* Figure 13.5: Self-attention via query, key, and value projections

*variants*, including **Informer**, **Autoformer**, **FEDformer**, and **Pyraformer,** each reducing cost through sparse or frequency-domain attention. All shared a *common assumption*: time series should be tokenized like language, with attention computing relationships between temporal positions. Whether attention genuinely extracted temporal structure (or merely overfitted to patterns that simpler models could match) remained an open question until the empirical challenge the next section takes up.

## 13.4 Linear baselines versus Transformers

Zeng et al. (2022) showed that simple linear models can outperform several Transformer-based architectures on standard long-term forecasting benchmarks. The result changed the field’s baseline expectations: a new architecture should not be taken seriously unless it improves on strong linear benchmarks.

At the same time, these results do not establish that attention is ineffective for time-series modeling in general. They show, rather, that on widely used benchmark tasks, additional architectural sophistication often failed to improve predictive accuracy.

The practical implication is clear: begin with strong, simple baselines, then add complexity only when it delivers robust out-of-sample gains.

### Permutation-invariance and temporal order

The core argument against using Transformers for time-series targets the mathematical properties treats input sequences as unordered sets. The dot product between a query at position 𝑖 and keys at of self-attention. The mechanism is **permutation-invariant**: without positional encodings, attention

all other positions produces the same values regardless of how those positions are arranged. **Positional encodings** inject order information, but Zeng et al. argue this is a patch on a mechanism that may underutilize temporal structure. The issue is not that Transformers *cannot* represent order (they can) but that order must be explicitly injected. It may be underweighted in practice depending on architecture and training dynamics.

There is a deeper point. Transformers excel in natural language processing because semantic relationships often transcend position: the words “king” and “queen” relate conceptually regardless of where 𝑡 depends causally on values at 𝑡, 𝑡, and so on, so the temporal ordering itself may carry the they appear in a sentence. Time series forecasting is about positional relationships: the value at time

predictive signal. Applying a permutation-invariant architecture to a task where order is paramount is a mismatch of tool and problem.

### The LTSF-Linear benchmark

To empirically test their critique, Zeng et al. (2022) introduced a family of “embarrassingly simple” one-layer linear models as new baselines for **Long-Term Series Forecasting** (LTSF), the task of predicting many steps rather than just the next value:

- **Linear** maps the lookback window directly to the forecast horizon via a single matrix multiplication, with no hidden layers, nonlinearities, or attention.
- **D-Linear** first decomposes the series into trend and seasonal components using a simple moving average, then applies separate linear layers to each component, and finally sums them. This adds minimal complexity while incorporating the decomposition insight that N-BEATS had validated.
- **N-Linear** normalizes for distribution shift: subtract the last input value before the linear layer, add it back to the output. This simple trick stabilizes predictions when test-time statistics differ from those used during training. `03_great_debate` implements all three LTSF-Linear variants, along with a vanilla Transformer encoder, for direct comparison.

The models require seconds to train on standard hardware. Their total parameter counts are orders of magnitude smaller than those of Transformer variants.

### Results and diagnostic evidence

In the Zeng et al. (2022) experiments, these linear models outperformed Transformer architectures across all nine LTSF benchmark datasets, achieving 20-50% improvements over Informer, Autoformer, and FEDformer. Perceived progress had been illusory, driven by comparisons against weak baselines rather than genuine architectural advances.

Diagnostic experiments revealed specific failure modes. Forecasting error *increased* for most Transformers as the lookback window grew: they were confused by additional noise rather than extracting useful signals. When input sequences were randomly shuffled (destroying all temporal structure), D-Linear’s MSE more than quadrupled (0.345 to 1.406), while the Transformer’s barely changed (0.379 to 0.391; *Figure 13.6*). D-Linear exploits temporal structure and collapses without it; the Transformer relies primarily on channel correlations, so removing the time axis incurs little cost. Attention was not learning temporal patterns. **Implementation**: `03_great_debate` replicates these diagnostics for ETF return data.

![Figure 13.6](assets/figure_13_6.jpeg)

*Figure 13.6: The Shuffle experiment: D-Linear versus Transformer under shuffled inputs*

### Counterpoints and caveats

The paper by Zeng et al. (2022) is not the final word. The original study focused on univariate forecasting; Transformers may do better in *multivariate settings* with rich covariates, where cross-series attention can extract meaningful relationships (see the next section). The comparison may also understate the Transformer’s potential: with careful *hyperparameter tuning*, attention-based models can match or exceed linear baselines on some of the same datasets, indicating sensitivity to the training budget. Finally, **the benchmarks are small** by language model standards. Time-series foundation models trained on millions of diverse series might unlock scaling benefits. However, *Section 13.6* indicates that the evidence regarding financial data is mixed.

Subsequent large-scale evaluations reinforce this message while generalizing it. Small experimental changes (hyperparameter budgets, train/test splits, metric choice) can flip model rankings on standard LTSF suites; across multiple cross-domain forecasting benchmarks, statistical and linear baselines frequently match or beat deep models deemed state-of-the-art on individual datasets. The problem extends beyond “Transformers versus Linear” to the **fragility of benchmark conclusions** themselves.

A closer look at *why* some Transformers succeed reveals that **tokenization and normalization choices matter more than the attention mechanism itself**. Performance on these benchmarks is largely driven by intra-variate dependencies and dataset stationarity, partially explaining why shuffle critiques hold on standard suites yet prove less predictive on harder tasks. The practical comparison is not “Transformers versus Linear” but “strong linear baselines versus time-series-aware tokenization with careful tuning,” and the latter’s advantage is conditional on problem characteristics. The paper served as a reset rather than an indictment. Any new architecture must now show clear improvement over LTSF-Linear, not just over previous Transformers. The goal shifted from making Transformers bigger to making them *smarter*, and the next generation of architectures, from **PatchTST**’s temporal patching to **iTransformer**’s variate inversion, responded directly to this challenge.

## 13.5 Modern Transformer variants

The Zeng et al. (2022) critique catalyzed a new wave of innovation rather than ending Transformer-based forecasting. Modern architectures address the identified weaknesses by incorporating stronger temporal inductive biases while retaining the attention mechanism’s ability to model complex dependencies. Three models exemplify this evolved approach: PatchTST, iTransformer, and the Temporal Fusion Transformer (TFT).

### Patching as inductive bias with PatchTST

**PatchTST**, introduced by Nie et al. (2023), offers a simple, effective response to the critique: rather than fight the Transformer’s permutation invariance, design the input representation to work with it. non-overlapping patches of fixed length 𝑃. A lookback window of 𝐿 time steps becomes roughly ܮȀܲ **PatchTST formalizes patching as the central inductive bias**. The input time series is segmented into

patches, each treated as a single token. This design has several implications: **Computational efficiency**: Attention complexity is ܱ൫ܰ ) rather than 𝑂(ܮ2), a large re-2 ௣௔௧௖௛௘௦ • duction that enables much longer lookback windows

- **Local pattern preservation**: Each patch encapsulates local temporal structure, information that raw pointwise tokenization would scatter across the attention mechanism
- **Semantic coherence**: Patches represent meaningful units of temporal evolution, more analogous to words in language than individual characters

For *multivariate time series*, PatchTST ignores cross-channel dependencies during training. Each channel (each time series in a multivariate set) is processed independently through shared Transformer weights. Forecasts are generated per-channel without attempting to model correlations between variables.

This seems counterintuitive, since multivariate data presumably contains cross-series information worth exploiting. But channel-independence acts as a regularizer. On many benchmarks, cross-channel attention overfits to spurious correlations in the training data that do not generalize. By forcing the model to predict each series solely from its own history, PatchTST avoids this trap while still benefiting from shared representations learned across all channels.

PatchTST often employs **Reversible Instance Normalization** (RevIN) to handle distribution shifts, neutralizing local distributional shifts before the attention mechanism processes the data. This normalization trick (subtracting the instance mean and dividing by the instance standard deviation, then reversing the transformation on the outputs) is crucial for time series that wander across different statistical regimes over time.

The combination of patching, channel-independence, and RevIN allowed PatchTST to achieve state-ofthe-art results on LTSF benchmarks, outperforming both earlier Transformer variants and the linear models that had previously matched them.

### Inverting the dimensions with iTransformer

Where PatchTST preserves the standard Transformer structure while changing its inputs, iTransformer (Liu et al., 2024) takes a more radical approach: it **inverts the dimensions on which attention operates**.

In a standard multivariate Transformer, each token represents a time step containing values from all variables. Attention operates across time, relating different moments to each other. iTransformer inverts this: **each token represents an entire univariate series** (or a substantial portion of it), and Concretely, given a multivariate series with 𝑁 variables, iTransformer creates 𝑁 tokens, each embed- attention operates across variables.

ding the full temporal history of one variable. Attention then learns relationships between variables: how one asset’s behavior relates to another’s, how a leading indicator connects to a lagging one.

Why does inversion work? This design sidesteps the temporal order critique entirely. The model does not apply permutation-invariant attention to time steps; temporal patterns within each variable are captured by the initial embedding process, which can use any time-aware architecture (such as an MLP or convolution applied along the time axis). Attention instead **models cross-variable dependencies**, a domain where permutation invariance is less problematic since there is often no intrinsic ordering among variables.

For *financial applications* involving portfolios of correlated assets, iTransformer offers a natural fit. The model can learn that technology stocks move together, that gold inversely correlates with the dollar, or that volatility spikes in one market presage spikes in others. These cross-asset dynamics represent the type of complex, non-local relationships that attention mechanisms excel at capturing.

### Covariate-rich forecasting with Temporal Fusion Transformers

While PatchTST and iTransformer optimize for pure time-series forecasting, the **Temporal Fusion Transformer** (TFT) addresses a different challenge: forecasting when rich metadata and known future inputs are available (Lim et al., 2021). Financial applications often include:

- Static covariates (sector, exchange, asset class)
- Time-varying observed inputs (volume, volatility, macroeconomic indicators)
- Known future inputs (calendar features, scheduled events, options expiry dates)

TFT provides a principled architecture for integrating all three: its distinguishing features are *learned variable selection* (a gating mechanism that assigns time-varying importance weights to inputs, providing interpretability absent from PatchTST and iTransformer) and multi-horizon quantile output, producing *calibrated prediction intervals* rather than point forecasts. For risk management and position sizing, this native uncertainty quantification is more principled than post-hoc methods applied to point-forecast architectures.

When is TFT the right choice? It is strongest in covariate-rich settings with heterogeneous input types, such as portfolio forecasting that incorporates sector metadata, macro indicators, and calendar effects. For pure time series without covariates, PatchTST’s simpler design typically matches or exceeds TFT. Covariate-aware forecasting is a design axis. TFT’s covariate handling is no longer unique. Recent benchmarks formalize covariate categories (static, past-only, dynamic, known-future) as a primary axis of evaluation. An emerging result shows that reframing forecasting as a tabular regression with temporal features achieves competitive covariate-aware performance without a time-series-specific architecture, which connects directly to the tabular methods of *Chapters 11–12*.

*Table 13.1* summarizes the comparison.

| Model | Key Innovation | Attention Domain | Best For |
| --- | --- | --- | --- |
| Vanilla Transformer | Self-attention | Time steps | Short sequences, simple baselines |
| TFT | Variable selection + covariates | Time steps | Covariate-rich, multi-horizon with uncertainty |
| PatchTST | Patching + Channel- independence | Time patches | Long-horizon univariate/ multivariate LTSF |
| iTransformer | Variate-as-token inversion | Variables | Cross-asset dynamics, portfolio forecasting |

*Table 13.1: Transformer summary*

There is no benchmark substitute for validation on the target data: ETTh1 electricity demand has little in common with cryptocurrency returns.

Transformers are far from the only response to the recurrent bottleneck: temporal convolutions, state-space models, and pretrained foundation models each offer distinct trade-offs that the next section surveys.

**Implementation**: `04_transformers` benchmarks PatchTST, iTransformer, and Ridge on ETF return prediction. On a 60/20/20 temporal split with cross-sectional IC, Ridge sits at +0.010, a stable anchor across reruns at the same random seed. Both Transformers clear that anchor by a small margin, but their relative ordering (PatchTST versus iTransformer) drifts across reruns of the same code on the same data. Single-split point estimates at this signal-to-noise ratio are not sufficiently stable to rank the architectures; the cuDNN non-determinism in the attention path is large relative to the ETF-panel IC scale. The walk-forward case studies in *Section 13.9* are the authoritative ranking. There, the marginal benefit of temporal modeling depends on how much signal the feature pipeline has already extracted.

## 13.6 Alternative architectures and foundation models

Beyond decomposition and attention, **several architectures address** **specific forecasting requirements**:

- **Temporal Convolutional Networks** (**TCNs**; Bai, Kolter, and Koltun, 2018) enforce causality through dilated causal convolutions, offering a middle ground between RNNs and Transformers with efficient parallelism and explicit temporal ordering.
- **TSMixer** (Chen et al., 2023) shows that well-designed MLPs that alternate time-mixing and feature-mixing layers can compete with attention on standard benchmarks.
- **CNN-based approaches** (Jiang et al., 2020) convert price histories into images using Gramian Angular Fields (GAFs), which transform a time series into a matrix capturing pairwise temporal relationships. This lets CNNs detect recurring market patterns by treating price dynamics as visual texture.

**Hybrid statistical-neural models** represent a design philosophy rather than a single architecture: let a classical model handle known structure such as level, seasonality, and trend, while a neural network learns residuals or time-varying parameters. The ES-RNN (Smyl, 2020), winner of the M4 forecasting competition, combines Holt-Winters exponential smoothing with an RNN. Holt-Winters is a classical forecasting method that recursively updates estimates of level, trend, and seasonality; ES-RNN uses this decomposition to remove non-stationary structure before the RNN learns multiplicative corrections.

**Deep state space models** (Rangapuram et al., 2018) take the probabilistic variant: neural networks generate time-varying parameters for a linear state space model, with inference performed via Kalman filtering (see *Chapter 9* for details), embedding well-understood Bayesian filtering structure while allowing non-stationary parameterization.

This hybrid philosophy lives on in N-BEATS’s decomposition priors (*Section 13.2*), TFT’s structured covariate handling (*Section 13.5*), and the modern SSMs below, but the explicit approach remains relevant in low-data regimes where pure neural networks overfit, and classical structure provides essential regularization (Lim and Zohren, 2021).

Three developments warrant deeper treatment: state space models for their different approach to sequence processing, foundation models for their implications about transfer learning in finance, and the emerging evaluation infrastructure that determines whether any claimed advance can be trusted.

### State space models

**State Space Models** (SSMs) have become a primary alternative to Transformers for sequence modeling. SSMs are inspired by **continuous-time dynamical systems** and are discretized for sequential processing. A latent state evolves according to learned dynamics: ℎ௧ൌܣℎ௧ିଵ൅ܤݔ௧

𝑦௧ൌܥ݄ ௧൅ܦݔ௧

In a conventional SSM, the matrices 𝐴, 𝐵, 𝐶, and 𝐷 are learned during training but then applied uni-

The **Mamba** architecture, introduced by Gu and Dao (2023), achieves linear 𝑂(ܮ) complexity while formly across sequence positions.

maintaining the ability to model long-range dependencies. Mamba’s innovation is **selective state** In particular, 𝐵௧, 𝐶௧, and the discretization step 𝛥௧ depend on the current input 𝑥௧; the base transition **spaces**: the model learns functions that generate input-dependent SSM coefficients for each time step. matrix 𝐴 remains learned and shared, while the effective discretized transition varies through 𝛥௧. This makes the recurrence content-aware, allowing the model to retain relevant information and filter out contextually noisy inputs. Mamba therefore combines the temporally structured memory of recurrent models with hardware-aware parallelization, making it efficient on very long sequences where even patched Transformers can become costly.

A limitation of base SSMs is their reliance on a single temporal resolution. Financial data contains signals across multiple timescales: microstructure dynamics within minutes, momentum over days, regime shifts over months. The **Multi-scale Mamba** architecture (**ms-Mamba**) addresses this by deploying parallel Mamba blocks operating at different sampling rates, capturing high-frequency and low-frequency patterns simultaneously without the quadratic cost of multi-scale attention (Karadag et al., 2025). This directly targets the long-memory properties discussed in *Section 13.1*: volatility clustering and persistent order flow produce dependencies spanning dozens or more trading days across multiple temporal granularities, precisely the setting where a single-resolution model loses information.

SSMs are best understood as part of a **sequence-mixing primitives toolkit** rather than an outright attention replacement. The practical question is: what is my context length, latency budget, and maintenance burden? For sequences exceeding 10,000 time steps, such as tick-level or minute-bar data spanning minutes to weeks, SSMs offer clear efficiency advantages because their linear complexity avoids the memory exhaustion that quadratic attention methods encounter. For shorter sequences with rich cross-variate structure, attention-based models retain their edge because cross-variable attention captures portfolio dynamics that SSMs do not natively model.

Hybrid architectures that interleave attention and SSM layers are an active research direction, but lack broad independent validation. The most promising finance-specific work, **FinMamba** (Hu et al., 2025), combines market-aware graph construction with multi-level Mamba blocks for stock movement prediction, but remains a single-paper result awaiting replication.

We evaluate each architecture for ETF return prediction against a Ridge baseline. On the single 60/20/20 split, Ridge anchors the comparison with a cross-sectional Spearman IC of +0.016 (MSE 0.0035), stable across reruns with the same seed. TCN and TSMixer point estimates sit within the rerun noise of Ridge: TCN has flipped from +0.051 to -0.013 across reruns of identical code, TSMixer from +0.013 to +0.001, with MSEs (TCN 0.0043, TSMixer 0.0039) close to Ridge’s. Single-split IC at this signal-to-noise ratio is an architectural sanity check, not a ranking signal; *Section 13.9*’s case studies offer a more thorough walk-forward comparison, and TSMixer reaches IC +0.029 at the 21-day horizon on the ETF panel under multi-step Darts retraining.

**Implementation**:

- `05_tcn` implements dilated causal convolutions with a receptive field spanning the full lookback window
- `06_tsmixer` alternates time-mixing and feature-mixing MLP layers
- `07_mamba_ssm` for a Mamba/SSM implementation on ETF data
- `08_cnn_image_encoding` converts price histories into Gramian Angular Field images and applies convolutional feature extraction

### Time series foundation models

**Foundation models** (**FMs**) for time series follow the LLM playbook: pretrain on massive corpora to learn universal temporal representations, then adapt to downstream tasks.

The first generation, comprising **Chronos** (Amazon), **TimesFM** (Google), **Lag-Llama**, and **MOMENT**, shared a common limitation: univariate-only inference with no native support for exogenous covariates or multivariate structure, making them poorly suited to quantitative finance where assets coevolve, and external signals are central.

Time-series foundation models import the *pretrain-then-adapt* logic of large language models into forecasting. Early systems showed that large pretrained models could produce useful zero-shot forecasts. However, many were evaluated primarily in univariate settings and were not designed for the multivariate, covariate-rich structure common in real applications.

The second generation addresses this directly. **Chronos-2** (Ansari et al., 2025) introduces a group attention mechanism that processes multivariate series and exogenous covariates through a unified pipeline, enabling covariate-informed zero-shot forecasting via in-context learning. By assigning group identifiers to targets and their associated covariates, the model shares dynamic information across related series without requiring task-specific fine-tuning. **Moirai-MoE** (Liu et al., 2024) replaces static frequency-based input projections with sparse **Mixture-of-Experts** routing, in which token-level gating automatically specializes to the data’s heterogeneous characteristics.

The result is **competitive performance with a fraction of the active parameters** of dense models: 11 million active parameters, achieving accuracy that previously required billion-parameter architectures. These advances do not invalidate the first-generation negative results discussed in this chapter, but they do substantially change the adaptation calculus.

#### Adaptation modes

How a foundation model is deployed matters as much as which model is chosen. The literature has converged on four adaptation modes, each with distinct compute, data, and maintenance requirements:

- **Zero-Shot Inference**: Deploy the pretrained model directly without weight updates. Viable when the target domain resembles the pretraining corpus: energy demand, retail sales, and weather. Empirical evidence suggests that TSFMs help most in data-scarce regimes but do not reliably dominate specialized models when sufficient training data is available, and that scaling laws are not strictly monotonic. For financial returns, zero-shot performance is unreliable (see below).
- **In-Context Learning (ICL)**: Feed historical reference series and covariates alongside the target within the context window; the model infers the current distribution without weight updates. Chronos-2’s group attention natively enables this. Recent work demonstrates that training TSFMs to use multiple related series in context at inference time can sometimes match the performance of explicit fine-tuning on target domains. ICL is most valuable for cold-start scenarios and rapid prototyping where gradient-based adaptation is impractical.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Adapt a subset of model weights (LoRA adapters, lowrank projections) on domain-specific data. Standard PEFT can lock adapters into suboptimal minima near initialization, particularly on noisy financial data. Two directions show promise: progressive training that stochastically deactivates adapters to encourage broader exploration, and prune-then-finetune, where removing up to 90% of parameters before adaptation yields superior accuracy. Both exploit the same insight: foundation models are overparameterized for any single task, and constraining the adaptation surface improves generalization.
- **Test-Time Model Selection** and **Ensembling**: Rather than fine-tuning one large model, build a portfolio of smaller pretrained models and combine them via selection or ensembling at inference time. Recent work demonstrates that this approach can match single-model accuracy while reducing inference cost, and can be more compute-efficient than test-time fine-tuning. This pathway suits production environments where latency, maintenance burden, and reproducibility matter more than marginal gains in accuracy.

The practical implication is to treat foundation models as *initializations* rather than solutions. The adaptation mode should match the compute budget, data availability, and deployment constraints, not the model’s marketing materials.

#### The limits of scaling

The NLP-imported assumption that performance scales monotonically with parameters does not hold for time series. Evidence from the **GIFT-Eval benchmark** (Aksu et al., 2024), which enforces strict pretrain/test separation across 28 datasets and 177 million data points, bears this out: hybrid architectures interleaving Fast-Fourier Transform (FFT) based long convolutions with linear RNNs achieve competitive zero-shot performance with under 3 million parameters, matching the accuracy of dense models with 1.5 billion parameters.

Independent evaluations confirm that a strict scaling law does not consistently hold for forecasting and that smaller models can occupy favorable positions on the accuracy-efficiency Pareto frontier. In practice, this means high-performance zero-shot forecasting is feasible on standard hardware without debilitating inference latency. The right evaluation metric is the **accuracy-efficiency frontier**, not the raw parameter count. When comparing TSFMs, report active parameter counts, wall-clock inference times, and memory footprints alongside predictive metrics.

#### The finance transfer gap

Despite these advances, foundation models face persistent challenges in financial applications. The gap between benchmark performance on standard forecasting tasks and actual utility for trading signals remains the central tension in applying TSFMs to finance.

Rahimikia et al. (2025) provide a *large-scale evaluation of TSFMs* in financial forecasting, covering daily excess returns across 94 countries over 34 years. Off-the-shelf TSFMs underperform tree-based ensembles on both statistical and economic metrics: sample 𝑅2 of -1.37%, while TimesFM 500M reaches -2.80%; by comparison, CatBoost achieves

- In the U.S. zero-shot setting with a 512-day input window, Chronos Large achieves an out-of-

-0.03% for the same window size, and -0.10% when averaged across all window sizes.

- **Directional accuracy remains close to random**. Chronos large is just above 51%, TimesFM is just below 50%, and CatBoost reaches 51.16% in the 512-day setting. Even the tree ensembles therefore operate close to the noise floor, reinforcing the weak-signal-plus-non-stationarity premise that runs throughout this book.
- Pretraining TSFMs on financial data substantially improves economic performance: the paper reports, for example, annualized returns and Sharpe ratios of 36.84% and 5.42 for Chronos small, and 30.36% and 3.66 for TimesFM 20M, both with a 512-day window, supporting the view that **domain-specific adaptation is required** rather than optional.

This poor transfer stems from a *distributional mismatch*: general time-series corpora emphasize seasonality, trend, and clear patterns that are largely absent in financial returns. The heavy tails, low signal-to-noise ratio, and non-stationarity of asset prices differ from the electricity demand, weather, and traffic data that dominate pretraining corpora.

TSFMs show promise in risk forecasting. The return-prediction results do not generalize to all financial tasks. For volatility and Value-at-Risk forecasting, where the target exhibits stronger and more transferable structure (volatility clustering, persistence, and mean reversion), TSFMs show credible evidence. Studies evaluating TimesFM against econometric baselines such as GARCH report that finetuned foundation models rank among the top performers across multiple VaR quantiles. Statistical forecast-comparison tests support these gains:

- The **Diebold-Mariano** test evaluates whether two models have equal average predictive accuracy under a chosen loss function
- The **Giacomini-White** test evaluates conditional predictive ability, asking whether performance differences remain significant given the information available at the time of forecasting

Zero-shot forecasting is insufficient even for these more structured targets; incremental fine-tuning is essential to learn volatility dynamics specific to the asset class and evaluation period.

Leakage risks specific to pretrained models are a concern. When evaluating TSFMs on financial data, standard temporal train/test splits are necessary but not sufficient. The pretraining corpus becomes a new leakage channel: if the TSFM was pretrained on data that overlaps with or is correlated with the evaluation period, zero-shot claims are compromised.

The **TIME** benchmark (Zou et al., 2025) formalizes this using “fresh datasets” and a human-in-the-loop construction pipeline designed to ensure strict zero-shot integrity. The growing training corpora used by TSFMs make test-set integrity increasingly difficult, analogous to the contamination challenges facing LLM evaluation. Practitioners must verify that evaluation data postdate the pretraining corpus or are excluded from it, and that no proxy leakage occurs via correlated series.

For return prediction, tree-based ensembles with engineered features (*Chapter 12*) remain a hard baseline; for risk forecasting (volatility, VaR), fine-tuned TSFMs are a credible addition to the toolkit. Prefer models with Student-t or mixture output distributions for financial data.

With this many architectures available, the question shifts from which model is best to which model fits the problem. The next section gives a decision framework for matching architecture to data characteristics, computational budget, and deployment constraints. **Implementation**: `09_foundation_models` compares zero-shot Chronos and TTM against trained baselines on ETF return prediction. Chronos achieves IC of −0.015, and TTM reaches −0.017, both worse than random, while a task-specific LSTM (50K parameters) achieves IC +0.025, and even Ridge regression (60 parameters) reaches +0.022. Chronos carries roughly 20 million parameters, and TTM 1–5 million, yet both produce negative zero-shot signals, confirming the distributional mismatch between general pretraining corpora and financial returns.

## 13.7 A practical framework

The chapter’s evidence and trade-offs translate into a practical model-selection process.

### Step 1: Establish strong baselines

Before evaluating any deep learning model, establish a baseline performance that any complex architecture must beat. The following models form a baseline ladder: each rung represents a stronger comparison that must be surpassed before adding complexity is justified:

1. **Seasonal naive**: For data with a seasonal pattern (not the case for, for example, return series), predict using the value at the same point in the previous cycle.
2. **LTSF-linear**: The D-Linear variant decomposes before applying linear layers; N-Linear normalizes by the last input value to handle distribution shift. If a sophisticated model cannot outperform D-Linear, its complexity is unjustified.
3. **Statistical models**: ARIMA and ETS remain strong baselines for univariate series with clear trend and seasonality (*Chapter 9*). For multivariate settings, VAR provides a comparable baseline.
4. **Gradient boosting models**: LightGBM or XGBoost with lagged features often match neural ing the sequential problem into a tabular regression problem that includes lag features (𝑡, approaches, particularly with limited data. Applying GBMs to time series requires transform𝑡, …, ݐെ݇), rolling statistics, and calendar features. GBMs excel at learning complex nonlin-

ear relationships in these engineered features but cannot extrapolate beyond the training range.

*When does DL outperform GBMs?* *Chapters 11* and *12* showed that gradient boosting tops many tabular cross-sectional problems. In the case study comparison, the paired daily IC delta between the DL and best-tabular models, with a HAC interval on the common-date difference, excludes zero on three of eight DL-covered case studies: Crypto perpetuals funding, where NLinear at IC +0.029 sits credibly above GBM at +0.011; US Equities, where LSTM at +0.007 sits credibly below GBM at +0.032; and CME Futures, where the LSTM near zero sits credibly below GBM at +0.032. The DL point estimate also exceeds the best tabular baseline on ETFs and FX, but the paired interval overlaps zero on both. On the remaining three panels, the baselines are comparable or stronger. In *Section 13.9, Table 13.4* reports the per-case-study evidence. The practical decision rule is unchanged: test whether temporal ordering adds incremental signal beyond lag-feature engineering, and keep the simpler model unless gains are robust out-of-sample. **TabM** (*Chapter 12*) extends tabular deep learning with batch-level feature interactions, offering a middle ground between GBMs and sequence architectures. On cross-sectional panels without a strong sequential structure, it is a useful additional baseline alongside LightGBM and Ridge.

Sample size thresholds are calibrated on the book’s daily-frequency, moderate-SNR case studies and should be adjusted for the target data’s signal strength and feature dimensionality:

- **Fewer than 5,000 samples**: Ridge regression or LightGBM. Deep learning risks overfitting with insufficient data.
- **5,000–20,000 samples**: LightGBM typically offers the best accuracy-to-compute ratio. Deep learning may match but rarely exceeds.
- **More than 20,000 samples**: LSTM and other deep architectures can be *tested at scale*, but superiority is not guaranteed. A large sample size should trigger model comparison, not model replacement.

Once established baselines are beaten, **foundation models offer a progressive adaptation ladder**:

1. Zero-shot inference (free but unreliable for financial returns).
2. In-context learning with domain examples.
3. PEFT/LoRA adaptation on target data.
4. Full domain-specific training.

Each step adds compute and data requirements and improves domain fit (*Section 13.6*). For covariate-rich problems, the chosen model should be verified to handle the relevant covariate types natively: TFT and Chronos-2 support static, past-only, and known-future covariates, while sequence-only models (PatchTST, Mamba) require feature engineering to incorporate external signals.

### Step 2: Diagnose the problem

The problem is likely to belong to one of the following categories:

- **Univariate** **versus** **multivariate**. For univariate forecasting, N-BEATS and TCNs are natural choices. For multivariate forecasting, the choice depends on whether cross-series dependencies are known or must be learned.
- **Known** **versus** **unknown** **relationships**. If the relationship structure is known (for example, a physical network or supply chain), GNNs (*Chapter 23*) exploit it directly. If relationships must be learned, iTransformer’s cross-variate attention or PatchTST’s channel independence is appropriate.
- **Forecast horizon**. For long-horizon (weeks to months), PatchTST and modern Transformers are appropriate. For short-horizon (minutes to days), performance gaps narrow, favoring simpler approaches.
- **Interpretability**. N-BEATS-I provides trend/seasonal decomposition; TFT offers learned variable importance; GBMs offer feature importance. Standard Transformers are black boxes. Note that post hoc methods such as SHAP and LIME (*Chapter 11*) typically ignore sequential dependencies among inputs, making them unreliable for attributing importance in recurrent or attention-based architectures (Lim and Zohren, 2021).
- **Covariate richness**. If the problem includes static metadata, known future inputs, or heterogeneous feature types, TFT’s variable selection architecture is purpose-built for this setting.
- **Sequence length**. For extremely long sequences (more than 10K steps), Mamba’s linear complexity is necessary. PatchTST handles moderately long sequences via patching.

### Step 3: Apply the selection matrix

With the problem diagnosed, match characteristics to recommended architectures (*Table 13.2*):

| Scenario | Primary Choice | Alternative | Baseline |
| --- | --- | --- | --- |
| Univariate, interpretability required | N-BEATS-I | Autoformer | Seasonal ARIMA |
| Univariate, black-box acceptable | PatchTST | N-BEATS (generic) | D-Linear |
| Multivariate, known graph structure | GNN (Ch. 23) | - | VAR |
| Multivariate, unknown correlations | PatchTST | iTransformer | D-Linear |
| High dimensionality (>50 series) | iTransformer | PatchTST | Linear |
| Long horizon LTSF task | PatchTST | D-Linear | Naive Repeat |
| Covariate-rich, multi-horizon | TFT | PatchTST | LightGBM |
| Extremely long sequences (more than 10K steps) | Mamba | TCN | Rolling statistics and GBM |
| Limited data, need robustness | ARIMA/LightGBM | N-Linear | Seasonal Naive |
| Zero adaptation budget | D-Linear | Zero-shot TSFM | Seasonal Naive |
| Moderate adaptation budget (PEFT) | PatchTST and LoRA | TSFM ensemble | LightGBM |

*Table 13.2: Architecture selection matrix. Match problem characteristics to recommended architectures*

Apply the walk-forward protocol (*Chapter 6*) for validation and evaluation. Always include non-deep baselines: omitting them inflates perceived gains. For TSFM evaluation, verify pretrain/test data separation to rule out leakage through pretraining corpora (*Section 13.6*). Report bootstrapped confidence intervals alongside economic metrics (IC, Sharpe, maximum drawdown).

### Forecasting paths versus predicting horizon labels

The architectures in this chapter originate in time-series forecasting, predicting future values of a series from its history. Our case study pipelines solve a related but distinct problem: given a lookback window of time series values and features for each asset, predict the forward return as a scalar. This sees a sequence, but the supervised target is a scalar label at horizon 𝐻, often multiple time steps is direct panel regression with sequential inputs, not classical forecasting. In other words, the model ahead, not the full path from 𝑡 to ݐ൅ܪ.

The distinction matters in practice:

- True forecasting predicts a path (daily returns over a multi-day horizon), then maps that path to a label
- Direct regression skips the path and predicts the label itself

For the cross-sectional ranking task central to this book, direct regression has two advantages: it matches the evaluation metric (Spearman IC on the predicted labels), and it avoids compounding prediction errors over the horizon.

This differs from both recursive and direct multi-horizon forecasting (see *Figure 13.4*):

- Recursive forecasting generates the path one step at a time
- Direct multi-horizon forecasting predicts the path in one pass
- Direct horizon-label prediction skips the intermediate path entirely

That last formulation is often the right choice for cross-sectional ranking because the evaluation metric is defined on the final label, not on the intermediate daily path. If we care about the path, say from a risk perspective, the choice might change.

We tested the forecasting formulation directly. For N-BEATS and TSMixer, we ran the faithful library implementations (via Darts), which predict daily return paths and compound them to the horizon label, the task these architectures were designed for. For ETFs (21-day horizon) and CME futures (5-day horizon), the forecasting formulations produced substantially weaker signals than the direct regression models (LSTM, NLinear) on the same data. The long horizons amplified per-step errors, and the forecasting objective was misaligned with the ranking metric.

This finding carries a practical lesson: architecture selection is necessary but not sufficient. The formulation of the prediction problem itself (direct regression versus multi-step forecasting, cross-sectional ranking versus per-series prediction) determines whether temporal modeling adds value. N-BEATS and TSMixer are faithful forecasting models for forecasting a future path. LSTM and NLinear can be better matched to the book’s direct horizon-label prediction task because they learn the scalar ranking target directly. The sophisticated forecasting components of N-BEATS, TSMixer, and PatchTST are valuable when the path matters; they can add unnecessary complexity when only the horizon label is evaluated.

There is a fourth formulation that we cover in *Chapter 17*: *direct learning of portfolio allocation*. Instead of predicting a path or horizon label, the model outputs positions or portfolio weights and is trained on an economic objective such as Sharpe ratio, drawdown, turnover, or transaction-cost-adjusted return.

### Forecasting libraries

This chapter implements architectures directly in PyTorch for demonstration. For standard forecasting tasks, several mature libraries reduce implementation effort substantially:

- **Darts** (Unit8) provides the broadest model coverage with a unified `fit()`/`predict()` API spanning classical and deep learning methods. Darts is the appropriate choice for rapidly benchmarking diverse approaches on standard forecasting tasks.
- **NeuralForecast** (Nixtla) offers native panel-data support and optimized training loops. Choosing NeuralForecast is the primary requirement when training a shared model across multiple related time series.
- **PyTorch Forecasting** (maintained under the sktime umbrella) is built around the Temporal Fusion Transformer. It is the appropriate choice when the problem involves rich covariates: static metadata, known future inputs, and multiple target variables.
- **GluonTS** (Amazon) anchors the foundation model ecosystem around Chronos and Chronos-2 with native probabilistic outputs. Choose GluonTS when zero-shot or fine-tuned foundation models are the deployment target.
- **sktime** provides a scikit-learn-compatible meta-framework that wraps Darts, NeuralForecast, HuggingFace, and classical methods under a single `fit`/`predict`/`score` API. sktime is the appropriate choice for swapping backends without rewriting data pipelines. time. For cross-sectional panel prediction (ranking 𝑁 assets by predicted return and evaluating via For standard per-series forecasting, Darts or NeuralForecast will save significant implementation

Spearman IC, as our case studies require), raw PyTorch with custom training loops remains the most natural fit. Our case study experiments illustrate this directly: running N-BEATS and TSMixer through Darts’s forecasting formulation produced a substantially weaker signal than the direct regression adaptations (see *Forecasting paths versus predicting horizon labels* above), confirming that library choice implies a modeling choice.

| Library | Strengths | Panel Support | Foundation Models |
| --- | --- | --- | --- |
| Raw PyTorch | Full control, custom losses, cross-sectional | Manual | Manual |
| Darts | Broad model coverage, unified API | Limited | TimesFM |
| NeuralForecast | Speed, native multi-series | Native (unique id) _ | No |
| PyTorch Forecasting | TFT, rich covariates | Via TimeSeriesDataSet | No |
| GluonTS / AutoGluon | Chronos-2, probabilistic | Via DeepAR / Chronos-2 | Yes |
| sktime | Composable pipelines, wraps backends | Via reduction | Via wrappers |

*Table 13.3: DL for time series library overview*

Selecting the right architecture and library is only the first step: a point forecast without a confidence measure gives no guidance on how aggressively to trade the signal. The next section addresses uncertainty quantification, the bridge between prediction and position sizing.

**Implementation**: `11_library_landscape` provides a hands-on comparison of raw Py-Torch, sktime, and Darts on the same forecasting task, with per-library API details and code-effort metrics.

## 13.8 Quantifying prediction uncertainty

Point forecasts are only part of the signal (see *Chapter 11*). For risk management, portfolio construction, and position sizing, the uncertainty surrounding a prediction is often as important as the prediction itself. A forecast of +2% warrants a different response when accompanied by a narrow predictive interval than when surrounded by wide uncertainty bands. In practice, useful uncertainty estimates can often be obtained from standard deep learning models without moving to fully Bayesian neural networks or other specialized probabilistic architectures.

### Estimating uncertainty with MC dropout and deep ensembles

Deep learning models can provide useful uncertainty estimates without requiring fully Bayesian neural networks. Two practical approaches are **Monte Carlo (MC) dropout** (Gal and Ghahramani, 2016) and **deep ensembles** (Lakshminarayanan et al., 2017). Both start from the same idea: instead of relying on a single deterministic prediction, generate several plausible predictions from closely related models and use both their average and their spread.

The average prediction often improves accuracy because idiosyncratic errors partially cancel out. The spread across predictions then serves as an uncertainty signal. When many plausible versions of the model produce similar forecasts, confidence is higher. When they disagree substantially, the prediction is less reliable. For trading applications, the main practical value is that uncertainty estimates help distinguish forecasts that warrant larger positions from those that call for smaller positions, tighter risk limits, or no trade at all.

**MC dropout** obtains this distribution from a single trained model. Dropout remains active during inference, and the model is evaluated repeatedly, typically over 50-100 forward passes. Each pass samples a different subnetwork, producing a distribution of predictions rather than a single output. The sample mean becomes the point forecast, and the sample standard deviation provides a simple measure of predictive uncertainty. This uncertainty is commonly interpreted as reflecting uncertainty in the fitted model itself: the model’s forecast changes because different plausible subnetworks produce different answers.

**Deep ensembles** generate the same kind of distribution in different ways. Instead of sampling many subnetworks from one trained model, they train several independent models, typically 3 to 10, with different random initializations and sometimes different data orderings or bootstrap samples. Each model converges to a somewhat different solution, so the ensemble captures a range of plausible predictive functions. Averaging across ensemble members often improves predictive performance, while disagreement among members provides a natural estimate of uncertainty.

In practice, **deep ensembles are often more robust than MC dropout**, but they also cost more because both training and inference scale approximately linearly with the number of ensemble members. The tradeoff is straightforward:

- MC dropout is cheaper and easier to add to an existing model because it requires little architectural change
- Deep ensembles usually demand more compute, but they often provide stronger predictive performance and better-calibrated uncertainty estimates

In many applied settings, an ensemble of about five models offers a reasonable compromise between performance and computational cost.

A more elaborate variant extends deep ensembles by having each model predict not only a mean forecast but also a variance through a **heteroscedastic output head**. This aims to capture the idea that some observations are intrinsically noisier than others: even a well-specified model may be less certain for some assets, horizons, or market regimes than for others. This extension can produce richer predictive intervals, but it also increases modeling complexity and is not necessary to obtain the main practical benefit of ensembles. The essential point is simpler: model averaging can improve forecasts, and model disagreement can identify less trustworthy predictions.

Whatever method is used, uncertainty estimates should be evaluated empirically rather than accepted at face value. Useful diagnostics include empirical coverage, interval width, sharpness, and reliability plots for predictive intervals. In trading applications, the most important test is whether forecasts flagged as more uncertain actually produce larger out-of-sample errors, and whether any chosen interval target achieves its intended coverage (see *Section 11.5*). Where nominal coverage is poor, conformal calibration on a held-out window can improve interval validity.

**Implementation**: `10_uncertainty` illustrates both MC dropout and a five-member LSTM ensemble on ETF data, together with **split-conformal calibration** on a held-out window. Without calibration, Gaussian intervals from either method are under-covered by roughly an order of magnitude at the 95% nominal level. After split-conformal post-processing, empirical coverage lands within about three percentage points of nominal at the 50, 80, and 95% levels for both methods. At the 95% target, MC dropout reaches 0.97 and the ensemble 0.97. The calibration step recovers interval validity without retraining the base model. Across reruns at the same random seed, the ensemble’s uncertainty-error rank correlation exceeds MC dropout’s, but the precise magnitudes drift; the *qualitative ordering* (ensemble disagreement tracks error better than dropout-subnetwork variance on this LSTM) is the result that survives rerunning.

### Foundation model calibration

Uncertainty estimation becomes more challenging for foundation models, especially under the regime shifts, heavy tails, and volatility clustering that characterize financial data. A model pretrained on broad time-series or textual corpora may produce useful point forecasts in finance, yet still be poorly calibrated in the tails or during market stress. In these settings, nominal predictive intervals may be too narrow exactly when risk control matters most.

**Conformal calibration** (see *Chapter 11*) offers a practical post hoc correction. By calibrating on a held-out window, conformal methods can improve empirical coverage without retraining the base model. This is particularly valuable when the pretraining distribution differs materially from the target financial environment.

For risk-sensitive tasks such as **Value-at-Risk** or **Expected Shortfall**, calibration is not secondary to point accuracy. A time-series foundation model may predict the conditional mean reasonably well while still understating tail risk. In such cases, *domain-specific fine-tuning* and *post-hoc calibration* are often essential not only for improving accuracy but also for obtaining usable uncertainty estimates.

With architectures selected and uncertainty estimates evaluated, the question becomes empirical: which approaches actually deliver useful signals across asset classes, horizons, and data regimes? The next section synthesizes the evidence from the case studies to answer that question.

## 13.9 Case study insights

*Chapters 11* and *12* measured how far regularized linear models, gradient boosting, and the TabM tabular network turn engineered features into a cross-sectional ranking. The deep sequence models of this chapter join that running comparison under the same test: reading the return sequence directly, do they rank assets better than the tabular families already do? The deep architectures cover eight of the nine case studies (all but US Firms) at each case study’s primary regression label, with NLinear fit on every one, the LSTM on all but the SP500 Options straddle, and TCN, TSMixer, and PatchTST contributing on a further subset. Each model is summarized by its average daily Spearman information coefficient (IC) and the 95% confidence interval around it, computed with a HAC correction for the autocorrelation in the daily series. An interval that excludes zero is the bar for distinguishable signal.

### Where the sequence models add ranking content

Two questions separate here. The first is whether a deep model clears zero; the second, sharper one is whether it clears the tabular baseline. *Table 13.4* reports, for each case study, the highest-IC deep architecture for the primary label, alongside the strongest tabular model from *Chapters 11* and *12*. The deep family’s interval excludes zero on four case studies (ETFs, Crypto, US Equities, and the NAS-DAQ-100). It overlaps zero on the SP500 Options straddle, CME Futures, the SP500 equity-and-options case study, and FX.

| Case study | Horizon | Best deep model | Deep IC (t) | Best tabular | Tabular IC | Δ IC |
| --- | --- | --- | --- | --- | --- | --- |
| ETFs | 21 days | NLinear | +0.062 (3.0) | Ridge | +0.054 | +0.009 |
| Crypto | 8 hours | NLinear | +0.029 (4.6) | GBM | +0.011 | +0.018 |
| SP500 Options | to expiry | PatchTST | +0.013 (1.8) | GBM | +0.018 | −0.005 |
| FX | 1 day | NLinear | +0.011 (1.3) | TabM | +0.007 | +0.004 |
| SP500 Eq+Opt | 5 days | PatchTST | +0.011 (1.1) | TabM | +0.011 | ≈ 0 |
| US Equities | 1 day | LSTM | +0.007 (5.4) | GBM | +0.032 | −0.025 |
| NASDAQ-100 | 15 minutes | NLinear | +0.005 (2.3) | GBM | +0.006 | −0.001 |
| CME Futures | 5 days | LSTM | −0.001 (−0.1) | GBM | +0.032 | −0.033 |

*Table 13.4: Highest-IC deep architecture per case study at the primary regression label, with its HAC t-statistic, beside the strongest tabular baseline (Ridge, GBM, or TabM) from Chapters 11 and 12. Δ IC is the deep-model point estimate minus the best tabular point estimate*

Clearing zero is not the same as improving on the baseline, and the two come apart sharply. On US Equities, the LSTM’s IC is credibly above zero yet, at +0.007, sits far below gradient boosting’s +0.032 on the same data. To compare the families, we compute the average difference between the two daily IC series and an HAC interval around this mean difference, pairing the models day by day. On this measure, the deep-minus-tabular delta excludes zero on three case studies (*Figure 13.7*): Crypto, where the deep model adds +0.018 in daily IC over a baseline near zero; US Equities, where it trails gradient boosting by 0.025; and CME Futures, where it trails by 0.033. On the other five case studies, the paired interval overlaps zero, so the deep model is statistically indistinguishable from the strongest tabular alternative.

![Figure 13.7](assets/figure_13_7.png)

*Figure 13.7: Deep-model-minus-best-tabular daily IC delta per case study, with paired HAC 95% confidence intervals on the common-date difference. The three intervals that exclude zero are Crypto (deep model above) and CME Futures and US Equities (deep model below)*

The result, stated plainly: across eight case studies, direct sequence modeling credibly improves the ranking on one, Crypto; credibly trails on two, US Equities and CME Futures; and matches the tabular families on the other five, at substantially higher training cost.

### Which architectures carry the signal

NLinear reaches the highest deep-model IC on four case studies (ETFs, Crypto, the NASDAQ-100, and FX), the LSTM on two (CME Futures and US Equities), and PatchTST on two (the SP500 Options straddle and the SP500 equity-and-options case study); the temporal convolutional network leads on none. That ordering follows from how the architectures are used here. The case studies frame each model as a direct predictor of the horizon-label return (see *Section 13.7*), rather than as a classical multi-step path forecaster. In that role, NLinear’s normalize-and-project structure and the LSTM’s gating retain enough temporal information to rank the cross-section without incurring the error-compounding cost of a multi-step objective. N-BEATS, TSMixer, PatchTST, and iTransformer carry inductive biases designed for path forecasting or cross-variate modeling, and those biases do not align with a cross-sectional rank label.

Training-budget trajectories fall into two regimes. On the high-frequency case studies (the NASDAQ-100, SP500 Options, and Crypto), daily IC peaks early and then decays. On the daily and longer-horizon case studies (FX, CME Futures, and US Equities), it drifts upward as training progresses. The late-trajectory spread is tightest where coverage is widest, so early stopping is best treated as a per-case-study tuning outcome rather than a fixed rule.

### Whether the predictive uncertainty is reliable

A credible ranking and a reliable uncertainty estimate are separate properties, and the deep models earn them unevenly. Calibrating a single split-conformal layer on one fold’s residuals and reading empirical coverage against the 90% nominal level splits the eight case studies three ways. On CME Futures, ETFs, and US Equities coverage lands near nominal, close to 0.90. The two SP500 case studies over-cover, returning intervals wider than the data warrant. The rest under-cover by varying margins: Crypto returns intervals well short of nominal, FX modestly short of it, and the NASDAQ-100 falls to near-zero empirical coverage. The spread reflects the fragility of a single calibration split in finite samples, whose validity rests on the calibration and evaluation residuals remaining exchangeable, something a stationary label does not guarantee.

Coverage refines the IC reading rather than restating it. A case study with a credibly nonzero IC and near-nominal coverage, such as ETFs or US Equities, sits in a different operational position from one with a credibly nonzero IC but unreliable coverage, such as Crypto or the NASDAQ-100, where the model may still rank assets while its uncertainty layer cannot yet be trusted. Multi-fold conformal calibration is the natural response where split noise dominates, and Mondrian, group-conditional calibration where coverage fails along known groupings (asset class, horizon, or volatility regime) at the cost of fewer calibration points per group.

### Direct prediction versus forecasting

The case studies treat sequence models as direct horizon-label predictors, and a controlled comparison shows why. Trained instead as a multi-step forecaster through Darts, TSMixer reaches an IC of +0.029 on ETFs at the 21-day horizon - real signal, but below both the direct-regression NLinear and Ridge on the same data (*Table 13.4*), because errors compound over the forecast path and degrade the eventual ranking. To remove compounding from the comparison, the US Equities case study was resampled to weekly, non-overlapping returns, in which predicting the five-day-ahead return becomes a single step.

| Model | Approach | Mean IC (4 folds) |
| --- | --- | --- |
| NLinear | Direct regression | +0.018 |
| LSTM | Direct regression | +0.006 |
| N-BEATS (Darts) | One-step forecasting | −0.013 |

*Table 13.5: Weekly-frequency US Equities experiment on non-overlapping Friday-to-Friday returns, where the five-day horizon reduces to a single forecast step*

Matching the data frequency to the horizon removes one source of error. However, it does not salvage the forecasting objective: direct regression still ranks better than one-step forecasting (*Table 13.5*), and both trail the daily tabular baselines, with gradient boosting reaching +0.032 on the primary one-day label. The finding restates *Section 13.7*: cross-sectional ranking is a direct prediction problem, not a path-forecasting one. It is also specific to this objective. Evaluating deep models for risk-adjusted position sizing instead and scoring on Sharpe ratio, downside risk, and cost robustness, Saly-Kaufmann et al. (2026) find hybrid recurrent and feature-selection architectures more competitive, a reminder that the target and the metric, not the architecture alone, determine what is being tested. *Chapter 17* returns to that risk-aware framing.

**Implementation**: `case_studies/us_equities_panel/12_dl_weekly` produces the results in this section.

### From IC to profitability

The case studies where the deep and tabular models part ways carry that result into the portfolio-construction and simulation chapters: Crypto, where the deep model adds content the tabular families miss, and US Equities and CME Futures, where gradient boosting holds the edge. On the rest, the choice between a deep and a tabular model is a choice between roughly equal ICs at very different training and inference costs. *Chapters 16* through *19* weigh that trade-off against turnover, capacity, and transaction costs to determine which ranking yields economic profit.

![Figure 13.8](assets/figure_13_8.png)

*Figure 13.8: Best validation IC by family (Ridge, GBM, TabM, and the best deep model) for each case study, with the deep-minus-best-tabular delta in the final column. Shade encodes IC magnitude*

## 13.10 Summary

Deep learning for time series has evolved from sequential RNNs through two competing philosophies, namely decomposition (N-BEATS) and attention (Transformers), into a mature toolkit where architecture choice depends on data characteristics, not benchmark rankings. The Zeng et al. critique remains a central lesson: any architecture must justify its complexity against a one-layer linear baseline. Modern responses like PatchTST and iTransformer succeed precisely because they embed stronger inductive biases rather than relying on raw model capacity.

Foundation models extend this toolkit with pretrained representations. However, in finance, they generally require domain adaptation: zero-shot inference on returns is limited, whereas parameter-efficient fine-tuning appears more promising for tasks such as volatility and risk forecasting. Evaluation also remains fragile because leakage, tuning budgets, and benchmark design can easily distort rankings. As a result, rolling-origin validation, strong non-deep baselines, and economic performance metrics remain essential. The cross-dataset case study experiments reinforce a practical point: in the book’s horizon-label, prediction-first setting, deep learning clearly beat the best tabular baseline in only one of eight case studies, while several others were statistically indistinguishable. This does not show that deep learning fails in finance; it shows that performance depends heavily on how the problem is formulated.

That leads to the chapter’s central conclusion: **architecture matters only after the target is defined correctly**. Recursive one-step forecasting, direct multi-horizon forecasting, horizon-label prediction, and direct allocation learning solve different problems.

In practice, begin with Ridge and LightGBM, test whether temporal structure is genuinely informative, and only then choose a more complex architecture that matches the data and objective. Uncertainty methods such as MC Dropout and Deep Ensembles can improve calibration for sizing and risk control. At the same time, generative and graph-based models extend the framework to scenario generation and relational data. *Chapter 14* next turns to latent factor models, which use dimensionality reduction to extract the underlying drivers of returns.
