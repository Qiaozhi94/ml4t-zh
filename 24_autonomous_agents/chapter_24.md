# Chapter 24: Autonomous Agents

Quantitative finance increasingly relies on systems that do more than evaluate fixed prediction functions. Earlier chapters focused on ML models that estimate returns and other targets from structured data. This chapter turns to agentic workflows: systems that gather evidence, use tools, maintain state, coordinate intermediate steps, and produce artifacts that can be reviewed, replayed, scored, and audited.

These workflows are useful beyond forecasting. In practice, agentic systems can support research, coding, data management, experiment tracking, report generation, monitoring, compliance review, and other tasks where the inputs are incomplete, heterogeneous, or changing. Their value does not come from replacing statistical models. It comes from organizing work around goals, evidence, tool use, provenance, and bounded autonomy.

The chapter therefore treats agents as engineering systems rather than chat interfaces. The core use case is decision support: collect market, document, and research evidence; turn that evidence into structured outputs; preserve enough state for replay and evaluation; and expose the controls that determine whether the result is trustworthy. This chapter does not cover autonomous trade execution. The systems here may search, read, compute, write sandboxed artifacts, and produce structured recommendations, but they do not place orders or commit capital. Execution and order routing require a stricter operational layer and are addressed in *Chapter 25*.

Two capstones make the architecture concrete. The first uses forecasting agents because event forecasting is evidence-rich, probabilistic, and externally scorable. A Bridgewater AIA-style system provides the motivating pattern: agents retrieve evidence, produce probability estimates, aggregate independent views, calibrate outputs, and preserve artifacts for later scoring. The second capstone generalizes the same primitives to the ML4T research process: an operator that uses coding, data, registry, and skill-retrieval tools to iterate on case-study research lines under bounded, replayable conditions.

After completing this chapter, you will be able to:

- Explain when agentic workflows add value in quantitative finance and when conventional statistical, rules-based, or batch pipelines remain the better choice.
- Distinguish the roles of ReAct, Tree of Thoughts, and Reflexion, and choose appropriate reasoning budgets for evidence-driven financial tasks.
- Design explicit state and memory schemas that support provenance, checkpointing, replay, schema evolution, and post-outcome evaluation.
- Specify robust tool contracts, structured outputs, source policies, and context-engineering rules for research, forecasting, and workflow agents.
- Compare single-agent, multi-agent, and operator-style architectures, and define a migration path from notebook prototypes to operational services without sacrificing visibility and control.
- Build an evidence-first forecasting workflow with structured output extraction, trace inspection, replayable artifacts, aggregation, calibration, and evaluation.
- Design a bounded research-iteration operator that uses coding, data, registry, and skill-retrieval tools while preserving auditability and human review.
- Define the operational, statistical, and security controls required to make agent outputs decision-grade, including point-in-time integrity, contamination-aware testing, observability, policy gates, sandboxing, and human approval boundaries.

The discussion moves from cognitive architectures to memory, tool integration, framework choices, and three implementations: a single-agent forecasting researcher, a multi-agent forecasting workflow, and an ML4T research-iteration operator. It closes with the controls that separate notebook demos from decision-grade systems, including point-in-time integrity, contamination-aware evaluation, observability, security policy, and human approval boundaries. *Section 24.1* begins with the structural shift from fixed prediction functions to adaptive agentic workflows.

## 24.1 From prediction functions to agentic workflows

Large language models change the structure of quantitative work (Korinek, 2025). Traditional pipelines map engineered features to outputs through fixed transformations: a model receives a feature vector and returns a prediction. **Agentic pipelines add an adaptive layer** that can inspect evidence, decide what information is missing, call tools to retrieve it, and update a persistent state before producing an output.

This shift does not replace statistical modeling; it changes where automation happens. Earlier chapters estimate returns, risk, and allocations using prepared datasets in which inputs are clean, and schemas are fixed. Agentic systems driven by large language models extend that pipeline upstream, into messier territory where raw evidence is noisy, heterogeneous, and incomplete. Kong et al. (2024) survey LLMs in investment management and identify retrieval augmentation and point-in-time discipline as prerequisites for reliable deployment, themes that recur throughout this chapter. In finance, this upstream layer includes filings, earnings transcripts, event calendars, policy statements, and market-implied probabilities: sources that vary in format, arrive on different schedules, and require relevance judgment before they can inform a decision.

*Figure 24.1* shows the high-level architecture as a five-phase loop, from Perception and Reasoning to Planning, Action, and Observation.

![Figure 24.1](assets/figure_24_1.png)

*Figure 24.1: Non-trading agent loop from perception through observation, with tools and memory attached to the core workflow*

In this chapter, “action” means **information actions**: calling a market-data API, querying a filing index, retrieving evidence from a vector store, requesting a calculation, or writing a structured artifact, not order execution. This boundary is a design choice: **read-only systems** are easier to secure and evaluate. It also aligns with the most credible public prototypes in the literature, including Bridgewater’s **AIA Forecaster** and **AlphaAgents** (by authors from BlackRock), which treat agent workflows as research and forecasting systems rather than as autonomous execution engines (Alur et al., 2025; Zhao et al., 2025).

Ang et al. (2026) present the **Self-Driving Portfolio**, an agentic architecture for institutional asset management that coordinates approximately fifty specialized agents through a full strategic asset allocation pipeline from macro regime classification, capital market assumptions, and portfolio construction to peer review with voting and CIO-level ensemble combination. The paper shows how multi-agent architectures can compress workflows that traditionally require teams of specialists working over days into automated runs completed in minutes. Their pipeline operates at the analysis-and-recommendation boundary, not at the execution boundary, reinforcing the non-trading design principle.

Financial decisions are constrained by three properties that generic chatbot applications rarely face:

1. Outputs should be probabilities with calibration diagnostics, not narratives. Uncertainty quantification is not optional when capital is at risk.
2. The system must respect what was knowable at the decision date; any leakage of future information invalidates the entire output.
3. Every claim must map to specific evidence and tool traces, as regulatory and compliance reviews require an audit trail that connects conclusions to their sources.

Agentic systems derive their complexity from decisions that depend on the acquisition and synthesis of evidence rather than on fixed feature transformations: tasks such as event forecasting, due diligence workflows, and structured research support. **A running example**

The running example estimates the probability that a prediction-market question resolves positively. The questions span macro events, corporate earnings, geopolitical developments, and technology milestones: heterogeneous, evidence-rich forecasting tasks where agentic workflows add genuine value over fixed pipelines. The complete workflow proceeds through **six stages**:

1. The system ingests the question and a decision cutoff date, enforcing point-intime discipline.
2. It then gathers evidence through a web search limited to the cutoff date.
3. Multiple research agents produce independent probability forecasts from the evidence they each discover.
4. Those forecasts are aggregated using principled combination rules and calibrated against historical reliability.
5. The system persists all artifacts (evidence, reasoning traces, and probability estimates) so they can be replayed.
6. Finally, after the event has resolved, scoring functions evaluate forecast accuracy using appropriate scoring rules.

Every major section of the chapter maps to one stage of this pipeline, and the accompanying notebooks implement each stage progressively:

- `01_react_reasoning` builds the provider abstraction and ReAct loop.
- `02_tool_contracts` adds typed schemas and provenance.
- `03_state_and_memory` introduces checkpointed state and quality gates.
- `04_research_agent` composes these into a single-agent research workflow.
- `05_aggregation_math` through `08_forecasting_pipeline` construct the multiagent system.
- `09_evaluation_and_governance` scores forecasts, runs ablations, and applies security controls.
- An optional `10_framework_comparison` contrasts the same pipeline in three framework styles.

### Where agentic workflows do not help

Agentic orchestration is not a universal replacement for traditional pipelines, and recognizing the boundary prevents costly overengineering. When labels are stable and fully structured, the adaptive evidence-gathering loop adds latency and non-determinism without improving accuracy. When the task is latency-critical (for example, execution at microsecond granularity), the overhead of language model inference is prohibitive. When no external evidence retrieval is required and model decisions can be implemented as fixed transformations, the entire reasoning loop is unnecessary machinery. In these settings, simpler statistical or rules-based systems deliver better reliability-to-cost tradeoffs. Agents are best treated as an additional layer for evidence-rich tasks, not as a default replacement for existing model stacks.

### Autonomy levels and realistic deployment boundaries

Agent deployments range from passive summarization (L0) through decision support with probability estimates (L1), constrained actions gated by human approval (L2), and progressively autonomous operation (L3–L4).

Most practical financial deployments remain at L1–L2 because the economics are asymmetric: a forecasting error is expensive, but a policy-free execution error can be catastrophic. A broad survey of agentic AI in finance confirms this pattern and documents the escalating systemic risk that accompanies higher autonomy (Aldridge et al., 2025). The notebooks in this chapter operate at L1, producing structured forecasts for human review.

### Interface with reinforcement learning

*Chapter 21* covers policy learning under reward signals, in which agents learn what to *do* from interactions with the environment. This chapter addresses a different layer: stateful research orchestration, in which the agent decides what to *know* before producing output.

The two layers are complementary: a research agent might produce probability estimates that feed an RL policy for position sizing, or an RL execution agent might query a research agent for updated fundamental context. Readers should treat this chapter’s outputs as structured inputs to the simulation and deployment stages in *Chapters 16–21* and *25*. With the agentic workflow framed, *Section 24.2* turns to the reasoning patterns that drive it.

## 24.2 Cognitive architectures – How agents reason

Reasoning patterns are foundational to agent design, but they are only one layer of the system. The patterns discussed here build on chain-of-thought prompting (Wei et al., 2023), which showed that externalizing intermediate reasoning steps improves multi-step accuracy without fine-tuning. Li et al. (2025) survey the full spectrum of reasoning architectures in LLMs, from fast pattern matching to deliberate multi-step search. In production, the quality of reasoning also depends on tool contracts, state visibility, and evaluation infrastructure, topics addressed in later sections. This section introduces three key frameworks as building blocks and clarifies where each one earns its complexity.

### The ReAct framework

**ReAct** (**Reason + Act**; Yao et al., 2023a) is the default pattern for evidence-grounded tasks. As *Figure 24.2* illustrates, it alternates between reasoning and tool use in a tight loop. The agent first identifies its next information need in a thought step, then selects and calls a tool, integrates the returned observation into its working context, and repeats until it judges the evidence sufficient for a response.

![Figure 24.2](assets/figure_24_2.png)

*Figure 24.2: A flow diagram showing how ReAct alternates thought, tool use, and observation until the evidence is sufficient for an answer*

The main benefit is **traceability**. Each intermediate claim can be linked to a specific tool call and observation, creating an audit trail that connects conclusions to sources. This property is valuable for compliance and debugging, especially when outputs are probabilistic forecasts that downstream systems consume.

**ReAct also has predictable failure modes** that practitioners should anticipate:

- **Repetitive tool loops** occur when the agent cannot determine that it has already retrieved the relevant evidence
- **Premature synthesis** occurs when the agent jumps to a conclusion based on a single observation rather than gathering corroborating data
- **Brittle behavior** follows from vague tool schemas: if the agent cannot distinguish between two tools with overlapping descriptions, it routes calls incorrectly
- **Hidden contradictions** accumulate when early and late observations conflict, but the agent never explicitly reconciles them

These failures are manageable when the workflow imposes iteration limits, evidence-sufficiency checks, and explicit state validators, all of which are shown in the accompanying notebooks.

**Implementation**: `01_react_reasoning` builds the full ReAct loop with a provider-agnostic `LLMClient` protocol that decouples agent logic from any specific LLM service. The agent emits structured JSON decisions at each step rather than relying on fragile text parsing, producing an execution trace of thought-action-observation triplets that the reader can inspect for auditability.

The same notebook switches providers without changing the agent code: the same loop runs against a deterministic mock, a local Ollama model, or a commercial API, establishing the abstraction that later notebooks inherit. Tree of Thoughts and Reflexion are presented as design patterns rather than dedicated notebook implementations; the capstone notebooks apply these ideas selectively within the multi-agent pipeline.

### Tree of thoughts

**Tree of Thoughts** (ToT) (Yao et al., 2023b) expands reasoning by exploring multiple candidate paths in parallel before selecting one. Where ReAct follows a single evidence chain, ToT branches at decision points and evaluates alternatives before committing. This is useful when decisions depend on strategic branches rather than linear accumulation of evidence: for instance, scenario planning around divergent policy regimes, stress-testing allocation alternatives under different macro assumptions, or evaluating multiple hypothesis families before deciding which retrieval queries to run.

For example, an agent analyzing the impact of a central-bank rate decision might branch into a hawkish scenario (such as a higher terminal rate or a sector rotation toward financials), and a dovish scenario (for example, rate cuts or duration-sensitive growth stocks), retrieving different evidence sets for each branch before selecting the better-supported path. Without branching, the agent would commit to a single narrative early and retrieve only confirming evidence.

The cost is compute and complexity. ToT multiplies the number of model calls by the branching factor and introduces state management to track, score, and prune candidate paths. It should be reserved for situations in which branch exploration materially changes decisions. In a typical financial research workflow, this means high-stakes strategic questions where the cost of premature commitment outweighs the cost of additional inference.

### Reflexion

**Reflexion** (Shinn et al., 2023) adds a post-run critique layer. After completing a task, the system records compact lessons (observations about what worked, what failed, and why) that can condition future behavior without full retraining. The method is attractive because it offers an adaptation mechanism that is lighter than fine-tuning but more persistent than single-conversation learning.

In practice, Reflexion helps only when memory policies are explicit. Lessons must carry provenance metadata that records when and under what conditions they were generated. Each lesson needs a validity horizon: a regime-change observation from a rising-rate environment should not persist indefinitely into a low-rate regime. Decay and pruning rules must retire stale lessons before they become persistent bias. And a rollback mechanism must handle the case where new evidence contradicts a stored lesson. Without these controls, Reflexion can convert temporary heuristics into durable analytical blind spots. *Section 24.3* formalizes the memory hierarchy that supports these policies.

### Choosing and composing reasoning patterns

A practical selection rule starts with ReAct as the baseline, adds ToT only for branch-heavy decisions where parallel hypothesis evaluation improves outcomes, and layers in Reflexion only after the core evaluation pipeline is stable enough to reliably identify which lessons are worth persisting. This sequence aligns engineering effort with measurable benefit: ReAct is cheap and auditable, ToT adds cost proportional to the number of branches, and Reflexion requires the most infrastructure investment. In real workflows, these patterns are usually composed rather than used in isolation. A common composition for forecasting tasks begins with ReAct for initial evidence gathering, branches into a brief ToT exploration when observations conflict and competing interpretations require explicit comparison, and optionally records a Reflexion note after scoring once outcomes are resolved. Composition should remain shallow (one layer of each) unless evaluation shows clear gains from deeper nesting. Deeply nested compositions increase trace complexity and make failure analysis substantially harder.

A practical guardrail is to define a maximum reasoning budget per run: a cap on total tool calls, a limit on branch count, and a ceiling on reconciliation rounds. When the budget is exhausted, the system should return a bounded output with explicit uncertainty rather than continue exploratory reasoning indefinitely. *Section 24.7* applies these budgets to the multi-agent capstone.

The following table compares the mechanisms, strengths, and best applications of each framework.

| Framework | Mechanism | Strengths | Best For |
| --- | --- | --- | --- |
| ReAct | Thought-Action-Observation loop | Grounded, auditable | Interactive research, due diligence |
| Tree of Thoughts | Parallel path exploration | Strategic depth | Scenario analysis, portfolio alternatives |
| Reflexion | Post-run critique and memory | Adaptive behavior | Iterative workflow refinement |

*Table 24.1: Reasoning frameworks: mechanism, strengths, and finance applications*

These frameworks are complementary. The chapter capstones use ReAct-style evidence loops as the default and add multi-agent and evaluation layers only where they provide measurable incremental value. Reasoning patterns determine how an agent thinks; *Section 24.3* addresses what it remembers.

## 24.3 Agent memory – State, persistence, and replay

After selecting a reasoning pattern, the next design task is memory. In agentic finance workflows, memory determines whether the system is testable and auditable. A one-shot prompt can produce a plausible answer, but a durable workflow requires explicit memory across multiple timescales: within a single reasoning step, across a task session, and between sessions over weeks or months.

### The memory hierarchy

*Figure 24.3* separates three timescales:

1. *Working* memory is the model context for the current reasoning step: prompt instructions, current state fields, recent tool outputs, and any retrieved evidence. It is bounded by the model’s context window and refreshed at every turn.
2. *Short-term* memory spans the task session, storing attempted actions, unresolved questions, temporary conclusions, and recent failures so the agent does not repeat dead-end searches or lose track of partial results.
3. *Long-term* memory is persistent storage that survives across sessions: retrieval indices, run artifacts, scored outcomes, and calibration history that the system can query as needed.

The memory types are shown in the following flow diagram:

![Figure 24.3](assets/figure_24_3.png)

*Figure 24.3: Working, session, and persistent memory support replay, audit, and disciplined reuse*

Yu et al. (2025) operationalize this hierarchy in **FINMEM**, a trading agent with layered memory and profiled character traits that adapts risk tolerance across market regimes. Their results show that explicit memory structure improves both performance and interpretability: the agent can explain which stored observations informed its current decision.

The main engineering risk is conflating these layers. When long-lived facts are kept only in context windows, reproducibility fails: the same question asked in a different conversation yields a different answer because the evidence has scrolled out of context. When transient reasoning is written directly to long-term memory without validation, stale assumptions accumulate, biasing future runs.

### Why explicit state is mandatory

Implicit chat history is convenient for prototypes but inadequate for production. A conversation transcript mixes instructions, evidence, reasoning, and tool outputs into an undifferentiated stream. Extracting any specific artifact (say, the evidence that supported a particular probability estimate) requires parsing free text, which is fragile and unreproducible.

Typed state objects solve this by making transitions explicit. A minimal state schema for a forecasting agent includes run and question identifiers, plus `as_of_date` and `cutoff_date,` which enforce point-intime discipline. The remaining fields record evidence, reasoning, and intermediate outputs: structured evidence records with source metadata and timestamps; a list of unresolved sub-tasks; a tool-call trace with status codes and error messages; intermediate probability estimates with associated confidence; and a decision artifact status that tracks whether the output is a draft, under review, or finalized.

These fields serve two purposes: they enable static and runtime validation before outputs are accepted, and they provide the substrate for replay and evaluation. Without them, debugging reduces to rereading conversation logs and guessing where the reasoning went wrong.

### Checkpointing, replay, and diagnosis

**Checkpointing** should be tied to decision boundaries rather than arbitrary intervals. The natural checkpoints for a forecasting workflow fall after initial evidence collection (when the agent has gathered its inputs), after contradiction resolution (when conflicting observations have been reconciled), before final synthesis (when intermediate probabilities are about to be combined), and before any high-impact downstream handoff (when the output will feed another system). **Each checkpoint captures the full state object**, all tool outputs received so far, and the metadata required for faithful replay. **Replay** is the core debugging mechanism for non-deterministic systems. Because language models are stochastic, a failing run cannot be reproduced by simply re-running the same prompt: the model may generate different reasoning paths. The objective of replay is to **isolate changes in model behavior** from changes in evidence. A useful protocol freezes the tool outputs from the target run, restores state from a checkpoint, reruns with updated prompts or policies while holding evidence constant, compares trace deltas and evaluation deltas, and accepts only changes that improve defined metrics. Without this discipline, teams routinely mistake evidence drift for prompt improvements, concluding that a reworded system prompt fixed a problem when in fact the underlying data shifted.

### Memory lifecycle and governance

Long-term memory requires governance because not all stored information should be reused indefinitely. Each artifact type needs a retention window: raw tool outputs might persist for the scoring horizon plus a buffer, while intermediate reasoning traces might expire after evaluation. Eviction rules must retire stale lessons before they become persistent bias. Provenance requirements ensure that reusable evidence carries enough metadata to verify its point-in-time validity. And conflict-handling policies must specify what happens when a new observation contradicts stored material: does the old artifact get flagged, archived, or deleted?

These policies are especially important for adaptive memory methods such as Reflexion (*Section 24.2*). A lesson tied to a specific volatility regime or policy environment should expire when conditions change. FinCon (Yu et al., 2024) takes a related approach: a multi-agent system that stores investment beliefs as natural-language summaries and re-injects them across episodes, using conceptual verbal reinforcement to improve sequential decisions without full retraining. The critical implementation detail is that stored beliefs carry regime tags and can be pruned when the tagged conditions no longer hold.

State schemas evolve as workflows mature. Every checkpoint should carry an explicit schema version with migration rules between adjacent versions, so that historical artifacts remain comparable across refactoring boundaries.

### Memory and evaluation are coupled

Forecast evaluation depends on memory quality in ways that are easy to underestimate. Calibration requires historical forecast-outcome pairs stored with enough resolution to verify whether events rated at 70% actually occurred roughly 70% of the time. Ablation analysis (testing whether removing a tool or evidence source changes outcomes) requires that intermediate artifacts from the original run be preserved. If artifact persistence is incomplete, evaluation shifts from systematic analysis to anecdotal review, rendering the agent’s claimed accuracy unverifiable. *Section 24.7* develops the full evaluation methodology.

Agent reliability depends on explicit state, disciplined persistence, and replayable diagnostics. Fluent output does not substitute for any of them, and memory design therefore belongs as much to model risk control as to software architecture. **Implementation**: `03_state_and_memory` makes these principles concrete. The notebook defines an `AgentState` dataclass that carries a run identifier, a forecast question, a cutoff date, an evidence list with source metadata and timestamps, a tool-call trace, open sub-questions, quality-gate results, and a synthesis status field.

Three quality gates enforce evidence discipline before the agent may advance to synthesis: a coverage gate that verifies all required evidence types are present, a freshness gate that rejects evidence outside the allowed time window, and a consistency gate that detects ticker mismatches and post-cutoff data leakage. The notebook also exercises checkpoint round-trip serialization (writing state to JSON, restoring it, and verifying that the restored object passes the same gates), establishing the replay substrate that *Section 24.6* applies in a full research agent workflow and that *Section 24.7* extends to multi-agent evaluation.

With state and memory in place, the next control surface is how the agent interacts with the outside world. *Section 24.4* formalizes tool contracts, provenance, and context engineering.

## 24.4 Tool integration – Contracts, controls, and context engineering

With an explicit state defined, tool integration becomes the next control surface. Tools convert language-model reasoning into verifiable operations: retrieving a stock quote, querying a filing index, or computing a rolling statistic. In finance, tool design is often the dominant determinant of agent quality, outweighing prompt engineering and even model selection.

### Tool categories for research and forecasting

A forecasting agent typically requires four classes of tools:

1. Market data tools retrieve prices, implied signals such as option-derived volatility, and event metadata like earnings dates and dividend announcements.
2. Document tools access filings, earnings transcripts, and analyst reports, either from local storage or through retrieval-augmented generation pipelines.
3. Search tools find recent external evidence (news articles, regulatory announcements, macroeconomic releases) subject to source controls that limit which domains the agent may query.
4. Deterministic calculation tools handle statistics and transformations that should not depend on language model stochasticity: computing returns, running regressions, or evaluating scoring functions.

Execution tools (order placement, portfolio rebalancing) constitute a separate layer with substantially stricter controls. They are not required for this chapter, which operates at the read-only boundary established in *Section 24.1*.

### Tool contracts and selection quality

A tool contract specifies purpose and scope, required arguments with their types and allowed value ranges, error semantics that distinguish transient failures from permanent ones, and provenance fields that every response must carry. Weak contracts push complexity into prompt text: if the model must infer from a vague description whether a tool returns adjusted or unadjusted prices, it will sometimes guess wrong. Strong contracts move constraints into typed interfaces and validation logic, where they can be checked statically before a call is dispatched.

High-quality tool descriptions include three parts: what the tool returns, when to call it, and when *not* to call it. The negative guidance matters because failures in tool selection are among the most common agent bugs. Typical patterns include selecting a broad tool when a scoped alternative exists (querying a general news API when a filing-specific tool would return more precise results), repeating calls after definitive errors, calling tools with incomplete arguments that trigger silent defaults rather than explicit failures, and mixing stale and current evidence without checking timestamps. Mitigations include argument validators that reject malformed calls before execution, retry policies that vary by error class, and state-based guardrails that block reasoning transitions until required fields are present.

### Context engineering

Context engineering is the controlled design of model-visible information across reasoning steps. Rather than sending the full accumulated context into every step (a common anti-pattern that wastes tokens and introduces noise), each step should expose only the state fields relevant to its task, make available only the tools appropriate for its phase, restrict evidence sources to those consistent with the current point-in-time boundary, and require a specific output schema that downstream steps can parse deterministically.

This principle applies at every scale. Within a single ReAct loop, the agent’s system prompt should specify which tools are active for that phase. Across a multi-agent pipeline, each specialist should receive only the evidence relevant to their analytical perspective, not the full research dossier. The result is reduced ambiguity, improved reproducibility, and lower cost: three properties that compound as agent complexity grows.

### Provenance and structured output

Whenever downstream components are code rather than humans reading prose, agent outputs should be structured objects: evidence records with source metadata, specialist forecast objects with probability estimates and reasoning summaries, uncertainty quantifications, and final decision artifacts. Structured outputs support deterministic parsing, schema validation, and consistent logging, all prerequisites for the replay infrastructure described in *Section 24.3*.

Tool responses should carry provenance by default. A practical response schema includes a source identifier, publication and retrieval timestamps, a document span or record key that localizes the evidence, policy flags such as allowlist status, and quality annotations when available. These fields make it possible to answer three operational questions quickly: Was this evidence available at the decision date? Which tool and source produced this value? Can the claim be reproduced from stored artifacts? When provenance is absent, errors propagate silently and are expensive to diagnose after the fact.

### Enforcing structured output

Typed output contracts are only as reliable as the enforcement mechanism. Major providers now offer built-in structured-output modes: JSON mode constrains generation to valid JSON, tool-use forcing requires the model to emit a function call rather than free text, and schema-validated generation checks outputs against a JSON Schema before returning them to the caller. These mechanisms move validation from post hoc parsing into the generation loop itself, eliminating an entire class of deserialization failures.

For the forecasting pipeline in this chapter, enforcement matters at two boundaries. Specialist agents must emit structured forecast objects with typed probability fields, not prose that downstream code must parse with regex. The supervisor must emit a final decision artifact that passes schema validation before it enters the persistence layer. When a model cannot satisfy the schema because the evidence is insufficient or the question is ambiguous, the enforcement layer should surface a structured error or abstention object rather than silently falling back to unstructured text. The chapter notebooks apply a lighter-weight version of this discipline: `01_react_reasoning.py` uses JSON-mode responses for tool decisions, while later notebooks move forecast artifacts into typed dataclass schemas and explicit validation logic in the orchestration layer. The design principle is the contract, not a particular validation library.

### Source policy

Search and retrieval tools require an explicit source policy. Domain allowlists restrict the agent to trusted sources. Date constraints tied to the run’s cutoff date enforce point-in-time discipline at the tool level rather than relying on the model to self-police. Source metadata must be present in every returned item so that downstream evaluation can trace claims to origins.

The FinDER benchmark (Choi et al., 2025) underscores this dependence in financial question answering over 10-K filings: retrieval quality is a binding constraint on downstream generation quality, which makes source policy a first-order determinant of agent output accuracy.

### Error handling and graceful degradation

Not all tool failures should trigger hard stops. A robust agent distinguishes transient failures that warrant retries, persistent tool failures that require a fallback path, policy violations that must be denied and escalated, and missing critical evidence that should result in an explicit abstention rather than a fabricated answer. Graceful degradation means returning a bounded output with explicit uncertainty rather than manufacturing false confidence, a property that matters more in financial applications than in most other domains.

**Model Context Protocol** (**MCP**) standardizes tool access across providers and runtimes. A pragmatic strategy is to start with direct integrations for core tools where latency matters, and to adopt MCP selectively where interoperability benefits outweigh the overhead. The *Chapter 22* RAG pipeline applies this pattern for document retrieval tools. In finance, tool integration is a systems-design problem. Reliable results require strong contracts, narrow permissions, explicit exposure of context, and structured outputs. Prompt quality matters, but tool quality sets the ceiling. With reasoning, memory, and tools defined, *Section 24.5* addresses the framework and engineering stack that hosts them.

**Implementation**: `02_tool_contracts` applies these principles with a ToolDefinition class that exports typed JSON schemas in both Anthropic and OpenAI formats from a single definition, eliminating provider-specific boilerplate. Every tool response is wrapped in a provenance-enriched result that carries a source identifier, publication, and retrieval timestamps, and policy-compliance metadata.

The notebook enforces a domain allowlist that restricts retrieval to approved sources and logs blocked calls with the policy that triggered the denial. Argument validation rejects malformed requests before execution, and the full execution log is structured for downstream replay and audit.

## 24.5 The engineering stack – Frameworks and migration

Given the tool and state requirements established in the preceding sections, the choice of framework becomes an implementation question. Framework discussion is often presented as a ranking problem: which library posts the highest benchmark score? For agentic finance workflows, this framing is usually unhelpful because the binding constraints are not benchmark accuracy but state visibility, replay capability, and policy enforcement. The better question is which framework best supports the controls required by the target workflow.

### Start from constraints, not preferences

Industry experience reinforces this constraint-first perspective. Recent practitioner write-ups describe agentic systems that are useful only when tightly integrated with proprietary tools, data controls, and evaluation loops, rather than being treated as stand-alone chat products (Fang and Moore, 2025). The common lesson is that workflow value comes from domain integration and governance, not from the choice of framework by itself.

A cautionary note: framework ecosystems evolve rapidly, and the landscape in early 2026 may differ from what readers encounter. The principles in *Sections 24.3-24.4*, namely explicit state, typed contracts, and replay support, outlast any specific library. If a framework enforces these properties with less boilerplate, adopt it; if it obscures them, avoid it regardless of benchmark claims.

Framework choice should follow from explicit constraints: the required state visibility; whether the team needs replay and checkpoint support for debugging non-deterministic runs; the complexity of the retrieval pipeline; whether a single agent with well-designed tools suffices or multi-agent orchestration is necessary; the debugging and observability infrastructure that must be in place before production; and deployment constraints such as team size, existing infrastructure, and maintenance budget. When these constraints are stated clearly, many choices become straightforward.
| Need | Preferred Style | Typical Fit |
| --- | --- | --- |
| Minimal overhead, full custom control | Native SDK and typed schemas | Small or tightly scoped agents |
| Quick ReAct/tool orchestration | Lightweight agent framework | Prototypes and teaching labs |
| Retrieval-centric applications | Retrieval-first framework abstractions | RAG-heavy workflows |
| Explicit state and durable execution | State-graph framework with persistence and replay | Production research pipelines |
| Complex multi-role collaboration | Multi-agent orchestration framework with role models | Specialized collaboration tasks |

*Table 24.2: Design-choice matrix*

### From notebook pattern to production demo

The notebooks build toward the production demo. Early notebooks isolate provider abstractions, tool contracts, state, and aggregation so readers can see each control surface clearly. Later notebooks compose those pieces into a forecasting workflow. The final step is to place the same logic within a stricter software boundary that includes configuration, connectors, storage, evaluation jobs, and publishing. In this chapter, that culmination is the `aia-forecaster` implementation, which turns the notebook patterns into a runnable forecasting service rather than a sequence of disconnected examples.

This progression also clarifies what a “framework comparison” can and cannot establish. `10_framework_` `comparison.py` compares how the same forecasting task looks when expressed as explicit Python, CrewAI-style role orchestration, and a LangGraph-style state graph. That comparison is useful because it explains the migration path into the production demo: start with explicit control flow, add stateful orchestration only where it helps, then wrap the resulting workflow in a packaged application boundary. The production forecaster is therefore not a different architectural idea; it is the end of the notebook sequence with stronger operational constraints.

### Migration sequence that reduces rework

A practical migration sequence keeps complexity proportional to measured benefit. Start by building a minimal baseline with native SDK calls and typed output schemas: this establishes the core data flow and makes the agent’s behavior testable from the first day. Next, add explicit state objects and trace capture so that every reasoning step is logged and inspectable. Then add checkpoints and replay hooks to support the debugging protocol from *Section 24.3*. Finally, add multi-agent orchestration only when evaluation shows that specialist diversity or adversarial debate produces measurable improvement over the single-agent baseline.

This incremental approach avoids a common set of anti-patterns: adopting multi-agent abstractions before a single-agent baseline shows the task is feasible; relying on framework defaults for security controls that should be explicit; postponing observability until after deployment, when debugging costs are highest; and treating framework migration as a substitute for evaluation methodology. Framework changes can improve ergonomics, such as better graph visualization and more convenient checkpointing APIs, but they rarely fix weak state design or poor scoring practices.

### Comparing frameworks locally

If a direct comparison is needed, compare one reproducible task using a single fixed dataset and metric suite, keeping prompts, tools, and policies constant so that differences reflect framework behavior rather than configuration drift. Report traceability quality, failure rates, and debug effort alongside task accuracy.

For this chapter, the comparison is intentionally narrow. The notebook contrasts a fully executable native SDK implementation with two framework-style expressions of the same pipeline. That makes it useful for reasoning about migration and state visibility, but not for producing a head-to-head vendor ranking.

| Pattern | State Handling | Chapter Status | Best Fit |
| --- | --- | --- | --- |
| Native SDK | Explicit Python objects and functions | Fully executable | Small or linear pipelines that need full control |
| CrewAI-style roles/tasks | Framework-managed task orchestration | Illustrative pseudocode | Role-based collaboration and fast prototyping |
| LangGraph-style state graph | Explicit graph state and node transitions | Illustrative pseudocode | Branching workflows that benefit from durable state |
| Packaged application layer | Config, storage, connectors, jobs, and publishing around an agent core | Implemented in companion repo | Operational forecasting services |

*Table 24.3: Framework patterns compared in the chapter*

Framework selection also has team-operating implications. Explicit graph-based orchestration often improves code review and incident triage because state transitions are visible in the graph definition rather than buried in conversation history. More implicit orchestration may accelerate prototyping, but can increase debugging costs when failures occur in opaque internal routing. Multi-agent abstractions improve modularity and separation of concerns, but they require stronger integration testing and clearer ownership boundaries to prevent coordination failures. Once a workflow becomes operational, another layer appears above the orchestration framework: profile-aware configuration, artifact versioning, connector health, storage migrations, and scheduled jobs. These are software-engineering concerns, but they determine whether the agent can be evaluated and maintained over time. **Implementation**: `10_framework_comparison` includes a fully executable native SDK, along with CrewAI- and LangGraph-style pseudocode for the same forecasting task. The native variant runs end-to-end with the same `LLMClient` and `ToolExecutor` used throughout the chapter; the framework variants are illustrative patterns that show how role definitions, task orchestration, and state-graph nodes can express the same logic. The comparison reveals concrete trade-offs: the native SDK offers maximum debuggability and state visibility at the cost of manual orchestration, while graph-based frameworks provide persistence and visualization at the cost of framework coupling.

Operational deployment is handled by the companion `aia-forecaster` repository, which wraps the same forecasting pipeline in configuration profiles, prediction-market connectors (such as Manifold or Polymarket), persistent run and forecast storage, scheduled evaluation commands, and a publishing layer that emits forecast feeds and evidence ledgers.

Framework choice is an architecture decision, not a performance claim. The right selection enforces state discipline, replayability, and policy controls for the specific workflow. *Section 24.6* applies all of these building blocks in the chapter’s first capstone: a stateful research agent.

## 24.6 Designing the research agent at the heart of the pipeline

This section operationalizes the design patterns from the preceding sections in a single-agent setting. The workflow remains read-only and evidence-driven. The objective is to produce a calibrated probability forecast for a prediction-market question using web search as the sole evidence tool. Every claim in the output must trace to a specific search query and its results, and the system must persist enough artifacts for replay and inspection. This notebook is the first capstone in the chapter’s progression toward the production forecasting demo: it keeps the scope narrow so that state, output extraction, and trace inspection remain legible before those same controls are embedded in a larger multi-agent pipeline.

### Architecture and toolset

The agent combines a single constrained tool, explicit ReAct-loop control flow, structured output extraction, and persistent traces for replay. The sole tool is web search: the same design choice made by the AIA Forecaster, where limiting the agent to a single tool class simplifies the contract surface and focuses evaluation on reasoning quality rather than tool-routing accuracy. The search tool enforces a cutoff date at the tool boundary, preventing post-cutoff evidence from entering the context. Execution tools are deliberately excluded. This agent produces probability estimates, not orders. The agent emits structured JSON at every step, choosing between two actions: `search` (issue a query) or `forecast` (produce a probability with rationale). This two-action schema is minimal by design: it eliminates the tool-selection ambiguity that causes routing failures in richer tool sets. The step prompt provides the question, optional market context, and the action schema; the agent must output exactly one valid JSON action per step.

### ReAct loop and rich output extraction

The workflow follows the ReAct pattern from *Section 24.2*. The agent begins by assessing what information it needs, issues search queries to gather evidence, and terminates with a forecast action when it judges the evidence sufficient. A configurable step limit (default: five) prevents unbounded exploration. If the agent exhausts its budget without forecasting, the system records a forced default with explicit uncertainty rather than manufacturing confidence.

After the agent produces a forecast, the system extracts rich metadata beyond the raw probabiliconfidence ൌʹ ڄ |݌yes −0.5|. Sentiment maps the probability to a five-level scale from strongly bearty. Confidence is either explicit (if the LLM includes it) or inferred from probability extremity

ish to strongly bullish. Key findings and uncertainties are extracted from the rationale text. Evidence quality is assessed from the number of search queries and sources consulted. The result is an `AgentForecastArtifact` that captures not just the prediction but the full provenance chain: traces, token usage, confidence, sentiment, evidence quality, and reasoning.

The quality gates from *Section 24.3* apply here as well. The state and memory notebook establishes the pattern: coverage gates verify that sufficient evidence was gathered, freshness gates check that evidence falls within the allowed window, and consistency gates detect post-cutoff leakage. These gates use the same mechanism as the multi-agent pipeline. The difference is that, at the single-agent level, gate failures are easy to diagnose and fix.

### Trace inspection and replay

Every step of the ReAct loop is captured in an `AgentTrace` object: the step number, the action taken, the search query (if any), the results returned, and the raw LLM output. This trace makes the agent’s reasoning path fully inspectable after the run completes. The full artifact (including traces, extracted metadata, and token usage) can be serialized to JSON for checkpoint round-tripping and comparison across runs.

In the production demo, the same discipline is carried into persistent run and forecast records. The important continuity is not the storage mechanism itself but the fact that every later scoring and audit step depends on artifacts that can be reconstructed without re-reading free-form transcripts.

### Walkthrough of a typical run

A typical run unfolds as a short sequence of bounded transitions. The agent receives a prediction-market question and an optional market-implied probability, searches for relevant evidence using one to three queries, and produces a forecast with a rationale grounded in the evidence it finds. The full trace and serialized artifact are then available for inspection. This sequence matters because it isolates failure points. If forecast quality is poor, the trace reveals whether the root cause was insufficient search queries, irrelevant results, JSON parsing failures, or flawed reasoning. In practice, the most common failures in this implementation are search queries that miss the key evidence, LLM outputs that fail JSON parsing, or rationales that ignore retrieved evidence in favor of prior knowledge. Those are exactly the failure classes that later grow more expensive in a multi-agent system, which is why the chapter addresses them here first.

### Evaluation and acceptance criteria

Before promoting to multi-agent workflows, the single-agent baseline should show stable task success across repeated runs, evidence groundedness above a predefined threshold, a low forced-default rate, reproducible artifacts from stored traces, and reasonable token efficiency. These criteria prevent premature escalation to more complex architectures, a common mistake when prototype results look promising but reproducibility has not been verified. *Section 24.7* develops the full evaluation methodology and extends this single-agent pattern into the production-oriented forecasting workflow.

When results are weak, debug in this order: tool contracts and search quality first, then JSON parsing and output extraction, then prompt wording. Most reliability failures originate in tool and parsing design, not in language generation. *Section 24.7* extends this single-agent baseline to a multi-agent forecasting system that incorporates agent diversity, aggregation, debate, and calibrated evaluation.

**Implementation**: `04_research_agent` builds the complete `ResearchAgent` class that is reused throughout the rest of the chapter. The agent runs a ReAct loop on a live prediction-market question (fetched from Polymarket or a resolved evaluation panel), using web search as its sole evidence tool. Each step produces structured JSON: either a search action with a query or a forecast action with a probability and rationale.

The notebook records rich output extraction (confidence, sentiment, key findings, uncertainties, and evidence quality), all captured in an `AgentForecastArtifact` with full provenance. The execution trace shows every search query, its results, and the final reasoning path. When run in mock mode (the default for CI), responses are deterministic and reproducible; when run with a live LLM provider, the agent performs real web searches and produces genuine forecasts. Readers can modify the provider, step limit, or search parameters through the Papermill parameters cell and observe how the agent adapts its search strategy and forecast quality.

## 24.7 Multi-agent forecasting systems

*Section 24.6* established a single-agent research baseline. This section extends that baseline to a multiagent forecasting architecture designed for probability quality, traceability, and reproducible evaluation. The capstone is a forecasting system, not an autonomous trading engine. Its design draws on two public architectures:

- **AIA Forecaster** combines agentic search, supervisor reconciliation, and calibration in a read-only prediction-market workflow (Alur et al., 2025)
- **AlphaAgents** emphasizes specialist roles and debate for equity decision support (Zhao et al., 2025)

The chapter implements the AIA-style pipeline: parallel research agents, aggregation, optional debate, supervisor review, calibration, and persisted forecast artifacts. The outputs remain probability forecasts and scored artifacts, and their value is measured after event resolution rather than at the time of narrative generation.

### Architecture – From agent ensemble to production pipeline

The architecture is easiest to understand in terms of six layers, each producing artifacts that make the system evaluable and debuggable at every stage. The intake layer loads open questions, resolution criteria, and market baseline signals from prediction-market connectors. It records the decision timestamp and the cutoff date used for the downstream retrieval policy. That step defines the point-in-time envelope that every subsequent tool call must respect.

The research layer then runs parallel agents under the same cutoff policy. The AIA Forecaster proposes that **agents** sharing a single prompt still produce useful variation because each issues its own search queries and reasons over whatever those queries return (Alur et al., 2025); stochastic sampling sends them down different retrieval paths rather than role specialization. The chapter notebooks test how much that variation matters and find the answer depends on the question. On `06_multi_agent_research`, three Claude Sonnet 4 agents forecast whether the US will enter a recession by the end of 2026, a question whose public evidence points in one direction. Their search paths still differed: one agent weighted current low-risk model readouts, the others leaned on historical base rates. The three forecasts nonetheless clustered at probabilities of 0.12, 0.22, and 0.22, close to the market price of 0.175. When the evidence is one-directional, independent agents reach similar probabilities, and that agreement is the expected outcome.

A genuinely contested question produces a wider spread for the same reason. On `07_adversarial_debate`, the agents forecast whether the Fed will hike in 2026, where credible evidence supports both sides; the same three-agent setup returned 0.35, 0.58, and 0.38, and the assigned bull and bear roles held a gap near 0.4 across three rounds without reaching consensus. The condition worth watching for is the opposite of either outcome: identical search results, identical reasoning, and an identical forecast across agents would signal a broken sampling temperature or a structural bug that suppressed the variation. That uniformity is the case to debug; similar numbers on a one-directional question are not. Forecast spread tracks how contested a question’s evidence is, not a sampling setting the operator can turn up on demand.

Debate and role specialization are therefore configuration choices to evaluate against the target question set, not fixes for a failure mode. An adversarial bull-bear stage and specialist mandates incur added latency on two-sided questions, where surfacing and stress-testing the disagreement move the aggregate; on one-directional questions, they add cost without changing the answer. Specialist-role architectures such as AlphaAgents (Zhao et al., 2025) and the institutional SAA pipeline of Ang, Azimbayev, and Kim (2026), which assigns distinct macro, asset-class, portfolio-construction, and risk mandates across roughly fifty agents, extend the same logic: explicit structure buys diversity that sampling alone does not, and whether it pays is empirical. If specialist roles produce measurably better calibration on the target questions, they justify their complexity; otherwise, a single research role plus an optional debate stage is the simpler default.

The next four layers convert those agent outputs into a forecast that can be tracked over time:

- **Aggregation** combines the independent probabilities into a pre-supervisor estimate. An optional debate stage stress-tests that estimate when the configuration enables bull-bear reconciliation.
- The **supervisor** layer identifies disagreements, performs clarifying searches, and may override the aggregate only when its confidence clears a policy threshold. The final calibration layer transforms the probability into the published forecast.
- The **persistence** and**scoring** layers store all intermediate and final artifacts so that calibration monitoring, baseline comparison, and ablation analysis can proceed after events resolve. Without this final layer, the system cannot learn from its own history.

The pipeline is shown in the following figure:

![Figure 24.4](assets/figure_24_4.png)

*Figure 24.4: Multi-agent forecasting pipeline with research agents, aggregation, debate, and supervisor stages*

### Aggregation and extremization

Aggregation encodes assumptions about dependence and is not a cosmetic step. Options range from simple averaging to weighted combinations based on historical reliability or evidence quality. The notebook demo exposes these choices as configuration rather than burying them inside prompts: mean, median, trimmed mean, and **Neyman-style extremization** are all explicit policies. A more principled approach is to apply extremization, which amplifies deviations from the base rate when forecaster diversity is genuinely high. One form is: 𝑝extreme = base ൅݀ ൈ(𝑝mean −base) where 𝑑 increases the distance from the base rate. Neyman’s extremization factor under an equicorrelated model is 𝑑√݊Ȁ(1 + (݊ െ1)ߩ), where 𝑛 is the number of forecasters and 𝜌 is their average pairwise correlation. When agents are perfectly independent (𝜌), 𝑑√݊ and extremization is aggressive. When they are highly correlated (𝜌), 𝑑, and the mean passes through unmodified.

This correctly reflects the fact that correlated agents contribute less independent information. tion factor 𝑑 varies with correlation 𝜌 for different numbers of forecasters, making clear that adding `05_aggregation_math` visualizes this relationship directly: a sensitivity plot shows how the extremiza-

correlated agents provides diminishing marginal information. The notebook also implements Platt scaling and compares calibrated against uncalibrated aggregates. This relationship exposes a fundamental design tension: adding more agents improves forecasts only if they bring genuinely diverse evidence, not just different phrasings of the same analysis.

### The debate-supervisor pattern

Some systems insert an adversarial debate between aggregation and supervisor review. A minimal design has a bull agent presenting the strongest positive case, a bear agent challenging assumptions and the source’s quality, an optional second round focused on unresolved contradictions, and a supervisor who reconciles the outcome. Debate can surface weak assumptions and improve the quality of synthesis, but it should remain optional and measurable. If debate does not improve out-of-sample scoring metrics relative to its cost and latency overhead, it should be removed. This is why the notebook demo treats debate as a configuration flag rather than as mandatory architecture.

The supervisor itself is a reconciliation stage, not a replacement for agent evidence. It identifies the agreement and disagreement structure across the research agents, assesses the quality of the evidence and the severity of contradictions, determines whether additional clarification is needed, and produces the final forecast artifact with an explicit rationale and caveats. In the AIA-style implementation, supervisor override is policy-bound: the supervisor may replace the aggregate only when it returns a forecast with high confidence. That rule matters because it prevents the last model call in the pipeline from automatically dominating the rest of the evidence.

### Probability calibration

Raw model probabilities are often miscalibrated. As Alur et al. (2025) observe, LLMs are “fundamentally miscalibrated for probabilistic prediction under uncertainty” and tend to hedge toward base rates. Calibration aligns forecast probabilities with realized outcome frequencies so that events rated at 70% by the system actually occur roughly 70% of the time.

A common post hoc method is **Platt scaling**, which fits a logistic transformation: 𝑝Ƹ ൌߪ൫ܽ ڄ logit(𝑝) ൅ܾ൯

Calibration and aggregation are distinct operations, and their interaction matters. If aggregation already extremizes probabilities, applying Platt scaling on top can compound overconfidence unless the calibration window is disjoint from the aggregation training data. This is not only a statistical concern but also an implementation concern: the production demo explicitly warns when both extremizing aggregation and post hoc calibration are enabled, because the combination can overshoot without validation on held-out data. A practical calibration workflow reserves a historical window of resolved events, fits calibration parameters on that window only, evaluates on a disjoint holdout window, and monitors drift with an explicit refit cadence. Common pitfalls include fitting on too few resolved events, overlapping calibration and evaluation windows, recalibrating too frequently without evidence of drift, and interpreting improved **expected calibration error** (**ECE**) as overall improvement when sharpness has collapsed. The remedies are procedural: define minimum sample sizes for recalibration, isolate temporal windows, jointly monitor calibration and sharpness, and version-control the calibration model to ensure historical comparisons remain valid.

### Evaluation, ablation, and baselines

A forecasting system should be judged by the quality of its probabilities, not by the quality of its narrative. The core metrics are Brier score, log score, expected calibration error, reliability diagnostics, and sharpness. Lower Brier and log scores indicate better overall forecast accuracy. Lower ECE indicates better calibration. Higher sharpness (the variance of forecast probabilities) is desirable only when calibration remains acceptable; a sharp but miscalibrated system is dangerous.

Ablation analysis identifies which components add measurable value. At a minimum, the evaluation should compare raw agent probabilities against the aggregated pre-calibration forecast, the aggregated post-calibration forecast, a variant without supervisor review, a variant with fewer agents, and the market baseline. Useful ablation questions include whether debate improves scores after cost adjustment, whether supervisor override reduces large errors or adds noise, whether calibration improves reliability without collapsing sharpness, and whether agent diversity provides genuinely incremental information. Ablations should be interpreted jointly with cost and latency: a component that slightly improves score quality but doubles runtime may still be justified for low-frequency, high-value decisions, but not for high-throughput workflows.

**Diagnostic checklist: Beyond aggregate metrics**

Aggregate metrics can hide systematic errors. Diagnostic analysis should examine error concentration by event type, overconfidence rates in high-probability bins, the contribution of supervisor overrides to large misses, and patterns of agent disagreement preceding forecast failures. These diagnostics clarify whether the next improvement should target retrieval quality, aggregation assumptions, calibration methodology, or override policy.

The most defensible claim is incremental information relative to a robust baseline, typically market consensus for the same event universe. Two cautions about interpreting that comparison:

1. For unresolved questions, both the market price and the agent forecast are estimates of the same future probability; neither can be declared correct in advance without an unwarranted assumption that the market is the ground truth.
2. On a resolved-outcome panel sufficient to compute Brier or log score, contamination is the dominant operational risk: the agent’s search tools may retrieve articles dated after the question’s natural cutoff, so apparent outperformance can reflect leakage rather than skill.

The defensible workflow is to evaluate on time-disjoint windows with explicit cutoff enforcement, report ranges across calibration variants and seeds rather than point estimates, and treat single-panel “wins” as illustrative of the scoring methodology rather than as definitive evidence of the superior approach. If, after that discipline, the model does not add incremental information at a stable cost/ latency envelope, the architecture needs revision; calling it sophisticated is not a defense.

Apply a retention rubric to each component: Does it improve resolved-outcome metrics? Is the improvement stable across event subsets? Is the cost and latency overhead acceptable? Does it increase policy risk or operational fragility? Components that fail should be demoted or removed.

| Component | Description |
| --- | --- |
| Research agents | N parallel agents that search and reason independently; forecast spread reflects how contested the question’s evidence is, not the sampling temperature |
| Debate | Optional bull/bear reconciliation stage |
| Supervisor | Policy-bound synthesis and override logic |
| Aggregation | Mean/median/weighted/extremized combination rules |
| Calibration | Post hoc probability calibration |
| Search | Bounded, date-filtered retrieval with source policy |
| Persistence | Run artifacts, trace metadata, and scored outcomes |
| Evaluation | Brier/log/ECE, reliability diagnostics, and ablations |

*Table 24.4: Implementation architecture*

Production optimizations include asynchronous agent execution, response caching with cutoff-aware keys, and replay modes for rapid iteration. They also require persistent run records, forecast rows, and later-evaluation rows so that the system can transition from a single notebook run to a track record. *Section 24.9* turns to the operational controls that make these forecast quality claims decision-grade.

**Implementation**: The multi-agent forecasting pipeline is built progressively across five notebooks:

- `05_aggregation_math` covers the pure mathematics of Neyman extremization and Platt scaling with sensitivity analysis across correlation and forecaster-count parameters.
- `06_multi_agent_research` runs three `ResearchAgent` instances on the pinned recession question and reports the result described above: when the evidence points one way, the agents’ independent search paths still land on close forecasts (0.12 to 0.22), near the market price. The notebook compares aggregation methods (simple mean, Neyman extremization, and confidence-weighted Neyman) and analyzes sensitivity to the correlation parameter, so readers can see what aggregation can and cannot add when the inputs already agree.
- `07_adversarial_debate` implements the bull-bear debate, tracking the probability gap across rounds with consensus detection that terminates early when positions align.
- `08_forecasting_pipeline` assembles the full agent-aggregation-debate-supervisor pipeline into the `AIAForecaster` class, runs it across multiple forecast questions, and reports token-cost analysis per question alongside persisted state.
- `09_evaluation_and_governance` scores a ten-question resolved panel spanning US macro (Fed rate decisions, CPI), corporate milestones (NVIDIA earnings, Tesla deliveries, Apple product events), market levels (S&P 500 targets, tech IPO timing), crypto price thresholds (Bitcoin), geopolitical events (tariff policy), and technology releases (GPT-5). Metrics include Brier, log, ECE, and sharpness; the notebook also produces a reliability curve and runs ablation experiments comparing the full pipeline against mean-only aggregation, a single agent, and market-baseline prices.

When run in mock mode (the default for CI), responses are deterministic and the metric values illustrate the evaluation methodology. When run with a live LLM provider, the notebooks perform real web searches on current prediction-market questions and produce genuine forecasts. The companion `aia-forecaster` repository provides the supporting operational machinery (configuration profiles, prediction-market connectors, persistent storage, and scheduled evaluation) needed to build a real track record over time.

## 24.8 The ML4T research agent

The forecasting workflows in *Sections 24.6* and *24.7* stop when an agent returns a calibrated probability. They are valuable systems, and a live multi-agent forecaster runs on the book’s website, but their deliverables sit closer to event prediction than to the systematic-trading research process this book has developed across its earlier parts. *Chapter 20* ran that process to its first decision point on nine case studies.

Each case study now exists as a research artifact: a fixed dataset, a labeling and feature pipeline, a comparable set of models, a portfolio construction rule, a backtest with explicit cost and risk treatment, and a registry of validation and holdout results. *Section 20.9* closes each case study with a sketch of the most informative experiment to run next. The agent in this section is what executes that sentence end-to-end against the artifacts the case study left behind.

### The operator’s shape

Research-line iteration looks different from prediction-market forecasting. Every case study has its own follow-up, and that follow-up is rarely a parameter sweep: it is a small amount of code that wires together the chapter’s libraries against a real run-log registry. Pre-enumerating the catalog of moves a researcher might make across nine case studies would either grow without limit or constrain the agent from making the moves that matter. The shape that fits this task is the one that production coding agents have converged on, and that this chapter calls the **operator**: a small orchestrator that hands a language model a handful of general-purpose tools (read and write files, run bash, query a SQLite registry, read a Parquet file, list and read a corpus of how-to skills) and then steps out of the way. The operator that runs the notebook is roughly 880 lines and exposes 10 such tools. The language model accesses them via `run_bash` when it decides the moment has come.

### Skills are the task-specific knowledge layer

The skills repository is the second pillar of this section, and a key feature of the book as a whole. Earlier chapters teach the methodology (point-in-time discipline, deflated Sharpe, cross-sectional information coefficient computation, walk-forward design, cost-aware backtest specification) in long-form prose alongside production code. The skills repo distills that methodology into a corpus that the agent can consult at runtime. The companion `skills` repository ships 56 concept-first Markdown files organized into nine categories: concepts, validation, infrastructure, features, data, backtest, and so on. Each file is short and structured:

- A problem statement
- A WRONG and CORRECT example pair
- A Production Implementation block that names the exact `ml4t-data`, `ml4t-engineer`, `ml4t-` `diagnostic`, or `ml4t-backtest` function to call

The operator’s prompt does not embed any of this. Instead, two of its ten tools are `list_skills`, which returns a one-line summary across the catalog, and `read_skill`, which fetches the full text of one file. When the agent reaches a method-choice decision point (“should I deflate this Sharpe?”, “How do I compute IC across a panel without leakage?”, “Which library function applies the next-step suggestion?”), it reads the relevant skill on demand.

The discipline that prevents look-ahead bias, mishandled cross-section ICs, or naive Sharpe arithmetic lives in those files, and the libraries the operator’s bash subprocess imports enforce the rest. This is the inversion that makes the operator pattern work: the agent’s prompt stays short and stable while the corpus the agent can consult grows with the methodology the chapter has been teaching.

The forecasting workflows in *Sections 24.6* and*24.7* do not need this layer because their action space is bounded by the search and forecast schemas; the workflow agent in *Section 24.8* is exactly where skills incur their cost.

### Two example runs of the ML4T research loop

The notebook replays two complete runs against the captured DeepSeek v4 Pro traces.

**The first run targets the ETFs case study**, whose *Section 20.9* next step is to ensemble the strongest gradient-boosting, tabular deep-learning, and convolutional autoencoder predictions and check whether the combined signal stabilizes the holdout Sharpe of the LSTM baseline. The operator did exactly that: it queried the registry to identify the prediction set with the highest validation Sharpe in each model family, read the skills on IC computation and deflated Sharpe, z-score-averaged the three signals cross-sectionally, recovered the LSTM baseline’s exact backtest specification from the `backtest_runs.` `spec_json` field, and re-ran the cost-aware backtest under that identical configuration. The ensemble produced the highest information coefficient in the case study (0.065 versus the LSTM’s 0.052), but its validation-window Sharpe was lower (0.56 versus 0.92 on the same window). Per-fold IC instability (three of nine folds were negative) was amplified by the score-weighted top-k allocator, eroding the IC advantage at the portfolio level. The registry carries holdout predictions only for the LSTM, so this validation comparison cannot displace the holdout ranking; promoting the ensemble there would require retraining or generating holdout predictions for its three constituents. The agent diagnosed the gap and reported a clean negative. The run took 39 tool-use turns and roughly $1.25 in OpenRouter spend.

**The second run targets the US firm characteristics case study**, whose *Section 20.1* prose flags that the selected strategy’s long/short legs cluster heavily in small-cap names and whose *Section 20.9* suggests filtering the universe to its top three market-capitalization quartiles. The operator located the lagged market equity column in the case study’s feature parquet, computed monthly mcap-quartile cutoffs, signal-filtered the selected model’s predictions to the top three quartiles, recovered the matching backtest specification from the registry, and re-ran it.

Sharpe collapsed from 4.27 to 2.24 (a 48% drop) while maximum drawdown deepened from −15% to −52% and IC fell from 0.074 to 0.048. Turnover was unchanged, confirming the move was a signal-level filter rather than a portfolio-construction artifact. The chapter’s small-cap binding-constraint claim is now a quantified one: the strategy retains genuine alpha after the capacity filter (Probability of Backtest Overfitting near zero), but roughly half of the reported Sharpe is a small-cap residual the static cost assumption did not absorb. Twenty-seven turns, about $0.95.

### What the runs share and where the limits sit

The two outcomes deliver opposite results: a clean negative and a quantified concern, as one might obtain from regular research. The runs share the property that makes the operator pattern portable: extending it to a different case study is one entry in `CASE_STUDY_TASKS` and a different value for `RESEARCH_OPERATOR_CASE_STUDY`. The same agent loop, the same tool surface, and the same skills repository handle both runs without code changes. The reason this works is that the methodology is not in the operator. It is in the libraries that the operator’s bash subprocess imports, and in the skills that the model reads when a method choice arises. That separation of concerns is also what makes the loop auditable: the trace records every tool call, every skill read, and every script the model wrote, so a reviewer can replay the decision path just as the demo notebook replays the captured run.

The failure modes worth watching are model-side: a confabulated improvement, a `done()` declared before the experiment finished, and an overaggressive interpretation of an inconclusive result.

The structural protections (sandboxed writes, a read-only working copy of the registry, a bounded turn count, a fixed model) are necessary but not sufficient. Production deployment of this pattern requires the controls in *Section 24.9* to be wrapped around the operator: replay receipts, contamination tracking, deterministic seeds, and the operational gates that separate a notebook demonstration from a service.

### Key Takeaways

The Research Agent reuses the architectural primitives the rest of this chapter built (typed traces, structured artifacts, an explicit ReAct loop, a replay-first persistence model) and re-points them from the open web to the deterministic ml4t stack.

The deliverable changes: a forecast resolves and earns a Brier score, while a research-line iteration produces a recommendation about the line and a registry-attached experiment record that future iterations are accountable to. Both deliverables are trustworthy only when the system that produces them is bounded, replayable, and inspectable, and the case for the operator pattern is that these properties carry over from a narrow forecasting agent to a workflow agent operating on a real research artifact.

The next section turns to the production controls (replay, contamination tracking, and the operational gates) that any executor downstream of an iteration agent will need before it can be trusted to act on a research note without a human in the loop.

**Implementation**: `11_research_operator` replays the two runs end-to-end from saved traces. Toggling `REPLAY_TRACE = False` runs the operator live against a sandboxed copy of either case study (~$1, ~8 minutes per run on DeepSeek v4 Pro via OpenRouter).

Switching the model is a single parameter; switching the case study is a single entry in `CASE_STUDY_TASKS`. The notebook prints the tool surface, shows skill discovery, replays both runs, and tabulates the side-by-side outcomes.

## 24.9 Preparing for production

A forecasting-agent prototype can look strong in notebooks and still fail in production. Common causes are non-determinism, hidden data leakage, weak observability, and uncontrolled cost. After defining the capstone architecture, the next design challenge is operational robustness: the controls that make forecast quality claims decision-grade. In this chapter, that transition is not hypothetical. The notebook sequence culminates in a production demo that wraps the forecasting pipeline in configuration profiles, prediction-market connectors, persistent storage, evaluation commands, and publishing jobs.

### The reliability gap and observability

Agents are non-deterministic systems. Identical prompts can yield different tool sequences and different conclusions across runs. Reliability, therefore, requires process controls rather than single-run confidence. Bounded retries classified by failure type prevent the system from hammering a broken tool. Fallback strategies produce explicit low-confidence outputs when primary paths fail, instead of fabricating answers. Checkpoint recovery allows restarts from the last known-good state rather than re-running entire workflows. Deterministic validators around tool outputs catch schema violations and missing fields before they propagate downstream. The target is stable workflow behavior under variation, not exact token-level reproducibility. Every run should produce a complete operational trace. At minimum, traces must capture run identifiers and timestamps, the input prompt and policy context, every tool call with arguments and outputs, state transitions and gate outcomes, model outputs and structured artifacts, latency and token cost telemetry, and final status with error classification. Without this trace, incident diagnosis is mostly guesswork. The production demo turns this requirement into data structures rather than prose guidance: run records persist configuration snapshots, agent artifacts, aggregation and supervisor outputs, token counts, search counts, and duration; forecast records persist the published probabilities and the market price observed at forecast time; evaluation records persist post-resolution scores.

**Observability** should support two distinct views:

- An engineering view for tool failures, latency spikes, and state-transition anomalies
- A research view for calibration drift, baseline deltas, and ablation sensitivity

Separating these views prevents teams from conflating operational incidents with model-quality issues. Fabozzi and López de Prado (2025) reinforce this point: prompts, retrieval corpora, and model versions must be treated as regulated artifacts because an LLM-assisted output that cannot be reproduced months later creates compliance risk regardless of its accuracy.

### Statistical testing and point-in-time integrity

distribution-aware thresholds: a success rate above a defined level across 𝑁 runs, a policy-violation Deterministic assertions are insufficient for agent systems. Testing should include repeated trials with

rate below tolerance, the variance of forecast outputs under controlled inputs, and the stability of calibration metrics over rolling windows.

Point-in-time integrity is a first-class requirement. Every run should carry a `cutoff_date` that propagates to all tool calls; retrieval tools must apply date filters that exclude post-cutoff evidence, and downstream components should revalidate publication timestamps. The replay protocol from *Section 24.3* applies directly. In the production demo, the same boundary appears as an explicit backtest cutoff that is propagated into research-agent search and supervisor-clarifying queries, which is the correct level at which to enforce the rule: the tool boundary, not the model narrative.

### Contamination-resistant evaluation

Backtests can overstate performance when models have latent exposure to historical periods. Lopez-Lira, Tang, and Zhu (2025) show that LLMs can appear to forecast economic variables pre-cutoff by memorizing realized outcomes from their training data, rendering pre-cutoff evaluation fundamentally non-identifiable. Contamination-aware evaluation should therefore apply strict temporal splits with a gap between training and evaluation periods, time-shift tests that assess whether model performance degrades as the evaluation window moves further from the training data, event windows designed to avoid leakage from post-event coverage, and baseline comparisons on the same event set to anchor interpretation. Claims of outperformance should be rejected when they are not stable across these controls. When contamination analysis and demo performance disagree, the contamination analysis should take precedence.

### Evaluation metric stack

The evaluation stack spans three dimensions:

1. Research agent quality is measured by task success rate, citation faithfulness (does each claim link to stored evidence?), tool-call validity, abstention quality when gates fail, and latency and cost per successful run.
2. Forecasting quality is measured by Brier score, log score, expected calibration error, reliability diagnostics, and sharpness.
3. Operational health is tracked through incident rate, policy violation rate, replay pass/fail rate, and drift alerts on key metrics.

Benchmark evidence helps calibrate realistic thresholds. The FinBen evaluation (Xie et al., 2024) tests LLMs across 42 financial datasets and 24 tasks, finding that extraction and classification are relatively reliable, while forecasting and numeric reasoning remain weak. This is a useful reference for setting expectations by task type. However, static benchmarks measure isolated LLM capability, not agent behavior. Evaluating agents is fundamentally harder because outputs depend on dynamic tool interactions, state transitions, and multi-step reasoning chains that vary across runs.

Resolution-based scoring platforms, in which forecasts are evaluated against real-world event outcomes after a defined waiting period, provide a more rigorous evaluation protocol. The key advantage is that they test the full pipeline under conditions less prone to memorization-driven overstatement, making them a natural complement to the contamination controls described above.

**Implementation**: `09_evaluation_and_governance` illustrates this logic on a ten-question resolved panel spanning US macro (Fed rate decisions, CPI), corporate milestones (NVIDIA earnings, Tesla deliveries, Apple product events), market levels (S&P 500 targets, tech IPO timing), crypto (Bitcoin), geopolitical events (tariff policy), and technology releases (GPT-5). The notebook produces Brier, log, ECE, and sharpness scores, a reliability curve, and an ablation comparison that together make the evaluation workflow structure concrete. In mock mode, the specific metric values illustrate the scoring methodology; when run with a live LLM, the same scoring pipeline evaluates real forecasts. The production `aia-` `forecaster` provides the necessary machinery to accumulate a real track record over time.

### Cost, latency, and deployment fit

Current LLM latency makes agent workflows unsuitable for microsecond-sensitive execution. The stronger near-term fit is research support, event forecasting, and lower-frequency decision preparation. This constraint should be explicit in system design and reader expectations.

Cost should be measured at the workflow level, not by token price alone. The most effective lever is model cascading: routing sub-tasks to the cheapest model that meets quality requirements. In a typical forecasting pipeline, evidence extraction, formatting, and tool-argument construction are well-suited to smaller, faster models, while probability synthesis, contradiction resolution, and supervisor reconciliation benefit from frontier-model capabilities. A practical cascading architecture assigns a model tier to each pipeline phase and monitors per-phase error rates to detect when a cheaper tier degrades output quality. In practice, cascading often materially lowers costs, but the gain has to be measured phase by phase rather than assumed.

Additional cost controls include caching with cutoff-aware keys that avoid redundant retrieval, context compression before model calls, early-exit logic when confidence is sufficient, and per-forecast budgets that cap total expenditure. Cost optimization that degrades calibration or increases policy violations is not a valid trade.

### Release process and monitoring

A lightweight release protocol improves stability without adding excessive ceremony. Before deployment, run replay and ablation checks on a fixed validation panel, compare score, calibration, and policy metrics against the prior release, require explicit sign-off for any regressions, and deploy behind a canary or shadow run when possible. Incident response should classify failures by root cause: retrieval or policy failures, connector or runtime failures, state-transition failures, or calibration drift, so that remediation is targeted and postmortem quality improves over time.

Operational packaging helps here as much as model quality does. Configuration profiles separate local, backtest, and production settings without rewriting prompts or code. A scheduler can run pull, forecast, and track jobs on different cadences. A publisher can emit a feed, per-market histories, and evidence ledgers for external inspection. None of these features directly improves probability, but together they determine whether the workflow can be monitored, replayed, and governed as a living service rather than a one-off notebook run. The governance document that constrains the pipeline need not be bespoke: Ang, Azimbayev, and Kim (2026) show that the Investment Policy Statement (already a legal and regulatory requirement for institutional investors) can serve as the operational boundary for an entire multi-agent pipeline, analogous to the operational design domain in autonomous driving. Repurposing an existing governance artifact rather than inventing a new one reduces adoption friction and aligns agent constraints with the same document that constrains human portfolio managers.

A production workflow should expose a compact dashboard spanning forecast quality (including Brier, log score, ECE, and sharpness), baseline deltas, policy violations, latency, cost, and metric drift. Alert thresholds should be policy-driven (triggering on sustained ECE increases, rising unsupported-claim rates, post-cutoff retrieval attempts, or latency violations) and coupled with run-level trace links for triage.

### Collective effects and risk posture

As agent usage rises across the industry, feedback loops can amplify market fragility. Lopez-Lira (2025) shows this in market simulations where LLM agents built on similar foundation models exhibit correlated behavior, producing bubbles, and liquidity crises that would not emerge from heterogeneous agents. This motivates a conservative deployment posture: clear action boundaries, explicit approvals for high-impact decisions, and strong monitoring for correlated failure patterns across agent instances.

Production quality is achieved through controls that make behavior measurable: point-in-time integrity, replayability, contamination-resistant evaluation, trace-complete observability, and durable operational artifacts. Without these controls, claims about forecast quality are not decision-grade. *Section 24.10* formalizes the security and governance layer that protects these controls from adversarial interference.

## 24.10 Security and governance

Building on the production controls in *Section 24.9*, this section formalizes security and governance requirements for financial agent systems. Security begins with architecture: the chapter defaults to read-only forecasting and research, thereby reducing the attack surface and simplifying policy enforcement. The production demo preserves that boundary. It pulls markets, searches for evidence, stores forecasts, and publishes artifacts, but it does not place trades or commit capital automatically. The controls described here apply regardless of the chosen framework and should be treated as prerequisites for any deployment that touches financial data or produces outputs consumed by decision-makers.

### Threat model – Where failures originate

A practical threat model identifies attack surfaces by workflow layer. At the input layer, malicious user prompts and malformed queries can steer the agent toward unauthorized behavior. At the retrieval layer, prompt injection embedded in documents and poisoned indices can hijack reasoning, a risk amplified in finance because retrieved text often contains imperative language and speculative narratives. At the tool layer, excessive permissions, argument abuse, and tool shadowing (substituting a malicious tool for a legitimate one) create execution risks. At the state layer, corrupted artifacts, stale memory reuse, and loss of provenance undermine reproducibility. At the output layer, unsupported claims, overconfident probabilities, and policy bypass can mislead downstream consumers.

An additional risk cuts across all layers: latent model bias. Lee et al. (2025) show that LLMs exhibit systematic investment preferences (favoring certain sectors, styles, or contrarian positions) and display confirmation bias when evidence conflicts with these latent preferences. These biases are invisible to standard accuracy metrics and require dedicated stress tests that present the same evidence framed from opposing perspectives. Agent diversity in *Section 24.7* partly mitigates this risk, but only if agents use different foundation models or receive deliberately varied framing.

### Prompt injection and retrieval defenses

Prompt injection remains the primary risk in tool-connected systems, as documented by OWASP (2025). Defenses require strict separation of instructions from retrieved data so that injected text cannot override system prompts. Content sanitization and pattern filters should strip known injection patterns before evidence enters the reasoning context. Evidence gating should verify that claims in the synthesis phase link to specific retrieved artifacts rather than to injected instructions. Citation verification checks that every referenced source actually exists in the evidence store. The chapter notebook illustrates this with a simple pattern-based sanitizer. The production demo adds another practical layer by restricting search with explicit allowlists and blocklists, thereby reducing the set of documents that can reach the model in the first place.

Retrieval poisoning targets the evidence pipeline rather than the prompt. Mitigations include index governance: source controls on what enters the retrieval corpus, ingestion checks that validate document provenance, and periodic integrity scans that detect unauthorized modifications to the index.

### Least privilege and the warden pattern

Excessive capability is a preventable risk. If a workflow requires read-only data access, write and execution privileges should not exist in that runtime. The default control set includes read-only credentials for all data access, scoped API keys that limit each tool to its intended domain, tool-specific argument constraints that reject out-of-range values, domain allowlists for retrieval sources with optional denylists for known-bad domains, and network and filesystem sandboxing where the environment supports it. Least-privilege enforcement should be verified through automated policy checks, not trusted solely on documentation. In the production demo, this principle is concrete rather than aspirational: market connectors are read-only, search access is configurable and source-bound, and credentials are scoped per provider rather than shared across the entire workflow.

For operations that can change state outside the workflow boundary (sending a notification, writing to a shared database, or triggering a downstream process), the **Warden** pattern interposes a policy proxy between agent output and runtime execution. The Warden validates each operation against an allowlist, rejects blocklisted patterns, validates arguments and value ranges, enforces timeout and resource limits, and logs every allow or deny decision to immutable storage.

`09_evaluation_and_governance` implements this pattern with three concrete policies: a no-write policy that blocks trade-execution tools (such as `execute_trade`, `send_order`, and `modify_position`), a domain-allowlist policy that rejects search queries containing terms indicating misuse (such as “hack”, “exploit”, “bypass”, or “credentials”), and a rate-limit policy placeholder. The notebook exercises the Warden across four test calls: two benign search queries (one on NVIDIA earnings, one on the Federal Reserve rate decision) pass through, while a search query for “hack admin credentials” and a trade-execution request are blocked with logged denials. The same notebook tests prompt-injection defense with five payloads covering role-override attempts (“ignore all previous instructions”), toolcall injection (embedded JSON trade commands), and data-exfiltration prompts, detecting all three attack classes while passing clean financial text through unmodified. This pattern does not eliminate risk, but it reduces risk concentration and creates enforceable policy boundaries that are auditable in retrospect.

In this chapter, the Warden appears as a teaching example because the production demo remains read-only. In practice, a sensible order is: first, prove the forecasting workflow under bounded permissions, then add stronger action controls only if the use case truly requires side effects.

### Human-in-the-loop controls and governance artifacts

For consequential actions, human approval is a governance requirement, not an optional nicety. An approval queue routes high-impact actions to designated reviewers. Threshold gates trigger human review when risk, cost, or exposure exceeds defined levels. Dual sign-off is required for exceptional overrides that fall outside normal operating parameters. Anomaly escalation flags unusual action sequences, such as a sudden spike in tool calls or an unexpected topic shift, for human inspection. All approval decisions should be logged with a rationale to support later review. For the read-only forecasting workflow in this chapter, the practical approval points are release management and publication: configuration changes, provider changes, and forecast publication policies should be reviewed even when the system does not execute trades. A compliant workflow generates reusable governance artifacts: the full run trace; an evidence index with provenance metadata; the policy evaluation log showing which controls were applied and their outcomes; the forecast artifact with confidence assessments and caveats; and a post-resolution score record linking the forecast to its outcome. In the production demo, these artifacts extend to persisted run records, forecast histories, and evidence ledgers that can be published or audited independently of the model runtime. Later deployment and MLOps chapters (*Chapters 25-26*) scale these control patterns into full operating frameworks.

### Security testing and metrics

Security controls should be tested with adversarial scenarios, not only nominal runs. A practical test suite includes prompt-injection payloads embedded in retrieved text, malformed tool arguments that probe validation boundaries, tool-name collisions and shadowing attempts, stale-source and post-cutoff retrieval attempts, and supervisor-override requests outside policy bounds. Each test should specify the expected allow/deny behavior and the required log fields, thereby converting security posture from narrative assurance to measurable system behavior.

Note that several security metrics overlap with the operational health metrics in *Section 24.9*. In production, these should be monitored through a unified trace infrastructure rather than separate systems, with security-specific alert thresholds layered on top.

Security metrics close the feedback loop. Key measurements include prompt-injection success rate (ideally zero but monitored continuously), unsupported-claim rate, policy-violation attempt rate, blocked privileged-call rate, false-positive rejection rate for valid calls, and mean time to incident diagnosis from trace data. These metrics integrate with the same run artifacts used for forecasting evaluation, so security monitoring does not require a separate infrastructure.

In financial agent systems, security is a property of the workflow. Read-only boundaries, explicit policy gates, immutable traces, and measured control performance are required for decision-grade use.

| Category | Control | Implementation | Verification |
| --- | --- | --- | --- |
| Input | Sanitization | Strip injection patterns from user input | Adversarial prompt test suite |
| Permissions | Least privilege | Read-only tools and scoped credentials | Permission audit per deployment |
| Retrieval | Domain policy | Allowlisted sources and date constraints | Post-cutoff retrieval attempt rate |
| Execution | Warden proxy | Validate tool calls against policy rules | Blocked-call log review |
| High-stakes | HITL | Require human approval above thresholds | Approval queue latency monitoring |
| Observability | Full tracing | Log every message, tool call, and decision | Trace completeness checks |
| Audit | Immutable Logs | Write-once storage for compliance | Periodic integrity verification |

*Table 24.5: Financial agent security checklist*

## 24.11 Summary

This chapter argued that financial agents should be designed as workflows rather than personalities. Reasoning patterns such as ReAct, Tree of Thoughts, and Reflexion matter only when they are embedded in explicit state, typed tool contracts, bounded autonomy, and replayable artifacts. The chapter’s capstones applied that principle progressively: first in a single-agent research workflow that can be inspected step by step, then in a multi-agent forecasting pipeline that accumulates evidence, aggregates views, reconciles disagreement, and publishes probabilities that can later be scored.

The deeper lesson is that forecasting support and market execution are different system classes. This chapter remained deliberately read-only because evidence acquisition, probability estimation, calibration, persistence, and post-outcome evaluation already impose demanding engineering and governance requirements. Point-in-time integrity, contamination-aware testing, observability, and security policy are what make any claim of forecast quality defensible, whether the workflow is presented in a notebook or wrapped as a production forecasting service.

The companion `aia-forecaster` repository carries the notebook workflow into an operational forecasting service with configuration profiles, prediction-market connectors, persistent storage, evaluation commands, and a publishing layer - the operational starting point for moving from exercises to a running system.

The next chapter moves from research workflows to deployment infrastructure. *Chapter 25* shows how disciplined artifacts from agent and model pipelines enter live trading systems with stronger operational controls, monitoring, and failure handling.
