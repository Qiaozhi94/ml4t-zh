# Chapter 23: Knowledge Graphs

Retrieval-augmented generation (Lewis et al., 2020) has made evidence-grounded financial question answering practical, but vector retrieval still operates on isolated chunks. When a question depends on ownership chains, supplier networks, contagion paths, or the timing of relationship changes, similarity search alone is not enough. Those questions require explicit relational structure (see *Figure 23.1*).

A knowledge graph provides that structure by representing entities as nodes, relationships as typed edges, and identifiers, timestamps, and provenance as properties attached to nodes and edges. This chapter concentrates on explicit financial knowledge graphs built for deterministic traversal and auditable answers, rather than on broad community-summary GraphRAG systems whose primary value lies in thematic synthesis. After completing this chapter, you will be able to:

- Distinguish financial questions that genuinely require graph structure from those better served by tabular databases or vector retrieval.
- Design a compact, typed, and auditable financial knowledge graph with stable entity identity, finite relationship vocabularies, and edge-level provenance.
- Build and validate LLM-assisted extraction pipelines that convert disclosures into replayable graph objects while controlling schema, duplication, and temporal consistency.
- Explain how Graph RAG differs from vector retrieval and implement safe relational query workflows using constrained text-to-query generation and deterministic database execution.
- Transform graph structure into leakage-aware machine learning features, including topology, crowding, concentration, and cross-graph interaction features.
- Evaluate explicit knowledge graphs, statistical financial networks, and learned graph representations pragmatically for forecasting, portfolio construction, and risk analysis.
- Apply a three-timestamp framework and disclosure-time cutoff rules to prevent temporal leakage in graph queries, feature generation, and backtests.
- Make sound engineering choices about graph databases, ontology scope, query safety, and schema evolution for production-oriented financial systems.

*Section 23.1* establishes when graphs justify their overhead. *Sections 23.2-23.3* cover graph construction and relational retrieval. *Sections 23.4-23.5* turn graph structure into features and network-based portfolio inputs. *Section 23.6* addresses temporal integrity, and *Section 23.7* closes with the engineering decisions needed to keep the system auditable and stable.

## 23.1 When relational structure justifies a graph

Not every financial question benefits from a graph. Schema design, extraction, and graph maintenance carry overhead that is justified only when the analytical task is genuinely relational: when the answer depends on paths between entities, not on properties of entities in isolation.

### When graphs add value

Three conditions reliably justify the use of graph infrastructure: relationships that span multiple entities, involve complex links, and evolve over time.

#### Multi-hop dependency queries

Questions that require joining facts across multiple entities are the clearest use case. “Which companies share critical suppliers with Nvidia, and which of those suppliers sit in geopolitically concentrated regions?” requires traversing supplier edges, intersecting with competitor supplier edges, and filtering by geographic properties. A graph database executes this as a pattern match; a relational database requires recursive self-joins that grow unwieldy beyond two hops; a vector retrieval system must infer the path from disconnected text fragments, exactly where high-impact errors enter.

One financial motivation is **contagion analysis**. Haldane and May (2011) showed that the stability of financial systems depends on the architecture of interconnections, the “robust-yet-fragile” property of scale-free networks. A single node failure propagates through hub-and-spoke structures in ways that per-entity risk metrics cannot capture. Supply chain graphs, interbank lending networks, and cross-holding structures all exhibit this topology.

![Figure 23.1](assets/figure_23_1.png)

*Figure 23.1: Vector RAG retrieves similar chunks; Graph RAG traverses explicit relations*

#### Structural crowding and co-ownership

Institutional ownership data from SEC 13F filings form a bipartite network: institutions hold securities, and securities are held by institutions. Projecting this network onto the security dimension reveals co-ownership patterns that predict return correlations beyond what fundamental factors explain.

Antón and Polk (2014) showed that stocks connected through common institutional ownership exhibit excess return comovement: returns correlate not because the firms are fundamentally similar, but because the same investors trade them simultaneously. Greenwood and Thesmar (2011) extended this to price fragility: securities with concentrated institutional ownership face fire-sale risk because forced liquidation by a single holder triggers correlated selling by others holding similar portfolios.

Computing co-ownership Jaccard similarity, crowding scores, and ownership concentration requires traversing the institution-security bipartite graph. The features that encode fragility and crowding are inherently relational: they measure a stock’s position in the ownership network, not its standalone characteristics. Konstantinov and Fabozzi (2025) provide recent evidence that causal spillover effects across asset networks convey risk information beyond what standard correlation models capture.

#### Temporal relationship evolution

Relationships change. A supplier becomes a competitor after vertical integration. An institution builds a position over multiple quarters before it becomes visible. Executive migration between firms creates hidden information channels. These dynamics appear in edge patterns before they surface in price data.

Li and Passino (2024) constructed dynamic financial knowledge graphs from news corpora and showed that temporal edge patterns (relationship formation, dissolution, and type transitions) carry thematic signals useful for trend detection. Evolving graph structure encodes regime shifts, strategic repositioning, and emerging competitive dynamics that static cross-sectional features miss. For these signals to be tradable, edge timestamps must distinguish between when an event occurred, when it was publicly disclosed, and when the extraction pipeline observed it, a three-timestamp model detailed in *Section 23.6*.

What distinguishes temporal graph signals from conventional time-series features is that they encode **topological change**: not shifts in entity attributes, but changes in the structure of relationships. A new supplier link, a dissolved partnership, or a shift in institutional ownership alters the network’s information-propagation properties, and these structural events are difficult to capture in tabular features because they involve relationships between entities, not the entities themselves.

### When graphs do not help

Three conditions consistently fail to justify the need for graph infrastructure.

#### Single-entity attribute lookup

If the question is “What was Apple’s revenue last quarter?” the answer lives in a single record. A graph adds storage and query overhead without analytical benefit. A common mistake in early KG projects is encoding entity attributes as graph relationships (for example, `Company -HAS_REVENUE-> $500B`), which complicates queries without adding relational insight.

#### Narrative synthesis over broad corpora

When the task is “Summarize the main themes in 500 analyst reports,” the bottleneck is text comprehension, not relational traversal. Community-summary approaches like GraphRAG (edge et al., 2024) address this by building statistical summaries of entity communities, but their core value lies in summarization rather than in graph structure. Vector retrieval with clustering and re-ranking often achieves comparable quality with lower infrastructure overhead.

#### Sparse graphs with few relationships

Graph analytics assume reasonable density. If the knowledge graph has 10,000 entities but only 500 edges, centrality metrics become unstable, community detection produces degenerate clusters, and network features add noise rather than signal. As a rough heuristic, if the average node degree falls below 2–3, most topology metrics are unreliable.

Sparsity is often a symptom of extraction failure rather than a property of the domain: if the supply chain graph of S&P 100 companies has fewer than 200 edges, the extraction pipeline needs improvement before the graph infrastructure becomes useful.

### Practical decision framework

A useful test: restate the question as a path pattern. If it naturally decomposes into “find entities connected through relationship chains,” a graph representation is likely justified. If the question concerns the properties of individual entities or requires summarizing large collections of text, simpler tools are more appropriate.

The rest of this chapter assumes the graph path is justified and addresses the practical challenges: constructing reliable graph objects from financial disclosures (*Section 23.2*), querying them for relational reasoning (*Section* *23.3*), extracting features for machine learning (*Sections* *23.4–23.5*), maintaining temporal integrity (*Section* *23.6*), and engineering the supporting infrastructure (*Section 23.7*).

**Implementation**: See `09_knowledge_graph_features.py` for notebook-scale relational features derived from compact supply-chain and holdings graphs, and `10_network_` `portfolio_construction.py` for market-network structure derived from real return data.

## 23.2 Constructing financial knowledge graphs

The central change in knowledge-graph practice is not that LLMs can extract triples; older systems could do that. The change is that modern models can map long, noisy disclosures into structured outputs with enough flexibility to replace large portions of handcrafted parsing logic. That shift changes project economics and lets teams iterate on real corpora quickly.

Speed alone is not sufficient in finance. A graph that cannot be replayed, audited, and corrected is a liability. The practical requirement is twofold: use LLMs for extraction throughput and pair them with data contracts that keep outputs stable across model versions and data updates.

### From legacy pipelines to directed extraction

Before LLMs, most teams used one of two approaches. Rule-based systems used regular expressions and keyword grammars: precise on narrow templates, brittle outside them. Supervised relation-extraction models performed better across linguistic variation, but required expensive labeled data and recurring retraining as document styles changed. Elhammadi et al. (2020) achieved 78% precision on the top-100 extractions using a pre-LLM pipeline that combined NER, dependency parsing, and pattern matching on financial news, a strong result for its time but one that required substantial per-corpus engineering.

Both approaches shared a structural weakness: extraction and schema design were loosely coupled. Engineers often normalized text first and then tried to coerce it into graph objects. That ordering produced recurring failure modes: duplicate entities, drifting relation names, and sparse provenance fields.

LLM-based extraction reverses the order. The schema is defined first. Prompting and structured output constraints force the model to emit graph-ready objects directly, moving engineering effort from parsing cleanup to validation and monitoring.

### Governance before prompt

A robust extraction run starts with three contracts covering identity, schema, and provenance.

#### Identity contract

Mentions are not identities. “Apple,” “Apple Inc.,” and “AAPL” may refer to the same company and, therefore, should never persist as independent canonical nodes. The extraction layer should preserve mentions while resolving canonical identifiers (CIK, ticker, LEI, CUSIP/ISIN, where relevant) in a separate canonicalization step.

Identity resolution in financial data is harder than simple deduplication. Tickers change (Facebook → Meta), mergers create new entities from old ones (Dow/DuPont → Corteva + DowDuPont), and the same name can refer to different entities across jurisdictions. A practical canonicalization strategy uses stable identifiers (such as CIK or LEI) as primary keys and stores tickers as time-varying attributes with validity intervals. Corporate actions (mergers, spinoffs, name changes) should be treated as firstclass events that update identity attributes rather than exceptions handled ad hoc.

#### Schema contract

Relationship vocabulary should be finite and typed. In practice, this means an allowlist of relation types with explicit source-target constraints. If a generated edge violates those constraints, it is rejected or routed for review.

#### Provenance contract

Every accepted node and edge must carry enough metadata to answer two questions: “where did this claim come from?” and “when did it become publicly available?” At minimum, store source document ID, filing/public date, source location (section/page/span), extraction timestamp, and extractor version. These contracts make high-coverage extraction maintainable and keep downstream analytics interpretable.

### A five-stage extraction workflow

A production-oriented workflow usually includes five stages:

1. **Targeted document slicing.** Focus extraction on relationship-dense filing segments. For 10-K pipelines, Items 1, 1A, and 7 are common starting points. This excludes boilerplate sections that rarely contain relation-level evidence.
2. **Schema-constrained generation.** Use JSON Schema or typed function output modes so that emitted objects can be validated without fragile text parsing.
3. **Canonicalization and deduplication.** Resolve aliases, collapse duplicates, and preserve mention-level evidence.
4. **Rule validation.** Enforce relation allowlists, type constraints, date sanity checks, and provenance completeness.
5. **Human review queue.** Route low-confidence or rule-violating objects for analyst adjudication.

Each stage should emit diagnostics and failure counts to enable iterative quality improvement. The pipeline should be idempotent: reprocessing the same filing with the same extractor version should produce identical graph objects.

Reflection-driven correction, as shown by FinReflectKG (Arun et al., 2025a), can improve compliance by having the model critique and revise its own extractions before validation. The reflection loop works in three steps: extract candidate triples, evaluate them against schema rules, and propose fixes for violations. This iterate-until-convergence pattern is particularly effective for entity normalization (for example, splitting “John Ternus, CEO of Apple” into separate Person and Role entities) and for enforcing relationship type constraints.

### Error taxonomy and monitoring

Teams often monitor extraction precision and stop there. For graph workloads, four additional metrics matter:

- **Schema-valid rate**: Fraction of generated objects satisfying type constraints
- **Provenance coverage**: Fraction of accepted edges with complete evidence metadata
- **Duplicate-node rate**: Unresolved aliases mapping to separate canonical nodes
- **Temporal-consistency rate**: Edges with coherent event/public/extraction timestamps

These metrics reveal different failure classes. A model can produce high lexical quality but still degrade graph usability if identity or provenance quality collapses. Tracking these across extraction runs also reveals model regression: a prompt change that improves entity extraction but degrades temporal consistency would be invisible to standard NLP evaluation, but visible in graph quality metrics. A practical monitoring target: schema-valid rate above 90% and provenance coverage above 95% for edges entering production graphs. Beyond automated metrics, routinely sample extracted edges and have analysts score correctness and provenance adequacy. Maintain a frozen “gold subset” of filings and verify that extraction outputs remain stable across prompt and model updates. This regression testing catches quality degradation that aggregate metrics can mask.

### Supply chain schema example

Consider constructing a supply chain graph from 10-K filings. The objective is to extract three relationship families (supplier dependencies, customer concentration links, and competitive relationships) for the S&P 100.

The schema should be small, typed, and versioned:

| Object | Core Fields | Purpose |
| --- | --- | --- |
| Company node | canonical ID, name, ticker, sector | Identity stability across filings |
| Filing node | accession, form, filing date | Source traceability |
| HAS SUPPLIER edge _ | evidence span, public date, extractor version | Dependency analysis and auditability |
| HAS CUSTOMER edge _ | evidence span, public date | Concentration analysis |
| COMPETESWITH edge _ | evidence span, public date | Market structure queries |

*Table 23.1: Sample schema*

The flow is shown in the following diagram:

![Figure 23.2](assets/figure_23_2.png)

*Figure 23.2: Minimal supply-chain schema linking companies, filings, and typed relationship edges*

The design principle is to store evidence with the edge. A relationship without a source context is difficult to verify, correct, or trust in downstream models. This schema is deliberately small: five object types cover the core analytical use cases. Larger schemas can be built incrementally as extraction pipelines mature, but starting with a compact schema reduces validation complexity and makes error diagnosis tractable.

An important caveat: **supply chain relationships disclosed in 10-K filings are partial, selective, and sometimes intentionally vague**. Companies disclose relationships they are required to report or choose to highlight; they do not provide a complete supplier inventory. This makes 10-K extraction a high-precision, low-recall source. A separate noise source enters at the LLM extraction step: generic placeholder entities such as “leaf merchants,” “vendors,” or “third parties” surface as candidate suppliers and require a curated stop list during canonicalization to keep the graph clean. When a use case requires broader coverage, layer in structured or vendor datasets and reconcile them through the identity contract. The KG schema accommodates this: edges from different sources carry different provenance metadata, and downstream consumers can filter by source confidence.

The pipeline acquires 10-K filings from EDGAR, stages them as per-ticker parquet files, and rebuilds a consolidated reference artifact from the local cache before extraction. Text is sliced from sections that carry relationship signals, especially Item 1 (Business), Item 1A (Risk Factors), and Item 7 (MD&A). Targeted slicing reduces extraction noise and lowers per-filing cost. The extraction stage applies the workflow described above, with entity normalization collapsing common aliases, dropping generic placeholder entities, and linking each accepted edge to source filing metadata.

When the extracted graph is visualized, structural concentration becomes easier to inspect than in raw triples alone. In the current notebook-scale run, shared suppliers such as TSMC, Foxconn, and Samsung emerge as bridge nodes connecting multiple downstream firms. That pattern is exactly the kind of relational concentration the graph representation is intended to expose.

![Figure 23.3](assets/figure_23_3.png)

*Figure 23.3: Notebook-scale extraction highlights shared suppliers as concentration nodes linking multiple downstream firms*

Idempotent writes keep reruns stable and enable versioned refreshes. In practice, that means using merge-style loading logic keyed on canonical node identity and relationship type, then updating edge-level provenance fields such as source document, source section, source span, public date, extraction time, and extractor version. The important property is behavioral rather than syntactic: reprocessing the same filing should update the same relationship record, not create a parallel copy that inflates downstream path counts.

Graph RAG, graph-derived ML features, and temporal graph analytics all depend on this foundation. If extraction does not preserve identity and evidence, downstream systems inherit hidden error. If it does, the same filing corpus becomes a reusable relational substrate for many analyses. LLMs are not replacing data discipline: they make disciplined graph construction scalable. With graph objects in place, the next question is how to query them. *Section 23.3* shows how Graph RAG delegates relational logic to the database while keeping language generation in the model.

**Implementation**: See `01_sp100_sec_download.py` for SEC EDGAR filing acquisition and `02_supply_chain_kg_construction.py` for the extraction and loading pipeline. The notebook works with the staged EDGAR cache, a live Neo4j instance, and a local LLM extraction path.

The staged run processes 601 10-K filings covering 101 S&P 100 companies and extracts 1,057 unique typed relationships after deduplication (276 supplier, 472 competitor, and 309 customer edges) across 127 distinct subject companies, and surfaces TSMC, Foxconn, and Samsung as the three most connected suppliers, serving 61, 33, and 28 downstream companies, respectively. The cached parquet ships with an `extracted_triples.meta.` `json` file that pins the content hash, schema, row count, and extractor identity.

## 23.3 Deterministic relational reasoning with graph RAG

**Graph RAG** uses an LLM for language understanding and a graph database for relational retrieval. Multi-hop joins are executed by the database engine, so the analyst can iterate on questions without re-engineering prompt chains each time.

Graph RAG is strongest for questions whose answer depends on explicit relationship paths:

- “Which competitors share suppliers with company X?”
- “Which institutions hold both a company and its key suppliers?”
- “Which exposures were visible before date T?”

Vector retrieval remains better for narrative and open-ended questions. The question “What are the main risks facing semiconductor companies?” is better served by vector retrieval across risk-factor disclosures than by graph traversal. In practice, most production systems are hybrid: graph retrieval for relational structure, document retrieval for narrative evidence. The query routing stage determines which path is appropriate for each incoming question.

### End-to-end architecture

A robust Graph RAG path includes five stages:

1. **Query routing.** Classify incoming questions as graph, document, or hybrid. This reduces unnecessary text-to-Cypher generation and improves answer quality for non-relational questions.
2. **Text-to-Cypher generation.** Generate Cypher from a compact schema representation and explicit query policy. The schema should include only the labels, relationships, and properties that the query interface is allowed to use. Not all questions are well-suited to free-form query generation; many production teams use parameterized template libraries with an LLM router that selects and fills the appropriate template, rather than generating arbitrary Cypher. This reduces the attack surface and simplifies validation.
3. **Query safety validation.** Validate generated queries before execution. This is a structural requirement for financial deployment, not optional hardening. The validation layer should enforce: read-only database role, allowlisted labels and relationships, parameterized values (never string interpolation), timeout and row-limit guards, and write/admin keyword blocking.
4. **Deterministic retrieval.** Execute validated Cypher and return structured rows with provenance metadata. The database handles relational joins; the LLM does not need to infer connection paths from text fragments.
5. **Grounded synthesis.** Compose the final answer from retrieved rows and cite source evidence. The synthesis prompt should include retrieved rows verbatim and instruct the model to cite specific rows when making claims. Because edges carry evidence pointers (such as document ID, section, span offsets), the system can fetch the original text snippets that justified each relationship and include them alongside the graph result. This two-layer citation (graph row plus underlying disclosure text) prevents a common failure mode where “the KG says X” without showing the original evidence.

This division of labor keeps retrieval logic in the database and language generation in the model, making errors easier to localize during debugging.

### Example query flow

Consider the question, “Which companies compete with Apple and share its suppliers?” In a Graph RAG system, the model does not answer that by narrating over loosely related passages. It routes the request to a constrained text-to-Cypher path, generates a read-only query over the `COMPETES_WITH` and `HAS_SUPPLIER` relationships, and applies a cutoff-date filter so that only relationships public at the time of the query are eligible. The database returns structured rows such as competitor names and shared suppliers; the model then turns those rows into prose while preserving the underlying evidence links. The cutoff-date filter is not an implementation detail. In finance, it is the mechanism that keeps historical analysis from leaking post-cutoff information.

In the current teaching workflow, the same pattern is applied to the real institutional holdings graph. Questions such as “Which institutions hold both Apple and Microsoft?” or “What are Berkshire Hathaway’s largest positions?” are routed to a compact, validated set of Cypher templates rather than to unconstrained query generation. This is a narrower interface than a fully general Graph RAG agent, but it is closer to how audited production systems are usually deployed.

### Comparing explicit KG graph RAG to community-summary GraphRAG

Two graph retrieval families are common in practice, as surveyed by Peng et al. (2024). **Explicit domain KG Graph RAG** uses typed entities and deterministic traversal, well-suited for regulated financial workflows where provenance and replay are required. **Community-summary GraphRAG** (Edge et al., 2024) constructs graph abstractions from corpus themes to enable broad summarization. It excels at thematic synthesis but is less direct for schema-governed relational queries where audit trails matter. The two approaches are complementary. This chapter emphasizes explicit domain KGs because they map directly to auditable financial reasoning and downstream feature workflows.

### Evaluation and failure modes

A useful Graph RAG evaluation harness checks three properties: query validity (syntax and schema compliance of generated Cypher), provenance coverage (share of returned claims linked to source evidence), and temporal correctness (accuracy under historical cutoff constraints).

Two recent benchmarks provide standardized evaluation with real financial data. FinDER (Choi et al., 2025) offers 5,703 expert-annotated query-evidence-answer triplets from S&P 500 10-K filings. Its queries reflect real analyst search behavior: abbreviated, ambiguous, and domain-specific. FinDER evaluates retrieval broadly (both sparse and dense methods), not Graph RAG specifically, but its finding that generation quality is strongly limited by retrieval quality reinforces the case for structured graph retrieval on relational questions.

FinReflectKG-MultiHop (Arun et al., 2025b) tests the specific claim that KG-guided evidence outperforms text-window retrieval for multi-hop financial reasoning. Built on the FinReflectKG knowledge graph derived from S&P 100 filings (2022–2024), it generates analyst-style questions that require 2–3hop reasoning across companies, years, and filing sections. KG-guided retrieval improved correctness by approximately 24% while reducing token consumption by roughly 85% compared to page-window retrieval. The efficiency gain shows that structured retrieval is not just more accurate but also more economical: the model spends tokens reasoning over pre-selected facts rather than navigating noisy context.

**Common failure modes** include invented labels or properties in generated Cypher, overly broad traversals that return noisy rows, missing cutoff filters in historical queries, and synthesis drift, where generated text exceeds the evidence returned. These are engineering problems: strict validation gates, conservative query templates, and regression test suites address most of them. Most production deployments route relational questions to the graph path and narrative questions to vector retrieval. When graph and vector results conflict, temporal metadata usually resolves the discrepancy: the more recent evidence takes precedence, provided timestamps are reliable.

Graph RAG is a different retrieval contract from vector RAG: the database executes relational logic, and the model explains results. When that contract is enforced, multi-hop financial questions move from heuristic synthesis toward repeatable analysis. Graph RAG answers questions about the existing structure; the next section turns that same structure into tabular features that can feed gradient-boosted models and factor regressions. The same pattern is visible in the notebook’s head-to-head benchmark on the live 13F holdings graph. Across seven analyst-style questions spanning holder lookup, co-ownership joins, and portfolio composition, structured graph retrieval achieves perfect support recall (1.00) against an embedding baseline that averages 0.13. Vector retrieval fails entirely on holdings-lookup queries (recall 0.00) and returns only a fraction of the relevant co-ownership rows (0.17) and direct-holder rows (0.30). Graph retrieval also uses roughly a third as many tokens per query (82 versus 267) because the traversal returns only the rows that match the schema rather than a broad set of candidate chunks. The magnitude of that gap depends on corpus and question type, but the direction is consistent with the FinReflectKG-MultiHop finding: on relational questions, structured retrieval is both more accurate and cheaper per answer.

**Implementation**: See `03_graph_rag_qa.py` for the text-to-Cypher pipeline over the 13F holdings graph, and `04_rag_comparison_benchmark.py` for the graph-versus-embedding comparison on 10,624 holdings rows covering 10 institutions and 250 stocks.

## 23.4 From graphs to machine learning features

The knowledge graphs constructed in previous sections (supply chain relationships from 10-K filings, institutional holdings from 13F) encode structural information that traditional price-based features do not capture. Graph analytics transform that structure into tabular features for gradient boosting (*Chapter 12*) and factor models (*Chapter 14*).

These algorithms extract scalar features from network structure. Prior studies report predictive value from text-derived network features (Kertkeidkachorn et al., 2023), though effect sizes depend on dataset design and evaluation protocol.

### Network topology features

Core features derive from centrality measures that quantify each node’s structural importance:

| Feature | Calculation | Financial Interpretation |
| --- | --- | --- |
| PageRank | Iterative infulence propagation | Supply chain importance—companies depended upon by many others |
| Betweenness | Fraction of shortest paths through node | Bottleneck position where disruption propagates widely |
| Clustering Coefficient | Local connectivity density | Redundancy—high clustering suggests alternative paths exist |
| Degree | Direct connection count | Immediate network exposure—more connections, more transmission channels |

*Table 23.2: Network topology features*

Different centrality measures carry different information. **PageRank** identifies systemically important firms weighted by the importance of their dependents. **Betweenness** identifies bottleneck firms whose failure would disconnect parts of the network. Using multiple centrality measures as separate features lets the downstream model learn which structural role matters for a given prediction task.

### Supply chain risk features

Beyond topology, the supply chain KG enables the development of domain-specific risk indicators. In the notebook workflow, these are derived from the extracted graph itself rather than from a hand-labeled supplier master. Examples include supplier count, shared-supplier count, single-source count, supplier-overlap ratio, and supplier-dependency score. Together, these features distinguish firms with diversified supply relationships from firms whose risk is concentrated in a small set of disclosed counterparties.

A company that depends on a few shared upstream suppliers is structurally fragile, even if no single disclosure directly states that risk. The graph makes that concentration measurable, and it does so in a form that can be joined to other feature families.

### Institutional crowding features

The 13F holdings knowledge graph reveals ownership patterns that carry predictive information about price dynamics. **Four feature families** are derived from the institution-security bipartite network:

1. **Crowding score.** Institutional holder count relative to the cross-sectional median measures how crowded a position is. Stocks with high crowding face correlated selling pressure during market stress: forced liquidation by one holder triggers cascading sales by others (Greenwood and Thesmar, 2011).
2. **Smart money concentration.** Value-weighted share held by top-performing institutions signals informed conviction, reflecting analytical judgment rather than index tracking.
3. **Ownership Herfindahl Index** (see *Chapter 17*)**.** Concentration of holdings across institutions predicts fire-sale risk. A stock held by three large funds is more fragile than one held by thirty small funds, even at the same total institutional ownership.
4. **Co-ownership Jaccard.** The Jaccard similarity between two stocks’ institutional-holder sets predicts return correlation beyond what fundamental factors alone explain. Antón and Polk (2014) showed that this co-ownership effect is economically significant: stocks linked by common institutional ownership exhibit excess comovement driven by correlated trading activity rather than shared fundamentals.

Konstantinov and Fabozzi (2025) provide recent evidence that network-based causal spillover metrics carry risk information beyond simple correlation models. Treat exact magnitudes as study-specific unless reproduced on your own corpus and split definitions.

Three temporal considerations affect how these features are computed for backtesting. First, 13F filings are quarterly with a 45-day reporting delay: a holding is tradable information only after the public filing date, not the quarter-end date. Second, large institutions sometimes request confidential treatment for positions they are building; masked positions appear in later amendments, creating gaps in real-time analysis. Third, derived relationships, such as portfolio similarity, should be recomputed quarterly as holdings change; they represent computational artifacts, not disclosed relationships.

### Cross-graph integration and temporal dynamics

The distinctive value of knowledge graph features comes from combining information across graph layers. A stock with both concentrated supplier dependence (supply chain KG) and high crowding (holdings KG) faces compound risk: operational disruption would coincide with correlated institutional selling. The `supply_chain_crowding` interaction feature captures this by multiplying a supplier-overlap measure with institutional crowding.

Unlike traditional factor models, where factors are assumed to be independent, KG features can directly encode structural dependencies. Other cross-graph features include concentrated dependency risk (supplier dependency score multiplied by ownership concentration) and systemic exposure (betweenness multiplied by holder count).

Graph structure also encodes *dynamic* information propagation. Cheng et al. (2020) showed that events affecting upstream entities in a supply chain KG propagate to downstream firms at different speeds depending on the industrial chain: a raw material shock reaches component manufacturers before final assemblers. Their KG-based event embedding framework exploits these lead-lag relationships, showing that graph topology determines the timing of information transmission, not just whether it occurs. A supply chain graph with temporal event data can generate lead-lag features: the weighted average of upstream event severity, lagged by estimated propagation delay, becomes a predictive feature for downstream firms.

The supply-chain graph is a single snapshot, whereas the staged 13F holdings panel spans four quarterly vintages, covering Q1 through Q4 2025. The companion notebook uses this history to compute three vintage-aware ownership features: the mean Jaccard distance between the institutional-holder sets of consecutive vintages (ownership churn), the coefficient of variation of total reported position value across vintages, and the count of new institutional holders who entered the position between the two most recent vintages. These populate for the 78 of 127 entities with a 13F CUSIP match; the supply-chain churn columns remain at the zero baseline until multiple supply-graph vintages are materialized.

Features tracking network changes over time capture evolving structure that static snapshots miss:

- Relationship churn (edge turnover rate)
- Centrality momentum (change in PageRank over time)
- Supplier change (net change in supplier count)

A company whose supplier count declines while competitor connections grow signals that it is undergoing strategic repositioning, information not reflected in quarterly financials until later. The companion notebook produces features in wide format (one row per company, ready for gradient boosting or factor regression) and long format (stacked feature names and values, convenient for information coefficient calculation). The features derived so far rely on explicitly stated relationships; *Section 23.5* adds a complementary perspective from statistical financial networks built on return correlations and assesses where graph neural networks fit in practice. **Implementation**: See `09_knowledge_graph_features.py` for the complete feature engineering pipeline and `05_institutional_holdings_kg.py` for the ownership graph construction. Both notebooks operate on staged real data. `09_knowledge_graph_features.py` joins the live supply-chain graph in Neo4j (127 source companies, 1,057 typed relationships) with the staged 13F holdings artifacts (201,145 raw rows), matches 80 of the 127 supply companies to the holdings universe, and produces a 31-column feature matrix spanning five families: topology (5 columns), supply (7), holdings (9), temporal (6), and cross-graph (4), joinable with the price-based features from earlier chapters.

## 23.5 Correlations and portfolios in financial networks

Knowledge graphs encode explicitly stated relationships: supplier links from 10-K filings, holdings from 13F. But financial networks also arise from statistical relationships. Correlation-based networks, minimum spanning trees, and hierarchical clustering of return series have been studied for over 25 years and provide a complementary perspective on market structure. This section connects the established financial networks literature to the knowledge graph features developed earlier, then assesses where graph neural networks fit in practice.

### The established tradition

Mantegna (1999) introduced **minimum spanning trees** (MSTs) constructed from stock return correlations as a tool for revealing hierarchical structure in financial markets. The approach converts a correlation matrix into a distance metric, constructs the MST, and interprets the resulting tree as a map of market organization: clusters of related stocks, hub securities that connect sectors, and peripheral assets with distinct dynamics.

This line of research matured over two decades. Marti et al. (2021) reviewed the full trajectory, from correlation-based networks and hierarchical clustering to random matrix theory filtering and modern applications in portfolio construction and risk management. Their survey identifies what works (MSTs reliably recover sector structure and detect regime changes) and what remains unstable: community detection algorithms are sensitive to estimation window and filtering method, and hierarchical structures can shift substantially across adjacent time periods.

Correlation-based networks are useful for visualization, cluster identification, and diversification analysis, but their structural features require careful treatment in predictive models. Network topology metrics computed from correlation graphs are noisier than the same metrics from explicitly stated relationships because the underlying edges are estimated rather than observed. The choice of filtering method (MSTs, planar maximally filtered graphs, or random matrix theory) determines which relationships survive and, therefore, which community structures and centrality rankings emerge. Practitioners should match the filter to their analytical goal rather than treating this choice as a preprocessing detail.

### From minimum spanning trees to portfolio construction

The companion notebook builds MSTs from `us_equities` return data and shows how network structure informs portfolio construction, following Konstantinov, Aldridge, and Kazemi (2023), who showed that network-based allocation can improve diversification relative to naïve equal-weight and minimum-variance portfolios.

Stocks that occupy central positions in the correlation network are more connected to the broader market and contribute less marginal diversification. Peripheral stocks, those at the edges of the MST, tend to offer more independent return streams. Network-aware allocation exploits this by giving peripheral assets greater weight.

Whether this translates to superior risk-adjusted returns after transaction costs depends on rebalancing frequency, estimation window, and the stability of network structure across regimes. MST structure can shift substantially between adjacent estimation windows, creating turnover that erodes theoretical diversification gains.

The notebook `10_network_portfolio_construction.py` evaluates this empirically on a 100-stock universe from 2015–2018 (504-day correlation lookback, average pairwise correlation 0.24). Equalweight, inverse-volatility, and network-diversified portfolios produce Sharpe ratios of 1.58, 1.51, and 1.54, respectively, with maximum drawdowns of −10.3%, −10.6%, and −10.1%, close enough that the summary metrics do not separate the three at this universe size. The structural argument for the network-diversified construction emerges from the contagion simulation: a shock propagated from the highest-centrality node (BRK.B) affects the portfolio roughly 3.8 times more severely (−87.3%) than the same shock originating at a peripheral node (AAL, −23.0%), illustrating that topology governs contagion reach more than shock magnitude.

### GNN maturity assessment

**Graph Neural Networks** (**GNNs**) extend deep learning to graph-structured data through message-passing: a node’s representation is iteratively updated by aggregating information from its neighbors. Common architectures include Graph Convolutional Networks (Kipf and Welling, 2017), Graph Attention Networks (Velickovic et al., 2018), and GraphSAGE (Hamilton, Ying, and Leskovec, 2017).

GNN applications in finance exist on a maturity spectrum. The clearest production-adjacent success case remains **transaction-graph fraud detection**. Weber et al. (2019) showed that graph neural methods can separate suspicious Bitcoin flows from ordinary activity on the Elliptic dataset, which is why fraud detection falls into a different maturity bucket than alpha generation.

| Domain | Maturity | Evidence |
| --- | --- | --- |
| Fraud Detection | Production-ready | Elliptic Bitcoin dataset (Weber et al., 2019); deployed at scale |
| Systemic Risk | Emerging | Academic research, regulatory pilot programs |
| Alpha Generation | Experimental | Limited reproducible evidence after costs |

*Table 23.3: Graph Neural Network applications*

Fraud detection succeeds because transaction networks have high signal-to-noise ratios (distinctive fraudulent topologies), clear labels, and real-time requirements that GNN inference can meet.

Alpha generation from price correlations remains challenging due to market efficiency, regime non-stationarity, and the reproducibility gap in academic claims.

### The pragmatic recommendation

For the financial use cases in this book, hand-crafted graph features are the first-choice baseline. They are easier to audit, easier to debug, and more stable under governance constraints. Consider 13F holdings data: crowding scores, ownership HHI, and co-ownership Jaccard, computed directly from the graph structure, are transparent, explainable to regulators, and robust across regimes. GNNs add complexity without proportional benefit for these structured relationships.

The companion GNN notebook illustrates this on the correlation network of a 200-stock universe, with the graph estimated ex-ante on returns strictly before the 21-day target window: 480 edges at a correlation threshold of 0.5, a graph density of 2.41%, and pairwise correlations ranging from −0.32 to 0.99, with a mean of 0.21. Four GAT embedding dimensions are concatenated with eight tabular factors and fed to a ridge regression. On this sample, the hybrid model’s cross-sectional information coefficient (0.215) lands roughly 7.7% below the tabular-only baseline (0.233). The comparison is in-sample and scale-limited, but the direction aligns with the broader literature’s concern about GNN alpha generation: added architectural complexity does not translate into reliable improvements in structured equity forecasting problems at this scale.

For teams that want learned graph representations without the full GNN complexity, simpler graph embedding methods (such as node2vec or metapath2vec) provide an intermediate step: they produce vector representations from random walks on the graph, are faster to train than message-passing architectures, and can serve as features alongside hand-crafted metrics.

The **recommended path** is:

1. Start with graph analytics and hand-crafted features (*Section 23.4*).
2. Use network-based portfolio methods with explicit structure (such as MSTs or community detection).
3. Add GNN or graph embedding features only if they improve out-of-sample metrics after accounting for transaction costs and temporal leakage.
4. Keep learned graph features supplemental, not the default architecture.

**Dynamic knowledge graphs**, such as FinDKG (Li and Passino, 2024), represent an emerging direction in which temporal edge patterns carry thematic signals. The framework is promising but should be evaluated against simpler temporal features before adopting the full GNN apparatus, as discussed in the next section.

**Implementation**: See `10_network_portfolio_construction.py` for MST construction, centrality-driven portfolio weighting, and the contagion simulation; see `06_gnn_feature_` `engineering.py` for the GAT-on-correlation-network versus tabular-ridge comparison.

## 23.6 Temporal integrity and leakage-safe evaluation

Static knowledge graphs answer the question “What relationships exist?” Temporal graphs answer “when did those relationships become observable?” That second question is central in finance because the timing of availability determines whether a signal is tradable.

### The three-timestamp model

Every relationship in a financial knowledge graph should carry three timestamps:

- **Event time**: when the underlying event occurred (for example, an executive departed)
- **Disclosure time**: when the information became publicly available (for example, the 8-K was filed)
- **Extraction time**: when the pipeline processed the source document

The critical distinction is between event time and disclosure time. An acquisition may be agreed in March but filed in April; a supplier relationship may exist for years before appearing in a 10-K. Feature generation and historical queries must use disclosure time as the visibility gate: information that was true economically but not yet disclosed cannot enter the model.

### 8-K filings as a temporal case study

SEC Form 8-K illustrates the three-timestamp model concretely. Companies must disclose material events within 4 business days, including acquisitions, leadership changes, earnings announcements, and strategic shifts. The gap between event and disclosure varies by event type:

| Item | Event Type | Typical Disclosure Lag |
| --- | --- | --- |
| 1.01 | Material Agreement | 1–4 business days |
| 2.01 | Acquisition/Disposition | 1–4 business days |
| 5.02 | Executive Changes | 1–4 business days |
| 7.01 | Regulation FD | Same day (by definition) |
| 8.01 | Other Events | Variable |

*Table 23.4: Disclosure lag by event type*

The disclosure lag matters for backtesting: an executive departure on March 1 filed on March 5 should not be available to a model evaluated on March 3. The current notebook implementation persists event date, public filing date, and extraction date directly on the event graph, so leakage-safe filtering becomes a graph query condition rather than a manual bookkeeping step.

**Executive turnover clustering is a temporal signal** that surfaces from 8-K event graphs. When multiple companies in a sector replace executives simultaneously, it may signal industry-wide disruption or competitive pressure. This pattern is visible only when event timestamps are aggregated across the graph; individual filings reveal individual changes, but the cluster is a structural observation. The extraction methodology should follow reflection-driven approaches (Arun et al., 2025a) to improve compliance, particularly for entity normalization in event-dense filing periods.

### 13F temporal considerations

Institutional holdings data from 13F filings introduces additional temporal complexity through three mechanisms:

- **Quarterly lag.** 13F filings are reported quarterly with a 45-day delay. Holdings become tradable information only after the public filing date, not the quarter-end date. A December 31 snapshot filed on February 14 should be added to the graph on the filing date.
- **Position masking.** Large institutions sometimes request confidential treatment for positions they are in the process of building. Masked positions appear in later amendments, creating gaps in real-time analysis. Any crowding or co-ownership feature computed during the masking period will be biased downward.
- **Confidential treatment gaps.** When an institution’s full 13F is withheld and released months later, the delayed disclosure creates a temporal inconsistency: either exclude the institution during the gap or use the amendment date as the effective disclosure time.

### Snapshot construction and dynamic features

A common workflow uses rolling windows, treating each window as a dated analytical view:

1. Filter edges visible at cutoff (using disclosure time).
2. Build windowed subgraphs.
3. Compute additions and removals versus the previous window.
4. Compute window-level diagnostics such as churn, centrality drift, and concentration.

Quarterly windows align with fundamental disclosures. Shorter windows capture event-driven structure but increase noise and computation cost. The important constraint is consistency: once a window policy is chosen, keep it stable throughout evaluation.

The visibility-safe query pattern enforces temporal discipline by filtering every edge on disclosure time before any aggregation occurs. In concrete terms, a supplier query should return only relationships whose public timestamp is on or before the cutoff date, regardless of when the underlying relationship began economically. This distinction is what prevents post-cutoff information from entering the analysis.

Temporal graphs produce several candidate features for predictive modeling:

- **Relationship churn**: Edge turnover rate (new + dropped edges/total), measuring network stability and identifying restructuring periods
- **Centrality momentum**: Change in PageRank or betweenness over time, identifying entities gaining or losing structural importance
- **Concentration drift**: Shift in supplier or ownership concentration, capturing evolving risk exposure
- **Theme acceleration**: Tagged relation frequency over time, identifying emerging trends before they consolidate

Dynamic financial knowledge graphs, such as FinDKG (Li and Passino, 2024), and temporal relation construction approaches (Miao et al., 2019) systematically explore these dynamics. Both show that temporal edge patterns carry thematic and structural signals, though translating pattern detection into tradable alpha requires the same walk-forward validation and cost controls applied to any other feature candidate.

### Leakage-safe evaluation protocol

Temporal experiments fail most often due to evaluation hygiene. A minimum protocol includes five rules:

- **Split by disclosure time, not event time**: Prevents using information that was true but not yet public
- Embargo observations **near split boundaries**: Prevents leakage through the gap between event and disclosure
- **Enforce cutoff filtering in every query path**: No exceptions
- **Log snapshot hash and extractor version**: Enables replay and audit of historical feature values
- **Compare against simpler baselines**: Both non-graph variants (tabular only) and static-graph variants (no temporal features)

Without these controls, performance estimates are unreliable. Temporal KG features require more evaluation infrastructure than static features, but the infrastructure is reusable across feature families and strategies. Enforcing temporal discipline requires supporting infrastructure. *Section 23.7* addresses database selection, ontology strategy, query safety, and schema evolution.

**Implementation**: See `07_dynamic_kg_temporal.py` for temporal snapshot construction and leakage-safe query patterns, and `08_8k_event_extraction.py` for the event extraction pipeline. Both notebooks run on the staged 8-K event graph. In the current notebook-scale run, the temporal notebook loads roughly 2,400 timestamped events from 207 companies, applies a 2025-11-01 disclosure-time cutoff that hides about 200 events, and retains roughly 2,200 in the visible snapshot, measures a mean disclosure lag near a week against a mean extraction lag of roughly three years, and verifies that no post-cutoff disclosures enter the visible graph. Alongside the parquet outputs, the notebook writes a `snapshot_manifest.json` that records the cutoff, window, upstream-extractor identity, and per-parquet SHA-256, the audit anchor for the **Log snapshot hash and the extractor version** protocol bullet above.

The extraction notebook processes 50 8-K filings with the Qwen2.5-7B model under FinReflectKG reflection; the critic–corrector loop lifts the validation rate by several percentage points through tens of corrections across the extracted events, and the typed relationships that emerge load idempotently to Neo4j. Counts vary run-to-run at the temperature=0.1 setting; the directional shift, not the exact tally, is the methodological claim.

## 23.7 Building a KG-ready pipeline

Reliable knowledge-graph workflows depend on supporting engineering infrastructure: database selection, ontology strategy, query safety, and schema evolution.

### Engine and query language selection

For book workflows, Neo4j with Cypher is the default because it offers mature tooling and a low-friction path from prototype to medium-scale deployment. The decision framework for other contexts:

| Engine Type | When to Choose | Key Trade-off |
| --- | --- | --- |
| Neo4j (self-managed or Aura) | Teaching, research, medium- scale production | Strong developer tooling; operational tuning at scale |
| Managed cloud graph (Neptune, for example) | Teams optimizing for operations | Managed scaling and IAM; tighter cloud coupling |
| Distributed graph engine (TigerGraph, for example) | Very large graph analytics | High throughput on massive workloads; steeper complexity |

*Table 23.5: Graph database decision framework*

For teams with existing relational infrastructure, PostgreSQL with recursive **Common Table Expressions** (CTE) can handle moderate graph workloads, particularly when the graph is narrow (few relationship types) and queries rarely exceed 3–4 hops. This avoids adding a new database to the stack, though query syntax becomes unwieldy for complex path patterns.

The practical selection rule: choose the engine that supports strict access boundaries, query guards, and stable schema metadata for text-to-Cypher controls. If those requirements are met, the specific engine matters less than the governance layer built on top of it. Cypher is the default for teaching because path patterns are concise and readable; Gremlin provides imperative traversal control in the AWS/Azure ecosystems; SPARQL is standard in RDF and linked data contexts. The structural thinking (node identity, typed relationships, and constraint-aware query design) carries over across languages.

### Ontology strategy

Financial teams rarely adopt large-scale ontologies in a single step. The **Financial Industry Business Ontology** (FIBO) defines over 1,200 classes, a comprehensive standard but an impractical starting point for most implementations. Teams that attempt full FIBO adoption often stall during early iteration because the ontology’s breadth obscures the analytical goal.

A pragmatic middle path captures interoperability benefits without engineering paralysis: align core entity classes with established identifiers (CIK, CUSIP, LEI) where natural, keep task-specific relationship vocabulary compact (most financial KG implementations use approximately 8–15 relationship types), and codify type constraints and validation rules in code rather than ontology management tools. A supply chain KG might use `Company`, `Filing`, and three relationship types (supplier, competitor, customer), small enough to reason about, large enough to support multi-hop analysis. When integration with external systems requires FIBO alignment, the mapping can be added incrementally without restructuring the core graph.

### Query safety controls

Text-to-Cypher interfaces require runtime controls beyond database access management. A production-grade safety layer should enforce:

- **Read-only credentials for all natural-language query paths**: Write access should never be reachable from a text-to-Cypher interface
- **Allowlisted labels, relationships, and properties**: The LLM generates queries only against the declared schema, not the full database catalog
- **Query parameterization rather than string** **interpolation**: Prevents injection attacks and improves caching
- **Execution limits on timeout, row cap, and traversal depth**: Prevent runaway queries from degrading the database
- **Query logging for replay and audit**: Every generated and executed query should be traceable

These controls are architectural requirements, not optional hardening. Research on financial knowledge graph query interfaces (Zehra et al., 2021) confirms that query safety and explainability are first-order concerns for regulated deployment.

### Schema versioning

Knowledge graphs evolve as extraction pipelines improve and analytical needs expand. Schema evolution without breaking existing queries requires discipline: additive changes (such as new node labels, relationship types, properties) are safe and should be the default pattern; renames or removals require migration scripts and downstream query audits; version metadata on edges (such as extractor version, schema version) enable coexistence of old and new extraction formats during transition periods.

A practical pattern is to tag each extraction run with a schema version and include version filters in queries when reproducibility is required. The minimal production architecture comprises an ingestion layer, a schema-constrained extraction layer, a canonicalization service, a graph database with read/write separation, a retrieval layer for safe text-to-Cypher execution, and an evaluation harness for query validity, provenance coverage, and temporal correctness.

**Implementation**: See `02_supply_chain_kg_construction.py` for schema and loading patterns.

## 23.8 Summary

Knowledge graphs earn their complexity only when the question is genuinely relational. That is the chapter’s central discipline. When the answer depends on paths across entities, relationship types, and disclosure timing, graphs provide something vector retrieval cannot: explicit structure that can be traversed, audited, and reused. Graph construction is therefore as much a governance problem as an extraction problem. Identity resolution, schema constraints, provenance tracking, and temporal metadata are what turn noisy filings into graph objects that can support deterministic retrieval and downstream modeling. Once that foundation is in place, graph workflows serve two distinct purposes. Graph RAG supports multi-hop reasoning by delegating relational logic to the database and language generation to the model, while graph analytics convert network structure into features that capture concentration, crowding, and contagion pathways invisible to standalone entity attributes. The companion notebooks ground each of these workflows empirically: the supply-chain extraction pipeline produces over a thousand typed relationships from S&P 100 filings, the RAG benchmark compares graph and vector retrieval on real institutional holdings questions, and the cross-graph feature pipeline combines supply-chain topology with 13F ownership signals into a unified feature matrix.

Throughout, the three-timestamp model remains decisive: without disclosure-time discipline, graph-derived signals are no more trustworthy than any other feature contaminated by look-ahead bias.

The next chapter uses some of the same ingredients in a different role. *Chapter 24* turns structured retrieval, memory, and evidence tracking into autonomous agent workflows for financial research and forecasting.
