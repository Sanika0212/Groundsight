# How Groundsight works

An audit takes an answer, the passages the model was given (in prompt order) and,
optionally, the question and a reference answer. It runs in four stages. Each stage
is usable on its own.

## 1. Claims

The answer is split into sentences with exact character offsets (`groundsight.text`).
The splitter handles abbreviations ("Dr.", "e.g."), initials, decimals, list items and
headings. Sentences shorter than `min_claim_words` (default 3) are skipped. Each
remaining sentence is a claim.

## 2. Checking: supported, contradicted or unsupported

Each claim is scored against every passage by a *checker*: a model that reads
`(premise, claim)` and returns a support probability and, for three-way models, a
contradiction probability.

| Checker | What it is | Output |
|---|---|---|
| `minicheck+nli` (default) | MiniCheck (RoBERTa-large) for support, NLI DeBERTa-v3 for contradiction | 3-way |
| `nli`, `nli-large` | NLI cross-encoders (DeBERTa-v3 base / large) | 3-way |
| `minicheck` | MiniCheck, trained for checking claims against grounding documents | support |
| `hhem` | HHEM-2.1-open (runs the model's own remote code) | support |
| `cosine` | Embedding similarity: a baseline, not a grounding check | support |
| `lexical` | Word overlap with number and negation rules: no model, for tests and CI | 3-way |

Scoring details:

- **Long passages** are checked in windows of whole sentences (250 words by default,
  overlapping by one sentence). Text without sentence breaks, such as tables or JSON,
  is split into overlapping word windows.
- **Multi-hop claims** that combine facts from two passages are also checked against
  all passages joined together, when they fit.
- **Contradictions need a related passage.** A contradiction counts only from a passage
  that shares at least 30% of the claim's content words. Without this gate, NLI models
  report confident contradictions between unrelated statements on the same subject.
  The RAGTruth benchmark measures the gate against human "evident conflict" labels.
- **Evidence** is the window, refined to its single best sentence when that sentence
  carries most of the window's score. The report highlights it inside the passage.

The verdict follows from two thresholds:

- **supported** if support ≥ `support` and the claim is not contradicted;
- **contradicted** if contradiction ≥ `contradiction` and contradiction > support;
- **unsupported** otherwise.

## Calibration and the recall guarantee

The support threshold comes from a *calibration profile*: the checker's support scores
on claims that annotators labelled as hallucinated or not. Groundsight ships profiles
fitted on the RAGTruth training split.

- By default, the threshold maximises balanced accuracy on the calibration data.
- With `target_recall=0.9`, the threshold is the ⌈(n + 1) × 0.9⌉-th smallest support score
  among the *n* hallucinated calibration claims (split conformal prediction).
  - If new hallucinated claims are exchangeable with the calibration ones, a new
    hallucinated claim is flagged with probability at least 0.9, on average over
    calibration sets.
  - The price is more false alarms on supported claims, which the profile reports.
  - The guarantee is only as good as the match between your data and the calibration
    data. Use `groundsight calibrate` to fit a profile on labelled examples from your
    own domain.

## 3. Triage: why did the claim go wrong?

Every flagged claim gets a *probable cause*. The rules use three signals:

- which passages support or contradict the claim;
- how topically related the closest passage is (word overlap: a relevance signal,
  never a grounding signal);
- when a reference answer is given, whether any passage contains that answer.

| Cause | Pattern | Where to fix it |
|---|---|---|
| retrieval miss | nothing in the context covers the claim (or the reference) | corpus, retriever |
| misleading source | a passage supports the claim, but the claim contradicts the reference | the source document |
| conflicting sources | one passage supports the claim, another contradicts it | corpus hygiene, prompt |
| misread context | a passage contradicts the claim | prompt, decoding, model |
| unsupported addition | the topic is covered but the claim is not stated | prompt, citations |
| buried evidence | the needed passage sits mid-context (not among the first two or last two) | reranking, fewer passages |

The cause is a hypothesis until it is verified.

## 4. Verification: replay with the fix

`verify(audit, responder)` regenerates the answer in two ways:

- `samples` times **with** the intervention that the cause implies;
- `samples` times **without** any change (plain resampling at the same temperature,
  the noise floor).

| Cause | Intervention |
|---|---|
| misleading source / conflicting sources | drop the passage that supports the wrong claim |
| misread context / buried evidence | move the contradicting or evidence passage to the top; or instruct "only from the context" |
| unsupported addition | instruct "only from the context" |
| retrieval miss | add the missing document (`gold=`) if known, else instruct the model to abstain |

A sample counts as fixed in one of two ways:

- without a reference, the new answer no longer asserts the claim (the checker, with
  the new answer as premise);
- with a reference, the new answer states the reference.

A cause is **confirmed** when the intervention's fix rate exceeds the resampling rate,
and the 95% Newcombe interval on the difference excludes zero. `estimate()` and the
CLI print the number of calls before anything runs. The CLI asks before calling a paid
API.

The *responder* can be your own pipeline (`PipelineResponder`). It can also be a
built-in generator with the default RAG prompt: `OpenAIChat`, `AnthropicChat`, or a
local Hugging Face model (`LocalChat`).

## Attribution: which passage made the model say it?

`attribute(audit, scorer)` scores each claim's own tokens in the actual answer with
teacher forcing, with and without each passage in the prompt. The score is the
log-probability the passage adds, in nats.

- `loo` uses `n + 1` forward passes (default up to 12 passages).
- `surrogate` scores random subsets and fits a ridge model per claim, at a fixed cost
  for any number of passages.
- A small local model can stand in for a hosted one (a proxy). The fault benchmark
  measures how often the proxy agrees.

This replaces attention-based attribution, which reads attention weights of the
prompt rather than the probability of what the model actually said.

## Capture, storage and outputs

- **Capture.** `trace()` with `@retriever` / `@generator`, or `capture_http()` for
  OpenAI- and Anthropic-style chat calls made through httpx.
- **Storage.** Audits are appended to `.groundsight/audits.jsonl`.
- **Viewing.** `groundsight serve` shows the store live on localhost.
- **Reports.** `write_report()` writes a single HTML file.
- **Export.** `to_otlp()` exports one `gen_ai.evaluation.result` event per claim.
