# Benchmarks

All numbers in the README come from the two suites below. Both run locally with open
models. One extra run of the second suite uses gpt-4.1-mini as the answering model, to
check that triage and attribution transfer to a hosted model.

## B1: claim-level detection on RAGTruth

[RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) (Niu et al., ACL 2024)
has answers from six LLMs to question-answering, summarisation and data-to-text tasks,
with human span annotations of hallucinations.

Setup:

1. **Labels.** An answer sentence is labelled hallucinated if any annotated span
   overlaps it.
2. **Passages.** QA contexts are split into their retrieved passages (separated by
   blank lines). Summary and data-to-text contexts are one document each, checked in
   windows.
3. **Calibration.** Each checker's profile is fitted on a stratified sample of the
   **train** split: 300 answers per task, 900 in total.
4. **Evaluation.** Everything is evaluated on a stratified sample of the **test** split:
   300 answers per task, 900 in total (about 6,000 sentences). Model checkers run at a
   few pairs per second on a laptop GPU, which is what bounds the sample.

Metrics:

1. **AUROC** for ranking hallucinated sentences below supported ones (threshold-free).
2. **Balanced accuracy** at the threshold chosen on train.
3. For each conformal target recall (80%, 90%, 95%): the recall actually achieved on
   test, and the share of supported sentences flagged (false alarms).
4. **Response-level balanced accuracy**: a response is flagged if any sentence is.
5. **Contradiction vs annotated "evident conflict" spans**, as the overlap gate varies.
   This is how the default gate (0.3) was chosen.

Baselines:

1. `cosine`: embedding similarity (all-MiniLM-L6-v2), with its threshold calibrated
   like the others. A `cosine@0.4` row applies the fixed 0.4 cut-off that is common in
   RAG tooling.
2. `lexical`: word overlap.

Reproduce:

```bash
uv sync --extra models --extra bench
uv run groundsight bench detect --checkers cosine lexical nli minicheck \
    --train-per-task 300 --test-per-task 300 --out runs/detect \
    --profiles src/groundsight/profiles
```

Sentence scores are cached in `runs/detect/rows-*.jsonl`, so rerunning with other
checkers reuses them. `hhem` is supported (add it to `--checkers`) but was not part of
the reported run: it executes the model repository's own code and would have added
several hours on a laptop.

## B2/B3: triage, verification and attribution under injected faults

Questions come from HotpotQA's distractor setting: 10 Wikipedia paragraphs, 2 of them
gold. The answering model is Qwen2.5-1.5B-Instruct, run locally.

1. The model answers with the clean context (greedy decoding). Questions it gets wrong
   are dropped.
2. One fault is injected and the question is answered again. Only instances whose
   answer turns wrong are kept, so the fault is the cause by construction. An answer
   counts as right when it contains the gold answer, or at least two thirds of the
   words of a gold name of three or more words ("Muslim Magomayev" for "Muslim
   Magometovich Magomayev").

| Fault | Change | Expected cause |
|---|---|---|
| `retrieval_miss` | gold paragraphs removed | retrieval miss |
| `misleading_source` | the answer string in the gold paragraphs replaced by a same-shaped wrong answer | misleading source |
| `conflicting_sources` | an edited copy of the answer-bearing gold paragraph added at a random position | conflicting sources |
| `buried_evidence` | context grown to 20 paragraphs; gold moved from the front (still answered correctly) to the middle | buried evidence |

The model's clean answer is the reference. For each kept instance:

1. **Triage.** Does the first flagged claim get the expected cause?
2. **Verification.** The matching intervention and plain resampling are each replayed
   `--samples` times at temperature 0.7. Is the cause confirmed (fix rate above
   resampling, 95% Newcombe interval above zero)?
3. **Attribution** (edited-source faults only). Is the edited paragraph ranked first for
   the wrong claim? Methods compared:
   1. likelihood leave-one-out with the answering model;
   2. the same with Qwen2.5-0.5B-Instruct as a proxy;
   3. embedding similarity;
   4. word overlap;
   5. random.

   In `conflicting_sources` the edited copy and the original differ only in the answer
   string. That is the case where similarity-based attribution has nothing to go on.

Reproduce:

```bash
uv run groundsight bench faults --questions 250 --samples 3 --out runs/faults
```

The hosted-model run answers with gpt-4.1-mini (150 questions; the answers are cached in
`instances.json`, so rescoring makes no API calls). Verification is reported for the
local model only: its replays cost nothing, while replaying a hosted model costs a call
per sample.

```bash
uv run groundsight bench faults --model openai:gpt-4.1-mini --questions 150 \
    --skip-verify --out runs/faults-gpt-4.1-mini
```

## Limits

1. **Small, local answering model.** Hosted models fail differently and less often.
2. **Synthetic faults.** They are realistic edits, but synthetic. Natural failures
   usually mix causes, and triage names one.
3. **English Wikipedia (HotpotQA) and RAGTruth's three tasks only.**
4. **Calibration transfers imperfectly.** Recalibrate on your own data before relying on
   the recall guarantee.
