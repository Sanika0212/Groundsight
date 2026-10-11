# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-10

### Added

- `audit()`: claim-by-claim checking of an answer against its passages, with verdicts
  of supported, contradicted or unsupported and evidence spans refined to sentences.
  Long passages are checked in windows; multi-hop claims are also checked against the
  joined context. Contradictions count only from passages about the claim.
- Checkers: MiniCheck, NLI (DeBERTa-v3 base and large), HHEM-2.1-open, an
  embedding-similarity baseline and a dependency-free lexical checker, combinable as
  `support+contradiction`.
- Calibration profiles with best-balanced-accuracy thresholds and split-conformal
  thresholds that guarantee a chosen recall; profiles fitted on RAGTruth train are
  bundled, and `groundsight calibrate` fits your own.
- Root-cause triage: retrieval miss, misleading source, conflicting sources, misread
  context, unsupported addition, buried evidence.
- Counterfactual verification: replay the answer with the fix each cause implies and
  compare against plain resampling with a Newcombe interval.
- Per-claim attribution by likelihood ablation (leave-one-out or a ridge surrogate),
  with local models and proxies.
- Generators for OpenAI-compatible, Anthropic and local Hugging Face models, or your own
  pipeline.
- Capture with `trace()`, `@retriever`, `@generator` and `capture_http()`; a JSON Lines
  store; a localhost live viewer; a single-file HTML report; OTLP export with
  `gen_ai.evaluation.result` events.
- `assert_grounded` and a pytest plugin (`grounded` fixture, HTML reports of failures).
- Benchmarks: RAGTruth claim-level detection and a HotpotQA fault-injection suite for
  triage, verification and attribution.
- `groundsight` command line: `audit`, `report`, `serve`, `verify`, `calibrate`,
  `bench`, `export`, `demo`.
