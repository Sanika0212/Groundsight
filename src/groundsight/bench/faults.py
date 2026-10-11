"""B2/B3: triage, verification and attribution on answers broken by injected faults.

Questions come from HotpotQA (distractor setting: each question has ten Wikipedia
paragraphs, two of them gold). A local instruct model answers each question; only
questions it answers correctly with the clean context are kept. Then one fault is
injected into the context and the question is answered again. Instances where the
answer becomes wrong are kept, so **the injected fault is the cause by construction**:

==========================  =====================================================  ==============
fault                       change to the context                                  expected cause
==========================  =====================================================  ==============
``retrieval_miss``          gold paragraphs removed                                retrieval miss
``misleading_source``       answer string replaced by a wrong one in the gold      misleading
                            paragraphs                                             source
``conflicting_sources``     an edited copy of the gold paragraph added             conflicting
                                                                                   sources
``buried_evidence``         gold paragraphs moved from the front to the middle     buried
                            of a 20-paragraph context                              evidence
==========================  =====================================================  ==============

The model's clean answer serves as the reference. For every kept instance:

- **triage**: does the first flagged claim get the expected cause?
- **verification**: does the intervention for that cause fix the answer more often
  than resampling (``--samples`` replays each, temperature 0.7)?
- **attribution** (edited-source faults): is the edited paragraph the top-attributed
  chunk for the wrong claim? Compared methods: likelihood leave-one-out with the
  answering model, the same with a smaller proxy model, embedding cosine, word overlap,
  and random.

    groundsight bench faults --questions 300 --out runs/faults
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from groundsight.audit import audit, get_checker, resolve_thresholds
from groundsight.bench.detect import cache_dir
from groundsight.checkers import overlap
from groundsight.generate import Generator, PromptResponder, build_messages
from groundsight.metrics import wilson_interval
from groundsight.model import Audit, Chunk, ClaimCheck, Thresholds
from groundsight.verify import verify

DATA_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)
FAULTS = ("retrieval_miss", "misleading_source", "conflicting_sources", "buried_evidence")
INSTRUCTION = (
    "Answer the question using the numbered context passages. Reply with one short "
    "sentence that states the answer."
)


@dataclass
class Question:
    id: str
    question: str
    answer: str
    gold: list[str]  # gold paragraphs (title: text)
    distractors: list[str]


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def correct(answer: str, gold: str) -> bool:
    return normalize(gold) in normalize(answer)


def load_questions(limit: int, seed: int = 0) -> list[Question]:
    import urllib.request

    import pandas as pd

    path = cache_dir() / "hotpotqa-distractor-validation.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    frame = pd.read_parquet(path).sample(frac=1.0, random_state=seed)
    out = []
    for row in frame.itertuples():
        if row.answer.lower() in {"yes", "no"}:
            continue
        titles = list(row.context["title"])
        paragraphs = [
            f"{t}: {' '.join(s).strip()}"
            for t, s in zip(titles, row.context["sentences"], strict=True)
        ]
        gold_titles = set(row.supporting_facts["title"])
        gold = [p for t, p in zip(titles, paragraphs, strict=True) if t in gold_titles]
        if not any(row.answer.lower() in g.lower() for g in gold):
            continue  # the answer must be editable in a gold paragraph
        out.append(
            Question(
                str(row.id),
                row.question,
                row.answer,
                gold,
                [p for t, p in zip(titles, paragraphs, strict=True) if t not in gold_titles],
            )
        )
        if len(out) >= limit:
            break
    return out


def substitute(answer: str, pool: Sequence[str], rng: random.Random, context: str) -> str | None:
    """A wrong answer of the same shape that does not occur in the context."""
    if re.fullmatch(r"[\d,.]+", answer.strip()):
        digits = answer.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            return None
        delta = rng.choice([1, 2, 3, 5, 7, 10, 12]) * (1 if value < 3 else rng.choice([1, -1]))
        new = value + delta
        text = str(int(new)) if "." not in digits else f"{new:.1f}"
        return text if text not in context else None
    shape = (len(answer.split()), answer[:1].isupper(), any(ch.isdigit() for ch in answer))
    candidates = [
        a
        for a in pool
        if (len(a.split()), a[:1].isupper(), any(ch.isdigit() for ch in a)) == shape
        and a.lower() != answer.lower()
        and a.lower() not in context.lower()
    ]
    return rng.choice(candidates) if candidates else None


def edit(text: str, answer: str, replacement: str) -> str:
    return re.sub(re.escape(answer), replacement, text, flags=re.IGNORECASE)


@dataclass
class Instance:
    question: str
    answer: str
    fault: str
    chunks: list[str]
    clean_answer: str  # reference: the model's correct answer with clean context
    faulted_answer: str
    edited_index: int | None = None  # chunk carrying the wrong answer, for attribution
    gold_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def ask(generator: Generator, question: str, chunks: Sequence[str]) -> str:
    items = [Chunk(f"c{i}", c) for i, c in enumerate(chunks)]
    return generator(build_messages(question, items, INSTRUCTION), temperature=0.0)


def build_instances(
    questions: Sequence[Question],
    generator: Generator,
    *,
    seed: int = 0,
    long_context: int = 20,
    progress: Callable[[str], None] = print,
) -> tuple[list[Instance], dict[str, Any]]:
    rng = random.Random(seed)
    pool = [q.answer for q in questions]
    filler = [d for q in questions for d in q.distractors]
    instances: list[Instance] = []
    stats: Counter[str] = Counter()
    started = time.time()
    for n, q in enumerate(questions):
        clean = q.gold + q.distractors
        rng.shuffle(clean)
        clean_answer = ask(generator, q.question, clean)
        stats["questions"] += 1
        if not correct(clean_answer, q.answer):
            continue
        stats["clean_correct"] += 1
        context = " ".join(clean)
        wrong = substitute(q.answer, pool, rng, context)
        candidates: dict[str, tuple[list[str], int | None]] = {
            "retrieval_miss": ([c for c in clean if c not in q.gold], None)
        }
        if wrong is not None:
            edited = [edit(c, q.answer, wrong) if c in q.gold else c for c in clean]
            carrier = next(i for i, c in enumerate(edited) if wrong.lower() in c.lower())
            candidates["misleading_source"] = (edited, carrier)
            source = next(g for g in q.gold if q.answer.lower() in g.lower())
            copy = edit(source, q.answer, wrong)
            mixed = list(clean)
            position = rng.randrange(len(mixed) + 1)
            mixed.insert(position, copy)
            candidates["conflicting_sources"] = (mixed, position)
        extra = [
            d
            for d in rng.sample(filler, min(len(filler), long_context * 2))
            if d not in q.distractors and d not in q.gold
        ][: long_context - len(clean)]
        front = q.gold + q.distractors + extra
        if correct(ask(generator, q.question, front), q.answer):
            stats["front_correct"] += 1
            middle = len(front) // 2
            others = q.distractors + extra
            buried = others[: middle - 1] + q.gold + others[middle - 1 :]
            candidates["buried_evidence"] = (buried, None)
        for fault, (chunks, edited_at) in candidates.items():
            stats[f"{fault}_tried"] += 1
            faulted = ask(generator, q.question, chunks)
            if correct(faulted, q.answer):
                continue
            stats[f"{fault}_kept"] += 1
            instances.append(
                Instance(
                    q.question,
                    q.answer,
                    fault,
                    chunks,
                    clean_answer,
                    faulted,
                    edited_at,
                    gold_text=q.gold[0],
                    meta={"id": q.id, "wrong": wrong},
                )
            )
        if (n + 1) % 25 == 0:
            rate = (n + 1) / (time.time() - started)
            progress(
                f"  {n + 1}/{len(questions)} questions, {len(instances)} instances ({rate:.2f}/s)"
            )
    return instances, dict(stats)


def first_flag(result: Audit) -> ClaimCheck | None:
    flagged = result.flagged
    return flagged[0] if flagged else None


def evaluate_triage(
    instances: Sequence[Instance], checker: str, thresholds: Thresholds | None = None
) -> tuple[list[dict[str, Any]], list[Audit]]:
    check = get_checker(checker)
    rows, audits = [], []
    for inst in instances:
        result = audit(
            inst.faulted_answer,
            inst.chunks,
            question=inst.question,
            reference=inst.clean_answer,
            checker=check,
            thresholds=thresholds,
        )
        claim = first_flag(result)
        predicted = claim.diagnosis.cause.value if claim and claim.diagnosis else "not flagged"
        rows.append({"fault": inst.fault, "predicted": predicted, "flagged": claim is not None})
        audits.append(result)
    return rows, audits


def evaluate_verification(
    instances: Sequence[Instance],
    audits: Sequence[Audit],
    generator: Generator,
    *,
    samples: int,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    responder = PromptResponder(generator, temperature=0.7, instruction=INSTRUCTION)
    rows: list[dict[str, Any]] = []
    for n, (inst, result) in enumerate(zip(instances, audits, strict=True)):
        claim = first_flag(result)
        if claim is None or claim.diagnosis is None:
            rows.append({"fault": inst.fault, "verified": None})
            continue
        gold = inst.gold_text if inst.fault == "retrieval_miss" else None
        checks = verify(result, responder, samples=samples, claims=[claim.index], gold=gold)
        check = checks[0] if checks else None
        best = check.best if check else None
        rows.append(
            {
                "fault": inst.fault,
                "cause": claim.diagnosis.cause.value,
                "verified": bool(check and check.confirmed),
                "fix_rate": best.rate if best else 0.0,
                "baseline_rate": check.baseline.rate if check else 0.0,
                "intervention": best.intervention.label if best else None,
            }
        )
        if (n + 1) % 20 == 0:
            progress(f"  verified {n + 1}/{len(instances)}")
    return rows


def evaluate_attribution(
    instances: Sequence[Instance],
    audits: Sequence[Audit],
    scorers: dict[str, Any],
    *,
    cosine: Any | None = None,
    seed: int = 0,
) -> dict[str, list[bool]]:
    """Is the edited chunk the top-attributed chunk for the wrong claim?"""
    from groundsight.attribution import attribute

    rng = random.Random(seed)
    hits: dict[str, list[bool]] = defaultdict(list)
    for inst, result in zip(instances, audits, strict=True):
        if inst.edited_index is None:
            continue
        wrong = inst.meta.get("wrong") or ""
        claim = next(
            (c for c in result.claims if wrong and wrong.lower() in c.text.lower()), None
        ) or first_flag(result)
        if claim is None:
            continue
        target = f"c{inst.edited_index}"
        for name, scorer in scorers.items():
            ranked = attribute(result, scorer, claims=[claim.index], instruction=INSTRUCTION)
            hits[name].append(ranked.top(claim.index) == target)
        word = max(result.chunks, key=lambda c: overlap(c.text, claim.text))
        hits["word overlap"].append(word.id == target)
        if cosine is not None:
            sims = cosine.score([(c.text, claim.text) for c in result.chunks])
            best = int(np.argmax([s for s, _ in sims]))
            hits["embedding cosine"].append(result.chunks[best].id == target)
        hits["random"].append(rng.randrange(len(result.chunks)) == inst.edited_index)
    return dict(hits)


def summarize(
    triage: Sequence[dict[str, Any]],
    verification: Sequence[dict[str, Any]],
    attribution: dict[str, list[bool]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    by_fault: dict[str, Any] = {}
    for fault in FAULTS:
        rows = [r for r in triage if r["fault"] == fault]
        if not rows:
            continue
        hits = sum(r["predicted"] == fault for r in rows)
        low, high = wilson_interval(hits, len(rows))
        ver = [v for v in verification if v["fault"] == fault and v.get("verified") is not None]
        confirmed = sum(bool(v["verified"]) for v in ver)
        by_fault[fault] = {
            "n": len(rows),
            "flagged": sum(r["flagged"] for r in rows) / len(rows),
            "triage_accuracy": hits / len(rows),
            "triage_ci": [low, high],
            "confusion": dict(Counter(r["predicted"] for r in rows)),
            "verified": confirmed / len(ver) if ver else None,
            "verified_n": len(ver),
            "fix_rate": float(np.mean([v["fix_rate"] for v in ver])) if ver else None,
            "baseline_rate": float(np.mean([v["baseline_rate"] for v in ver])) if ver else None,
        }
    attribution_summary = {}
    for name, values in attribution.items():
        k = sum(values)
        low, high = wilson_interval(k, len(values))
        attribution_summary[name] = {
            "n": len(values),
            "top1": k / len(values) if values else 0,
            "ci": [low, high],
        }
    total = len(triage)
    overall = sum(r["predicted"] == r["fault"] for r in triage)
    majority = max(Counter(r["fault"] for r in triage).values(), default=0)
    return {
        "stats": stats,
        "instances": total,
        "triage_accuracy": overall / total if total else 0.0,
        "majority_baseline": majority / total if total else 0.0,
        "by_fault": by_fault,
        "attribution": attribution_summary,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"{summary['instances']} broken answers; triage accuracy "
        f"{summary['triage_accuracy']:.0%} (majority-class baseline "
        f"{summary['majority_baseline']:.0%}).",
        "",
        "| Injected fault | Answers | Flagged | Triage correct (95% CI) | Cause confirmed by "
        "replay | Fix rate vs resampling |",
        "|---|---|---|---|---|---|",
    ]
    for fault, r in summary["by_fault"].items():
        low, high = r["triage_ci"]
        confirmed = "n/a" if r["verified"] is None else f"{r['verified']:.0%}"
        fix = "n/a" if r["fix_rate"] is None else f"{r['fix_rate']:.0%} vs {r['baseline_rate']:.0%}"
        lines.append(
            f"| {fault} | {r['n']} | {r['flagged']:.0%} | {r['triage_accuracy']:.0%} "
            f"[{low:.0%}, {high:.0%}] | {confirmed} | {fix} |"
        )
    lines += [
        "",
        "| Attribution method | Edited source ranked first (95% CI) | Answers |",
        "|---|---|---|",
    ]
    for name, r in sorted(summary["attribution"].items(), key=lambda kv: -kv[1]["top1"]):
        lines.append(
            f"| {name} | {r['top1']:.0%} [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}] | {r['n']} |"
        )
    return "\n".join(lines) + "\n"


def estimate_calls(questions: int, samples: int, clean_rate: float = 0.5) -> int:
    """Rough number of model calls for a run (to show before using a paid API).

    Per question: a clean answer and a long-context check; for the questions answered
    correctly (``clean_rate``), up to four faulted answers and, for each kept instance,
    ``samples`` replays of the resampling baseline and of each of up to two fixes.
    """
    kept = questions * clean_rate * 4 * 0.6
    return int(questions * 2 + questions * clean_rate * 4 + kept * samples * 3)


def make_generator(spec: str) -> tuple[Generator, bool]:
    """``openai:MODEL``, ``anthropic:MODEL`` or a Hugging Face model id (run locally)."""
    from groundsight.generate import AnthropicChat, LocalChat, OpenAIChat

    provider, _, model = spec.partition(":")
    if provider == "openai" and model:
        return OpenAIChat(model=model, max_tokens=48), True
    if provider == "anthropic" and model:
        return AnthropicChat(model=model, max_tokens=48), True
    return LocalChat(spec, max_new_tokens=48), False


def main(argv: Sequence[str]) -> int:
    from groundsight.attribution import LocalScorer
    from groundsight.checkers import CosineChecker

    parser = argparse.ArgumentParser(prog="groundsight bench faults", description=__doc__)
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="answering model: a Hugging Face id (local), openai:MODEL or anthropic:MODEL",
    )
    parser.add_argument("--proxy", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checker", default="minicheck+nli")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--yes", action="store_true", help="don't ask before calling a paid API")
    parser.add_argument("--out", default="runs/faults")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chat, remote = make_generator(args.model)
    cached = out / "instances.json"
    if remote and not args.yes:
        calls = estimate_calls(args.questions, 0 if args.skip_verify else args.samples)
        print(f"about {calls:,} calls to {args.model} (roughly {calls * 1_100:,} input tokens)")
        if input("continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("cancelled")
            return 1
    if cached.exists():
        data = json.loads(cached.read_text())
        instances = [Instance(**i) for i in data["instances"]]
        stats = data["stats"]
    else:
        questions = load_questions(args.questions, args.seed)
        print(f"{len(questions)} questions; building instances with {args.model}")
        instances, stats = build_instances(questions, chat, seed=args.seed)
        cached.write_text(
            json.dumps({"instances": [asdict(i) for i in instances], "stats": stats}, indent=1)
        )
    print(f"{len(instances)} instances: {dict(Counter(i.fault for i in instances))}")
    # Resolve the calibrated thresholds once, so every audit in the run uses the same ones.
    thresholds = resolve_thresholds(get_checker(args.checker), None, None)
    print(f"thresholds: support {thresholds.support:.4f} ({thresholds.calibration})")
    triage, audits = evaluate_triage(instances, args.checker, thresholds)
    verification = (
        []
        if args.skip_verify
        else evaluate_verification(instances, audits, chat, samples=args.samples)
    )
    scorers: dict[str, Any] = {"likelihood (0.5B proxy)": LocalScorer(args.proxy)}
    if not remote:  # a hosted model's own likelihoods are not available
        scorers["likelihood (answering model)"] = LocalScorer(chat)  # type: ignore[arg-type]
    attribution = evaluate_attribution(
        instances, audits, scorers, cosine=CosineChecker(), seed=args.seed
    )
    summary = summarize(triage, verification, attribution, stats)
    summary["model"] = args.model
    summary["thresholds"] = asdict(thresholds)
    (out / "results.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "triage": triage,
                "verification": verification,
                "attribution": attribution,
            },
            indent=1,
        )
    )
    table = markdown(summary)
    (out / "results.md").write_text(table)
    print(table)
    return 0
