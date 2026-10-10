"""B1: claim-level hallucination detection on RAGTruth.

RAGTruth (Niu et al., ACL 2024) has human span annotations of hallucinations in
responses of six LLMs to question answering, summarisation and data-to-text tasks.
Each answer sentence is labelled hallucinated if any annotated span overlaps it; spans
are typed as evident *conflict* (contradicts the source) or *baseless* information.

For every checker the benchmark:

1. scores sentences of a stratified sample of the **train** split and fits a
   calibration profile (saved to ``--profiles``, which is how Groundsight's bundled
   profiles are made);
2. scores the **test** split and reports threshold-free AUROC, balanced accuracy at the
   train-chosen threshold, the recall and false-alarm rate achieved by the conformal
   threshold for each target, and response-level balanced accuracy (a response is
   flagged if any sentence is);
3. for checkers with a contradiction score, measures how well "contradicted" picks out
   the annotated conflicts as the topical-overlap gate varies.

The ``cosine`` checker reproduces the common embedding-similarity shortcut; the
``cosine@0.4`` row applies it with the popular fixed 0.4 cut-off.

    groundsight bench detect --train-per-task 600 --test-per-task 900 \\
        --checkers cosine lexical nli minicheck hhem --out runs/detect
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from groundsight.audit import audit, get_checker
from groundsight.calibrate import Profile, conformal_threshold, fit_profile
from groundsight.checkers import overlap
from groundsight.metrics import auroc, balanced_accuracy, precision_recall_f1
from groundsight.model import Thresholds

DATA_URL = (
    "https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)
TASKS = ("QA", "Summary", "Data2txt")
TARGETS = (0.8, 0.9, 0.95)


def cache_dir() -> Path:
    return Path(os.environ.get("GROUNDSIGHT_CACHE", Path.home() / ".cache" / "groundsight"))


def load_split(split: str) -> Any:
    """RAGTruth split as a DataFrame (downloaded once to the cache directory)."""
    import pandas as pd

    path = cache_dir() / f"ragtruth-{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=split), path)
    return pd.read_parquet(path)


@dataclass
class Example:
    id: str
    task: str
    question: str
    chunks: list[str]
    answer: str
    spans: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.spans)


def passages(context: str, task: str) -> list[str]:
    """QA contexts hold several retrieved passages separated by blank lines."""
    if task == "QA":
        parts = [p.strip() for p in context.split("\n\n") if p.strip()]
        return parts or [context]
    return [context.strip()]


def examples(frame: Any, per_task: int | None, seed: int = 0) -> list[Example]:
    out = []
    for task in TASKS:
        rows = frame[frame.task_type == task]
        if per_task is not None and per_task < len(rows):
            rows = rows.sample(per_task, random_state=seed)
        for row in rows.itertuples():
            labels = json.loads(row.hallucination_labels) if row.hallucination_labels else []
            spans = [
                (
                    int(s["start"]),
                    int(s["end"]),
                    "conflict" if "conflict" in s["label_type"].lower() else "baseless",
                )
                for s in labels
            ]
            out.append(
                Example(
                    str(row.id),
                    task,
                    str(row.query),
                    passages(row.context, task),
                    str(row.output),
                    spans,
                )
            )
    return out


@dataclass
class Row:
    example: str
    task: str
    start: int
    end: int
    support: float
    contradiction: float  # max over passages whose overlap with the claim >= gate_overlap
    raw_contradiction: dict[str, float]  # per-passage contradiction and overlap, ungated
    label: bool
    label_type: str | None


def score(items: Sequence[Example], checker: str, *, progress: bool = True) -> list[Row]:
    check = get_checker(checker)
    rows = []
    started = time.time()
    for n, ex in enumerate(items):
        result = audit(
            ex.answer,
            ex.chunks,
            question=ex.question,
            checker=check,
            thresholds=Thresholds(contradiction_overlap=0.0),
            diagnose=False,
            min_claim_words=1,
        )
        for claim in result.claims:
            hits = [t for s, e, t in ex.spans if s < claim.end and e > claim.start]
            raw = {
                f"{overlap(result.chunk(cid).text, claim.text):.4f}|{cid}": c
                for cid, (_, c) in claim.per_chunk.items()
            }
            rows.append(
                Row(
                    example=ex.id,
                    task=ex.task,
                    start=claim.start,
                    end=claim.end,
                    support=claim.support,
                    contradiction=claim.contradiction,
                    raw_contradiction=raw,
                    label=bool(hits),
                    label_type=("conflict" if "conflict" in hits else "baseless") if hits else None,
                )
            )
        if progress and (n + 1) % 100 == 0:
            rate = (n + 1) / (time.time() - started)
            print(f"  {checker}: {n + 1}/{len(items)} answers ({rate:.1f}/s)", flush=True)
    return rows


def gated(row: Row, gate: float) -> float:
    return max(
        (c for key, c in row.raw_contradiction.items() if float(key.split("|")[0]) >= gate),
        default=0.0,
    )


def _cached_rows(path: Path, items: Sequence[Example], checker: str) -> list[Row]:
    if path.exists():
        return [Row(**json.loads(line)) for line in path.read_text().splitlines() if line]
    rows = score(items, checker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(r)) for r in rows) + "\n")
    return rows


def evaluate_detection(train: Sequence[Row], test: Sequence[Row], name: str) -> dict[str, Any]:
    profile = fit_profile(
        name, "RAGTruth train", [r.support for r in train], [r.label for r in train]
    )
    s = np.array([r.support for r in test])
    y = np.array([r.label for r in test])
    best = profile.balanced().support
    result: dict[str, Any] = {
        "checker": name,
        "train_sentences": len(train),
        "test_sentences": len(test),
        "auroc": auroc(-s, y),
        "threshold": best,
        "balanced_accuracy": balanced_accuracy(s < best, y),
        "by_task": {},
        "conformal": [],
    }
    for target in TARGETS:
        t = conformal_threshold(profile.positives, target)
        result["conformal"].append(
            {
                "target": target,
                "threshold": t,
                "recall": float(np.mean(s[y] < t)) if y.any() else float("nan"),
                "false_alarms": float(np.mean(s[~y] < t)) if (~y).any() else float("nan"),
            }
        )
    for task in TASKS:
        mask = np.array([r.task == task for r in test])
        if mask.any():
            result["by_task"][task] = {
                "auroc": auroc(-s[mask], y[mask]),
                "balanced_accuracy": balanced_accuracy(s[mask] < best, y[mask]),
            }
    result["response_level"] = response_level(test, lambda r: r.support < best)
    return result


def response_level(rows: Sequence[Row], flag: Any) -> dict[str, float]:
    flagged: dict[str, bool] = {}
    truth: dict[str, bool] = {}
    for r in rows:
        flagged[r.example] = flagged.get(r.example, False) or bool(flag(r))
        truth[r.example] = truth.get(r.example, False) or r.label
    keys = sorted(truth)
    pred = [flagged[k] for k in keys]
    gold = [truth[k] for k in keys]
    _, _, f1 = precision_recall_f1(pred, gold)
    return {"responses": len(keys), "balanced_accuracy": balanced_accuracy(pred, gold), "f1": f1}


def evaluate_contradiction(
    test: Sequence[Row], gates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
) -> list[dict[str, float]]:
    """How well 'contradicted' (score >= 0.5) singles out annotated conflicts."""
    out = []
    for gate in gates:
        pred = np.array([gated(r, gate) >= 0.5 and gated(r, gate) > r.support for r in test])
        conflict = np.array([r.label_type == "conflict" for r in test])
        clean = np.array([not r.label for r in test])
        precision, recall, f1 = precision_recall_f1(pred, conflict)
        out.append(
            {
                "gate": gate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_contradictions_on_clean": float(pred[clean].mean()) if clean.any() else 0.0,
            }
        )
    return out


def markdown(results: list[dict[str, Any]], contradiction: dict[str, Any]) -> str:
    lines = [
        "| Checker | AUROC | Balanced acc. | QA | Summary | Data2txt | Response bal. acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tasks = " | ".join(
            f"{r['by_task'].get(t, {}).get('auroc', float('nan')):.3f}" for t in TASKS
        )
        lines.append(
            f"| {r['checker']} | {r['auroc']:.3f} | {r['balanced_accuracy']:.3f} | {tasks} | "
            f"{r['response_level']['balanced_accuracy']:.3f} |"
        )
    lines += [
        "",
        "Per-task columns are AUROC.",
        "",
        "| Checker | Target recall | Recall | False alarms |",
        "|---|---|---|---|",
    ]
    for r in results:
        for c in r.get("conformal", []):
            lines.append(
                f"| {r['checker']} | {c['target']:.0%} | {c['recall']:.1%} | "
                f"{c['false_alarms']:.1%} |"
            )
    for name, rows in contradiction.items():
        lines += [
            "",
            f"Contradiction ({name}) vs annotated conflicts:",
            "",
            "| Overlap gate | Precision | Recall | F1 | False contradictions on clean |",
            "|---|---|---|---|---|",
        ]
        for g in rows:
            lines.append(
                f"| {g['gate']:.1f} | {g['precision']:.3f} | {g['recall']:.3f} | {g['f1']:.3f} | "
                f"{g['false_contradictions_on_clean']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="groundsight bench detect", description=__doc__)
    parser.add_argument("--checkers", nargs="+", default=["cosine", "lexical", "nli", "minicheck"])
    parser.add_argument("--train-per-task", type=int, default=600)
    parser.add_argument("--test-per-task", type=int, default=None, help="default: all 900")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/detect")
    parser.add_argument("--profiles", help="also save calibration profiles here")
    args = parser.parse_args(argv)

    out = Path(args.out)
    train_items = examples(load_split("train"), args.train_per_task, args.seed)
    test_items = examples(load_split("test"), args.test_per_task, args.seed)
    print(f"train {len(train_items)} answers, test {len(test_items)} answers")
    results, contradiction, rows_by = [], {}, {}
    for name in args.checkers:
        train = _cached_rows(out / f"rows-{name}-train.jsonl", train_items, name)
        test = _cached_rows(out / f"rows-{name}-test.jsonl", test_items, name)
        rows_by[name] = (train, test)
        results.append(evaluate_detection(train, test, name))
        if name == "cosine":
            s = np.array([r.support for r in test])
            y = np.array([r.label for r in test])
            results.append(
                {
                    "checker": "cosine@0.4",
                    "auroc": auroc(-s, y),
                    "threshold": 0.4,
                    "balanced_accuracy": balanced_accuracy(s < 0.4, y),
                    "by_task": {},
                    "response_level": response_level(test, lambda r: r.support < 0.4),
                }
            )
        if any(r.contradiction > 0 for r in test) and name != "lexical":
            contradiction[name] = evaluate_contradiction(test)
        if args.profiles:
            profile = fit_profile(
                name,
                f"RAGTruth train ({len(train_items)} answers)",
                [r.support for r in train],
                [r.label for r in train],
            )
            Path(args.profiles).mkdir(parents=True, exist_ok=True)
            profile.save(Path(args.profiles) / f"{name.replace('+', '_')}.json")
            if name == "minicheck":
                Profile("minicheck+nli", profile.source, profile.positives, profile.negatives).save(
                    Path(args.profiles) / "minicheck_nli.json"
                )
        print(json.dumps(results[-1], indent=1, default=float)[:600])
    (out / "results.json").write_text(
        json.dumps({"detection": results, "contradiction": contradiction}, indent=1)
    )
    table = markdown(results, contradiction)
    (out / "results.md").write_text(table)
    print(table)
    return 0
