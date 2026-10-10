"""Command line: ``groundsight {audit,report,serve,verify,calibrate,bench,export,demo,version}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from groundsight._version import __version__
from groundsight.audit import DEFAULT_CHECKER
from groundsight.capture import DEFAULT_STORE

ICONS = {"supported": "✓", "contradicted": "✗", "unsupported": "?"}

DEMO: dict[str, Any] = {
    "question": "How long does the Pro plan keep snapshots, and can I get phone support?",
    "chunks": [
        "Acme Backup keeps daily snapshots for 30 days on the Standard plan.",
        "The Pro plan keeps daily snapshots for 90 days and adds hourly snapshots for the "
        "last 48 hours.",
        "Restores can be started from the web console or with the acme command-line tool.",
        "Deleted files stay recoverable for 30 days after deletion on every plan.",
        "Snapshots are encrypted with AES-256 before they leave your device.",
        "Support is available by email around the clock. Phone support is included with Pro.",
    ],
    "answer": "The Pro plan keeps daily snapshots for 60 days, plus hourly snapshots for the "
    "last 48 hours. Phone support is included with the Pro plan. Snapshots are stored in "
    "three separate data centres.",
}


def _records(path: str) -> Iterator[dict[str, Any]]:
    """Read one JSON object, a JSON list of objects, or JSON Lines."""
    text = Path(path).read_text() if path != "-" else sys.stdin.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    if isinstance(data, list):
        yield from data
    else:
        yield data


def _print_audit(result: Any) -> None:
    print(f"{result.groundedness:.0%} of {len(result.claims)} claims supported  [{result.id}]")
    for claim in result.claims:
        print(f"  {ICONS[claim.verdict.value]} {claim.text}")
        if claim.diagnosis is not None:
            d = claim.diagnosis
            print(f"      {d.cause.value} ({d.confidence}): {d.reason}")


def cmd_audit(args: argparse.Namespace) -> int:
    from groundsight.audit import audit
    from groundsight.capture import Store
    from groundsight.report import write_report

    store = Store(args.store) if args.store else None
    worst = 0
    for i, record in enumerate(_records(args.input)):
        result = audit(
            record["answer"],
            record.get("chunks") or record.get("contexts") or [],
            question=record.get("question"),
            reference=record.get("reference"),
            checker=args.checker,
            target_recall=args.target_recall,
            diagnose=not args.no_diagnose,
        )
        _print_audit(result)
        if store is not None:
            store.save(result)
        if args.html:
            path = Path(args.html)
            if i:
                path = path.with_name(f"{path.stem}-{i}{path.suffix}")
            print(f"  report: {write_report(result, path)}")
        if args.json:
            Path(args.json).write_text(json.dumps(result.to_dict(), indent=1))
        worst = max(worst, len(result.flagged))
    return 1 if worst and args.fail_on_flags else 0


def cmd_report(args: argparse.Namespace) -> int:
    from groundsight.capture import Store
    from groundsight.report import write_report

    print(write_report(Store(args.store).get(args.audit), args.out))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocks until interrupted
    from groundsight.serve import make_server

    server = make_server(args.store, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Groundsight viewer on {url} (Ctrl+C to stop)")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def make_responder(spec: str, temperature: float) -> Any:
    """``openai:MODEL``, ``anthropic:MODEL`` or ``local:HF_MODEL_ID``."""
    from groundsight.generate import AnthropicChat, LocalChat, OpenAIChat, PromptResponder

    provider, _, model = spec.partition(":")
    generator: Any
    if provider == "openai":
        generator = OpenAIChat(model=model or "gpt-4.1-mini")
    elif provider == "anthropic":
        generator = AnthropicChat(model=model) if model else AnthropicChat()
    elif provider == "local":
        generator = LocalChat(model or "Qwen/Qwen2.5-1.5B-Instruct")
    else:
        raise SystemExit(f"unknown model {spec!r}; use openai:MODEL, anthropic:MODEL or local:ID")
    return PromptResponder(generator, temperature=temperature)


def cmd_verify(args: argparse.Namespace) -> int:
    from groundsight.capture import Store
    from groundsight.verify import estimate, verify

    store = Store(args.store)
    result = store.get(args.audit)
    cost = estimate(result, samples=args.samples, gold=args.gold)
    print(
        f"verification needs {cost['calls']} model calls (~{cost['prompt_tokens']:,} prompt tokens)"
    )
    remote = not args.model.startswith("local:")
    if remote and not args.yes:
        reply = input("these calls go to a paid API; continue? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("cancelled")
            return 1
    checks = verify(
        result,
        make_responder(args.model, args.temperature),
        samples=args.samples,
        gold=args.gold,
        progress=print,
    )
    store.save(result)
    if args.html:
        from groundsight.report import write_report

        print(f"report: {write_report(result, args.html, verifications=checks)}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from groundsight.audit import audit, get_checker
    from groundsight.calibrate import fit_profile

    check = get_checker(args.checker)
    scores, labels = [], []
    for record in _records(args.input):
        result = audit(
            record["claim"], record["chunks"], checker=check, diagnose=False, min_claim_words=1
        )
        scores.append(max((c.support for c in result.claims), default=0.0))
        labels.append(bool(record["hallucinated"]))
    profile = fit_profile(check.name, args.source or Path(args.input).name, scores, labels)
    profile.save(args.out)
    print(f"{profile.n} claims, AUROC {profile.auroc():.3f}; saved {args.out}")
    best = profile.balanced()
    print(
        f"  best balanced accuracy: threshold {best.support:.3f}, "
        f"false alarms {profile.false_alarm_rate(best):.0%}"
    )
    for target in (0.8, 0.9, 0.95):
        t = profile.conformal(target)
        print(
            f"  recall >= {target:.0%}: threshold {t.support:.3f}, "
            f"false alarms {profile.false_alarm_rate(t):.0%}"
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from groundsight.capture import Store
    from groundsight.otel import send_otlp, write_otlp

    result = Store(args.store).get(args.audit)
    if args.endpoint:
        print(f"collector answered HTTP {send_otlp(result, args.endpoint)}")
    else:
        print(write_otlp(result, args.out))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from groundsight.audit import audit
    from groundsight.report import write_report

    result = audit(DEMO["answer"], DEMO["chunks"], question=DEMO["question"], checker=args.checker)
    _print_audit(result)
    print(f"report: {write_report(result, args.out)}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from groundsight.bench import run

    return run(args.suite, args.rest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundsight",
        description="See which claims in a RAG answer your sources don't support, and why.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def store_opt(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--store", default=str(DEFAULT_STORE), help="audit store (JSON Lines)")
        return p

    p = sub.add_parser("audit", help="audit answers from JSON / JSON Lines")
    p.add_argument("input", help='file with {"answer", "chunks", "question"?, "reference"?}')
    p.add_argument("--checker", default=DEFAULT_CHECKER)
    p.add_argument("--target-recall", type=float, help="flag at least this share of bad claims")
    p.add_argument("--html", help="write an HTML report")
    p.add_argument("--json", help="write the audit as JSON")
    p.add_argument("--store", help="also append audits to this store")
    p.add_argument("--no-diagnose", action="store_true")
    p.add_argument("--fail-on-flags", action="store_true", help="exit 1 if any claim is flagged")
    p.set_defaults(func=cmd_audit)

    p = store_opt(sub.add_parser("report", help="write the HTML report of a stored audit"))
    p.add_argument("audit")
    p.add_argument("--out", default="groundsight-report.html")
    p.set_defaults(func=cmd_report)

    p = store_opt(sub.add_parser("serve", help="live local view of stored audits"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--open", action="store_true", help="open a browser")
    p.set_defaults(func=cmd_serve)

    p = store_opt(sub.add_parser("verify", help="confirm diagnoses by replaying with fixes"))
    p.add_argument("audit")
    p.add_argument("--model", required=True, help="openai:MODEL, anthropic:MODEL or local:ID")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gold", help="a document that should have been retrieved")
    p.add_argument("--html", help="write a report with the verification results")
    p.add_argument("--yes", action="store_true", help="don't ask before calling a paid API")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("calibrate", help="fit a calibration profile from labelled claims")
    p.add_argument("input", help='JSON Lines of {"claim", "chunks", "hallucinated"}')
    p.add_argument("--checker", default=DEFAULT_CHECKER)
    p.add_argument("--out", default="profile.json")
    p.add_argument("--source", help="name of the labelled dataset")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("bench", help="run a benchmark suite (detect, triage, attribution)")
    p.add_argument("suite", choices=["detect", "triage", "attribution"])
    p.add_argument("rest", nargs=argparse.REMAINDER, help="options for the suite")
    p.set_defaults(func=cmd_bench)

    p = store_opt(sub.add_parser("export", help="export a stored audit as OTLP JSON"))
    p.add_argument("audit")
    p.add_argument("--out", default="audit.otlp.json")
    p.add_argument("--endpoint", help="POST to an OTLP/HTTP collector instead")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("demo", help="audit a built-in example answer")
    p.add_argument("--checker", default=DEFAULT_CHECKER)
    p.add_argument("--out", default="groundsight-demo.html")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("version")
    p.set_defaults(func=lambda _: print(__version__) or 0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
