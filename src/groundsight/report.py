"""Single-file HTML report for an audit.

The page needs no server, build step or network: styles, data and a few lines of
JavaScript are inline. The answer is shown with every claim marked by verdict (icon,
label and colour, never colour alone); selecting a claim shows its diagnosis and
highlights its evidence inside the source passage. Attribution and verification
results are included when given. The page follows the system light/dark preference.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from groundsight._version import __version__
from groundsight.attribution import Attribution
from groundsight.model import Audit, ClaimCheck, Verdict
from groundsight.verify import Verification

ICONS = {"supported": "✓", "contradicted": "✗", "unsupported": "?"}
LABELS = {"supported": "Supported", "contradicted": "Contradicted", "unsupported": "Unsupported"}
CAUSES = {
    "retrieval_miss": "Retrieval miss",
    "misleading_source": "Misleading source",
    "conflicting_sources": "Conflicting sources",
    "misread_context": "Misread context",
    "unsupported_addition": "Unsupported addition",
    "buried_evidence": "Buried evidence",
}

_CSS = """
:root { color-scheme: light;
  --bg:#f9f9f7; --surface:#fcfcfb; --raised:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --good:#0ca30c; --warn:#fab219; --bad:#d03b3b; --accent:#2a78d6;
  --good-bg:rgba(12,163,12,.10); --warn-bg:rgba(250,178,25,.22); --bad-bg:rgba(208,59,59,.13);
  --hl:rgba(42,120,214,.16); --seq0:#eef4fc; --seq1:#cde2fb; --seq2:#86b6ef; --seq3:#3987e5; --seq4:#1c5cab; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19; --raised:#222220; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a;
  --accent:#3987e5; --good-bg:rgba(12,163,12,.20); --warn-bg:rgba(250,178,25,.20); --bad-bg:rgba(208,59,59,.26);
  --hl:rgba(57,135,229,.28); --seq0:#202a36; --seq1:#184f95; --seq2:#256abf; --seq3:#3987e5; --seq4:#86b6ef; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19; --raised:#222220; --ink:#ffffff;
  --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; --good-bg:rgba(12,163,12,.20); --warn-bg:rgba(250,178,25,.20);
  --bad-bg:rgba(208,59,59,.26); --hl:rgba(57,135,229,.28); --seq0:#202a36; --seq1:#184f95; --seq2:#256abf;
  --seq3:#3987e5; --seq4:#86b6ef; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding:22px 24px 6px; max-width:1240px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 4px; } h2 { font-size:15px; margin:0 0 12px; } h3 { font-size:14px; margin:14px 0 6px; }
.sub { color:var(--ink2); font-size:13.5px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:12px 0 4px; }
.big { font-size:30px; font-weight:700; margin-right:6px; }
.chip { display:inline-flex; gap:6px; align-items:center; border:1px solid var(--line); background:var(--surface);
  border-radius:999px; padding:2px 10px; font-size:13px; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.d-supported { background:var(--good); } .d-unsupported { background:var(--warn); } .d-contradicted { background:var(--bad); }
main { max-width:1240px; margin:0 auto; padding:10px 24px 40px; display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:18px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 18px; min-width:0; }
.answer { white-space:pre-wrap; font-size:15.5px; line-height:1.75; }
.claim { cursor:pointer; border-radius:4px; padding:1px 2px; border-bottom:3px solid transparent; }
.claim:focus { outline:2px solid var(--accent); outline-offset:1px; }
.v-supported { border-bottom-color:var(--good); }
.v-unsupported { border-bottom-color:var(--warn); background:var(--warn-bg); }
.v-contradicted { border-bottom-color:var(--bad); background:var(--bad-bg); }
.claim.sel { box-shadow:0 0 0 2px var(--accent); }
.tag { font-size:11px; font-weight:700; vertical-align:super; margin-left:1px; color:var(--ink2); }
.flag { border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:10px 0; background:var(--raised); cursor:pointer; }
.flag.sel { border-color:var(--accent); }
.badge { display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:600; border-radius:6px;
  padding:1px 7px; border:1px solid var(--line); }
.cause { font-weight:600; } .fix { color:var(--ink2); font-size:13.5px; }
.conf { font-size:12px; color:var(--muted); }
.conf.confirmed { color:var(--ink); font-weight:600; }
blockquote { margin:6px 0; padding:4px 10px; border-left:3px solid var(--line); color:var(--ink2); font-size:13.5px; }
.chunk { border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:0 0 10px; background:var(--raised); }
.chunk.involved { border-color:var(--accent); }
.chunk-head { display:flex; justify-content:space-between; gap:8px; font-size:12.5px; color:var(--ink2); margin-bottom:4px; flex-wrap:wrap; }
.chunk-text { font-size:13.8px; white-space:pre-wrap; overflow-wrap:anywhere; }
mark { background:var(--hl); color:inherit; border-radius:3px; padding:0 1px; }
.meter { display:inline-flex; align-items:center; gap:6px; font-variant-numeric:tabular-nums; }
.bar { width:70px; height:6px; background:var(--line); border-radius:3px; overflow:hidden; display:inline-block; }
.bar > i { display:block; height:100%; background:var(--accent); }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:5px 6px; text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; }
.heat td.cell { text-align:center; min-width:44px; }
details summary { cursor:pointer; color:var(--ink2); font-size:13px; margin-top:8px; }
.wide { grid-column:1 / -1; }
footer { max-width:1240px; margin:0 auto; padding:0 24px 30px; color:var(--muted); font-size:12.5px; }
@media (max-width: 860px) { main { grid-template-columns:1fr; padding:10px 16px 30px; } header, footer { padding-left:16px; padding-right:16px; } }
"""

_JS = """
const data = JSON.parse(document.getElementById('gs-data').textContent);
const chunks = Object.fromEntries(data.chunks.map(c => [c.id, c]));
function renderChunk(el, chunk, ev) {
  const t = el.querySelector('.chunk-text'); t.textContent = '';
  if (!ev || ev.chunk_id !== chunk.id) { t.textContent = chunk.text; return; }
  t.append(document.createTextNode(chunk.text.slice(0, ev.start)));
  const m = document.createElement('mark'); m.textContent = chunk.text.slice(ev.start, ev.end); t.append(m);
  t.append(document.createTextNode(chunk.text.slice(ev.end)));
}
function select(i) {
  const claim = data.claims[i]; if (!claim) return;
  document.querySelectorAll('.sel').forEach(e => e.classList.remove('sel'));
  document.querySelectorAll(`[data-claim="${i}"]`).forEach(e => e.classList.add('sel'));
  const involved = new Set((claim.diagnosis && claim.diagnosis.chunk_ids) || []);
  if (claim.evidence) involved.add(claim.evidence.chunk_id);
  document.querySelectorAll('.chunk').forEach(el => {
    const c = chunks[el.dataset.chunk]; renderChunk(el, c, claim.evidence);
    el.classList.toggle('involved', involved.has(c.id));
    const s = claim.per_chunk[c.id] || [0, 0];
    el.querySelector('.s-sup').textContent = s[0].toFixed(2);
    el.querySelector('.s-sup-bar').style.width = (s[0] * 100).toFixed(0) + '%';
    el.querySelector('.s-con').textContent = s[1].toFixed(2);
    const att = el.querySelector('.s-att');
    if (att && data.attribution) {
      const row = data.attribution.claim_indices.indexOf(i);
      const j = data.attribution.chunk_ids.indexOf(c.id);
      att.textContent = row >= 0 && j >= 0 ? data.attribution.scores[row][j].toFixed(2) : 'n/a';
    }
  });
  document.getElementById('sel-claim').textContent = `Claim ${i + 1}: “${claim.text}”`;
}
document.querySelectorAll('[data-claim]').forEach(el => {
  el.addEventListener('click', () => select(+el.dataset.claim));
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(+el.dataset.claim); } });
});
select(data.initial);
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(verdict: str) -> str:
    return (
        f'<span class="badge"><span class="dot d-{verdict}"></span>'
        f"{ICONS[verdict]} {LABELS[verdict]}</span>"
    )


def _answer_html(audit: Audit) -> str:
    out, cursor = [], 0
    for claim in audit.claims:
        out.append(_e(audit.answer[cursor : claim.start]))
        v = claim.verdict.value
        title = (
            f"{LABELS[v]} (support {claim.support:.2f}, contradiction {claim.contradiction:.2f})"
        )
        out.append(
            f'<span class="claim v-{v}" data-claim="{claim.index}" tabindex="0" role="button" '
            f'title="{_e(title)}">{_e(claim.text)}<span class="tag" aria-hidden="true">'
            f"{ICONS[v]}</span></span>"
        )
        cursor = claim.end
    out.append(_e(audit.answer[cursor:]))
    return "".join(out)


def _flag_html(claim: ClaimCheck, verification: Verification | None) -> str:
    parts = [
        f'<div class="flag" data-claim="{claim.index}" tabindex="0" role="button">',
        f"{_badge(claim.verdict.value)} <b>Claim {claim.index + 1}</b>",
        f"<div>“{_e(claim.text)}”</div>",
    ]
    if claim.reference_conflict:
        parts.append('<div class="fix">Contradicts the reference answer.</div>')
    if claim.diagnosis is not None:
        d = claim.diagnosis
        confirmed = d.confidence == "confirmed"
        parts.append(
            f'<div><span class="cause">{_e(CAUSES[d.cause.value])}</span> '
            f'<span class="conf{" confirmed" if confirmed else ""}">({_e(d.confidence)})</span></div>'
            f"<div>{_e(d.reason)}</div><div class='fix'>Fix: {_e(d.fix)}</div>"
        )
    if claim.evidence is not None and claim.verdict is not Verdict.UNSUPPORTED:
        parts.append(f"<blockquote>{_e(claim.evidence.text)}</blockquote>")
    if verification is not None:
        rows = []
        for trial in [verification.baseline, *verification.trials]:
            diff = (
                ""
                if trial is verification.baseline
                else "{:+.0%} [{:+.0%}, {:+.0%}]".format(*verification.effect(trial))
            )
            rows.append(
                f"<tr><td>{_e(trial.intervention.label)}</td>"
                f"<td>{trial.successes}/{trial.samples}</td><td>{diff}</td></tr>"
            )
        parts.append(
            "<table><thead><tr><th>Replay</th><th>Fixed</th><th>Effect vs resampling (95% CI)"
            f"</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    parts.append("</div>")
    return "".join(parts)


def _chunks_html(audit: Audit, attribution: Attribution | None) -> str:
    out = []
    for i, chunk in enumerate(audit.chunks):
        meta = f"passage {i + 1} of {len(audit.chunks)}"
        if chunk.source:
            meta += f", {chunk.source}"
        if chunk.score is not None:
            meta += f", retriever score {chunk.score:.3f}"
        att = ', attribution <span class="s-att">n/a</span> nats' if attribution is not None else ""
        out.append(
            f'<div class="chunk" data-chunk="{_e(chunk.id)}"><div class="chunk-head">'
            f"<span><b>{_e(chunk.id)}</b>: {_e(meta)}</span>"
            '<span class="meter">support <span class="bar"><i class="s-sup-bar" style="width:0%"></i>'
            '</span><span class="s-sup">0.00</span>, contradiction <span class="s-con">0.00</span>'
            f"{att}</span></div>"
            f'<div class="chunk-text">{_e(chunk.text)}</div></div>'
        )
    return "".join(out)


def _heatmap_html(audit: Audit, attribution: Attribution) -> str:
    peak = max(float(np.abs(attribution.scores).max()), 1e-9)
    head = "".join(f"<th>{_e(cid)}</th>" for cid in attribution.chunk_ids)
    rows = []
    for r, ci in enumerate(attribution.claim_indices):
        cells = []
        for j in range(len(attribution.chunk_ids)):
            value = float(attribution.scores[r][j])
            step = min(4, max(0, round(4 * max(value, 0.0) / peak)))
            ink = "#ffffff" if step >= 3 else "var(--ink)"
            cells.append(
                f'<td class="cell" style="background:var(--seq{step});color:{ink}">{value:.1f}</td>'
            )
        text = audit.claims[ci].text
        short = text if len(text) <= 70 else text[:67] + "…"
        rows.append(f'<tr><td data-claim="{ci}">{ci + 1}. {_e(short)}</td>{"".join(cells)}</tr>')
    return (
        '<section class="card wide"><h2>Which passage made the model say each claim</h2>'
        f'<div class="sub">Log-probability each passage adds to the claim (nats; {attribution.method}, '
        f"{attribution.passes} forward passes). 2.3 nats = ten times more likely.</div>"
        f'<table class="heat"><thead><tr><th>Claim</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def render(
    audit: Audit,
    *,
    attribution: Attribution | None = None,
    verifications: Sequence[Verification] = (),
    title: str = "Groundsight audit",
) -> str:
    counts = audit.counts()
    by_claim = {v.claim_index: v for v in verifications}
    flagged = audit.flagged
    initial = flagged[0].index if flagged else 0
    data = audit.to_dict()
    data["attribution"] = attribution.to_dict() if attribution is not None else None
    data["initial"] = initial
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    limits = audit.thresholds
    calibration = _e(limits.calibration or "default thresholds")
    chips = "".join(
        f'<span class="chip"><span class="dot d-{v}"></span>{ICONS[v]} {counts[v]} {v}</span>'
        for v in ("supported", "unsupported", "contradicted")
    )
    flags = "".join(_flag_html(c, by_claim.get(c.index)) for c in flagged) or (
        '<div class="sub">No flagged claims: every claim is supported by a passage.</div>'
    )
    question = f'<div class="sub">Question: {_e(audit.question)}</div>' if audit.question else ""
    reference = (
        f'<div class="sub">Reference answer: {_e(audit.reference)}</div>' if audit.reference else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head><body>
<header><h1>{_e(title)}</h1>{question}{reference}
<div class="stats"><span class="big">{audit.groundedness:.0%}</span><span class="sub">of claims supported</span>{chips}</div>
<div class="sub">Checker <b>{_e(audit.checker)}</b>, support threshold {limits.support:.3f}, {calibration}</div>
</header>
<main>
<section class="card"><h2>Answer</h2><div class="answer">{_answer_html(audit)}</div>
<h3>Flagged claims</h3>{flags}</section>
<section class="card"><h2>Sources</h2><div class="sub" id="sel-claim"></div>
<div style="margin-top:10px">{_chunks_html(audit, attribution)}</div>
<details><summary>All claims as a table</summary><table><thead><tr><th>#</th><th>Claim</th>
<th>Verdict</th><th>Support</th><th>Contradiction</th><th>Cause</th></tr></thead><tbody>
{"".join(f"<tr><td>{c.index + 1}</td><td>{_e(c.text)}</td><td>{ICONS[c.verdict.value]} {_e(c.verdict.value)}</td><td>{c.support:.2f}</td><td>{c.contradiction:.2f}</td><td>{_e(CAUSES[c.diagnosis.cause.value]) if c.diagnosis else ''}</td></tr>" for c in audit.claims)}
</tbody></table></details></section>
{_heatmap_html(audit, attribution) if attribution is not None else ""}
</main>
<footer>Groundsight {_e(__version__)}, audit {_e(audit.id)}. Verdicts are model judgements; check flagged claims against the sources.</footer>
<script type="application/json" id="gs-data">{payload}</script>
<script>{_JS}</script></body></html>
"""


def write_report(
    audit: Audit,
    path: str | Path,
    *,
    attribution: Attribution | None = None,
    verifications: Sequence[Verification] = (),
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(audit, attribution=attribution, verifications=verifications))
    return out
