"""Human-readable rendering of a drift report.

Logged as an MLflow artifact next to the machine-readable JSON. Self-contained
by necessity: MLflow serves artifacts from an S3-backed store through its own
viewer, so anything fetched from a CDN at render time simply would not load.
That rules out a charting library, which is why the distribution comparisons
below are drawn as plain proportional bars in table cells - readable in the
artifact viewer, readable in a browser, and readable after being saved to a
ticket six months from now.

The layout puts the concept-drift verdict first and the per-feature table last,
because the first question anyone opening this has is "is the model still
right", not "which of thirty features moved".
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from src.monitoring.drift_monitor import DriftReport

_STATUS_COLOURS = {
    "OK": "#1a7f37",
    "WARN": "#9a6700",
    "ALERT": "#b42318",
    "SKIPPED": "#57606a",
}

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 24px; color: #1f2328; background: #ffffff; line-height: 1.5; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; text-transform: uppercase;
     letter-spacing: .04em; color: #57606a; }
.muted { color: #57606a; font-size: 13px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff;
         font-weight: 600; font-size: 12px; letter-spacing: .03em; }
.verdict { border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-top: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.kpi { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #57606a; }
.kpi .value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #d0d7de;
         vertical-align: middle; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #57606a; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { height: 8px; border-radius: 4px; background: #d0d7de; min-width: 2px; }
.scroll { overflow-x: auto; }
"""


def _badge(status: str) -> str:
    colour = _STATUS_COLOURS.get(status, "#57606a")
    return f'<span class="badge" style="background:{colour}">{html.escape(status)}</span>'


def _kpi(label: str, value: str) -> str:
    return (
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_html_report(report: DriftReport) -> str:
    concept = report.concept_drift
    confidence = report.confidence_drift

    kpis = [
        _kpi("Predictions", str(report.coverage.get("n_predictions", 0))),
        _kpi("Labelled", str(report.coverage.get("n_labelled", 0))),
        _kpi("Label coverage", f"{float(report.coverage.get('label_coverage', 0.0)):.1%}"),
        _kpi("Features drifted", f"{report.drift_share:.0%}"),
    ]

    concept_rows = [
        ("Baseline weighted F1 (held-out, at training)", _fmt(concept.baseline_f1)),
        ("Live weighted F1 (labelled window)", _fmt(concept.live_f1)),
        (
            "Live F1 95% interval",
            f"[{_fmt(concept.live_f1_lower)}, {_fmt(concept.live_f1_upper)}]",
        ),
        ("Drop versus baseline", _fmt(concept.f1_drop)),
        ("Live accuracy", _fmt(concept.live_accuracy)),
        ("Labelled predictions", str(concept.n_labelled)),
        (
            "Whole interval below baseline",
            "yes" if concept.statistically_significant else "no",
        ),
    ]
    concept_html = "".join(
        f"<tr><td>{html.escape(label)}</td><td class='num'>{html.escape(value)}</td></tr>"
        for label, value in concept_rows
    )

    per_class = "".join(
        f"<tr><td>{html.escape(segment)}</td><td class='num'>{score:.4f}</td></tr>"
        for segment, score in sorted(concept.per_class_f1.items(), key=lambda kv: kv[1])
    )

    # Bar width is scaled against the ALERT threshold rather than the maximum
    # observed PSI, so the same value always draws the same length and two
    # reports from different days can be compared by eye.
    def _bar(psi: float) -> str:
        if psi != psi:  # NaN
            return ""
        width = min(100.0, (psi / 0.25) * 100.0)
        colour = _STATUS_COLOURS["ALERT" if psi >= 0.25 else "WARN" if psi >= 0.10 else "OK"]
        return f'<div class="bar" style="width:{width:.0f}%;background:{colour}"></div>'

    feature_rows = "".join(
        "<tr>"
        f"<td>{html.escape(d.feature)}</td>"
        f"<td>{html.escape(d.kind)}</td>"
        f"<td class='num'>{'n/a' if d.psi != d.psi else f'{d.psi:.4f}'}</td>"
        f"<td style='width:120px'>{_bar(d.psi)}</td>"
        f"<td class='num'>{'n/a' if d.jensen_shannon != d.jensen_shannon else f'{d.jensen_shannon:.4f}'}</td>"
        f"<td>{_badge(d.status)}</td>"
        f"<td class='muted'>{html.escape(d.detail)}</td>"
        "</tr>"
        for d in report.data_drift
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Drift report - {html.escape(report.model_name)} v{html.escape(report.model_version or '?')}</title>
<style>{_CSS}</style></head><body>

<h1>Drift report &mdash; {html.escape(report.model_name)}
    <span class="muted">v{html.escape(report.model_version or '?')}
    (@{html.escape(report.model_alias)})</span></h1>
<p class="muted">Window {html.escape(report.window_start)} &rarr; {html.escape(report.window_end)}
   &middot; generated {html.escape(report.generated_at)}</p>

<div class="verdict">
  <div>{_badge(report.status)} &nbsp; <strong>{html.escape(report.reason)}</strong></div>
  <p class="muted" style="margin-bottom:0">
    Consecutive alerting windows for this model version: {report.consecutive_alerts}
  </p>
</div>

<h2>Coverage</h2>
<div class="grid">{''.join(kpis)}</div>

<h2>Concept drift &mdash; has P(y|X) moved? {_badge(concept.status)}</h2>
<p class="muted">{html.escape(concept.detail)}</p>
<div class="scroll"><table>{concept_html}</table></div>
{'<h2>Per-class F1 (live)</h2><div class="scroll"><table><tr><th>Segment</th><th>F1</th></tr>'
 + per_class + '</table></div>' if per_class else ''}

<h2>Prediction drift &mdash; has P(y-hat) moved? {_badge(report.prediction_drift_status)}</h2>
<p class="muted">PSI of the predicted-segment mix against the training target
   distribution: <strong>{_fmt(report.prediction_drift_psi)}</strong></p>

<h2>Confidence drift {_badge(str(confidence.get('status', 'SKIPPED')))}</h2>
<p class="muted">{html.escape(str(confidence.get('reason', '')))}
   &middot; baseline {_fmt(confidence.get('baseline_mean_confidence'))}
   &rarr; live {_fmt(confidence.get('live_mean_confidence'))}</p>

<h2>Data drift &mdash; has P(X) moved?</h2>
<p class="muted">Population Stability Index per feature, against the baseline logged with this
   model version. Bands: &lt;0.10 stable, 0.10&ndash;0.25 moderate, &gt;0.25 significant.</p>
<div class="scroll"><table>
<tr><th>Feature</th><th>Type</th><th>PSI</th><th></th><th>JS distance</th><th>Status</th><th>Detail</th></tr>
{feature_rows or '<tr><td colspan="7" class="muted">No feature-level analysis in this run.</td></tr>'}
</table></div>

</body></html>
"""
