"""Render persona case files as human-readable reviewer summaries.

Produces a three-block text summary per case:

    CASE <id> — <name>
    <model> (<group>) · <score> · <threshold> · <context line>
    <context line: transaction / sender+subject / anomaly flags>
    ══════════════════════════════════════════════════════════════════════

      SHAP EXPLANATION — Why was <name> flagged?

      #1  V14     = -3.10          SHAP +0.42  ↑↑↑  (dominant)
          V14 is 3.1 standard deviations below the mean; on this
          dataset, that is a strong fraud signal.

      #2  V12     = -0.98          SHAP +0.19  ↑↑  (strong)
          V12 is below the mean; combined with V14, this is the
          textbook fraud-onset pattern.

      (24 features collapsed — see JSON for full list)

    ────────────────────────────────────────────────────────────────

      RECOMMENDATION — What to do about <name>'s transaction?

      ACTION: ESCALATE                                    PRIORITY: HIGH
      CONFIDENCE: HIGH

      Why: Score 0.983 is 2.9× the threshold, and V14/V12 match a
           known card-not-present fraud pattern. ...

      Next steps:
        1. Contact Priya Nakamura via the phone number on file
           (not SMS, not email) to confirm the €1.03 transaction.
        2. If unrecognized: block card ···7361, reverse the charge,
           open investigation ticket.

      Estimated cost if missed: €1.03 + up to €5,000 ...

      Reviewer notes:
        Case F-012453 — Priya Nakamura (Visa Credit ···7361, Porto).
        Reviewed by: [NAME] on [DATE]. Outcome: [PENDING ...].

    ══════════════════════════════════════════════════════════════════════

Pure Python — no dependency on rich, tabulate, or any external library.
Uses textwrap and Unicode box-drawing characters.
"""
from __future__ import annotations

import textwrap


BOX_DOUBLE = "═"
BOX_SINGLE = "─"


def _wrap(text: str, width: int, indent: int = 4) -> str:
    """Wrap ``text`` to ``width`` columns with a hanging indent."""
    if not text:
        return ""
    subsequent = " " * indent
    return textwrap.fill(text, width=max(20, width - indent),
                         subsequent_indent=subsequent)


def _arrow(shap_value: float) -> str:
    """Return an arrow character for SHAP magnitude/direction."""
    mag = abs(shap_value)
    if mag >= 0.25:
        base = "↑↑↑"
    elif mag >= 0.10:
        base = "↑↑"
    elif mag >= 0.02:
        base = "↑"
    else:
        base = "·"
    if shap_value < 0:
        base = base.replace("↑", "↓")
    return base


def _strength_label(shap_value: float) -> str:
    """Return a strength label for SHAP magnitude."""
    mag = abs(shap_value)
    if mag >= 0.25:
        return "dominant"
    if mag >= 0.10:
        return "strong"
    if mag >= 0.02:
        return "moderate"
    return "weak"


def _fmt_value(value: float, persona: str) -> str:
    """Format a feature value for display."""
    if persona == "spam":
        # tokens show count
        return f"×{int(round(value))}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:+.2f}"


def _fmt_score_line(casefile: dict) -> str:
    """Build the score line for the CASE header."""
    persona = casefile.get("persona", "fraud")
    score = casefile.get("score")
    threshold = casefile.get("threshold", 0.5) or 0.5
    error_type = casefile.get("error_type")
    disagreement = casefile.get("disagreement_type")
    anomaly_score = casefile.get("anomaly_score")
    supervised_score = casefile.get("supervised_score")
    raw_score = casefile.get("raw_score")

    if persona == "anomaly":
        raw_part = f" (raw {raw_score:+.3f})" if raw_score is not None else ""
        sup_part = f" · supervised {supervised_score:.3f}" if supervised_score is not None else ""
        disc = disagreement.upper() if disagreement else ""
        a = anomaly_score if anomaly_score is not None else 0.0
        return f"anomaly {a:.3f}{raw_part}{sup_part} · {disc}"
    score_str = f"{score:.3f}" if score is not None else "n/a"
    et = error_type.upper() if error_type else ""
    return f"score {score_str} · threshold {threshold:.2f} · {et}"


def _fmt_context_line(casefile: dict) -> str:
    """Build the person-specific context line."""
    persona = casefile.get("persona", "fraud")
    subject = casefile.get("subject", {}) or {}

    if persona == "spam":
        sender_display = subject.get("sender_display", "?")
        sender_email = subject.get("sender_email", "?")
        subject_line = subject.get("subject_line", "")
        return f"{sender_display} <{sender_email}> · \"{subject_line}\""

    # fraud and anomaly share transaction context
    amount = subject.get("amount_eur", 0.0)
    try:
        amount_str = f"€{float(amount):.2f}"
    except (TypeError, ValueError):
        amount_str = f"€{amount}"
    city = subject.get("city", "?")
    card_product = subject.get("card_product", "?")
    last4 = subject.get("account_last4", "?")
    extra = []
    if persona == "anomaly":
        if subject.get("night_transaction"):
            extra.append("night txn")
        if subject.get("cross_border"):
            extra.append("cross-border")
    ctx = f"{amount_str} {city} · {card_product} ···{last4}"
    if extra:
        ctx = f"{ctx} ({', '.join(extra)})"
    return ctx


def _top_feature_lines(top_features: list[dict], persona: str,
                       width: int, max_features: int = 5) -> list[str]:
    """Build the ranked feature lines for the explanation block."""
    lines = []
    top_features = top_features or []
    total = len(top_features)
    shown = top_features[:max_features]
    for i, f in enumerate(shown, start=1):
        name = f.get("name", f.get("token", "?"))
        value = f.get("value", 0.0)
        sv = f.get("shap_value", 0.0)
        arrow = _arrow(sv)
        strength = _strength_label(sv)
        value_str = _fmt_value(value, persona)
        if persona == "spam":
            count = f.get("count", 0)
            line = (f"#{i}  \"{name}\"   (×{count})  SHAP {sv:+.2f}  "
                    f"{arrow}  ({strength})")
        else:
            line = (f"#{i}  {name:<10} = {value_str:>12}  SHAP {sv:+.2f}  "
                    f"{arrow}  ({strength})")
        lines.append(line)
        explanation = f.get("explanation", "")
        if explanation:
            lines.append(_wrap(explanation, width, indent=6))
    if total > max_features:
        lines.append(f"({total - max_features} features collapsed — see JSON for full list)")
    return lines


def _recommendation_block(casefile: dict, width: int) -> list[str]:
    """Build the recommendation block."""
    rec = casefile.get("recommendation", {}) or {}
    action = rec.get("action", "?")
    priority = rec.get("priority", "?")
    confidence = rec.get("confidence", "?")
    reason = rec.get("reason", "")
    next_steps = rec.get("suggested_next_steps", []) or []
    cost = rec.get("estimated_cost_if_missed", "")
    template = rec.get("reviewer_notes_template", "")

    lines = []
    lines.append("")
    lines.append(BOX_SINGLE * width)
    lines.append("")
    case_type = casefile.get("case_type", "empirical")
    full_name = casefile.get("subject", {}).get("full_name", "?")
    if case_type == "hypothetical":
        header = f"RECOMMENDATION — What to do about {_noun(casefile)}?"
    else:
        header = f"RECOMMENDATION — What to do about {full_name}'s {_noun(casefile)}?"
    lines.append(f"  {header}")

    # ACTION / PRIORITY line
    action_part = f"ACTION: {action.upper()}"
    priority_part = f"PRIORITY: {priority.upper()}"
    gap = width - len(action_part) - len(priority_part) - 6
    if gap < 2:
        gap = 2
    lines.append(f"  {action_part}{' ' * gap}{priority_part}")
    lines.append(f"  CONFIDENCE: {confidence.upper()}")
    lines.append("")

    if reason:
        lines.append(_wrap(f"Why: {reason}", width))
        lines.append("")

    if next_steps:
        lines.append("  Next steps:")
        for i, step in enumerate(next_steps, start=1):
            lines.append(f"    {i}. {step}")
        lines.append("")

    if cost:
        lines.append(f"  Estimated cost if missed: {cost}")
        lines.append("")

    if template:
        lines.append("  Reviewer notes:")
        lines.append(_wrap(template, width, indent=4))

    return lines


def _noun(casefile: dict) -> str:
    """Return the noun for the recommendation header."""
    case_type = casefile.get("case_type", "empirical")
    if case_type == "hypothetical":
        return "this hypothetical customer"
    persona = casefile.get("persona", "fraud")
    if persona == "spam":
        return "message"
    if persona == "anomaly":
        return "anomaly"
    return "transaction"


def _render_fraud(casefile: dict, width: int) -> list[str]:
    """Render a fraud case file."""
    subject = casefile.get("subject", {}) or {}
    name = subject.get("full_name", "?")
    subject_id = subject.get("subject_id", "?")
    case_type = casefile.get("case_type", "empirical")
    model = casefile.get("provenance", {}).get("model", "?")
    group = casefile.get("provenance", {}).get("group", "?")

    lines = []
    lines.append(BOX_DOUBLE * width)
    if case_type == "hypothetical":
        lines.append(f"CASE {subject_id} (hypothetical) — {name}")
    else:
        lines.append(f"CASE {subject_id} — {name}")
    lines.append(f"{model} ({group}) · {_fmt_score_line(casefile)}")
    lines.append(f"  {_fmt_context_line(casefile)}")
    lines.append(BOX_DOUBLE * width)
    lines.append("")

    # Explanation block
    title_name = name
    if case_type == "hypothetical":
        header = f"SHAP EXPLANATION — Would {title_name} be flagged?"
    else:
        header = f"SHAP EXPLANATION — Why was {title_name} flagged?"
    lines.append(f"  {header}")
    lines.append("")
    top = casefile.get("top_features", []) or casefile.get("top_tokens", []) or []
    lines.extend(_top_feature_lines(top, "fraud", width))
    lines.append("")
    lines.append(BOX_SINGLE * width)
    lines.append("")
    lines.extend(_recommendation_block(casefile, width))
    lines.append("")
    lines.append(BOX_DOUBLE * width)
    return lines


def _render_spam(casefile: dict, width: int) -> list[str]:
    """Render a spam case file."""
    subject = casefile.get("subject", {}) or {}
    name = subject.get("full_name", "?")
    subject_id = subject.get("subject_id", "?")
    case_type = casefile.get("case_type", "empirical")
    model = casefile.get("provenance", {}).get("model", "?")
    group = casefile.get("provenance", {}).get("group", "?")

    lines = []
    lines.append(BOX_DOUBLE * width)
    if case_type == "hypothetical":
        lines.append(f"CASE {subject_id} (hypothetical) — {name}")
    else:
        lines.append(f"CASE {subject_id} — {name}")
    lines.append(f"{model} ({group}) · {_fmt_score_line(casefile)}")
    lines.append(f"  {_fmt_context_line(casefile)}")
    lines.append(BOX_DOUBLE * width)
    lines.append("")

    title_name = name
    if case_type == "hypothetical":
        header = f"SHAP EXPLANATION — Would {title_name}'s message be classified as spam?"
    else:
        header = f"SHAP EXPLANATION — Why was {title_name}'s message classified as spam?"
    lines.append(f"  {header}")
    lines.append("")
    top = casefile.get("top_features", []) or casefile.get("top_tokens", []) or []
    lines.extend(_top_feature_lines(top, "spam", width))
    lines.append("")
    lines.append(BOX_SINGLE * width)
    lines.append("")
    lines.extend(_recommendation_block(casefile, width))
    lines.append("")
    lines.append(BOX_DOUBLE * width)
    return lines


def _render_anomaly(casefile: dict, width: int) -> list[str]:
    """Render an anomaly case file."""
    subject = casefile.get("subject", {}) or {}
    name = subject.get("full_name", "?")
    subject_id = subject.get("subject_id", "?")
    case_type = casefile.get("case_type", "empirical")
    model = casefile.get("provenance", {}).get("model", "?")
    group = casefile.get("provenance", {}).get("group", "?")

    lines = []
    lines.append(BOX_DOUBLE * width)
    if case_type == "hypothetical":
        lines.append(f"CASE {subject_id} (hypothetical) — {name}")
    else:
        lines.append(f"CASE {subject_id} — {name}")
    lines.append(f"{model} ({group}) · {_fmt_score_line(casefile)}")
    lines.append(f"  {_fmt_context_line(casefile)}")
    lines.append(BOX_DOUBLE * width)
    lines.append("")

    title_name = name
    if case_type == "hypothetical":
        header = f"SHAP EXPLANATION — Would {title_name} be flagged as anomalous?"
    else:
        header = f"SHAP EXPLANATION — Why was {title_name} flagged as anomalous?"
    lines.append(f"  {header}")
    lines.append("")
    top = casefile.get("top_features", []) or casefile.get("top_tokens", []) or []
    lines.extend(_top_feature_lines(top, "anomaly", width))
    lines.append("")
    lines.append(BOX_SINGLE * width)
    lines.append("")
    lines.extend(_recommendation_block(casefile, width))
    lines.append("")
    lines.append(BOX_DOUBLE * width)
    return lines


def render_case_summary(casefile: dict, width: int = 78) -> str:
    """Render a persona case file as a reviewer-facing multi-block summary.

    Produces three sections per case:

        CASE <id> — <name>          (title block)
        SHAP EXPLANATION — ...      (feature ranking)
        RECOMMENDATION — ...        (action / priority / next steps)

    Dispatches on ``casefile["persona"]`` which must be one of
    ``"fraud"``, ``"spam"``, ``"anomaly"``.

    Returns the rendered string. No ``print`` calls — the caller decides
    where to write.
    """
    persona = casefile.get("persona", "fraud")
    if persona not in ("fraud", "spam", "anomaly"):
        raise ValueError(f"unknown persona: {persona}")
    if persona == "spam":
        return "\n".join(_render_spam(casefile, width))
    if persona == "anomaly":
        return "\n".join(_render_anomaly(casefile, width))
    # default to fraud
    return "\n".join(_render_fraud(casefile, width))


__all__ = ["render_case_summary"]
