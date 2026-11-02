"""Recommendation engine for persona case files.

Given a case file (empirical or hypothetical) and a persona, produce a
``recommendation`` block with:

    action: "block" | "escalate" | "review" | "monitor" | "clear"
    priority: "high" | "medium" | "low"
    reason: str
    suggested_next_steps: list[str]
    estimated_cost_if_missed: str
    confidence: "high" | "medium" | "low"
    reviewer_notes_template: str

The rules are persona-specific and documented in FRAMEWORK.md.
"""
from __future__ import annotations

from typing import Any


def recommend_fraud(casefile: dict) -> dict:
    """Recommendation for a fraud case file (empirical or hypothetical)."""
    score = float(casefile.get("score", 0.0))
    threshold = float(casefile.get("threshold", 0.5))
    ratio = score / threshold if threshold > 0 else 0.0
    error_type = casefile.get("error_type")
    case_type = casefile.get("case_type", "empirical")
    subject = casefile.get("subject", {}) or {}
    amount = float(subject.get("amount_eur", 0.0) or 0.0)
    city = subject.get("city", "?")
    merchant = subject.get("merchant", "?")
    top = casefile.get("top_features", [])[:3]
    top_str = ", ".join(f"{f['name']} ({f['shap_value']:+.2f})" for f in top)

    is_hypo = case_type == "hypothetical"

    # --- Empirical cases ---
    if error_type == "true_positive" and ratio >= 2.0:
        action, priority, confidence = "escalate", "high", "high"
        reason = (
            f"Score {score:.3f} is {ratio:.1f}× the threshold, and "
            f"{top[0]['name'] if top else 'top features'} matches a known fraud pattern."
        )
        next_steps = [
            f"Contact {subject.get('full_name', '?')} via phone (not SMS/email) "
            f"to confirm the €{amount:.2f} transaction.",
            f"If unrecognized: block card ···{subject.get('account_last4', '?')}, "
            "reverse the charge, open investigation ticket.",
            f"If confirmed: add customer-specific allowlist rule for {top_str}.",
            f"Check for related transactions from {merchant} in last 72h.",
        ]
        cost = (
            f"€{amount:.2f} (this transaction) + up to €{max(amount * 5, 5000):.0f} "
            "(typical follow-up fraud on the same card after a test transaction)"
        )
    elif error_type == "true_positive":
        action, priority, confidence = "review", "medium", "medium"
        reason = f"Score {score:.3f} exceeds threshold {threshold:.2f} ({ratio:.1f}×); standard manual review."
        next_steps = [
            f"Review {subject.get('full_name', '?')}'s transaction history for related activity.",
            f"Cross-check {merchant} against the known-merchant allowlist.",
            "Confirm via the customer's registered contact method.",
        ]
        cost = f"€{amount:.2f} + follow-up risk"
    elif error_type == "false_positive":
        action, priority, confidence = "review", "low", "medium"
        reason = f"Flagged at {score:.3f} but ground truth is legitimate — analyze why the model false-positived."
        next_steps = [
            f"Confirm {subject.get('full_name', '?')}'s transaction is legitimate.",
            f"Identify whether {top_str} are false-positive-prone features.",
            "Consider adding this row to the training set as a labeled legitimate example.",
        ]
        cost = "€0 (legitimate transaction) but reputational cost of blocking a real customer"
    elif error_type == "false_negative":
        action, priority, confidence = "escalate", "high", "high"
        reason = f"Score {score:.3f} is BELOW threshold {threshold:.2f}, but ground truth is fraud — the model missed it."
        next_steps = [
            f"Investigate why {top_str} did not trigger the model.",
            "Add this row to the training set with the fraud label.",
            f"Consider lowering the threshold or adjusting the feature engineering for {top[0]['name'] if top else 'top features'}.",
        ]
        cost = f"€{amount:.2f} + risk of model deploying similar misses in production"
    else:
        action, priority, confidence = "review", "medium", "medium"
        reason = f"Score {score:.3f} (threshold {threshold:.2f}); standard review."
        next_steps = ["Confirm transaction with customer.", "Log outcome."]
        cost = "n/a"

    # --- Hypothetical cases: recommendation is about model validation ---
    if is_hypo:
        action = "review"
        priority = "medium"
        confidence = "high"
        reason = (
            f"Hypothetical scenario '{casefile.get('scenario_description', '?')}' scored "
            f"{score:.3f}. Verify the model behaves as expected for this feature combination."
        )
        next_steps = [
            f"Confirm with the analytics team that '{casefile.get('scenario_description', '?')}' is representative "
            "of real fraud patterns.",
            "If it matches a known novel pattern, add it as a labeled training example.",
            "Add a monitoring rule if this scenario is production-relevant.",
            "Re-run the scenario after any model change to verify stability.",
        ]
        cost = "n/a — hypothetical customer, not a real one"

    return {
        "action": action,
        "priority": priority,
        "reason": reason,
        "suggested_next_steps": next_steps,
        "estimated_cost_if_missed": cost,
        "confidence": confidence,
        "reviewer_notes_template": (
            f"Case {subject.get('subject_id', '?')} — "
            f"{subject.get('full_name', '?')}. "
            f"Score {score:.3f} (threshold {threshold:.2f}). "
            f"Top features: {top_str}. "
            f"Recommended action: {action}. "
            f"Reviewed by: [NAME] on [DATE]. Outcome: [PENDING / CONFIRMED / OTHER]."
        ),
    }


def recommend_spam(casefile: dict) -> dict:
    """Recommendation for a spam case file."""
    score = float(casefile.get("score", 0.0))
    threshold = float(casefile.get("threshold", 0.5))
    ratio = score / threshold if threshold > 0 else 0.0
    case_type = casefile.get("case_type", "empirical")
    subject = casefile.get("subject", {}) or {}
    sender = subject.get("sender_email", "?")
    subject_line = subject.get("subject_line", "?")
    top = casefile.get("top_tokens", [])[:3]
    top_str = ", ".join(
        f"{t.get('token', t.get('name', '?'))} (×{t.get('count', 1)}, {t.get('shap_value', 0):+.2f})"
        for t in top
    )

    if case_type == "hypothetical":
        return {
            "action": "review",
            "priority": "medium",
            "reason": (
                f"Hypothetical message '{casefile.get('scenario_description', '?')}' scored {score:.3f}. "
                "Verify the spam filter's sensitivity to this token pattern."
            ),
            "suggested_next_steps": [
                f"Search training set for messages with the pattern {top_str}; if < 20 exist, add examples.",
                "Test variations to ensure score degrades gracefully.",
                "Add to the spam filter regression suite.",
            ],
            "estimated_cost_if_missed": "n/a — hypothetical message",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Hypothetical case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')}. "
                f"Score {score:.3f}. Top tokens: {top_str}. Reviewed by: [NAME] on [DATE]."
            ),
        }

    if ratio >= 1.8:
        return {
            "action": "block",
            "priority": "high",
            "reason": f"Score {score:.3f} is {ratio:.1f}× the threshold; multiple strong spam tokens present.",
            "suggested_next_steps": [
                f"Quarantine the message from {subject.get('full_name', '?')}'s inbox.",
                f"Check sender domain {sender} — if it matches 10+ messages in last 7d, add to blocklist.",
                "Do not bounce the message (confirms address validity to spammers).",
                "Downgrade to 'review' if user reports as legitimate within 7 days.",
            ],
            "estimated_cost_if_missed": "low — annoyance, but repeated spam increases user churn",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')} "
                f"received '{subject_line}' from {sender}. Score {score:.3f}. "
                f"Top tokens: {top_str}. Action: block. Reviewed by: [NAME] on [DATE]."
            ),
        }

    return {
        "action": "monitor",
        "priority": "low",
        "reason": f"Score {score:.3f} is near threshold {threshold:.2f}; monitor but do not block.",
        "suggested_next_steps": [
            "Add sender to watchlist for 14 days.",
            "Review if 2+ more messages from same sender are flagged.",
        ],
        "estimated_cost_if_missed": "very low",
        "confidence": "medium",
        "reviewer_notes_template": (
            f"Case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')} "
            f"received '{subject_line}' from {sender}. Score {score:.3f} near threshold {threshold:.2f}. "
            f"Action: monitor. Reviewed by: [NAME] on [DATE]."
        ),
    }


def recommend_anomaly(casefile: dict) -> dict:
    """Recommendation for an anomaly case file."""
    disagreement = casefile.get("disagreement_type", "novel")
    case_type = casefile.get("case_type", "empirical")
    anomaly_score = float(casefile.get("anomaly_score", 0.0))
    supervised_score = float(casefile.get("supervised_score", 0.0))
    subject = casefile.get("subject", {}) or {}
    amount = float(subject.get("amount_eur", 0.0) or 0.0)
    city = subject.get("city", "?")
    top = casefile.get("top_features", [])[:3]
    top_str = ", ".join(f"{f['name']} ({f['shap_value']:+.2f})" for f in top)

    if case_type == "hypothetical":
        return {
            "action": "review",
            "priority": "medium",
            "reason": (
                f"Hypothetical scenario '{casefile.get('scenario_description', '?')}' — "
                f"anomaly score {anomaly_score:.3f}, supervised score {supervised_score:.3f}. "
                f"Disagreement type: {disagreement}. Verify model behavior."
            ),
            "suggested_next_steps": [
                "Confirm this scenario matches a known novel-anomaly pattern.",
                "If so, add to the training set as a labeled example.",
                "Add to anomaly-detector regression suite.",
            ],
            "estimated_cost_if_missed": "n/a — hypothetical",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Hypothetical case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')}. "
                f"Anomaly {anomaly_score:.3f}, supervised {supervised_score:.3f}. "
                f"Disagreement: {disagreement}. Reviewed by: [NAME] on [DATE]."
            ),
        }

    if disagreement == "novel":
        return {
            "action": "escalate",
            "priority": "high",
            "reason": (
                f"Anomaly score {anomaly_score:.3f} is high but supervised model scored "
                f"{supervised_score:.3f} (below threshold). This is a novel anomaly — "
                "the supervised model missed it; a human must review."
            ),
            "suggested_next_steps": [
                "Escalate to the fraud analyst team with a 'novel anomaly' tag.",
                f"Contact {subject.get('full_name', '?')} via registered phone to confirm the €{amount:.2f} transaction.",
                f"If unrecognized: block card ···{subject.get('account_last4', '?')}, reverse, open 'novel fraud pattern' ticket.",
                "If confirmed: add customer-specific allowlist, mark as 'legitimate high-value' for anomaly detector retraining.",
            ],
            "estimated_cost_if_missed": (
                f"€{amount:.2f} + up to €{max(amount * 2, 15000):.0f} (typical follow-up) "
                "+ reputational risk (supervised model missed it)"
            ),
            "confidence": "medium",
            "reviewer_notes_template": (
                f"Case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')}. "
                f"Anomaly {anomaly_score:.3f} but supervised {supervised_score:.3f}. "
                f"Novel anomaly. Top features: {top_str}. Action: escalate. Reviewed by: [NAME] on [DATE]."
            ),
        }
    elif disagreement == "sophisticated":
        return {
            "action": "review",
            "priority": "high",
            "reason": (
                f"Supervised score {supervised_score:.3f} is high but anomaly score {anomaly_score:.3f} is low — "
                "this looks like a sophisticated attack that mimics normal behavior."
            ),
            "suggested_next_steps": [
                "Flag for senior analyst review (sophisticated fraud ring pattern).",
                f"Cross-reference {city} with known mule account locations.",
                "Check for related customer accounts with similar patterns.",
            ],
            "estimated_cost_if_missed": f"€{amount:.2f} and risk of organized fraud ring undetected",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')}. "
                f"Supervised {supervised_score:.3f} but anomaly {anomaly_score:.3f}. "
                f"Sophisticated pattern. Action: review. Reviewed by: [NAME] on [DATE]."
            ),
        }
    elif disagreement == "agree_high":
        return {
            "action": "escalate",
            "priority": "high",
            "reason": (
                f"Both anomaly ({anomaly_score:.3f}) and supervised ({supervised_score:.3f}) scores are high. "
                "This is a high-confidence fraud flag."
            ),
            "suggested_next_steps": [
                f"Immediate escalation. Contact {subject.get('full_name', '?')} via phone.",
                "Consider pre-emptive card block if customer cannot be reached within 1 hour.",
                "Preserve all logs for potential law enforcement referral.",
            ],
            "estimated_cost_if_missed": f"€{amount:.2f} and possible organized fraud",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Case {subject.get('subject_id', '?')} — {subject.get('full_name', '?')}. "
                f"Both models agree high risk. Action: escalate. Reviewed by: [NAME] on [DATE]."
            ),
        }
    else:  # agree_low
        return {
            "action": "clear",
            "priority": "low",
            "reason": f"Both anomaly ({anomaly_score:.3f}) and supervised ({supervised_score:.3f}) scores are low. No fraud evidence.",
            "suggested_next_steps": ["Close as legitimate.", "Log outcome for retraining."],
            "estimated_cost_if_missed": "n/a — low risk",
            "confidence": "high",
            "reviewer_notes_template": (
                f"Case {subject.get('subject_id', '?')} — both models low. Cleared. "
                f"Reviewed by: [NAME] on [DATE]."
            ),
        }


def recommend(persona: str, casefile: dict) -> dict:
    """Dispatch to the persona-specific recommender."""
    if persona == "fraud":
        return recommend_fraud(casefile)
    elif persona == "spam":
        return recommend_spam(casefile)
    elif persona == "anomaly":
        return recommend_anomaly(casefile)
    raise ValueError(f"unknown persona: {persona}")


__all__ = ["recommend_fraud", "recommend_spam", "recommend_anomaly", "recommend"]
