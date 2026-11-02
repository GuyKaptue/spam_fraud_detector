"""Tests for case_renderer.py (v4.4.8)."""
import pytest

from spam_fraud_detector.core.case_renderer import render_case_summary


def _make_fraud_casefile():
    return {
        "schema_version": "1.0.0",
        "case_type": "empirical",
        "persona": "fraud",
        "subject": {
            "subject_id": "F-012453",
            "full_name": "Priya Nakamura",
            "city": "Porto",
            "account_last4": "7361",
            "card_product": "Visa Credit",
            "merchant": "Tidewater Books",
            "merchant_category": "Online",
            "amount_eur": 1.03,
        },
        "score": 0.983,
        "threshold": 0.34,
        "y_true": 1,
        "y_pred": 1,
        "error_type": "true_positive",
        "top_features": [
            {
                "name": "V14",
                "value": -3.10,
                "shap_value": 0.42,
                "direction": "increases_fraud_risk",
                "explanation": "V14 is 3.1 standard deviations below the mean; on this dataset, that is a strong card-not-present fraud signal.",
            },
            {
                "name": "V12",
                "value": -0.98,
                "shap_value": 0.19,
                "direction": "increases_fraud_risk",
                "explanation": "V12 is below the mean; combined with V14, this is the textbook fraud-onset pattern.",
            },
            {
                "name": "Amount",
                "value": 1.03,
                "shap_value": -0.08,
                "direction": "decreases_fraud_risk",
                "explanation": "A €1.03 amount is unusually small; weakly pushes toward legitimate.",
            },
        ],
        "provenance": {"model": "XGBoost", "group": "unscaled"},
        "recommendation": {
            "action": "escalate",
            "priority": "high",
            "confidence": "high",
            "reason": "Score 0.983 is 2.9× the threshold, and V14/V12 match a known card-not-present fraud pattern.",
            "suggested_next_steps": [
                "Contact Priya Nakamura via the phone number on file.",
                "If unrecognized: block card ···7361, reverse the charge.",
            ],
            "estimated_cost_if_missed": "€1.03 + up to €5,000",
            "reviewer_notes_template": "Case F-012453 — Priya Nakamura. Action: escalate. Reviewed by: [NAME] on [DATE].",
        },
    }


def _make_spam_casefile():
    return {
        "schema_version": "1.0.0",
        "case_type": "empirical",
        "persona": "spam",
        "subject": {
            "subject_id": "S-000412",
            "full_name": "Mateo Alvarez",
            "sender_display": "Weekly Deals",
            "sender_email": "offers@promo.example",
            "subject_line": "Congratulations — claim your prize",
        },
        "score": 0.941,
        "threshold": 0.50,
        "y_true": 1,
        "y_pred": 1,
        "error_type": "true_positive",
        "top_tokens": [
            {
                "token": "click",
                "count": 3,
                "shap_value": 0.31,
                "direction": "increases_spam_probability",
                "explanation": "'click' appears 3 times; in the training data, messages with >=2 clicks are spam 89% of the time.",
            },
            {
                "token": "prize",
                "count": 1,
                "shap_value": 0.24,
                "direction": "increases_spam_probability",
                "explanation": "'prize' is a strong spam token.",
            },
        ],
        "provenance": {"model": "RandomForest", "group": "unscaled"},
        "recommendation": {
            "action": "block",
            "priority": "high",
            "confidence": "high",
            "reason": "Score 0.941 is 1.9× the threshold; multiple strong spam tokens present.",
            "suggested_next_steps": [
                "Quarantine the message to Mateo Alvarez's spam folder.",
                "Check sender domain offers@promo.example — add to blocklist if 10+ messages.",
            ],
            "estimated_cost_if_missed": "low — annoyance, repeated spam increases user churn",
            "reviewer_notes_template": "Case S-000412 — Mateo Alvarez. Action: block. Reviewed by: [NAME] on [DATE].",
        },
    }


def _make_anomaly_casefile():
    return {
        "schema_version": "1.0.0",
        "case_type": "empirical",
        "persona": "anomaly",
        "subject": {
            "subject_id": "A-013483",
            "full_name": "Talia Lindgren",
            "city": "Malmö",
            "account_last4": "2291",
            "card_product": "Visa Credit",
            "merchant": "Hôtel Bellevue",
            "merchant_category": "Travel",
            "amount_eur": 7200.0,
            "night_transaction": True,
            "cross_border": True,
        },
        "anomaly_score": 1.000,
        "supervised_score": 0.000,
        "disagreement_type": "novel",
        "threshold": 0.98,
        "y_true": None,
        "y_pred": 1,
        "top_features": [
            {
                "name": "Amount",
                "value": 7200.0,
                "shap_value": 0.34,
                "direction": "increases_anomaly_score",
                "explanation": "Amount is 3.4 standard deviations above the customer's typical amount.",
            },
            {
                "name": "V14",
                "value": -3.10,
                "shap_value": 0.29,
                "direction": "increases_anomaly_score",
                "explanation": "V14 is in the tail of the legitimate distribution.",
            },
        ],
        "provenance": {"model": "IsolationForest", "group": "anomaly"},
        "recommendation": {
            "action": "escalate",
            "priority": "high",
            "confidence": "medium",
            "reason": "Anomaly score 1.000 is high but supervised model scored 0.000 (below threshold). Novel anomaly.",
            "suggested_next_steps": [
                "Escalate to the fraud analyst team with a 'novel anomaly' tag.",
                "Contact Talia Lindgren via registered phone.",
            ],
            "estimated_cost_if_missed": "€7,200.00 + reputational risk",
            "reviewer_notes_template": "Case A-013483 — Talia Lindgren. Anomaly 1.000, supervised 0.000. Reviewed by: [NAME] on [DATE].",
        },
    }


def test_fraud_casefile_has_three_blocks():
    out = render_case_summary(_make_fraud_casefile())
    assert "CASE F-012453 — Priya Nakamura" in out
    assert "SHAP EXPLANATION — Why was Priya Nakamura flagged?" in out
    assert "RECOMMENDATION — What to do about Priya Nakamura's transaction?" in out
    assert "ACTION: ESCALATE" in out
    assert "PRIORITY: HIGH" in out


def test_spam_casefile_has_three_blocks():
    out = render_case_summary(_make_spam_casefile())
    assert "CASE S-000412 — Mateo Alvarez" in out
    assert "SHAP EXPLANATION — Why was Mateo Alvarez's message classified as spam?" in out
    assert "RECOMMENDATION — What to do about Mateo Alvarez's message?" in out
    assert "ACTION: BLOCK" in out
    assert "PRIORITY: HIGH" in out


def test_anomaly_casefile_has_three_blocks():
    out = render_case_summary(_make_anomaly_casefile())
    assert "CASE A-013483 — Talia Lindgren" in out
    assert "SHAP EXPLANATION — Why was Talia Lindgren flagged as anomalous?" in out
    assert "RECOMMENDATION — What to do about Talia Lindgren's anomaly?" in out
    assert "ACTION: ESCALATE" in out
    assert "PRIORITY: HIGH" in out


def test_fraud_feature_lines():
    out = render_case_summary(_make_fraud_casefile())
    assert "#1  V14" in out
    assert "SHAP +0.42" in out
    assert "↑↑↑" in out
    assert "(dominant)" in out
    # explanation wrapped below feature line
    assert "3.1 standard deviations" in out


def test_spam_token_lines():
    out = render_case_summary(_make_spam_casefile())
    assert '#1  "click"   (×3)' in out
    assert "SHAP +0.31" in out
    assert "89% of the time" in out


def test_anomaly_score_line():
    out = render_case_summary(_make_anomaly_casefile())
    assert "anomaly 1.000" in out
    assert "supervised 0.000" in out
    assert "NOVEL" in out


def test_line_width_bounded():
    out = render_case_summary(_make_fraud_casefile(), width=78)
    for line in out.split("\n"):
        # allow small tolerance for box-drawing chars (each counts as 1 cell)
        assert len(line) <= 78 + 10, f"line too long ({len(line)}): {line!r}"


def test_hypothetical_casefile():
    cf = _make_fraud_casefile()
    cf["case_type"] = "hypothetical"
    cf["scenario_id"] = "H-FRAUD-001"
    cf["scenario_description"] = "High-amount cross-border night transaction"
    out = render_case_summary(cf)
    assert "(hypothetical)" in out
    assert "Would Priya Nakamura be flagged?" in out
    assert "What to do about this hypothetical customer?" in out


def test_dispatch_unknown_persona_raises():
    cf = _make_fraud_casefile()
    cf["persona"] = "not_a_persona"
    with pytest.raises(ValueError):
        render_case_summary(cf)


def test_all_feature_lines_present():
    out = render_case_summary(_make_fraud_casefile())
    # 3 features → 3 #<i> lines plus wrapped explanations
    assert out.count("#1  V14") == 1
    assert out.count("#2  V12") == 1
    assert out.count("#3  Amount") == 1
