"""Tests for the recommendations engine (v4.4.6)."""
import pytest

from spam_fraud_detector.core.recommendations import (
    recommend,
    recommend_anomaly,
    recommend_fraud,
    recommend_spam,
)

VALID_ACTIONS = {"block", "escalate", "review", "monitor", "clear"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_CONFIDENCES = {"high", "medium", "low"}


def _assert_valid(rec):
    assert rec["action"] in VALID_ACTIONS
    assert rec["priority"] in VALID_PRIORITIES
    assert rec["confidence"] in VALID_CONFIDENCES
    assert isinstance(rec["reason"], str) and rec["reason"]
    assert isinstance(rec["suggested_next_steps"], list) and rec["suggested_next_steps"]
    assert isinstance(rec["reviewer_notes_template"], str)
    assert "[NAME]" in rec["reviewer_notes_template"]
    assert "[DATE]" in rec["reviewer_notes_template"]


def test_fraud_true_positive_high_ratio_escalate():
    cf = {"persona": "fraud", "case_type": "empirical", "score": 0.983, "threshold": 0.34,
          "error_type": "true_positive", "subject": {"full_name": "Priya", "amount_eur": 1.03,
          "account_last4": "7361"}, "top_features": []}
    rec = recommend_fraud(cf)
    _assert_valid(rec)
    assert rec["action"] == "escalate"
    assert rec["priority"] == "high"


def test_fraud_false_negative_escalate():
    cf = {"persona": "fraud", "case_type": "empirical", "score": 0.2, "threshold": 0.5,
          "error_type": "false_negative", "subject": {"full_name": "X", "amount_eur": 10.0},
          "top_features": []}
    rec = recommend_fraud(cf)
    _assert_valid(rec)
    assert rec["action"] == "escalate"
    assert rec["priority"] == "high"


def test_fraud_false_positive_review_low():
    cf = {"persona": "fraud", "case_type": "empirical", "score": 0.6, "threshold": 0.5,
          "error_type": "false_positive", "subject": {"full_name": "X", "amount_eur": 5.0},
          "top_features": []}
    rec = recommend_fraud(cf)
    _assert_valid(rec)
    assert rec["action"] == "review"
    assert rec["priority"] == "low"


def test_fraud_hypothetical_review_medium():
    cf = {"persona": "fraud", "case_type": "hypothetical", "score": 0.874, "threshold": 0.34,
          "scenario_description": "night transaction", "subject": {"full_name": "Priya"},
          "top_features": []}
    rec = recommend_fraud(cf)
    _assert_valid(rec)
    assert rec["action"] == "review"
    assert rec["priority"] == "medium"
    assert rec["confidence"] == "high"
    assert rec["estimated_cost_if_missed"] == "n/a — hypothetical customer, not a real one"


def test_spam_high_ratio_block():
    cf = {"persona": "spam", "case_type": "empirical", "score": 0.941, "threshold": 0.5,
          "subject": {"full_name": "Mateo", "sender_email": "offers@promo.example",
          "subject_line": "claim your prize"}, "top_tokens": []}
    rec = recommend_spam(cf)
    _assert_valid(rec)
    assert rec["action"] == "block"
    assert rec["priority"] == "high"


def test_spam_near_threshold_monitor():
    cf = {"persona": "spam", "case_type": "empirical", "score": 0.55, "threshold": 0.5,
          "subject": {"full_name": "Mateo", "sender_email": "a@b.c", "subject_line": "hi"},
          "top_tokens": []}
    rec = recommend_spam(cf)
    _assert_valid(rec)
    assert rec["action"] == "monitor"
    assert rec["priority"] == "low"


def test_spam_hypothetical_review():
    cf = {"persona": "spam", "case_type": "hypothetical", "score": 0.978, "threshold": 0.5,
          "scenario_description": "promo message", "subject": {"full_name": "Hattie"},
          "top_tokens": []}
    rec = recommend_spam(cf)
    _assert_valid(rec)
    assert rec["action"] == "review"
    assert rec["estimated_cost_if_missed"] == "n/a — hypothetical message"


def test_anomaly_novel_escalate():
    cf = {"persona": "anomaly", "case_type": "empirical", "anomaly_score": 0.91,
          "supervised_score": 0.03, "disagreement_type": "novel",
          "subject": {"full_name": "Talia", "amount_eur": 7200.0, "account_last4": "2291"},
          "top_features": []}
    rec = recommend_anomaly(cf)
    _assert_valid(rec)
    assert rec["action"] == "escalate"
    assert rec["priority"] == "high"


def test_anomaly_sophisticated_review_high():
    cf = {"persona": "anomaly", "case_type": "empirical", "anomaly_score": 0.2,
          "supervised_score": 0.8, "disagreement_type": "sophisticated",
          "subject": {"full_name": "X", "amount_eur": 500.0}, "top_features": []}
    rec = recommend_anomaly(cf)
    _assert_valid(rec)
    assert rec["action"] == "review"
    assert rec["priority"] == "high"


def test_anomaly_agree_high_escalate():
    cf = {"persona": "anomaly", "case_type": "empirical", "anomaly_score": 0.9,
          "supervised_score": 0.874, "disagreement_type": "agree_high",
          "subject": {"full_name": "Gustav", "amount_eur": 7200.0}, "top_features": []}
    rec = recommend_anomaly(cf)
    _assert_valid(rec)
    assert rec["action"] == "escalate"
    assert rec["priority"] == "high"


def test_anomaly_agree_low_clear():
    cf = {"persona": "anomaly", "case_type": "empirical", "anomaly_score": 0.1,
          "supervised_score": 0.05, "disagreement_type": "agree_low",
          "subject": {"full_name": "Rosa", "amount_eur": 15.0}, "top_features": []}
    rec = recommend_anomaly(cf)
    _assert_valid(rec)
    assert rec["action"] == "clear"
    assert rec["priority"] == "low"


def test_anomaly_hypothetical_review():
    cf = {"persona": "anomaly", "case_type": "hypothetical", "anomaly_score": 0.93,
          "supervised_score": 0.874, "disagreement_type": "agree_high",
          "scenario_description": "night transaction", "subject": {"full_name": "Gustav"},
          "top_features": []}
    rec = recommend_anomaly(cf)
    _assert_valid(rec)
    assert rec["action"] == "review"
    assert rec["confidence"] == "high"


def test_recommend_dispatch():
    cf = {"persona": "fraud", "case_type": "empirical", "score": 0.9, "threshold": 0.5,
          "error_type": "true_positive", "subject": {"full_name": "X", "amount_eur": 1.0},
          "top_features": []}
    assert recommend("fraud", cf)["action"] == recommend_fraud(cf)["action"]
    cf2 = {"persona": "spam", "case_type": "empirical", "score": 0.9, "threshold": 0.5,
           "subject": {"full_name": "X", "sender_email": "a@b.c", "subject_line": "x"},
           "top_tokens": []}
    assert recommend("spam", cf2)["action"] == recommend_spam(cf2)["action"]
    cf3 = {"persona": "anomaly", "case_type": "empirical", "anomaly_score": 0.9,
           "supervised_score": 0.1, "disagreement_type": "novel",
           "subject": {"full_name": "X", "amount_eur": 1.0}, "top_features": []}
    assert recommend("anomaly", cf3)["action"] == recommend_anomaly(cf3)["action"]
    with pytest.raises(ValueError):
        recommend("not_a_persona", {})


def test_all_fields_present():
    cf = {"persona": "fraud", "case_type": "empirical", "score": 0.9, "threshold": 0.5,
          "error_type": "true_positive", "subject": {"full_name": "X", "amount_eur": 1.0},
          "top_features": []}
    rec = recommend_fraud(cf)
    for f in ["action", "priority", "reason", "suggested_next_steps",
              "estimated_cost_if_missed", "confidence", "reviewer_notes_template"]:
        assert f in rec
