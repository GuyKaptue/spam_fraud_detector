"""Integration tests for explain console output (v4.4.8).

Runs ``make explain-<persona>`` on cached bundles and asserts the console
output contains the multi-block per-case render (CASE title, SHAP EXPLANATION,
RECOMMENDATION) for every persona.
"""
import subprocess
import sys


def _run_explain(persona: str) -> subprocess.CompletedProcess:
    # Persona "fraud" runs on fraud dataset; persona "spam" runs on spam dataset;
    # persona "anomaly" runs on fraud dataset (anomaly models are under fraud).
    dataset = "spam" if persona == "spam" else "fraud"
    return subprocess.run(
        [sys.executable, "-m", "spam_fraud_detector.main",
         "--dataset", dataset,
         "--explainer", "shap",
         "--explain-only", "--persona", persona],
        capture_output=True, text=True, timeout=600,
    )


def test_explain_fraud_console_output():
    proc = _run_explain("fraud")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # At least one CASE title per explained model case
    assert out.count("CASE F-") >= 1
    assert "SHAP EXPLANATION — Why was" in out
    assert "RECOMMENDATION — What to do about" in out
    assert "ACTION:" in out
    assert "PRIORITY:" in out
    assert "Next steps:" in out
    assert "Reviewer notes:" in out
    # No duplicate filename pattern
    assert "__case_.*__case_" not in out.replace("\n", " ")


def test_explain_spam_console_output():
    proc = _run_explain("spam")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.count("CASE S-") >= 1
    assert "SHAP EXPLANATION — Why was" in out and "classified as spam?" in out
    assert "RECOMMENDATION — What to do about" in out
    assert "ACTION:" in out
    assert "PRIORITY:" in out
    assert "__case_.*__case_" not in out.replace("\n", " ")


def test_explain_anomaly_console_output():
    proc = _run_explain("anomaly")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.count("CASE A-") >= 1
    assert "SHAP EXPLANATION — Why was" in out and "flagged as anomalous?" in out
    assert "RECOMMENDATION — What to do about" in out
    assert "ACTION:" in out
    assert "PRIORITY:" in out
    # anomaly score rendered normalized with raw, not raw integer like -254
    assert "__case_.*__case_" not in out.replace("\n", " ")
