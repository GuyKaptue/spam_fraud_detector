"""Hypothetical scenario loading and synthetic subject generation for persona case files."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Union

FRAUD_NAMES = [
    "Priya Nakamura", "Mateo Alvarez", "Talia Lindgren", "Gustav Weber",
    "Rosa Petrov", "Ibrahim Haddad", "Noor Singh", "Hattie Kowalski",
    "Diego Costa", "Elena Fischer", "Marcus Chen", "Aisha Rahman",
    "Lars Müller", "Yuki Tanaka", "Sofia Rossi", "Omar Benali",
    "Lena Schmidt", "Arjun Patel", "Freya Andersen", "Carlos Mendez",
]

SPAM_NAMES = [
    "Hattie Kowalski", "Ibrahim Haddad", "Noor Singh", "Priya Nakamura",
    "Mateo Alvarez", "Talia Lindgren", "Gustav Weber", "Rosa Petrov",
    "Diego Costa", "Elena Fischer", "Marcus Chen", "Aisha Rahman",
    "Lars Müller", "Yuki Tanaka", "Sofia Rossi", "Omar Benali",
    "Lena Schmidt", "Arjun Patel", "Freya Andersen", "Carlos Mendez",
]

ANOMALY_NAMES = [
    "Gustav Weber", "Rosa Petrov", "Priya Nakamura", "Mateo Alvarez",
    "Talia Lindgren", "Diego Costa", "Elena Fischer", "Marcus Chen",
    "Aisha Rahman", "Lars Müller", "Yuki Tanaka", "Sofia Rossi",
    "Omar Benali", "Lena Schmidt", "Arjun Patel", "Freya Andersen",
    "Carlos Mendez", "Hattie Kowalski", "Ibrahim Haddad", "Noor Singh",
]


def _normalize_row_id(row_id: Union[int, str]) -> int:
    """Normalize a row_id (int or str) to a stable integer seed.

    When given a string, hash it to a stable integer. This ensures that
    empirical cases with string-based scenario_ids (e.g. file-like names)
    get deterministic names instead of falling back to placeholders.
    """
    if isinstance(row_id, str):
        return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) % 100_000
    return int(row_id)


def _name_for_row(rid: int, names: list[str], seed: int = 42) -> str:
    """Pick a deterministic name from ``names`` using rid and seed as RNG seed."""
    rng = random.Random((rid + seed) % 100_000)
    return names[rng.randint(0, len(names) - 1)]


def load_scenarios(persona: str) -> list[dict]:
    """Load hypothetical scenarios for a persona.

    Reads ``spam_fraud_detector/core/hypothetical_scenarios.json``. Returns an
    empty list if the file does not exist or the persona has no entries.
    """
    p = Path(__file__).parent / "hypothetical_scenarios.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get(persona, [])


class SyntheticSubject:
    """A fabricated subject (customer / message sender) for a case file.

    Attributes are all derived deterministically from the scenario_id + seed so
    two runs produce identical subject data.
    """

    def __init__(self, subject_id: str, full_name: str, city: str,
                 seed: int = 42, **extra: Any):
        self.subject_id = subject_id
        self.full_name = full_name
        self.city = city
        self.seed = seed
        self.extra = extra

    def to_dict(self) -> dict:
        d = {
            "subject_id": self.subject_id,
            "full_name": self.full_name,
            "city": self.city,
        }
        d.update(self.extra)
        return d


def subject_generator(scenario_id: Union[int, str], seed: int = 42,
                      persona: str = "fraud") -> SyntheticSubject:
    """Build a deterministic synthetic subject from a scenario hint.

    The subject name / city come from ``subject_hint`` in the scenario; other
    fields (account_last4, customer_tenure_months, prior_chargebacks, etc.) are
    derived from a hash of scenario_id so they are stable across runs.

    ``scenario_id`` can be an int or str. When given a string (e.g. an
    empirical case's case filename), it is hashed to a stable integer and used
    to select a proper name from the persona's name pool, instead of falling
    back to a ``Subject NNN`` placeholder.
    """
    from .fake_subjects import load_scenarios  # local import to avoid cycles

    rid = _normalize_row_id(scenario_id)

    # Find the scenario to read subject_hint.
    hint: dict[str, Any] = {}
    for p in load_scenarios(persona):
        if p.get("scenario_id") == str(scenario_id):
            hint = p.get("subject_hint", {}) or {}
            break

    rng = random.Random(int(hashlib.sha256(f"{scenario_id}:{seed}".encode("utf-8")).hexdigest()[:8], 16))

    hint_name = hint.get("name")
    if hint_name:
        full_name = hint_name
    else:
        # Scenario not found (empirical case): pick a proper name from the
        # persona's name pool, deterministically derived from the scenario_id
        # hash so the same subject always gets the same name.
        name_pool = FRAUD_NAMES if persona == "fraud" else (
            SPAM_NAMES if persona == "spam" else ANOMALY_NAMES
        )
        full_name = _name_for_row(rid, name_pool, seed)
    city = hint.get("city", "Unknown")
    last4 = f"{rng.randint(0, 9999):04d}"

    extra: dict[str, Any] = {
        "account_last4": last4,
        "account_last4_full": f"····{last4}",
    }

    if persona == "fraud":
        extra.update({
            "card_product": hint.get("card_product", "Visa Credit"),
            "merchant": hint.get("merchant", "Unknown Merchant"),
            "merchant_category": hint.get("merchant_category", "Retail"),
            "amount_eur": float(hint.get("features", {}).get("Amount", 0.0)),
            "transaction_utc": hint.get("transaction_utc",
                                        "2026-03-14T09:22:00Z"),
            "customer_tenure_months": rng.randint(6, 120),
            "prior_chargebacks": rng.randint(0, 3),
        })
    elif persona == "spam":
        extra.update({
            "sender_email": hint.get("sender_email", "unknown@example.com"),
            "sender_display": hint.get("sender_display", "Unknown Sender"),
            "subject_line": hint.get("subject_line", ""),
            "inbox_age_days": rng.randint(30, 2000),
            "prior_messages_from_sender": rng.randint(0, 10),
            "has_unsubscribe_header": rng.random() < 0.5,
        })
    elif persona == "anomaly":
        extra.update({
            "card_product": hint.get("card_product", "Visa Credit"),
            "merchant": hint.get("merchant", "Unknown Merchant"),
            "merchant_category": hint.get("merchant_category", "Retail"),
            "amount_eur": float(hint.get("features", {}).get("Amount", 0.0)),
            "night_transaction": bool(hint.get("features", {}).get("Time", 0) < 3600 or hint.get("features", {}).get("Time", 86400) > 82800),
            "cross_border": hint.get("merchant_category") in ("Travel", "Online", "Retail"),
            "amount_zscore": round(rng.uniform(-4.0, 4.0), 2),
            "minutes_since_previous_txn": rng.randint(5, 480),
        })

    return SyntheticSubject(
        subject_id=scenario_id,
        full_name=full_name,
        city=city,
        seed=seed,
        **extra,
    )


__all__ = ["load_scenarios", "SyntheticSubject", "subject_generator"]
