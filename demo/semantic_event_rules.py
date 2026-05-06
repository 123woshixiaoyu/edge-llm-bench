from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SemanticEventRule:
    rule_name: str
    natural_language_rule: str
    roi_name: str
    privacy_mode: str
    verifier_backend: str
    min_trigger_interval_s: float
    risk_level_if_verified: str


def default_semantic_event_rule() -> dict[str, Any]:
    return {
        "rule_name": "chemical cabinet approach",
        "natural_language_rule": "Alert me when someone approaches the chemical cabinet.",
        "roi_name": "chemical_cabinet",
        "privacy_mode": "allow_remote",
        "verifier_backend": "mock_final_line",
        "min_trigger_interval_s": 10,
        "risk_level_if_verified": "high",
    }


def rule_from_config(config: dict[str, Any]) -> SemanticEventRule:
    defaults = default_semantic_event_rule()
    merged = {**defaults, **config}
    return SemanticEventRule(
        rule_name=str(merged["rule_name"]),
        natural_language_rule=str(merged["natural_language_rule"]),
        roi_name=str(merged["roi_name"]),
        privacy_mode=str(merged["privacy_mode"]),
        verifier_backend=str(merged["verifier_backend"]),
        min_trigger_interval_s=float(merged["min_trigger_interval_s"]),
        risk_level_if_verified=str(merged["risk_level_if_verified"]),
    )


def build_verifier_prompt(rule: SemanticEventRule, proposal: dict[str, Any]) -> str:
    return (
        f'User rule: "{rule.natural_language_rule}"\n'
        f"Candidate trigger: {proposal.get('proposal_reason', '')}\n"
        f"ROI: {proposal.get('roi_name') or rule.roi_name}\n"
        f"Objects: {proposal.get('objects', [])}\n"
        "Question: Does this candidate frame show that event?\n"
        "Answer with exactly one FINAL_ANSWER line: YES, NO, or UNKNOWN, "
        "then one short REASON line."
    )
