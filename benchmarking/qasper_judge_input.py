"""Portable, auditable construction of per-annotation QASPER judge input."""

import json

from benchmarking.evaluation_config import sha256_json


NO_GOLD_EVIDENCE_MESSAGE = "No gold evidence was supplied."
ANNOTATION_EVIDENCE_POLICY = {
    "policy_version": "qasper-annotation-evidence-v1",
    "preferred_source": "highlighted_evidence",
    "fallback_source": "evidence",
    "empty_source": "none",
    "ordering": "original",
    "usable_text": "non-empty after whitespace trimming",
    "reference_transformation": "none",
    "evidence_transformation": "none",
    "empty_evidence_message": NO_GOLD_EVIDENCE_MESSAGE,
}


def semantic_evaluation_contract(evaluator_version):
    """Return the portable methodology contract for a v1.6 pipeline."""
    return {
        "evaluator_version": evaluator_version,
        "execution": "one independent provider request per annotation",
        "selection": "maximum annotation score; first annotation wins ties",
        "placeholder_policy": "qasper-placeholder-markup-v1",
        "annotation_evidence_policy": ANNOTATION_EVIDENCE_POLICY,
        "abbreviated_reference_absence_is_fabrication": False,
    }


def select_annotation_evidence(annotation):
    """Return the preferred usable evidence strings without modifying them."""
    if not isinstance(annotation, dict):
        raise ValueError("Gold annotation must be an object.")
    for source in ("highlighted_evidence", "evidence"):
        values = annotation.get(source)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"Gold annotation {source} must be a list of strings.")
        usable = [value for value in values if value.strip()]
        if usable:
            return {
                "source": source,
                "ordered_values": usable,
                "sha256": sha256_json(usable),
            }
    return {
        "source": "none",
        "ordered_values": [],
        "sha256": sha256_json([]),
    }


def evidence_for_reference(gold_bundle, reference):
    """Select evidence belonging to exactly the converted reference annotation."""
    index = reference.get("annotation_index")
    annotations = gold_bundle.get("answer") if isinstance(gold_bundle, dict) else None
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not isinstance(annotations, list)
        or not 0 <= index < len(annotations)
    ):
        raise ValueError("Converted reference is not aligned with its annotation.")
    return select_annotation_evidence(annotations[index])


def validate_serialized_evidence(value):
    """Validate the portable evidence representation and return a safe copy."""
    if not isinstance(value, dict):
        raise ValueError("Judge evidence must be an object.")
    source = value.get("source")
    ordered_values = value.get("ordered_values")
    if source not in {"highlighted_evidence", "evidence", "none"}:
        raise ValueError("Judge evidence has an invalid source.")
    if not isinstance(ordered_values, list) or any(
        not isinstance(item, str) or not item.strip() for item in ordered_values
    ):
        raise ValueError("Judge evidence has invalid ordered values.")
    if (source == "none") != (not ordered_values):
        raise ValueError("Judge evidence source and values are inconsistent.")
    if value.get("sha256") != sha256_json(ordered_values):
        raise ValueError("Judge evidence digest is invalid.")
    return {
        "source": source,
        "ordered_values": list(ordered_values),
        "sha256": value["sha256"],
    }


def format_expected_output(reference, evidence):
    """Serialize one reference and its matching evidence for EXPECTED_OUTPUT."""
    evidence = validate_serialized_evidence(evidence)
    lines = [
        f"Answer type: {reference['answer_type']}",
        f"Reference answer: {reference['answer']}",
        f"Gold evidence source: {evidence['source']}",
    ]
    if evidence["ordered_values"]:
        lines.append("Gold evidence (ordered JSON array):")
        lines.append(json.dumps(
            evidence["ordered_values"],
            ensure_ascii=False,
            separators=(",", ":"),
        ))
    else:
        lines.append(f"Gold evidence: {NO_GOLD_EVIDENCE_MESSAGE}")
    return "\n".join(lines)
