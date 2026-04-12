"""
Step 8 – Validate synthesized output before exposing to user.
Structural validation only. Fully deterministic.
"""

from __future__ import annotations

from models import ScoredSignal, SynthesizedOutput, ValidationResult, ValidationStatus


_REQUIRED_SECTIONS = ["what_matters_now", "drivers", "monitoring", "conclusion"]


def validate(
    output: SynthesizedOutput,
    scored_signals: list[ScoredSignal],
) -> ValidationResult:
    """
    Validate structure and signal references.
    """
    flags: list[str] = []
    valid_ids = {s.signal_id for s in scored_signals}

    # 1. Structural: all sections present and non-empty
    for section in _REQUIRED_SECTIONS:
        val = getattr(output.sections, section, None)
        if not val or not val.strip():
            flags.append(f"empty_section:{section}")

    # 2. Signal-ID traceability
    referenced = output.signal_ids_used
    invalid_refs = [sid for sid in referenced if sid not in valid_ids]
    if invalid_refs:
        flags.append(f"invalid_signal_ids:{','.join(invalid_refs)}")

    # 3. Status
    structural_errors = len([f for f in flags if f.startswith("empty_section")])
    has_invalid_refs = any(f.startswith("invalid_signal_ids") for f in flags)

    if structural_errors == 0 and not has_invalid_refs:
        status = ValidationStatus.APPROVED
    elif structural_errors <= 1:
        status = ValidationStatus.NEEDS_REVISION
    else:
        status = ValidationStatus.BLOCKED

    return ValidationResult(
        status=status,
        flags=flags,
    )