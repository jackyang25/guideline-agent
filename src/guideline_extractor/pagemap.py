import re
from collections import Counter

_STANDALONE_INT = re.compile(r"^\s*(\d{1,4})\s*$")


def detect_printed_number(raw_text: str) -> int | None:
    found: int | None = None
    for line in raw_text.splitlines():
        m = _STANDALONE_INT.match(line)
        if m:
            found = int(m.group(1))  # keep the last standalone int (footer)
    return found


def calibrate_offset(samples: list[tuple[int, int | None]]) -> int | None:
    deltas = [printed - idx for idx, printed in samples if printed is not None]
    if not deltas:
        return None
    return Counter(deltas).most_common(1)[0][0]


def assign_page_numbers(pdf_indices: list[int], offset: int | None) -> list[int]:
    if offset is None:
        return list(pdf_indices)
    return [idx + offset for idx in pdf_indices]


def check_monotonic(page_numbers: list[int]) -> list[int]:
    flags: list[int] = []
    for i in range(1, len(page_numbers)):
        if page_numbers[i] <= page_numbers[i - 1]:
            flags.append(i)
    return flags


def resolve_page_numbers(
    raw_texts: list[str], pdf_indices: list[int]
) -> tuple[list[int], list[int]]:
    """Figure out printed page numbers and which sheets are suspect.

    Composes detect -> calibrate -> assign -> QC into one stage. Returns
    ``(page_numbers, flags)`` where ``flags`` are 0-based positions whose
    printed numbering broke: it is not strictly increasing, or a detected
    printed number disagrees with its calibrated page number.
    """
    printed = [detect_printed_number(t) for t in raw_texts]
    offset = calibrate_offset(list(zip(pdf_indices, printed)))
    page_numbers = assign_page_numbers(pdf_indices, offset)

    # Monotonic break among the sheets that carry a printed number, mapped
    # back to their positions in the full list.
    present = [(i, n) for i, n in enumerate(printed) if n is not None]
    monotonic_flags = {present[i][0] for i in check_monotonic([n for _, n in present])}

    # Calibration disagreement: a detected number that doesn't match its
    # assigned page number (restart, mis-detected offset).
    mismatch_flags = {
        i for i, n in enumerate(printed) if n is not None and n != page_numbers[i]
    }

    return page_numbers, sorted(monotonic_flags | mismatch_flags)
