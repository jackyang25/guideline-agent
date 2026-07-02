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
