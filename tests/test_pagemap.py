from page2prose.pagemap import (
    detect_printed_number,
    calibrate_offset,
    assign_page_numbers,
    check_monotonic,
    resolve_page_numbers,
)


def test_detect_printed_number_returns_last_standalone_integer():
    text = "TB treatment\nStart HRZE regimen\n38"
    assert detect_printed_number(text) == 38


def test_detect_printed_number_ignores_numbers_inside_text():
    text = "Give 500 mg amoxicillin\nsome prose here"
    assert detect_printed_number(text) is None


def test_detect_printed_number_none_when_empty():
    assert detect_printed_number("") is None


def test_calibrate_offset_picks_most_common_delta():
    # printed = pdf_index - 2 on the pages that have a number
    samples = [(3, 1), (4, 2), (5, None), (6, 4)]
    assert calibrate_offset(samples) == -2


def test_calibrate_offset_none_when_no_numbers():
    assert calibrate_offset([(1, None), (2, None)]) is None


def test_assign_page_numbers_applies_offset():
    assert assign_page_numbers([3, 4, 5], -2) == [1, 2, 3]


def test_assign_page_numbers_falls_back_to_pdf_index():
    assert assign_page_numbers([1, 2, 3], None) == [1, 2, 3]


def test_check_monotonic_flags_breaks():
    # index 2 breaks (5 -> 4), index 3 is fine again (4 -> 9)
    assert check_monotonic([1, 5, 4, 9]) == [2]


def test_check_monotonic_clean_sequence_has_no_flags():
    assert check_monotonic([1, 2, 3]) == []


def test_resolve_page_numbers_clean_sequence_has_no_flags():
    page_numbers, flags = resolve_page_numbers(["Cough\n1", "TB\n2"], [1, 2])
    assert page_numbers == [1, 2]
    assert flags == []


def test_resolve_page_numbers_flags_numbering_break():
    # printed 5 then 3 on sheets 1,2 -> calibrated page_numbers [5,6];
    # sheet 2's printed 3 disagrees with 6 -> flagged.
    page_numbers, flags = resolve_page_numbers(["A\n5", "B\n3"], [1, 2])
    assert flags == [1]


def test_resolve_page_numbers_skips_pages_without_printed_number():
    # middle sheet has no detectable number; offset 9 from the numbered pages.
    page_numbers, flags = resolve_page_numbers(
        ["A\n10", "B body with no page number", "C\n12"], [1, 2, 3]
    )
    assert page_numbers == [10, 11, 12]
    assert flags == []


def test_resolve_page_numbers_flags_restart():
    # printed 1,2,1 -> offset 0, page_numbers [1,2,3]; sheet 3's printed 1
    # both breaks monotonicity and disagrees with 3 -> flagged.
    page_numbers, flags = resolve_page_numbers(["A\n1", "B\n2", "C\n1"], [1, 2, 3])
    assert page_numbers == [1, 2, 3]
    assert flags == [2]
