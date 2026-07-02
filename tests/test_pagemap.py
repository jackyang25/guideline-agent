from guideline_extractor.pagemap import (
    detect_printed_number,
    calibrate_offset,
    assign_page_numbers,
    check_monotonic,
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
