from page2prose.util import slugify


def test_slugify_basic():
    assert slugify("Adult Primary Care (APC) 2023") == "adult-primary-care-apc-2023"


def test_slugify_collapses_and_trims():
    assert slugify("  Hello,  World!! ") == "hello-world"


def test_slugify_empty():
    assert slugify("") == ""
    assert slugify("!!!") == ""
