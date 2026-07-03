import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Filesystem-safe slug from arbitrary text.

    'Adult Primary Care (APC) 2023' -> 'adult-primary-care-apc-2023'.
    Returns '' if there are no usable characters.
    """
    return _NON_ALNUM.sub("-", (text or "").lower()).strip("-")
