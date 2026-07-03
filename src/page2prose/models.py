from dataclasses import dataclass, asdict


@dataclass
class PageRecord:
    guideline_id: str
    page_number: int
    pdf_index: int
    title: str
    prose: str
    image_path: str
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageMapEntry:
    page_number: int
    title: str
    pdf_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    guideline_id: str
    title: str
    jurisdiction: str | None
    publisher: str | None
    version: str | None
    effective_date: str | None
    source_file: str
    page_count: int
    pages: list[PageMapEntry]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pages"] = [p.to_dict() for p in self.pages]
        return d
