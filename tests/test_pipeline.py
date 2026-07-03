import json

from page2prose.render import RenderedPage
from page2prose.pipeline import extract


def _fake_describe(client, image_bytes, raw_text):
    # title echoes the raw text's first line so assertions are meaningful
    return raw_text.splitlines()[0], f"prose for {raw_text.splitlines()[0]}"


def test_extract_writes_manifest_and_one_record_per_page(tmp_path):
    rendered = [
        RenderedPage(1, b"\x89PNG1", "Cough\n1"),
        RenderedPage(2, b"\x89PNG2", "TB treatment\n2"),
    ]
    manifest, flags, _failed = extract(
        "ignored.pdf",
        str(tmp_path),
        guideline_id="APC_2023_ZA",
        guideline_title="Adult Primary Care 2023",
        jurisdiction="South Africa",
        describe_fn=_fake_describe,
        rendered=rendered,
    )
    assert manifest.page_count == 2
    assert flags == []
    # page numbers calibrated: printed 1,2 on sheets 1,2 -> offset 0
    assert [p.page_number for p in manifest.pages] == [1, 2]
    assert manifest.pages[1].title == "TB treatment"

    rec = json.loads((tmp_path / "APC_2023_ZA" / "pages" / "p002.json").read_text())
    assert rec["page_number"] == 2
    assert rec["title"] == "TB treatment"
    assert rec["image_path"] == "pages/p002.png"
    assert (tmp_path / "APC_2023_ZA" / "pages" / "p002.png").read_bytes() == b"\x89PNG2"
    assert (tmp_path / "APC_2023_ZA" / "manifest.json").exists()


def test_extract_flags_broken_numbering(tmp_path):
    # printed numbers 5 then 3 -> not strictly increasing -> flag index 1
    rendered = [
        RenderedPage(1, b"a", "A\n5"),
        RenderedPage(2, b"b", "B\n3"),
    ]
    _, flags, _failed = extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == [1]


def test_extract_flags_printed_number_mismatch_with_calibrated_offset(tmp_path):
    # Sheets 1,2,3 with printed numbers 1,2,10. Calibration picks offset 0
    # (majority: two of three deltas are 0), so page 3 (printed 10) disagrees
    # with its calibrated page_number of 3 -> flagged, even though 1,2,10 is
    # still monotonically increasing.
    rendered = [
        RenderedPage(1, b"a", "A\n1"),
        RenderedPage(2, b"b", "B\n2"),
        RenderedPage(3, b"c", "C\n10"),
    ]
    _, flags, _failed = extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == [2]


def test_extract_names_records_by_pdf_index_even_with_nonzero_offset(tmp_path):
    # printed 40, 41 on sheets 1, 2 -> offset 39 -> page_numbers 40, 41,
    # but files are still named by pdf_index (p001, p002).
    rendered = [
        RenderedPage(1, b"a", "A\n40"),
        RenderedPage(2, b"b", "B\n41"),
    ]
    manifest, flags, _failed = extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == []
    assert [p.page_number for p in manifest.pages] == [40, 41]

    rec1 = json.loads((tmp_path / "g" / "pages" / "p001.json").read_text())
    assert rec1["page_number"] == 40
    rec2 = json.loads((tmp_path / "g" / "pages" / "p002.json").read_text())
    assert rec2["page_number"] == 41


def test_extract_respects_limit(tmp_path):
    rendered = [
        RenderedPage(1, b"a", "P1\n1"),
        RenderedPage(2, b"b", "P2\n2"),
        RenderedPage(3, b"c", "P3\n3"),
    ]
    manifest, _flags, _failed = extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
        limit=2,
    )
    assert manifest.page_count == 2
    assert [p.pdf_index for p in manifest.pages] == [1, 2]
    assert not (tmp_path / "g" / "pages" / "p003.json").exists()


def test_extract_concurrent_writes_all_pages_in_order(tmp_path):
    import threading

    seen = []
    lock = threading.Lock()

    def describe(client, image_bytes, raw_text):
        with lock:
            seen.append(raw_text.splitlines()[0])
        return raw_text.splitlines()[0], "prose"

    rendered = [RenderedPage(i, f"img{i}".encode(), f"Page{i}\n{i}") for i in range(1, 6)]
    manifest, _flags, _failed = extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=describe, rendered=rendered,
        concurrency=4,
    )
    # all pages processed
    assert len(seen) == 5
    # manifest map stays in page order regardless of completion order
    assert [p.pdf_index for p in manifest.pages] == [1, 2, 3, 4, 5]
    assert [p.title for p in manifest.pages] == [f"Page{i}" for i in range(1, 6)]
    for i in range(1, 6):
        assert (tmp_path / "g" / "pages" / f"p00{i}.json").exists()


def test_extract_reports_progress_via_on_page(tmp_path):
    rendered = [RenderedPage(i, f"i{i}".encode(), f"P{i}\n{i}") for i in range(1, 4)]
    calls = []
    extract(
        "ignored.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
        on_page=lambda done, total: calls.append((done, total)),
    )
    # an initial (0, N) then one call per completed page, ending at (N, N)
    assert calls[0] == (0, 3)
    assert calls[-1] == (3, 3)
    assert sorted(d for d, _ in calls) == [0, 1, 2, 3]


def test_extract_auto_derives_id_and_title_from_detection(tmp_path):
    # No guideline_id/title given: a fake client triggers detection, and id is a
    # slug of the detected title. Output folder is named by the derived id.
    rendered = [RenderedPage(1, b"a", "cover"), RenderedPage(2, b"b", "Body\n2")]

    def fake_detect(client, image_bytes, raw_text):
        return {"title": "Adult Primary Care (APC) 2023", "jurisdiction": "South Africa",
                "publisher": None, "version": "2023", "effective_date": None}

    manifest, _flags, _failed = extract(
        "APC_2023_Clinical_tool.pdf", str(tmp_path),
        client=object(),  # truthy so detection runs; fake_detect ignores it
        describe_fn=_fake_describe, detect_fn=fake_detect, rendered=rendered,
    )
    assert manifest.guideline_id == "adult-primary-care-apc-2023"
    assert manifest.title == "Adult Primary Care (APC) 2023"
    assert manifest.jurisdiction == "South Africa"
    assert manifest.version == "2023"
    assert (tmp_path / "adult-primary-care-apc-2023" / "manifest.json").exists()


def test_extract_explicit_values_override_detection(tmp_path):
    rendered = [RenderedPage(1, b"a", "cover")]

    def fake_detect(client, image_bytes, raw_text):
        return {"title": "Detected Title", "jurisdiction": "Nowhere",
                "publisher": None, "version": None, "effective_date": None}

    manifest, _flags, _failed = extract(
        "x.pdf", str(tmp_path), guideline_id="my-id", guideline_title="My Title",
        client=object(), describe_fn=_fake_describe, detect_fn=fake_detect, rendered=rendered,
    )
    assert manifest.guideline_id == "my-id"
    assert manifest.title == "My Title"
    # jurisdiction was not supplied, so detection still fills it
    assert manifest.jurisdiction == "Nowhere"


def test_extract_detection_sees_front_matter_not_just_page_one(tmp_path):
    # Metadata detection should receive text from the first few pages (cover +
    # imprint), so a publisher printed on page 2 is available to it.
    rendered = [
        RenderedPage(1, b"cover", "APC 2023"),
        RenderedPage(2, b"imprint", "Published by Western Cape Government Health"),
        RenderedPage(3, b"toc", "Contents"),
        RenderedPage(4, b"body", "Body\n4"),
    ]
    seen = {}

    def fake_detect(client, image_bytes, raw_text):
        seen["image"] = image_bytes
        seen["raw_text"] = raw_text
        return {"title": "APC 2023", "jurisdiction": None,
                "publisher": "Western Cape Government Health", "version": None, "effective_date": None}

    manifest, _flags, _failed = extract(
        "x.pdf", str(tmp_path), client=object(),
        describe_fn=_fake_describe, detect_fn=fake_detect, rendered=rendered,
    )
    # cover image is page 1; text spans the front matter (page 2's publisher line present)
    assert seen["image"] == b"cover"
    assert "Western Cape Government Health" in seen["raw_text"]
    assert "Contents" in seen["raw_text"]          # page 3 included
    assert "Body" not in seen["raw_text"]          # page 4 (body) excluded
    assert manifest.publisher == "Western Cape Government Health"


def test_extract_resumes_existing_pages(tmp_path):
    # Pre-write page 1's record; extract should reuse it (not call describe for it)
    # and only process page 2.
    import json as _json, os as _os
    d = tmp_path / "g" / "pages"
    d.mkdir(parents=True)
    (d / "p001.json").write_text(_json.dumps(
        {"guideline_id": "g", "page_number": 1, "pdf_index": 1, "title": "Existing",
         "prose": "kept", "image_path": "pages/p001.png", "raw_text": "r"}))
    (tmp_path / "g" / "manifest.json").write_text("{}")  # prior partial run

    calls = []

    def describe(client, image_bytes, raw_text):
        calls.append(raw_text)
        return raw_text.splitlines()[0], "prose"

    rendered = [RenderedPage(1, b"a", "P1\n1"), RenderedPage(2, b"b", "P2\n2")]
    manifest, _flags, failed = extract(
        "x.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=describe, rendered=rendered,
    )
    assert calls == ["P2\n2"]                       # page 1 skipped, only page 2 described
    assert failed == []
    titles = {p.page_number: p.title for p in manifest.pages}
    assert titles[1] == "Existing"                  # reused from disk
    assert titles[2] == "P2"


def test_extract_tolerates_a_failing_page(tmp_path):
    def describe(client, image_bytes, raw_text):
        if raw_text.startswith("P2"):
            raise RuntimeError("boom on page 2")
        return raw_text.splitlines()[0], "prose"

    rendered = [RenderedPage(1, b"a", "P1\n1"), RenderedPage(2, b"b", "P2\n2")]
    manifest, _flags, failed = extract(
        "x.pdf", str(tmp_path), guideline_id="g",
        guideline_title="t", describe_fn=describe, rendered=rendered,
    )
    assert failed == [2]                            # reported, not raised
    assert [p.pdf_index for p in manifest.pages] == [1]   # only the good page
    assert manifest.page_count == 1
    assert (tmp_path / "g" / "manifest.json").exists()    # manifest still written
    assert not (tmp_path / "g" / "pages" / "p002.json").exists()  # failed page left for retry


def test_extract_auto_id_collision_gets_suffix(tmp_path):
    (tmp_path / "adult-primary-care" / "pages").mkdir(parents=True)
    (tmp_path / "adult-primary-care" / "manifest.json").write_text("{}")  # a different guideline

    def fake_detect(client, image_bytes, raw_text):
        return {"title": "Adult Primary Care", "jurisdiction": None,
                "publisher": None, "version": None, "effective_date": None}

    rendered = [RenderedPage(1, b"a", "cover")]
    manifest, _flags, _failed = extract(
        "x.pdf", str(tmp_path), client=object(),
        describe_fn=_fake_describe, detect_fn=fake_detect, rendered=rendered,
    )
    assert manifest.guideline_id == "adult-primary-care-2"   # did not overwrite
    assert (tmp_path / "adult-primary-care-2" / "manifest.json").exists()
