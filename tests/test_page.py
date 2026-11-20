"""The data of the published page against the results; the JavaScript port is tested by tests/web_model.test.js."""

import json
import re

import pytest

from intercalation_voltage import paths, summary

PAGE = paths.WEB / "intercalation-voltage.html"


@pytest.fixture(scope="module")
def block():
    text = PAGE.read_text(encoding="utf-8")
    m = re.search(r"window\.VOLTAGE_DATA = (\{.*?\});</script>", text, re.S)
    assert m, "no data block"
    return text, json.loads(m.group(1))


def test_page_numbers_are_the_summary_numbers(block):
    _, D = block
    N = summary.compute()
    assert D["summary"] == {k: N[k] for k in D["summary"]}
    dist = json.loads(paths.result("distance.json").read_text(encoding="utf-8"))
    assert D["model"]["threshold"] == dist["threshold"]
    assert D["model"]["n_train"] == json.loads(paths.result("learning.json").read_text(encoding="utf-8"))["n_li"]


def test_page_rows_and_histograms(block):
    _, D = block
    assert len(D["rows"]) == 3430
    assert sum(D["hist"]["li"]) == 2043 and sum(D["hist"]["na"]) == 327


def test_data_block_cannot_close_its_script_tag(block):
    text, _ = block
    start = text.index("window.VOLTAGE_DATA = ")
    end = text.index("</script>", start)
    assert "<script" not in text[start:end]


def test_page_inlines_the_current_sources(block):
    # the Node test runs the core script of the page; this keeps that script equal to web/src/core.js
    text, _ = block
    assert (paths.WEB / "src" / "core.js").read_text(encoding="utf-8") in text
    src = (paths.WEB / "src" / "intercalation-voltage.src.html").read_text(encoding="utf-8")
    for part in src.replace("/*__DATA__*/", "/*__CORE__*/").split("/*__CORE__*/"):
        assert part in text, "page out of date: run scripts/build_web_page.py"


def test_page_never_writes_user_input_as_html():
    src = (paths.WEB / "src" / "intercalation-voltage.src.html").read_text(encoding="utf-8")
    # formulas typed by the user reach the DOM only through textContent or esc()
    for line in src.splitlines():
        if "innerHTML" in line and ("f-dis" in line or "f-chg" in line or ".value" in line):
            pytest.fail(f"user input into innerHTML: {line.strip()}")
