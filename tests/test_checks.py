import pytest

from llm_eval.checks import (
    CheckConfig,
    check_keywords,
    check_length,
    check_nonempty,
    check_script,
    expected_script,
    is_refusal,
    keyword_coverage,
    run_checks,
    script_of,
    script_profile,
)
from llm_eval.records import ResponseRecord


@pytest.mark.parametrize("text,ok", [("", False), ("   \n\t", False), ("...!?", False), ("ok", True), ("नमस्ते", True), ("42", True)])
def test_nonempty(text, ok):
    assert check_nonempty(text).passed is ok


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I’m sorry, but I cannot assist with this request.",  # curly apostrophe
        "As an AI language model, I have no opinions.",
        "I cannot provide that information.",
        "I must decline to answer.",
        "माफ़ कीजिए, मैं इसमें आपकी मदद नहीं कर सकता।",
        "معاف کیجیے، میں اس میں مدد نہیں کر سکتا۔",
    ],
)
def test_refusals_detected(text):
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Water boils at 100 °C.",
        "I can help with that: the answer is 4.",
        "Sorry for the delay — here is the summary you asked for.",
        "भारत की राजधानी नई दिल्ली है।",
    ],
)
def test_non_refusals(text):
    assert not is_refusal(text)


def test_extra_refusal_pattern():
    assert is_refusal("Request blocked by policy.", [r"blocked by policy"])


def test_length_bounds():
    assert check_length("one two three", 3, 3).passed
    assert not check_length("one two", 3, None).passed
    r = check_length("a b c d e", 1, 4)
    assert not r.passed and r.value == 5


def test_keyword_coverage_case_and_unicode():
    cov, missing = keyword_coverage("New DELHI is the capital; नई दिल्ली", ["new delhi", "नई दिल्ली", "Mumbai"])
    assert cov == pytest.approx(2 / 3)
    assert missing == ["Mumbai"]
    assert check_keywords("abc", ["a", "z"], threshold=0.5).passed
    assert not check_keywords("abc", ["a", "z"], threshold=1.0).passed


def test_keyword_coverage_nfc_normalisation():
    # "é" precomposed vs. "e" + combining acute must match.
    assert keyword_coverage("café", ["café"])[0] == 1.0


def test_script_of_ranges():
    assert script_of("क") == "devanagari"
    assert script_of("A") == "latin"
    assert script_of("ب") == "arabic"
    assert script_of("த") == "tamil"
    assert script_of("1") is None


def test_script_profile_devanagari_with_matras():
    # Matras (vowel signs, category Mc/Mn) are counted with Devanagari; danda and spaces ignored.
    prof = script_profile("भारत की राजधानी नई दिल्ली है।")
    assert prof == {"devanagari": 1.0}


def test_script_profile_mixed_and_digits_ignored():
    prof = script_profile("abcd 1234 कखगघ")
    assert prof["latin"] == pytest.approx(0.5)
    assert prof["devanagari"] == pytest.approx(0.5)
    assert script_profile("123 !!!") == {}


def test_combining_mark_inherits_base_script():
    assert script_profile("é") == {"latin": 1.0}


def test_expected_script_mapping():
    assert expected_script("hi") == "devanagari"
    assert expected_script("mr-IN") == "devanagari"
    assert expected_script("ur") == "arabic"
    assert expected_script("Tamil") == "tamil"
    with pytest.raises(ValueError):
        expected_script("xx")


def test_check_script_threshold():
    assert check_script("پاکستان کا قومی پھول چنبیلی ہے۔", "ur").passed
    r = check_script("The capital is New Delhi (नई दिल्ली).", "hi", threshold=0.6)
    assert not r.passed and r.value < 0.6


def test_run_checks_only_applicable():
    rec = ResponseRecord(id="1", prompt="p", response="hello world there", model="m")
    names = [c.name for c in run_checks(rec, CheckConfig(min_words=1))]
    assert names == ["nonempty", "no_refusal", "length"]
    rec.keywords, rec.language = ["hello"], "en"
    results = run_checks(rec, CheckConfig())
    assert [c.name for c in results][-2:] == ["keywords", "script"]
    assert all(c.passed for c in results)


def test_check_config_validation():
    with pytest.raises(ValueError):
        CheckConfig.from_dict({"min_words": 10, "max_words": 5})
    with pytest.raises(ValueError):
        CheckConfig.from_dict({"script_threshold": 1.5})
