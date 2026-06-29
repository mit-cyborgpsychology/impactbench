"""prompts.load() — substitution and error reporting."""
import pytest
from lib.core import prompts


def test_loads_raw_when_no_kwargs():
    # Templates carry $-placeholders; loading raw must not substitute.
    text = prompts.load("turn_first.txt")
    assert "Begin the conversation" in text


def test_substitutes_kwargs():
    out = prompts.load("expand_demographic.txt",
                       scenario="a tired student", demographic="gender: female")
    assert "a tired student" in out and "gender: female" in out


def test_missing_key_names_template_and_key():
    with pytest.raises(KeyError) as e:
        prompts.load("expand_demographic.txt", scenario="x")  # demographic missing
    msg = str(e.value)
    assert "expand_demographic.txt" in msg
    assert "demographic" in msg
