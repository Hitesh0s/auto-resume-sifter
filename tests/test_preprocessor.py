from src.preprocessor import preprocess


def test_lowercases_text():
    result = preprocess("Python DJANGO Flask")
    assert result == result.lower()


def test_lemmatizes_running():
    result = preprocess("I am running the tests")
    assert "run" in result
    assert "running" not in result


def test_removes_stopwords():
    result = preprocess("the quick brown fox")
    assert "the" not in result.split()


def test_removes_punctuation():
    result = preprocess("Python, Django; Flask.")
    assert "," not in result
    assert ";" not in result
    assert "." not in result


def test_removes_digits():
    result = preprocess("Python 3 years experience")
    assert "3" not in result.split()


def test_empty_string():
    assert preprocess("") == ""


def test_whitespace_only():
    assert preprocess("   ") == ""
