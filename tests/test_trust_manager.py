"""Tests for trust_manager.py."""

import tempfile
import os
from trust_manager import TrustManager, _slugify_model


def _tmp_env() -> str:
    fd, path = tempfile.mkstemp(suffix=".env")
    os.close(fd)
    return path


def test_slugify_model():
    assert _slugify_model("nomic-ai/nomic-embed-text-v1.5") == "NOMIC_AI_NOMIC_EMBED_TEXT_V1_5"
    assert _slugify_model("sentence-transformers/all-MiniLM-L6-v2") == "SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2"


def test_slugify_no_collision():
    """Different model names should produce different slugs."""
    models = [
        "nomic-ai/nomic-embed-text-v1.5",
        "nomic-ai/nomic-embed-text-v1",
        "sentence-transformers/all-MiniLM-L6-v2",
        "microsoft/codebert-base",
    ]
    slugs = [_slugify_model(m) for m in models]
    assert len(slugs) == len(set(slugs)), f"Collision detected: {slugs}"


def test_model_to_key():
    tm = TrustManager(env_file=_tmp_env())
    key = tm._model_to_key("nomic-ai/nomic-embed-text-v1.5")
    assert key == "TRUST_REMOTE_CODE_NOMIC_AI_NOMIC_EMBED_TEXT_V1_5"


def test_set_and_get_approval():
    env_path = _tmp_env()
    tm = TrustManager(env_file=env_path)

    # No approval yet
    assert tm._get_model_approval("test-model") is None

    # Set approval
    tm.set_model_approval("test-model", True)

    # Re-read (new instance to ensure it reads from file)
    tm2 = TrustManager(env_file=env_path)
    # We need to reload the env var since it was written to a non-standard path
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)
    assert tm2._get_model_approval("test-model") is True

    os.unlink(env_path)


def test_check_model_needs_trust():
    tm = TrustManager(env_file=_tmp_env())

    # Known to need trust
    assert tm.check_model_needs_trust("nomic-ai/nomic-embed-text-v1.5") is True

    # Known safe
    assert tm.check_model_needs_trust("sentence-transformers/all-MiniLM-L6-v2") is False

    # Unknown defaults to True
    assert tm.check_model_needs_trust("some-random/model") is True


def test_set_approval_no_duplicate_entries():
    """Setting approval twice should not create duplicate entries."""
    env_path = _tmp_env()
    tm = TrustManager(env_file=env_path)

    tm.set_model_approval("test-model", True)
    tm.set_model_approval("test-model", False)

    with open(env_path) as f:
        lines = f.readlines()

    key = tm._model_to_key("test-model")
    # Count lines that are actual settings (not comments)
    setting_lines = [l for l in lines if l.strip().startswith(f"{key}=")]
    assert len(setting_lines) == 1, f"Found {len(setting_lines)} setting entries, expected 1:\n{''.join(lines)}"
    # The value should be the latest one (false)
    assert "false" in setting_lines[0]

    os.unlink(env_path)
