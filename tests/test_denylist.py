import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "denylist_check", Path(__file__).resolve().parent.parent / "scripts" / "denylist_check.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_hashed_terms_match_words_and_dotted_hosts():
    hashes = {hashlib.sha256(b"secretco").hexdigest(), hashlib.sha256(b"api.secretco.io").hexdigest()}
    assert dl.scan("nothing to see", hashes) == []
    assert dl.scan("Built at SecretCo", hashes)
    assert len(dl.scan("call https://api.secretco.io/v1", hashes)) == 2  # word + host


def test_patterns_catch_private_network_tokens_and_emails():
    # literals are assembled at runtime so this file passes its own scan
    assert "tailnet/CGNAT address" in dl.scan("host " + "100." + "101.12.3", set())
    assert dl.scan("host 100.12.1.1", set()) == []  # public range, not CGNAT
    assert "tailnet hostname" in dl.scan("box.tail1234" + ".ts" + ".net", set())
    assert "secrets-manager token" in dl.scan("dp" + ".st.dev." + "A" * 24, set())
    assert "private email" in dl.scan("mail me: someone" + "@" + "gmail.com", set())
    assert dl.scan("someone@users.noreply.github.com or a@example.com", set()) == []


def test_repo_is_clean():
    assert dl.main() == 0
