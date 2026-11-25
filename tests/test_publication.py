from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts.check_publication import (
    audit,
    candidate_files,
    check_path,
    export_files,
    scan_bytes,
)


def test_secret_findings_are_redacted_and_paths_are_caught():
    token = b"ghp_" + b"A" * 36
    raw = b"line one\ncredential=" + token
    findings = scan_bytes("example.txt", raw)
    assert findings == [{"file": "example.txt", "line": 2, "reason": "github_token"}]
    assert token.decode() not in str(findings)
    assert scan_bytes("example.txt", b"D:" + b"\\Users\\example\\project")


def test_excludes_runtime_artifacts_but_allows_retained_source():
    for name in (
        "data/raw.json",
        ".venv/python.exe",
        "weights.pkl",
        ".env",
        "keys/private.pem",
        "../escape.py",
    ):
        assert check_path(name)
    assert not check_path("configs/retained_sources/id/source.zip")
    assert not check_path(".env.example")


def test_publication_audit_checks_links_and_exports_only_audited_files(tmp_path):
    (tmp_path / "README.md").write_text("[guide](docs/guide.md)\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "private.txt").write_text("not selected", encoding="utf-8")
    files = ["README.md", "docs/guide.md"]
    result = audit(tmp_path, files)
    assert result["status"] == "passed"
    folder = export_files(tmp_path, result)
    assert (folder / "README.md").is_file()
    assert not (folder / "private.txt").exists()
    assert len(list(folder.rglob("*.md"))) == 2
    assert set(candidate_files(folder)) == set(files)
    (folder / "README.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        candidate_files(folder)
    (tmp_path / "docs/guide.md").unlink()
    broken = audit(tmp_path, ["README.md"])
    assert broken["findings"][0]["reason"] == "broken_local_link"


def test_runtime_files_fail_even_if_explicitly_selected(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"not public")
    assert audit(tmp_path, ["model.pkl"])["status"] == "failed"
    assert not check_path(Path("scripts/check_publication.py").as_posix())


def test_retained_archive_allows_code_and_lock_but_rejects_credentials(tmp_path):
    name = "configs/retained_sources/id/source.zip"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("module.py", "x = 1\n")
        archive.writestr("requirements-lock.txt", "numpy==2.2.6\n")
        archive.writestr("pyproject.toml", "[project]\n")
    assert audit(tmp_path, [name])["status"] == "passed"
    with ZipFile(path, "a") as archive:
        archive.writestr(".env", "TOKEN=hidden\n")
    assert (
        audit(tmp_path, [name])["findings"][0]["reason"]
        == "unsafe_retained_archive_member"
    )


def test_export_manifest_cannot_hide_git_candidate_files(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".publication-manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.check_publication.subprocess.run",
        lambda *a, **kw: SimpleNamespace(stdout=b"README.md\0secret.env\0"),
    )
    assert candidate_files(tmp_path) == ["README.md", "secret.env"]
