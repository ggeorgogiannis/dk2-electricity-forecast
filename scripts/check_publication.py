"""Audit Git candidate files without staging, committing or publishing anything."""

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b"
    ),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "personal_windows_path": re.compile(rb"[A-Za-z]:[\\/]Users[\\/][^\s\\/]+", re.I),
}
FORBIDDEN_PARTS = {
    "data",
    ".venv",
    ".reproduction-env",
    ".git",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {
    ".parquet",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".log",
    ".pem",
    ".key",
    ".whl",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def scan_bytes(label, raw):
    """Report only locations/categories, never the matched credential text."""
    return [
        {"file": label, "line": raw[: match.start()].count(b"\n") + 1, "reason": kind}
        for kind, pattern in PATTERNS.items()
        for match in pattern.finditer(raw)
    ]


def check_path(name):
    path = PurePosixPath(name)
    return (
        path.is_absolute()
        or ".." in path.parts
        or any(p in FORBIDDEN_PARTS for p in path.parts)
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
        or (path.name.startswith(".env") and path.name != ".env.example")
    )


def check_links(root, name, text):
    failures = []
    # Repository Markdown uses inline links; remote links and anchors are not fetched.
    for target in re.findall(r"\]\(([^\s)]+)\)", text):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0]).strip("<>")
        path = (root / name).parent / target
        if not path.resolve().is_relative_to(root.resolve()) or not path.exists():
            failures.append(
                {"file": name, "reason": "broken_local_link", "target": target}
            )
    return failures


def audit(root, files):
    failures, hashes = [], {}
    for name in sorted(set(files)):
        path = root / name
        if check_path(name) or path.is_symlink():
            failures.append({"file": name, "reason": "excluded_or_unsafe_public_file"})
            continue
        if not path.is_file():
            failures.append({"file": name, "reason": "missing_candidate_file"})
            continue
        raw = path.read_bytes()
        hashes[name] = sha(raw)
        failures.extend(scan_bytes(name, raw))
        if len(raw) > 5_000_000:
            failures.append({"file": name, "reason": "file_over_5MB_review_required"})
        if path.suffix == ".md":
            failures.extend(check_links(root, name, raw.decode("utf-8")))
        if path.suffix == ".zip":
            if not name.startswith("configs/retained_sources/"):
                failures.append({"file": name, "reason": "unexpected_archive"})
                continue
            with ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) > 100 or sum(m.file_size for m in members) > 5_000_000:
                    failures.append(
                        {"file": name, "reason": "oversized_retained_archive"}
                    )
                    continue
                for member in members:
                    if (
                        check_path(member.filename)
                        or not (
                            member.filename.endswith(".py")
                            or member.filename
                            in ("pyproject.toml", "requirements-lock.txt")
                        )
                        or member.file_size > 1_000_000
                    ):
                        failures.append(
                            {"file": name, "reason": "unsafe_retained_archive_member"}
                        )
                    else:
                        failures.extend(
                            scan_bytes(
                                f"{name}!{member.filename}", archive.read(member)
                            )
                        )
    provenance_path = root / "configs/annual-replay-provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_bytes())
        for label, field in (("report", "sha256"), ("source_bundle", "archive_sha256")):
            entry = provenance[label]
            if hashes.get(entry["path"]) != entry[field]:
                failures.append(
                    {
                        "file": entry["path"],
                        "reason": "retained_provenance_hash_mismatch",
                    }
                )
        bundle = root / provenance["source_bundle"]["path"]
        with ZipFile(bundle) as archive:
            content = b"\0".join(
                name.encode() + b"\0" + archive.read(name)
                for name in archive.namelist()
            )
        if sha(content) != provenance["source_bundle"]["content_identity"]:
            failures.append(
                {
                    "file": str(bundle.relative_to(root)),
                    "reason": "retained_content_identity_mismatch",
                }
            )
    manifest = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "passed" if not failures else "failed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(hashes),
        "manifest_sha256": sha(manifest),
        "files": hashes,
        "findings": failures,
        "scope": "Git tracked and nonignored untracked files; retained archive members; local Markdown file links. Pattern scan is not a guarantee of absence of secrets. External links and remote security settings are not validated.",
    }


def candidate_files(root):
    manifest = root / ".publication-manifest.json"
    if manifest.exists() and not (root / ".git").exists():
        files = json.loads(manifest.read_bytes())
        if any(
            check_path(name)
            or not (root / name).resolve().is_relative_to(root.resolve())
            for name in files
        ):
            raise ValueError("Unsafe public export manifest path")
        if any(
            sha((root / name).read_bytes()) != expected
            for name, expected in files.items()
        ):
            raise ValueError("Public export differs from its original manifest")
        return list(files)
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def export_files(root, result):
    folder = root / "data" / "publication" / result["manifest_sha256"] / "source"
    for name, expected in result["files"].items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha(path.read_bytes()) != expected:
            raise ValueError(
                "Existing public export was changed; use a separate directory"
            )
        shutil.copyfile(root / name, path)
    (folder / ".publication-manifest.json").write_text(
        json.dumps(result["files"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Copy audited candidate files to ignored data/publication/<manifest>/source",
    )
    args = parser.parse_args()
    result = audit(ROOT, candidate_files(ROOT))
    output = ROOT / "data/publication-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Publication audit: {result['status']}; {result['file_count']} files")
    for finding in result["findings"]:
        print(json.dumps(finding))
    print(f"Manifest/report: {output.relative_to(ROOT)}")
    if result["findings"]:
        return 1
    if args.export:
        print(f"Export: {export_files(ROOT, result).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
