"""Retain a reproducible source bundle while a checkout has uncommitted edits."""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from dk2_forecast.storage import digest, publish_bytes


def save_code_snapshot(root: Path) -> str:
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "dk2_forecast").glob("*.py"))
    paths += sorted((project / "scripts").glob("*.py"))
    paths += [project / "pyproject.toml", project / "requirements-lock.txt"]
    files = [
        (path.relative_to(project).as_posix(), path.read_bytes()) for path in paths
    ]
    version = digest(
        b"\0".join(name.encode() + b"\0" + content for name, content in files)
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files:
            item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            archive.writestr(item, content)
    publish_bytes(root / "code" / version / "source.zip", buffer.getvalue())
    return version
