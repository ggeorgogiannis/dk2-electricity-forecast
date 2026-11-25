from contextlib import contextmanager
from hashlib import sha256
from os import name as operating_system
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb


def digest(content: bytes) -> str:
    return sha256(content).hexdigest()


def _publish_file(temporary_path: Path, path: Path) -> None:
    # Unix rename would overwrite an existing artifact. Windows rename is exclusive.
    try:
        if operating_system == "nt":
            temporary_path.rename(path)
        else:
            path.hardlink_to(temporary_path)
    except FileExistsError:
        pass


def publish_bytes(path: Path, content: bytes) -> bytes:
    """Publish a fully written file and return the first capture at that path."""
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        _publish_file(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path.read_bytes()


@contextmanager
def connect():
    with duckdb.connect(config={"threads": 1}) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        yield connection


def write_parquet(connection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        connection.execute(
            f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(temporary_path)],
        )
        _publish_file(temporary_path, path)
        if temporary_path.exists() and temporary_path.read_bytes() != path.read_bytes():
            raise ValueError(
                f"Existing Parquet differs from regenerated data at {path}"
            )
    finally:
        temporary_path.unlink(missing_ok=True)
