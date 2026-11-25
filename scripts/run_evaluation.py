"""Standard-library bootstrap, copied immutably into each research evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from zipfile import ZipFile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(args):
    plan = json.loads(args.plan.read_bytes())
    candidate = plan["candidate"]
    if sha(Path(__file__).read_bytes()) != candidate["launcher_sha256"]:
        raise ValueError("Retained launcher hash mismatch")
    for package, expected in candidate["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(
                f"Frozen dependency mismatch for {package}; restore the pinned environment"
            )
    archive = args.root / "code" / candidate["code_snapshot_sha256"] / "source.zip"
    if sha(archive.read_bytes()) != candidate["archive_sha256"]:
        raise ValueError("Frozen source archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dk2-execution-") as directory:
        with ZipFile(archive) as z:
            entries = [(name, z.read(name)) for name in z.namelist()]
        if (
            sha(b"\0".join(name.encode() + b"\0" + raw for name, raw in entries))
            != candidate["code_snapshot_sha256"]
        ):
            raise ValueError("Source content identity mismatch")
        for name, raw in entries:
            path = Path(directory) / name
            if not path.resolve().is_relative_to(Path(directory).resolve()):
                raise ValueError("Unsafe archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        sys.path.insert(0, directory)
        from dk2_forecast.research_ops import worker

        print(
            json.dumps(
                worker(
                    args.action,
                    args.plan,
                    args.root,
                    day=args.day,
                    actuals=args.actuals,
                ),
                default=str,
                indent=2,
            )
        )


def main():
    from datetime import date

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "publish", "inspect", "score", "collect-score", "weather"),
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actuals", type=Path)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.plan, args.root = args.plan.resolve(), args.root.resolve()
    if args.worker:
        execute(args)
        return
    stamp = datetime.now(UTC)
    prefix = (
        args.plan.parent
        / "logs"
        / f"{stamp.strftime('%Y%m%dT%H%M%S%fZ')}_{args.action}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock = args.plan.parent / "locks" / "execution.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "started_at_utc": stamp.isoformat(),
        "action": args.action,
        "exit_code": 1,
    }
    owned = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at_utc": stamp.isoformat(),
                        "action": args.action,
                    }
                )
            )
        owned = True
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
        ]
        with (
            prefix.with_suffix(".stdout.log").open("wb") as out,
            prefix.with_suffix(".stderr.log").open("wb") as err,
        ):
            result["exit_code"] = subprocess.run(
                command, stdout=out, stderr=err, check=False
            ).returncode
        if result["exit_code"]:
            result["failure_reason"] = prefix.with_suffix(".stderr.log").read_text(
                errors="replace"
            )[-4000:]
    except Exception as error:
        result["failure_reason"] = str(error)
        prefix.with_suffix(".stderr.log").write_text(traceback.format_exc())
    finally:
        if owned:
            lock.unlink()
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(prefix.with_suffix(".json"))
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
