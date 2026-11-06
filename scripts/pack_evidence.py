"""Export only the requested run, bounded to 128 MiB; no raw images or weights."""

import argparse
import hashlib
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(path for path in args.run.rglob("*") if path.is_file())
    files = [path for path in files if path.suffix in {".json", ".txt", ".npz"}]
    if any(path.is_symlink() for path in files):
        raise SystemExit("symlink exports are not supported")
    if sum(path.stat().st_size for path in files) > 128 * 1024 * 1024:
        raise SystemExit(
            "artifact cap exceeded; inspect/reduce trace, never export dataset/weights"
        )
    with tarfile.open(args.output, "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(args.run)))
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest + "\n")
    print(digest)


if __name__ == "__main__":
    main()
