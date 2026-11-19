"""Compare two synthesis output directories file by file (SHA-256).

    python compare_outputs.py DIR_A DIR_B

Every regular file under either directory is compared by relative path.
Files present on only one side are reported as differences. Exit status:
0 identical, 1 any difference, 2 usage error.
"""
import argparse
import hashlib
import os
import sys


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _list_files(root: str):
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            files.add(os.path.relpath(path, root).replace(os.sep, "/"))
    return files


def compare(dir_a: str, dir_b: str, max_listed: int = 50) -> bool:
    files_a = _list_files(dir_a)
    files_b = _list_files(dir_b)
    only_a = sorted(files_a - files_b)
    only_b = sorted(files_b - files_a)
    common = sorted(files_a & files_b)

    differing = []
    for rel in common:
        if _sha256(os.path.join(dir_a, rel)) != _sha256(os.path.join(dir_b, rel)):
            differing.append(rel)

    def _report(title, items):
        print(f"{title}: {len(items)}")
        for rel in items[:max_listed]:
            print(f"  {rel}")
        if len(items) > max_listed:
            print(f"  ... and {len(items) - max_listed} more")

    print(f"A: {dir_a} ({len(files_a)} files)")
    print(f"B: {dir_b} ({len(files_b)} files)")
    print(f"identical: {len(common) - len(differing)} / {len(common)} common files")
    _report("only in A", only_a)
    _report("only in B", only_b)
    _report("content differs", differing)

    same = not (only_a or only_b or differing)
    print("RESULT: IDENTICAL" if same else "RESULT: DIFFERENT")
    return same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    args = parser.parse_args()
    for d in (args.dir_a, args.dir_b):
        if not os.path.isdir(d):
            print(f"Not a directory: {d}", file=sys.stderr)
            return 2
    return 0 if compare(args.dir_a, args.dir_b) else 1


if __name__ == "__main__":
    sys.exit(main())
