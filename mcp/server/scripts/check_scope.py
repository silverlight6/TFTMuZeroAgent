"""Check the complete worktree against the accepted extension path scope."""

import argparse
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("base", help="Fixed reviewed Git base revision")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]


def git_paths(*arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments],
                            check=True, capture_output=True)
    return {path.decode() for path in result.stdout.split(b"\0") if path}


paths = git_paths("diff", "--no-renames", "--name-only", "-z", args.base, "HEAD", "--")
paths |= git_paths("diff", "--no-renames", "--name-only", "-z", args.base, "--")
paths |= git_paths("diff", "--cached", "--no-renames", "--name-only", "-z", args.base, "--")
paths |= git_paths("diff", "--no-renames", "--name-only", "-z", "--")
paths |= git_paths("ls-files", "--others", "--exclude-standard", "-z")
outside = sorted(path for path in paths if not path.startswith("mcp/"))
if outside:
    raise SystemExit("Out-of-scope paths:\n" + "\n".join(outside))
print(f"Scope passed for {len(paths)} changed paths against {args.base}.")
