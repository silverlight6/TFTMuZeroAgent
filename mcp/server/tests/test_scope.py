from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("change,accepted", [
    ("extension", True), ("staged_outside_restored", False),
    ("committed_outside_restored", False), ("outside_rename", False),
])
def test_scope_checks_every_git_layer_and_rename_source(tmp_path, change, accepted):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(tmp_path), *arguments], text=True).strip()

    scripts = tmp_path / "mcp/server/scripts"
    scripts.mkdir(parents=True)
    script = scripts / "check_scope.py"
    shutil.copyfile(Path(__file__).parents[1] / "scripts/check_scope.py", script)
    outside = tmp_path / "simulator.py"
    outside.write_text("original simulator\n")
    git("init", "-q")
    git("config", "user.name", "Scope acceptance")
    git("config", "user.email", "scope@example.invalid")
    git("add", ".")
    git("commit", "-qm", "reviewed base")
    base = git("rev-parse", "HEAD")
    if change == "extension":
        (tmp_path / "mcp/server/allowed.py").write_text("extension\n")
    elif change == "staged_outside_restored":
        outside.write_text("staged change\n")
        git("add", "simulator.py")
        outside.write_text("original simulator\n")
    elif change == "committed_outside_restored":
        outside.write_text("committed change\n")
        git("add", "simulator.py")
        git("commit", "-qm", "outside change")
        outside.write_text("original simulator\n")
        git("add", "simulator.py")
    else:
        git("mv", "simulator.py", "mcp/server/moved.py")
    result = subprocess.run([sys.executable, str(script), base], capture_output=True, text=True)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    if not accepted:
        assert "simulator.py" in result.stderr
