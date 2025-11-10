import subprocess
import sys
from pathlib import Path

import pytest

paths = list(Path(__file__).parent.glob("*.py"))


@pytest.mark.parametrize("path", paths, ids=[d.name for d in paths])
def test(path) -> None:
    subprocess.run([sys.executable, path])


@pytest.mark.parametrize("path", paths, ids=[d.name for d in paths])
def test_np3(path) -> None:
    subprocess.run(["mpiexec", "-np", "3", sys.executable, path])
