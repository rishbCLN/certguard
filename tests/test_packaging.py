import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_built_wheel_contains_loadable_grammar_resources(tmp_path) -> None:
    project = Path(__file__).parents[1]
    output = tmp_path / "dist"
    environment = {**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")}
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(output.glob("certguard-*.whl"))
    expected = {
        "certguard/data/grammar/coursera.example.yaml",
        "certguard/data/grammar/nptel.example.yaml",
        "certguard/data/grammar/swayam.example.yaml",
        "certguard/data/issuers.json",
    }
    with zipfile.ZipFile(wheel) as archive:
        assert expected <= set(archive.namelist())

    script = (
        "from importlib.resources import files; "
        "root=files('certguard.data').joinpath('grammar'); "
        "assert root.joinpath('nptel.example.yaml').is_file()"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**environment, "PYTHONPATH": str(wheel)},
        check=True,
        capture_output=True,
        text=True,
    )
