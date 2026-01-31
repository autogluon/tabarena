from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Resolve paths relative to *this file*
_THIS_FILE = Path(__file__).resolve()

# /home/ubuntu/workspace/code/tabarena/tabflow/tabflow/cli
_CLI_DIR = _THIS_FILE.parent

# /home/ubuntu/workspace/code/tabarena/tabflow
_TABFLOW_DIR = _CLI_DIR.parent.parent

# /home/ubuntu/workspace/code/tabarena/tabflow/docker/build_docker.sh
DEFAULT_SCRIPT_PATH = _TABFLOW_DIR / "docker" / "build_docker.sh"

# /home/ubuntu/workspace/code
DEFAULT_WORKDIR = _TABFLOW_DIR.parent.parent


def build_docker(
    *,
    tag: str,
    source_account: str,
    target_account: str,
    repo_name: str = "tabarena",
    region: str = "us-west-2",
    script_path: str | Path = DEFAULT_SCRIPT_PATH,
    workdir: str | Path = DEFAULT_WORKDIR,
    extra_env: dict[str, str] | None = None,
) -> None:
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"builder script not found: {script_path}")

    if workdir is None:
        workdir = script_path.parent
    else:
        workdir = Path(workdir).resolve()
        if not workdir.exists():
            raise FileNotFoundError(f"working directory does not exist: {workdir}")

    cmd = [
        "bash",
        str(script_path),
        repo_name,
        tag,
        source_account,
        target_account,
        region,
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    output = []
    for line in process.stdout:
        print(line, end="")
        output.append(line)

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={returncode})\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- output ---\n{''.join(output)}"
        )
