"""Runs one MetaGPT job in a child process so a timeout can kill it outright."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from worker.jobs import Job

# Only these variables reach the child: no unrelated server secrets.
PASSTHROUGH = (
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "PYTHONPATH",
    "METAGPT_API_TYPE",
    "METAGPT_MODEL",
    "METAGPT_API_KEY",
    "METAGPT_BASE_URL",
    "METAGPT_N_ROUND",
    "METAGPT_INVESTMENT",
)


async def subprocess_runner(job: Job, job_dir: Path) -> dict:
    (job_dir / "request.json").write_text(json.dumps({"idea": job.idea}), encoding="utf-8")
    env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
    # MetaGPT resolves its workspace and storage relative to this root.
    env["METAGPT_PROJECT_ROOT"] = str(job_dir)
    # A per-job home keeps the server's own ~/.metagpt config (if any) out of the run.
    env["HOME"] = env["USERPROFILE"] = str(job_dir)
    env["PYTHONUNBUFFERED"] = "1"
    log = open(job_dir / "log.txt", "wb")
    kwargs = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "worker.metagpt_job",
        str(job_dir),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
        **kwargs,
    )
    try:
        code = await proc.wait()
    except asyncio.CancelledError:
        _kill_tree(proc.pid)
        await proc.wait()
        raise
    finally:
        log.close()
    result_path = job_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    if code != 0:
        raise RuntimeError(result.get("error") or f"MetaGPT exited with code {code}; see log.txt")
    return result


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
