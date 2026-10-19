"""Job queue for MetaGPT runs: one job at a time, bounded queue, hard timeout,
state persisted to disk so status survives a worker restart."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "timed_out"}
MAX_IDEA_CHARS = 1200
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
SKIP_DIRS = {"storage", ".git", "__pycache__", "node_modules"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    id: str
    owner: str
    idea: str
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cost_usd: float | None = None
    files: list[str] = field(default_factory=list)

    def public(self) -> dict:
        data = asdict(self)
        data.pop("owner")
        return data


class QueueFullError(RuntimeError):
    pass


class JobTimeoutError(RuntimeError):
    pass


# A runner executes one job in `job_dir` and returns {"cost_usd": float | None}.
Runner = Callable[[Job, Path], Awaitable[dict]]


class JobManager:
    def __init__(self, root: Path, runner: Runner, timeout_s: float, max_queued: int = 5, keep: int = 50):
        self.root = root
        self.runner = runner
        self.timeout_s = timeout_s
        self.max_queued = max_queued
        self.keep = keep
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    # -- persistence -------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def workspace(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "workspace"

    def _save(self, job: Job) -> None:
        path = self.job_dir(job.id) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError):
                logger.warning("Skipping unreadable job file %s", path)
                continue
            if job.status not in TERMINAL:
                job.status, job.error, job.finished_at = "failed", "Worker restarted before the job finished.", _now()
                self._save(job)
            self.jobs[job.id] = job

    def _prune(self) -> None:
        done = sorted((j for j in self.jobs.values() if j.status in TERMINAL), key=lambda j: j.created_at)
        for job in done[: max(0, len(done) - self.keep)]:
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
            self.jobs.pop(job.id, None)

    # -- public API --------------------------------------------------------
    def submit(self, owner: str, idea: str) -> Job:
        idea = idea.strip()
        if not idea or len(idea) > MAX_IDEA_CHARS:
            raise ValueError(f"Idea must be 1-{MAX_IDEA_CHARS} characters.")
        if not owner.strip():
            raise ValueError("owner is required")
        pending = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
        if pending >= self.max_queued:
            raise QueueFullError("The worker is busy. Try again later.")
        job = Job(id=uuid.uuid4().hex, owner=owner, idea=idea)
        self.jobs[job.id] = job
        self._save(job)
        self._queue.put_nowait(job.id)
        return job

    def get(self, job_id: str, owner: str) -> Job | None:
        job = self.jobs.get(job_id)
        return job if job and job.owner == owner else None

    def archive(self, job: Job) -> bytes:
        workspace = self.workspace(job.id)
        buffer = io.BytesIO()
        total = 0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in job.files:
                path = workspace / rel
                total += path.stat().st_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Generated output is too large to download.")
                zf.write(path, rel)
        return buffer.getvalue()

    def pending_count(self) -> int:
        return self._queue.qsize()

    # -- execution ---------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is not None:
                await self.run_one(job)
            self._queue.task_done()

    async def run_one(self, job: Job) -> None:
        job.status, job.started_at = "running", _now()
        self._save(job)
        workspace = self.workspace(job.id)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            result = await asyncio.wait_for(self.runner(job, self.job_dir(job.id)), timeout=self.timeout_s)
            job.cost_usd = result.get("cost_usd")
            job.files = collect_files(workspace)
            job.status = "succeeded" if job.files else "failed"
            if not job.files:
                job.error = "MetaGPT finished without producing any files."
        except TimeoutError:
            job.status, job.error = "timed_out", f"Job exceeded the {int(self.timeout_s)}s limit."
            job.files = collect_files(workspace)
        except Exception as exc:  # recorded on the job; details go to the log
            logger.exception("Job %s failed", job.id)
            job.status, job.error = "failed", str(exc)[:500] or type(exc).__name__
            job.files = collect_files(workspace)
        job.finished_at = _now()
        self._save(job)
        self._prune()


def collect_files(workspace: Path) -> list[str]:
    if not workspace.exists():
        return []
    files = []
    for path in sorted(workspace.rglob("*")):
        rel = path.relative_to(workspace)
        if path.is_file() and not (set(rel.parts) & SKIP_DIRS):
            files.append(rel.as_posix())
    return files
