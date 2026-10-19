"""Child-process entry point: run a MetaGPT software team for one idea.

Usage: python -m worker.metagpt_job <job_dir>

Uses the classic MetaGPT pipeline (product manager, architect, project manager,
engineer with code review). These roles write documents and code but never execute
it; QaEngineer, Engineer2 and DataAnalyst are deliberately left out because they run
generated code or shell commands.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path


def llm_config() -> dict:
    api_key = os.getenv("METAGPT_API_KEY")
    if not api_key:
        raise RuntimeError("METAGPT_API_KEY is not set on the worker.")
    config = {
        "api_type": os.getenv("METAGPT_API_TYPE", "gemini"),
        "model": os.getenv("METAGPT_MODEL", "gemini-3.8-flash"),
        "api_key": api_key,
        "timeout": 120,
    }
    if os.getenv("METAGPT_BASE_URL"):
        config["base_url"] = os.environ["METAGPT_BASE_URL"]
    return config


async def run(job_dir: Path) -> dict:
    from metagpt.config2 import Config
    from metagpt.configs.workspace_config import WorkspaceConfig
    from metagpt.context import Context
    from metagpt.roles import Architect, Engineer, ProductManager, ProjectManager
    from metagpt.team import Team

    idea = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))["idea"]
    config = Config.from_llm_config(llm_config())
    config.workspace = WorkspaceConfig(path=job_dir / "workspace")
    ctx = Context(config=config)

    # use_mgx=False: MetaGPT 1.0 defaults to the MGX environment, which routes every
    # message through a TeamLeader role that this classic pipeline doesn't hire.
    team = Team(context=ctx, use_mgx=False)
    # use_fixed_sop=True: in MetaGPT 1.0 these are tool-using RoleZero agents by default;
    # the fixed SOP restores the classic PRD -> design -> tasks -> code pipeline.
    team.hire(
        [
            ProductManager(context=ctx, use_fixed_sop=True),
            Architect(context=ctx, use_fixed_sop=True),
            ProjectManager(context=ctx, use_fixed_sop=True),
            Engineer(context=ctx, n_borg=5, use_code_review=True),
        ]
    )
    team.invest(float(os.getenv("METAGPT_INVESTMENT", "3.0")))
    history = await team.run(n_round=int(os.getenv("METAGPT_N_ROUND", "5")), idea=idea)
    # Team.run's serialize_decorator logs exceptions and returns None instead of raising.
    if history is None:
        raise RuntimeError("MetaGPT raised an error during the run (traceback in the job's log.txt on the worker).")
    return {"cost_usd": round(team.cost_manager.total_cost, 4)}


def write_home_config(job_dir: Path) -> Path:
    """MetaGPT builds a global default Config from ~/.metagpt/config2.yaml at import
    time and fails if it has no LLM section. The runner points HOME at the job dir, so
    write a per-job file there; it's deleted when the job ends."""
    import yaml

    path = job_dir / ".metagpt" / "config2.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"llm": llm_config()}), encoding="utf-8")
    path.chmod(0o600)
    return path


def main() -> int:
    job_dir = Path(sys.argv[1]).resolve()
    result_path = job_dir / "result.json"
    home_config = None
    try:
        home_config = write_home_config(job_dir)
        result = asyncio.run(run(job_dir))
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0
    except Exception as exc:
        traceback.print_exc()
        result_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}), encoding="utf-8")
        return 1
    finally:
        if home_config is not None:
            home_config.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
