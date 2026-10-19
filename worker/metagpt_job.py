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

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def llm_config() -> dict:
    api_key = os.getenv("METAGPT_API_KEY")
    if not api_key:
        raise RuntimeError("METAGPT_API_KEY is not set on the worker.")
    api_type = os.getenv("METAGPT_API_TYPE", "gemini")
    config = {
        "api_type": api_type,
        "model": os.getenv("METAGPT_MODEL", "gemini-3.5-flash"),
        "api_key": api_key,
        "timeout": 120,
        # MetaGPT answers with long JSON documents; Gemini's thinking tokens also count
        # against this limit, and 4096 truncated the PRD mid-string in testing.
        "max_token": int(os.getenv("METAGPT_MAX_TOKENS", "16384")),
    }
    if api_type == "gemini":
        # MetaGPT's native Gemini client sets use_system_prompt=False and silently drops
        # every system instruction (including "return JSON"), so its first action fails.
        # Gemini's OpenAI-compatible endpoint keeps system prompts intact.
        config["api_type"] = "openai"
        config["base_url"] = GEMINI_OPENAI_BASE_URL
        # MetaGPT's cost tables don't know Gemini models; token counts would be wrong.
        config["calc_usage"] = False
    if os.getenv("METAGPT_BASE_URL"):
        config["base_url"] = os.environ["METAGPT_BASE_URL"]
    return config


async def run(job_dir: Path) -> dict:
    from metagpt.actions import WritePRD
    from metagpt.actions.prepare_documents import PrepareDocuments
    from metagpt.config2 import Config
    from metagpt.configs.workspace_config import WorkspaceConfig
    from metagpt.context import Context
    from metagpt.roles import Architect, Engineer, ProductManager, ProjectManager
    from metagpt.team import Team
    from metagpt.utils.common import any_to_str

    idea = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))["idea"]
    config = Config.from_llm_config(llm_config())
    config.workspace = WorkspaceConfig(path=job_dir / "workspace")
    ctx = Context(config=config)

    # use_mgx=False: MetaGPT 1.0 defaults to the MGX environment, which routes every
    # message through a TeamLeader role that this classic pipeline doesn't hire.
    team = Team(context=ctx, use_mgx=False)

    # use_fixed_sop=True: in MetaGPT 1.0 these are tool-using RoleZero agents by default;
    # the fixed SOP restores the classic PRD -> design -> tasks -> code pipeline.
    # RoleZero._react still runs its "quick think" shortcut even with a fixed SOP, and for
    # small ideas that answers in chat instead of starting the pipeline, so turn it off.
    class NoQuickThink:
        async def _quick_think(self):
            return None, ""

    class SopProductManager(NoQuickThink, ProductManager):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # By default PrepareDocuments asks the LLM for a "project path" found in the idea and
            # writes there, so a crafted idea could steer files outside the job's workspace.
            # Ask for a harmless key instead; the project is created inside the workspace.
            self.set_actions(
                [
                    PrepareDocuments(
                        send_to=any_to_str(self),
                        key_descriptions={"project_summary": "a one-line summary of the project"},
                    ),
                    WritePRD,
                ]
            )

    class SopArchitect(NoQuickThink, Architect):
        pass

    class SopProjectManager(NoQuickThink, ProjectManager):
        pass

    team.hire(
        [
            SopProductManager(context=ctx, use_fixed_sop=True),
            SopArchitect(context=ctx, use_fixed_sop=True),
            SopProjectManager(context=ctx, use_fixed_sop=True),
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
