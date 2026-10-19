"""Client for the MetaGPT worker service."""

from __future__ import annotations

import httpx


class JobClientError(RuntimeError):
    pass


class JobClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 30):
        self._base_url = base_url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout_s

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, headers=self._headers, timeout=self._timeout
            ) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise JobClientError(
                "Couldn't reach the MetaGPT worker. It may be starting up; try again shortly."
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            raise JobClientError(str(detail or f"Worker returned HTTP {response.status_code}"))
        return response

    async def submit(self, owner: str, idea: str) -> dict:
        return (await self._request("POST", "/jobs", json={"owner": owner, "idea": idea})).json()

    async def status(self, owner: str, job_id: str) -> dict:
        return (await self._request("GET", f"/jobs/{job_id}", params={"owner": owner})).json()

    async def archive(self, owner: str, job_id: str) -> bytes:
        return (await self._request("GET", f"/jobs/{job_id}/archive", params={"owner": owner})).content
