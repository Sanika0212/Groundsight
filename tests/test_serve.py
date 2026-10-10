from pathlib import Path

import httpx
import pytest

import groundsight as gs
from groundsight.serve import make_server, serve_in_background


def test_viewer_lists_and_renders_audits(tmp_path: Path) -> None:
    store = gs.Store(tmp_path / "audits.jsonl")
    with gs.trace("Refund window?", store=store, checker="lexical") as t:
        t.retrieved(["Refunds are accepted within 30 days."])
        t.answered("Refunds are accepted within 30 days.")
    assert t.audit is not None
    server, url = serve_in_background(store, port=0)
    try:
        index = httpx.get(url)
        assert index.status_code == 200
        assert "Groundsight" in index.text
        assert "default-src 'none'" in index.headers["content-security-policy"]
        listing = httpx.get(url + "api/audits").json()
        assert listing[0]["id"] == t.audit.id
        assert listing[0]["groundedness"] == 1.0
        page = httpx.get(url + "audit/" + t.audit.id[:8])
        assert page.status_code == 200
        assert "Refund window?" in page.text
        assert httpx.get(url + "audit/nope").status_code == 404
        assert httpx.get(url + "elsewhere").status_code == 404
        rebinding = httpx.get(url, headers={"Host": "evil.example:7860"})
        assert rebinding.status_code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_refuses_public_interfaces() -> None:
    with pytest.raises(ValueError, match="local machine"):
        make_server(host="0.0.0.0", port=0)
