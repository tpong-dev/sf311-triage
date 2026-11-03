"""API tests with stub predictors. Uses a synthetic image; no real 311 photos."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import api
from app.predictors import MedianResolutionPredictor, PriorRouter, ResolutionPredictor

STATS = {
    "label_prior": {"Graffiti Public": 60, "Encampment": 30, "Other": 10},
    "agency_by_label": {"Graffiti Public": "PW - Street and Environmental Services"},
    "overall": {"n": 1000, "median_hours": 12.0, "p_late": 0.2},
    "by_category": {"Graffiti Public": {"n": 100, "median_hours": 30.0, "p_late": 0.3}},
    "by_subtype": {"bulky_items": {"n": 200, "median_hours": 8.0, "p_late": 0.1}},
    "by_subtype_source": {"bulky_items|Phone": {"n": 80, "median_hours": 5.0, "p_late": 0.05}},
}


@pytest.fixture
def client():
    api.app.state.router = PriorRouter(STATS)
    api.app.state.resolver = MedianResolutionPredictor(STATS)
    with TestClient(api.app) as c:
        yield c
    api.app.state.router = api.app.state.resolver = None


def png_bytes(color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "router": "prior-stub",
        "router_is_stub": True,
        "resolver": "subtype-median-stub",
        "resolver_is_stub": True,
    }


def test_route_returns_top3(client):
    r = client.post(
        "/route",
        files={"image": ("x.png", png_bytes(), "image/png")},
        data={"lat": "37.7793", "lon": "-122.4193", "timestamp": "2026-10-01T09:30:00"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "Graffiti Public"
    assert [t["label"] for t in body["top3"]] == ["Graffiti Public", "Encampment", "Other"]
    assert body["top3"][0]["prob"] == pytest.approx(0.6)
    assert body["agency_hint"] == "PW - Street and Environmental Services"
    assert body["is_stub"] is True


def test_route_timestamp_optional_and_outside_sf_noted(client):
    r = client.post(
        "/route", files={"image": ("x.png", png_bytes(), "image/png")}, data={"lat": "40.7", "lon": "-74.0"}
    )
    assert r.status_code == 200
    assert any("outside San Francisco" in n for n in r.json()["notes"])


def test_route_rejects_non_image(client):
    r = client.post(
        "/route", files={"image": ("x.txt", b"hello", "text/plain")}, data={"lat": "37.77", "lon": "-122.42"}
    )
    assert r.status_code == 415


def test_route_rejects_corrupt_image(client):
    r = client.post(
        "/route", files={"image": ("x.png", b"not a png", "image/png")}, data={"lat": "37.77", "lon": "-122.42"}
    )
    assert r.status_code == 422


def test_route_validates_coordinates(client):
    r = client.post(
        "/route", files={"image": ("x.png", png_bytes(), "image/png")}, data={"lat": "123", "lon": "-122.42"}
    )
    assert r.status_code == 422


def test_route_does_not_write_files(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client.post("/route", files={"image": ("x.png", png_bytes(), "image/png")}, data={"lat": "37.77", "lon": "-122.42"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "payload, hours, basis",
    [
        ({"subtype": "bulky_items", "source": "Phone"}, 5.0, "subtype+source"),
        ({"subtype": "bulky_items", "source": "Web"}, 8.0, "subtype"),
        ({"subtype": "unknown_thing", "category": "Graffiti Public"}, 30.0, "category"),
        ({"subtype": "unknown_thing"}, 12.0, "all requests"),
    ],
)
def test_resolution_fallback_chain(client, payload, hours, basis):
    r = client.post("/resolution", json={"lat": 37.77, "lon": -122.42, "timestamp": "2026-10-01T09:30:00", **payload})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["median_hours"] == hours
    assert body["basis"].startswith(basis)
    assert body["late_threshold_hours"] == 168
    assert 0 <= body["p_late"] <= 1


def test_resolution_rejects_bad_source(client):
    r = client.post("/resolution", json={"subtype": "bulky_items", "lat": 37.77, "lon": -122.42, "source": "Fax"})
    assert r.status_code == 422


def test_default_loaders_fall_back_to_stubs(tmp_path, monkeypatch):
    monkeypatch.delenv("SF311_ROUTER", raising=False)
    from app.predictors import PhotoRouter

    assert PhotoRouter.load(tmp_path).is_stub
    assert ResolutionPredictor.load(tmp_path).is_stub


def test_torchscript_artifact_is_picked_up(tmp_path, monkeypatch):
    """The documented models/image/ format loads and predicts (tiny fake model)."""
    torch = pytest.importorskip("torch")
    import json

    from app.predictors import PhotoRouter

    labels = ["Graffiti Public", "Encampment", "Sewer", "Other"]

    class Tiny(torch.nn.Module):
        def forward(self, x):
            m = x.mean(dim=(2, 3))  # (N, 3)
            return torch.cat([m, m[:, :1] * 0], dim=1) * torch.tensor([1.0, 5.0, 0.0, 0.0])

    (tmp_path / "image").mkdir()
    torch.jit.script(Tiny()).save(str(tmp_path / "image" / "model.pt"))
    (tmp_path / "image" / "labels.json").write_text(json.dumps(labels))
    monkeypatch.delenv("SF311_ROUTER", raising=False)

    router = PhotoRouter.load(tmp_path)
    assert not router.is_stub
    res = router.predict(png_bytes((255, 255, 255)), 37.77, -122.42)
    assert res.label == "Encampment"
    assert len(res.top3) == 3
    assert sum(p for _, p in res.top3) <= 1.0 + 1e-6
