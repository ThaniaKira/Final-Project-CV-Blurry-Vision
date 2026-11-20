"""Website smoke tests. Do not re-train or alter the user's model files."""
from fastapi.testclient import TestClient
from src.site import app

client = TestClient(app)


def test_pages_are_separate_routes():
    landing = client.get("/")
    studio = client.get("/studio")
    guide = client.get("/guide")
    assert landing.status_code == studio.status_code == guide.status_code == 200
    assert 'id="heroHeading"' in landing.text
    assert 'id="analyzeBtn"' not in landing.text
    assert 'id="analyzeBtn"' in studio.text
    assert 'id="heroHeading"' not in studio.text
    assert 'SIX POSES' in guide.text


def test_static_assets_and_status():
    for path in ("/static/css/site.css", "/static/js/studio.js", "/static/img/holo-hand.svg"):
        assert client.get(path).status_code == 200
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["mode"] in ("rules", "trained")
    assert "label" in status.json()


def test_image_validation_does_not_run_model():
    response = client.post("/api/analyze", files={"image": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()
