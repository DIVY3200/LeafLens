import io

import pytest

from app import create_app

from conftest import CARD_CLASSES

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(carded_model):
    return create_app(carded_model).test_client()


def post_image(client, data, name="leaf.jpg"):
    return client.post("/api/predict", data={"image": (data, name)}, content_type="multipart/form-data")


def test_home_page_shows_the_models_own_numbers(client):
    html = client.get("/").get_data(as_text=True)
    assert "90.0%" in html and "80.0%" in html  # lab and field accuracy from the card
    assert "Tiny test model" in html
    assert html.count('class="plant-card"') == 3  # Cassava, Corn, Tomato


def test_a_leaf_photo_gets_a_diagnosis(client, photo):
    with open(photo(), "rb") as f:
        response = post_image(client, f)
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"]["label"] in CARD_CLASSES
    assert len(body["top"]) == 3


def test_missing_or_unreadable_images_are_rejected(client):
    assert client.post("/api/predict").status_code == 400
    response = post_image(client, io.BytesIO(b"not an image"))
    assert response.status_code == 400
    assert "couldn't be read" in response.get_json()["error"]


def test_uploads_over_10_mb_are_rejected(client):
    response = post_image(client, io.BytesIO(b"0" * (11 * 1024 * 1024)))
    assert response.status_code == 413


def test_health_check_reports_the_model(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["model"] == "Tiny test model"
    assert body["classes"] == 3
    assert body["input_size"] == [32, 32]


def test_health_check_reports_the_deployed_commit(client, monkeypatch):
    assert client.get("/api/health").get_json()["commit"] is None
    monkeypatch.setenv("RENDER_GIT_COMMIT", "d17c005a1b2c3d4e5f60718293a4b5c6d7e8f901")
    assert client.get("/api/health").get_json()["commit"] == "d17c005"
