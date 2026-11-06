import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from PIL import Image

from prismserve.api import create_app


class Backend:
    metadata = {"model_id": "test-contract-backend", "device": "cpu"}

    def predict(self, images):
        return [
            {"top5": [{"label": "fixture", "probability": 1}], "embedding": [1.0]}
            for image in images
        ]


def encoded():
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), "red").save(buffer, "PNG")
    return buffer.getvalue()


def test_http_image_contract_metadata_and_readiness():
    with TestClient(create_app(lambda: Backend())) as client:
        assert client.get("/ready").status_code == 200
        result = client.post(
            "/v1/predict", content=encoded(), headers={"content-type": "image/png"}
        )
        assert result.status_code == 200
        assert result.json()["model_id"] == "test-contract-backend"
        assert "embedding" not in result.json()
        assert "x-request-id" in result.headers
        assert client.get("/v1/stats").json()["completed"] == 1
        assert client.post("/v1/predict", content=b"bad").status_code == 415
        assert (
            client.post(
                "/v1/predict", content=b"bad", headers={"content-type": "image/png"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/predict?timeout_ms=0", content=encoded(), headers={"content-type": "image/png"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/predict", content=b"x" * (2097152 + 1), headers={"content-type": "image/png"}
            ).status_code
            == 413
        )


def test_failed_model_load_is_never_ready():
    def fail():
        raise ValueError("corrupt artifact")

    with TestClient(create_app(fail)) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
        assert (
            client.post(
                "/v1/predict", content=encoded(), headers={"content-type": "image/png"}
            ).status_code
            == 503
        )


def test_extreme_aspect_ratio_returns_422_and_normal_request_recovers():
    buffer = io.BytesIO()
    Image.new("RGB", (1, 16384), "red").save(buffer, "PNG")
    with TestClient(create_app(lambda: Backend())) as client:
        response = client.post(
            "/v1/predict", content=buffer.getvalue(), headers={"content-type": "image/png"}
        )
        assert response.status_code == 422
        assert "resize" in response.json()["detail"]
        assert client.get("/v1/stats").json()["admitted"] == 0
        assert client.get("/ready").status_code == 200
        assert (
            client.post(
                "/v1/predict", content=encoded(), headers={"content-type": "image/png"}
            ).status_code
            == 200
        )


def test_backend_failure_disables_readiness():
    class Broken(Backend):
        def predict(self, images):
            raise RuntimeError("device failure")

    with TestClient(create_app(lambda: Broken())) as client:
        assert (
            client.post(
                "/v1/predict", content=encoded(), headers={"content-type": "image/png"}
            ).status_code
            == 503
        )
        assert client.get("/ready").status_code == 503


def test_numerical_model_failure_disables_readiness():
    import torch

    from prismserve.model import VisionModel

    class Features(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x), 512, 1, 1), 2.0)

    model = VisionModel.__new__(VisionModel)
    model.device = torch.device("cpu")
    model.normalize = lambda x: x
    model.features = Features()
    model.classifier = torch.nn.Linear(512, 10)
    model.metadata = {"model_id": "overflow-regression"}
    with torch.no_grad():
        model.classifier.weight.fill_(3e38)
        model.classifier.bias.zero_()
    with TestClient(create_app(lambda: model)) as client:
        assert client.get("/ready").status_code == 200
        assert (
            client.post(
                "/v1/predict", content=encoded(), headers={"content-type": "image/png"}
            ).status_code
            == 503
        )
        assert client.get("/ready").status_code == 503


def test_timed_out_decode_keeps_its_capacity_until_thread_finishes(monkeypatch):
    from prismserve import api

    entered = threading.Barrier(3)
    release = threading.Event()
    real_decode = api.decode_image

    def slow_decode(value):
        entered.wait(timeout=5)
        release.wait(timeout=5)
        return real_decode(value)

    monkeypatch.setattr(api, "decode_image", slow_decode)
    app = create_app(lambda: Backend(), max_batch=1, max_pending=2)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:

        def request():
            return client.post(
                "/v1/predict?timeout_ms=100",
                content=encoded(),
                headers={"content-type": "image/png"},
            )

        tasks = [pool.submit(request) for _ in range(2)]
        try:
            entered.wait(timeout=5)
            assert [task.result().status_code for task in tasks] == [504, 504]
            assert request().status_code == 429
            assert app.state.decoding == 2
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while app.state.decoding and time.monotonic() < deadline:
            time.sleep(0.005)
        assert app.state.decoding == 0
