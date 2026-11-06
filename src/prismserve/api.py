import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from prismserve.preprocess import MAX_IMAGE_BYTES, decode_image
from prismserve.scheduler import BatchScheduler, DeadlineExceeded, Overloaded, Unavailable


def create_app(loader, *, max_batch=8, max_pending=64, wait_ms=5):
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = None
        app.state.scheduler = None
        app.state.receiving = 0
        app.state.decoding = 0
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
        decode_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="decode")
        app.state.executor = executor
        app.state.decode_executor = decode_executor
        try:
            backend = await asyncio.get_running_loop().run_in_executor(executor, loader)
            app.state.backend = backend

            async def infer(images):
                return await asyncio.get_running_loop().run_in_executor(
                    executor, backend.predict, images
                )

            scheduler = BatchScheduler(
                infer, max_batch=max_batch, max_pending=max_pending, wait_ms=wait_ms
            )
            await scheduler.start()
            app.state.scheduler = scheduler
        except Exception:
            app.state.startup_failed = True
        yield
        if app.state.scheduler:
            await app.state.scheduler.close()
        # CUDA work cannot be preempted safely; finish the current launch on shutdown.
        executor.shutdown(wait=True, cancel_futures=True)
        decode_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="PrismServe", lifespan=lifespan)

    @app.middleware("http")
    async def request_limits(request, call_next):
        request.state.started = time.monotonic()
        request_id = uuid.uuid4().hex
        if request.url.path == "/v1/predict":
            if app.state.receiving >= max_pending:
                return JSONResponse(
                    {"detail": "request admission full"},
                    status_code=429,
                    headers={"Retry-After": "1", "X-Request-ID": request_id},
                )
            app.state.receiving += 1
            try:
                response = await call_next(request)
            finally:
                app.state.receiving -= 1
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(
            round((time.monotonic() - request.state.started) * 1000, 3)
        )
        return response

    def scheduler():
        current = app.state.scheduler
        if not current or not current.ready:
            raise HTTPException(503, "model unavailable")
        return current

    @app.get("/live")
    def live():
        return {"live": True}

    @app.get("/ready")
    def ready():
        scheduler()
        return {"ready": True, "model_id": app.state.backend.metadata["model_id"]}

    @app.get("/v1/model")
    def model():
        scheduler()
        return app.state.backend.metadata

    @app.get("/v1/stats")
    def stats():
        current = app.state.scheduler
        return current.stats() if current else {"ready": False}

    @app.post("/v1/predict")
    async def predict(
        request: Request, timeout_ms: int = Query(1000, ge=1, le=30000), embedding: bool = False
    ):
        current = scheduler()
        if request.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "send raw image/jpeg or image/png bytes")
        body = bytearray()
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_IMAGE_BYTES:
                        raise HTTPException(413, "image byte cap exceeded")
                    body.extend(chunk)
                # Canceled HTTP tasks must not release slots while decoder threads still run.
                if app.state.decoding >= max_pending:
                    raise Overloaded()
                app.state.decoding += 1
                decode = asyncio.get_running_loop().run_in_executor(
                    app.state.decode_executor, decode_image, bytes(body)
                )

                def release_decode(future):
                    app.state.decoding -= 1
                    # Retrieve failed abandoned work without leaking unhandled exceptions.
                    if not future.cancelled():
                        future.exception()

                decode.add_done_callback(release_decode)
                image = await asyncio.shield(decode)
                remaining = timeout_ms - (time.monotonic() - request.state.started) * 1000
                if remaining < 1:
                    raise DeadlineExceeded()
                work = asyncio.create_task(current.submit(image, remaining))
                try:
                    while not work.done():
                        await asyncio.wait({work}, timeout=0.01)
                        if await request.is_disconnected():
                            work.cancel()
                            raise HTTPException(499, "client disconnected")
                    result = await work
                finally:
                    if not work.done():
                        work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (DeadlineExceeded, TimeoutError) as exc:
            raise HTTPException(504, "request deadline exceeded") from exc
        except Overloaded as exc:
            raise HTTPException(429, "model admission full", headers={"Retry-After": "1"}) from exc
        except Unavailable as exc:
            raise HTTPException(503, "model unavailable") from exc
        if not embedding:
            result.pop("embedding", None)
        return {"model_id": app.state.backend.metadata["model_id"], **result}

    return app
