# Architecture and invariants

## Input and model identity

`POST /v1/predict` accepts one raw JPEG/PNG, never an image URL or filesystem path. It limits the body while streaming, before a complete body is materialized. An image may contain at most 2 MiB of bytes and 4 million source pixels; animated and unsupported formats are rejected. Before RGB conversion, the projected shorter-side resize is also capped at 1,048,576 pixels. This rejects extreme aspect ratios with HTTP 422 before a tiny image can amplify into a multi-gigabyte intermediate. Accepted images retain the authentic transform: PIL converts to RGB, bilinearly resizes the shorter side to 256 with antialiasing, and center crops to 224. EXIF orientation is not applied, matching the stock transform contract on the opened image. The decoder does not fetch external resources.

The selected torchvision `ResNet18_Weights.IMAGENET1K_V1` contract uses this geometry and per-channel mean `(0.485, 0.456, 0.406)` / standard deviation `(0.229, 0.224, 0.225)`. [Official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Native division/subtraction/division order matches the framework transform exactly in the CPU tests.

Weights are loaded from one local file only, capped at 50 MiB and checked against a full pinned SHA-256 before `torch.load(..., weights_only=True)`. Weight bytes are read once for verification and deserialization. Head files are trusted local NPZ artifacts with fixed keys, finite values, dimension checks and a 1 MiB cap; hashes identify their content but do not authenticate the author. The evaluation loader also checks the frozen training report's head hash. An API request cannot choose a head or weights file. Serving a changed model requires a new process and produces a new model ID; there is no hot-swap or rollout controller.

## Admission and cancellation

One event loop owns queue state. At most `max_pending` images are queued or executing, including images whose request has timed out while their GPU operation is still running. At most the same number of HTTP requests are reading bodies or awaiting results. Four decoder threads have a separate bounded pending count. Their underlying futures are shielded: an HTTP timeout does not release a decoder slot while the thread is still decoding.

Queued cancellations remove the work and reclaim its slot immediately. In-flight cancellation marks its future canceled; the batch still executes, keeps its capacity until completion, and discards the result. Deadlines include upload, geometry, queue wait, model and response preparation in the endpoint. The scheduler also checks deadlines before dispatch and before publishing results. An earliest deadline can shorten the first request's batching window.

Batch execution is serial on one dedicated model thread. Request/result correspondence is positional and checked by output cardinality. A backend exception or cardinality mismatch fails all admitted futures with unavailable, rejects subsequent work and changes readiness to 503. It does not retry an unknown CUDA failure or claim the device is safe to reuse. Normal malformed input affects only its request.

Shutdown fails outstanding clients, cancels the scheduler, and waits for active model/decoder threads to finish. There is no guarantee of bounded shutdown under a hung native GPU call; the external process/VM deadline is the last stop. The HTTP client load tool has a 30-second process termination grace period before forcefully killing its owned subprocess.

## Kernel and strong baselines

Each Triton program writes a spatial tile in one `(batch, channel)` output plane. Output writes are contiguous; RGB input reads have channel stride three for ordinary NHWC and use supplied strides for views. Programs mask spatial tails. FP32 output and disabled FP fusion make the numerical intent explicit, although the GPU gate still permits small floating-point differences and checks downstream logits.

This kernel targets fusion of conversion, normalization and layout, not convolution or resizing. Coalesced output versus strided RGB reads is a deliberate tradeoff to examine in profiling. Fixed block size 256 is an unverified initial implementation, not a tuned optimum. A meaningful next iteration uses the recorded trace and bandwidth/timing results to compare alternatives; no optimization claim precedes that evidence.

`torch.compile(normalize_native, fullgraph=True)` can already fuse the reference operations. CUDA timing excludes first calls, uses warmups and synchronized events, and retains three shuffled-order measurement rounds. Tensor-pipeline timing excludes decoding, upload, CPU geometry and HTTP; the HTTP experiment includes them. Input/output allocation is included consistently in the normalization comparison. NVIDIA/CUDA/Triton/PyTorch metadata and a Chrome profiler trace are saved.

## Operational boundaries

This is a single-process local research service, with no public authentication, TLS termination, persistent telemetry store, autoscaling, multi-GPU scheduler, durable queue, distributed rollout or trained-from-scratch backbone. Bind it to loopback; an approved cloud run uses SSH access and local load generation. Softmax scores are descriptive model outputs, not calibrated risk estimates. Metrics aggregate counts and timing, never raw image bytes.


## Numerical and metric failure boundaries

Research head archives are limited to1MiB compressed and128KiB expanded, with exactly weight/bias arrays of the expected shape. Only finite float32-representable floating parameters are accepted. Warmup and each inference check finite features, logits, probabilities and embeddings; a numerical execution error disables readiness through the scheduler failure path.

These are model-work lifecycle counters, not guaranteed HTTP delivery counts: a connection can fail after a result is published. Terminal scheduler outcomes are counted once per admitted work item, even when a worker and caller notice the same deadline. `pending` measures retained execution/admission capacity; timed-out active GPU work may still occupy it. `unresolved` counts admitted items without a terminal outcome. Thus `admitted = completed + expired + cancelled + failed + unresolved`; rejected work is not admitted.
