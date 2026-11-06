# Independent automated engineering review

Reviewed 2026-10-23 by a separate automated reviewer. This evaluates the artifact relative to an intern/new-grad portfolio; it is not an employer endorsement, hiring guarantee, or proof of the applicant's personal understanding.

**Verdict after independent rechecks: accept the bounded CPU vision and model-serving artifact.** Three material correctness findings were reproduced and fixed during this review. A subsequent owner-run load experiment exposed double-counted deadline expirations; the accounting fix also passed an independent recheck. The CUDA/Triton branch, GPU performance, GPU HTTP operation and cloud deployment remain unverified. GPU execution remains deferred. Do not present this as a completed GPU project before those gates execute on NVIDIA hardware.

## Findings reproduced and independently closed

### High: source-image caps did not bound resize intermediates

The original decoder accepted a valid 141-byte PNG of size 1×16,384: only 16,384 source pixels, well within the four-million-pixel cap. Resizing the shorter side to 256 would create 256×4,194,304 pixels—over one billion RGB pixels. I intercepted the resize call before it allocated that memory. Four decoder threads and request deadlines would not make such an allocation safe.

The current decoder computes the exact projected shorter-side resize dimensions and rejects intermediates over 1,048,576 pixels before RGB conversion or resize. I independently checked 1×16,384, 16,384×1, 1×17 and 17×1 inputs: all return HTTP 422. A 1×16 boundary image still matches the official torchvision preprocessing exactly. A separate real TCP server with the pinned pretrained ResNet18 rejected the extreme image, then returned 200 for a normal prediction and remained ready. This closes the input-resource finding without replacing accepted-image geometry with a different transform.

### Medium: a finite artifact could become nonfinite at execution precision

The original head loader accepted finite float64 weights of 1e100 with valid dimensions. Converting them to the classifier's float32 precision overflowed to infinity. Warmup returned NaN probabilities without raising, allowing a nominally ready model whose result could not be serialized as strict JSON. The artifact path is trusted local input, not an arbitrary remote upload, but this contradicted the fail-closed readiness contract.

The revised loader bounds expanded NPZ contents and requires float32/float64 arrays with finite, float32-representable values. Execution additionally rejects nonfinite features, logits, norms, probabilities and embeddings. I independently reran two actual model constructions: 1e100 float64 parameters now fail validation; 3e38 float32 parameters pass representation checks but fail real warmup when matrix multiplication overflows. Both loaders leave readiness at 503. The permanent suite also exercises numerical failure during service operation and verifies that the scheduler disables readiness. This closes the finding for the tested contracts.

The compact probe outcomes are retained in [independent-review.json](../evidence/independent-review.json). The original large-allocation probe did not actually allocate the projected image.

### Medium: a tiny malformed NPY member bypassed the expanded-byte bound

The final packaging pass found that the ZIP member-size limit did not bound NumPy's allocation: a 506-byte NPZ with 296 expanded ZIP bytes contained an NPY header declaring one billion float32 values. `np.load` attempted a four-billion-byte array allocation before the loader could reject its shape. I patched NumPy's allocation call to intercept this safely; no large array was allocated. This affects the trusted local optional head artifact, not the HTTP upload interface.

The revised loader reads bounded member bytes and validates the NPY version, header, exact shape, supported dtype and exact payload length before constructing an array. My repeated huge-shape and truncated-payload probes rejected both inputs before guarded array-construction calls. Eight combinations of float32/float64, C/Fortran storage and NPY v1/v2 preserved the expected values exactly. The actual retained learned-head arrays also exactly matched the previous loader's output. This closes the bounded-allocation finding for the reviewed format.

### Deadline metrics: duplicate terminal observations

An owner-run final HTTP experiment recorded 98 admitted items but 96 completed plus four expired items after draining. The worker deadline check and the submit-side timeout could both count one expiration. The revised scheduler records one terminal outcome per work item and exposes unresolved work separately from occupied capacity. My independent probe deliberately blocked the event loop past a two-millisecond deadline: it returned one deadline exception with admitted=1, expired=1, completed=0, pending=0 and unresolved=0. The permanent regression and final full suite passed. These are model-work lifecycle counts, not a guarantee that an HTTP client received a response after a successful future publication. I inspected the original owner result; I did not independently rerun the unfixed implementation for this third finding.

## Checks actually performed

| Check | Independent result |
|---|---|
| Initial `make test-cpu` | Wheel installation, lint/format and 27 CPU tests passed; 11 GPU tests deselected. |
| Final CPU suite after all fixes | From the extracted final archive: `pytest -q -m 'not gpu' -W error`: 40 passed, 11 GPU tests deselected, 3.73 seconds. Lint and format checks also passed. |
| Source provenance | All 35 final manifest entries match both working files and archived bytes. |
| Public input integrity | Complete cached Imagenette archive and ResNet18 weights matched their pinned SHA-256 values. This checks captured bytes, not publisher signatures. |
| Learned parameters and selection | Reran all three head candidates from the retained real feature cache. Every exported parameter and every selection-report field matched exactly. |
| Full assessment | Recomputed the entire 300-image assessment, including actual blurred-image inference. Every report field matched the retained JSON exactly. |
| Image boundary | Four rejected shapes, exact accepted-boundary preprocessing, and real TCP 422→200 recovery passed. |
| Artifact/numerical boundary | Both precision overflow and real float32 matrix-multiplication overflow were rejected; readiness remained 503. |
| NPY allocation boundary | Forged huge shape and truncated payload rejected before array construction; eight valid storage/version combinations and the real learned artifact retained exact values. |
| Deadline accounting | Controlled event-loop delay produced one expiration for one admitted item, with no unresolved work after draining. |
| Final source upload bundle | Exact 36-file allowlist: 35 sources plus manifest, all regular files, no missing or extra entries. CLI import/help, full CPU suite and lint/format pass from extracted source using existing dependencies. No parent resumes, raw data or environment directories. |
| GPU run plan | Inspected command generator and lifecycle limits. No billing, quota, image-resolution, provisioning or CUDA command was executed by this reviewer. |

I inspected the owner's final normal-load and overload/recovery records but did not independently repeat its complete HTTP throughput matrix. The retained final matrix has 360/360 normal responses at 200; all 16 statistics snapshots reconcile terminal outcomes to admissions and have zero pending or unresolved work. My real TCP check verifies behavior, not a throughput figure. The tests inject controlled backends for some lifecycle contracts; they are supplemented by actual pretrained model execution and the real subprocess HTTP checks, rather than represented as entirely hardware-backed tests.

The final source archive is 42,608 bytes with SHA-256 `deef83868f8c25a78277c35bb5ca6b1f9c0f5ed1bac3d60f9ad386de640a9059`. It includes the source, requirements, scripts, tests and runbook needed for a standalone preparation and GPU attempt. This packaging check reused the local macOS dependencies; it is not a clean Linux/CUDA environment installation. The review attestation is deliberately outside the archive to avoid circular source hashes.

## Evaluation interpretation

The exact replay confirms 95.33% learned-head accuracy versus 97.33% for the stronger pretrained ten-class baseline, a −2.0 percentage-point difference with the recorded paired bootstrap interval of −4.0 to −0.33 points. The experiment correctly withholds an improvement claim. The unrestricted 1,000-class result is useful context but is not the fair baseline for a model given the ten-class label space.

Train-only feature standardization, separate selection rows and the assessment-mutation test support the head-training separation. However, Imagenette originates from ImageNet: source-training images overlap backbone pretraining, and official validation was part of model-family development. This is held out for this head selection process, not independent evidence of generalization to new deployments. Exact duplicate QA does not establish absence of similar images or label errors. The paired interval assumes independent images. The reports state these limits appropriately.

The GPU run uses CUDA for feature extraction and inference; the small linear-head optimization still runs on CPU. No GPU training claim follows from running that shell script.

## GPU evidence remains pending

The original reviewer inspected a private provisioning proposal without exercising cloud APIs. That account-specific helper and billing context are excluded from this export. No CUDA correctness, profiling, GPU HTTP, physical camera, multi-GPU or production result follows. The generic GPU execution guide describes unexecuted checks on a separately supplied NVIDIA host.

## Interview ownership

The applicant should personally reproduce the source-image expansion bug, explain why cancellation cannot preempt native inference, trace a request's capacity across a decoder timeout, and justify the mapped baseline and negative head result. A useful next independent modification is a profiling-driven kernel change or a new input-contract regression with measured consequences. The artifact can be strong before its author personally understands every decision; an automated review does not close that gap.
