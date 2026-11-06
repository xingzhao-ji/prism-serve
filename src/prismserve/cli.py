import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="prism")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("prepare")
    download.add_argument("--data", type=Path, default=Path("data"))
    extract = commands.add_parser("extract")
    extract.add_argument("--data", type=Path, default=Path("data"))
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train = commands.add_parser("train")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    serve = commands.add_parser("serve")
    serve.add_argument("--weights", type=Path, default=Path("data/resnet18-f37072fd.pth"))
    serve.add_argument("--head", type=Path)
    serve.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    serve.add_argument("--preprocess", choices=["native", "triton", "compiled"], default="native")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--max-batch", type=int, default=8)
    serve.add_argument("--max-pending", type=int, default=64)
    serve.add_argument("--wait-ms", type=float, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        from prismserve.data import prepare

        result = prepare(args.data)
        print(f"Verified {len(result['records'])} selected images, pinned data and weights")
    elif args.command == "extract":
        from prismserve.experiment import extract

        extract(args.data, args.output, args.device)
    elif args.command == "train":
        from prismserve.experiment import train

        train(args.features, args.output)
    elif args.command == "evaluate":
        from prismserve.experiment import evaluate

        result = evaluate(args.data, args.features, args.artifact, args.output, args.device)
        print(
            f"Head accuracy: {result['trained_head']['accuracy']:.3f}; "
            f"GPU evidence: {result['gpu_evidence']}"
        )
    else:
        import uvicorn

        from prismserve.api import create_app
        from prismserve.model import VisionModel

        app = create_app(
            lambda: VisionModel(
                args.weights, device=args.device, preprocess=args.preprocess, head=args.head
            ),
            max_batch=args.max_batch,
            max_pending=args.max_pending,
            wait_ms=args.wait_ms,
        )
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            workers=1,
            limit_concurrency=128,
            timeout_keep_alive=5,
            access_log=False,
        )
