# ============================================================
# run.py
# GalaxEye Space — Unified Entry Point
# ============================================================

import argparse
import sys
import os


def main():

    parser = argparse.ArgumentParser(
        description="GalaxEye Binary Change Detection — Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "mode",
        choices=["train", "eval", "viz", "infer", "remap"],
        help=(
            "Pipeline stage to run:\n"
            "  train — train model\n"
            "  eval  — evaluate checkpoint\n"
            "  viz   — generate visualizations\n"
            "  infer — run single inference\n"
            "  remap — verify/apply label remapping\n"
        ),
    )

    parser.add_argument(
        "--config",
        default="config.yaml"
    )

    parser.add_argument(
        "--weights",
        default=None,
        help="Path to checkpoint"
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training checkpoint"
    )

    # UPDATED: Added train support
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"]
    )

    parser.add_argument(
        "--data_path",
        default=None
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10
    )

    parser.add_argument(
        "--pre",
        default=None
    )

    parser.add_argument(
        "--post",
        default=None
    )

    parser.add_argument(
        "--out_dir",
        default="outputs/predictions/demo"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--verify",
        action="store_true"
    )

    parser.add_argument(
        "--apply",
        action="store_true"
    )

    parser.add_argument(
        "--save_preds",
        action="store_true"
    )

    args = parser.parse_args()

    # ========================================================
    # TRAIN
    # ========================================================

    if args.mode == "train":

        from src.train import main as train_main

        train_main(args)

    # ========================================================
    # EVAL
    # ========================================================

    elif args.mode == "eval":

        if not args.weights:
            print("❌ ERROR: --weights required for eval")
            sys.exit(1)

        from src.evaluate import main as eval_main

        eval_main(args)

    # ========================================================
    # VISUALIZATION
    # ========================================================

    elif args.mode == "viz":

        if not args.weights:
            print("❌ ERROR: --weights required for viz")
            sys.exit(1)

        from src.visualize import main as viz_main

        viz_main(args)

    # ========================================================
    # INFERENCE
    # ========================================================

    elif args.mode == "infer":

        if not all([args.weights, args.pre, args.post]):
            print("❌ ERROR: --weights --pre --post required")
            sys.exit(1)

        from src.inference import main as infer_main

        infer_main(args)

    # ========================================================
    # REMAP
    # ========================================================

    elif args.mode == "remap":

        from src.remap_labels import main as remap_main

        import sys as _sys

        argv = ["remap_labels.py"]

        if args.verify:
            argv.append("--verify")

        if args.apply:
            argv.append("--apply")

        if args.split:
            argv.extend(["--split", args.split])

        _sys.argv = argv

        remap_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()