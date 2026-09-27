import argparse
import os

from env_utils import require_env_python, run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")


def main():
    parser = argparse.ArgumentParser(description='Setup verification -- run after setup_env.py and before the experiment.')
    parser.add_argument("data_root", nargs="?",
                        default=os.path.join(ROOT, "dataset"),
                        help="MVTec AD root (default: ./dataset)")
    args = parser.parse_args()

    py = require_env_python()

    print("=== [1/2] Dataset check ===")
    run([py, os.path.join(SCRIPTS, "check_dataset.py"),
         "--data-root", args.data_root])

    print("=== [2/2] Smoke test ===")
    run([py, os.path.join(SCRIPTS, "smoke_test.py"),
         "--data-root", args.data_root])

    print("\nSetup test PASSED. Launch the main experiment with:")
    print("  python scripts/run_all.py")


if __name__ == "__main__":
    main()
