import os
import subprocess

from env_utils import ENV_NAME, conda_base, env_python, run

PYTHON_VERSION = "3.11"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def env_exists():
    out = subprocess.run(["conda", "env", "list"], check=True,
                         capture_output=True, text=True).stdout
    names = {line.split()[0] for line in out.splitlines()
             if line.strip() and not line.startswith("#")}
    return ENV_NAME in names


def main():
    import argparse
    argparse.ArgumentParser(
        description="Create the conda env 'anomaly' and install requirements.txt."
    ).parse_args()
    base = conda_base()

    if env_exists():
        print(f"conda env '{ENV_NAME}' already exists -- reusing it.")
    else:
        run(["conda", "create", "-y", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "pip"])

    py = env_python(base)

    print(">>> Installing dependencies (including the CUDA build of torch)...")
    run([py, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])

    print(">>> Verifying installation...")
    run([py, os.path.join(HERE, "_verify_env.py")])

    print("\nSetup complete. Next steps:")
    print("  1. Place MVTec AD under ./dataset (see README).")
    print("  2. python setup/run_setup_test.py")


if __name__ == "__main__":
    main()
