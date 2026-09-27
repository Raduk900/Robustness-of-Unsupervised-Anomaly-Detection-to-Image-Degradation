import os
import platform
import shutil
import subprocess
import sys

ENV_NAME = "anomaly"


def conda_base():
    if not shutil.which("conda"):
        sys.exit(
            "ERROR: conda not found on PATH. Load it first (e.g. 'module load "
            "anaconda3' on most university clusters, or open an Anaconda Prompt "
            "on Windows) and try again."
        )
    out = subprocess.run(["conda", "info", "--base"], check=True,
                         capture_output=True, text=True)
    return out.stdout.strip()


def env_python(base=None):
    base = base or conda_base()
    if platform.system() == "Windows":
        return os.path.join(base, "envs", ENV_NAME, "python.exe")
    return os.path.join(base, "envs", ENV_NAME, "bin", "python")


def require_env_python():
    py = env_python()
    if not os.path.exists(py):
        sys.exit(f"ERROR: conda env '{ENV_NAME}' not found (expected {py}).\n"
                 f"Run setup_env.py first.")
    return py


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
