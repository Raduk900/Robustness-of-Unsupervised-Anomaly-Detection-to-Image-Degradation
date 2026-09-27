import torch
from anomalib.models import EfficientAd, Ganomaly, Padim, Patchcore, ReverseDistillation

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    sys_exit_msg = (
        "ERROR: CUDA not available. A dependency likely replaced torch with a "
        "CPU build -- re-run setup_env.py, or check that this node has a GPU "
        "and a recent NVIDIA driver."
    )
    raise SystemExit(sys_exit_msg)
print("All five models import correctly.")
