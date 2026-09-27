import _paths  # puts scripts/ on sys.path

from anomalib.models import EfficientAd

from methods.spec import MethodSpec, method_main

SPEC = MethodSpec(
    name="EfficientAD",
    paradigm="Student-teacher",
    build=lambda: EfficientAd(visualizer=False),
    # anomalib's EfficientAD trains with batch size 1 only
    train_batch_size=1,
    eval_batch_size=4,
    max_epochs=50,
    notes="ImageNet normalization inside the model; downloads ImageNette on "
          "first run",
)

if __name__ == "__main__":
    method_main(SPEC)
