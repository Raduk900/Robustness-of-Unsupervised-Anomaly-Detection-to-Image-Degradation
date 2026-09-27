import _paths  # puts scripts/ on sys.path

from anomalib.models import ReverseDistillation

from methods.spec import MethodSpec, method_main

SPEC = MethodSpec(
    name="ReverseDistillation",
    paradigm="Student-teacher",
    build=lambda: ReverseDistillation(visualizer=False),
    train_batch_size=8,
    eval_batch_size=8,
    max_epochs=50,
)

if __name__ == "__main__":
    method_main(SPEC)
