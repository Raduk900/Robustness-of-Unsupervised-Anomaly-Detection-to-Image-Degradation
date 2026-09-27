import _paths  # puts scripts/ on sys.path

from anomalib.models import Padim

from methods.spec import MethodSpec, method_main

SPEC = MethodSpec(
    name="PaDiM",
    paradigm="Embedding (Gaussian)",
    build=lambda: Padim(visualizer=False),
    train_batch_size=8,
    eval_batch_size=8,
    max_epochs=1,
)

if __name__ == "__main__":
    method_main(SPEC)
