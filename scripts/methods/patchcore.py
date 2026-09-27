import _paths  # puts scripts/ on sys.path

from anomalib.models import Patchcore

from methods.spec import MethodSpec, method_main

SPEC = MethodSpec(
    name="PatchCore",
    paradigm="Embedding (memory)",
    build=lambda: Patchcore(coreset_sampling_ratio=0.01, visualizer=False),
    train_batch_size=4,
    eval_batch_size=4,
    max_epochs=1,
    notes="coreset ratio 1% so the memory bank fits a 6 GB GPU",
)

if __name__ == "__main__":
    method_main(SPEC)
