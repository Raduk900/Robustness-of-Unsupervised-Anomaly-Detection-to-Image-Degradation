import _paths  # puts scripts/ on sys.path

from anomalib.models import Ganomaly

from methods.spec import MethodSpec, method_main

SPEC = MethodSpec(
    name="GANomaly",
    paradigm="Reconstruction (GAN)",
    build=lambda: Ganomaly(batch_size=8, visualizer=False),
    train_batch_size=8,
    eval_batch_size=8,
    max_epochs=50,
    pixel_output=False,
    notes="detection only (no pixel-level output, so no checkpoint is kept); "
          "high inter-seed variance",
)

if __name__ == "__main__":
    method_main(SPEC)
