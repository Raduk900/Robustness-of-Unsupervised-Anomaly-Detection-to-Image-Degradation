import torch

from common import INPUT_SIZE, make_resize

NATIVE_SIZES = [700, 800, 840, 900, 1000, 1024]
SIGMA = 1.0
REPEATS = 8
SEED = 0


def measure(native, resize, generator):
    ratios = []
    for _ in range(REPEATS):
        noise = torch.randn(3, native, native, generator=generator) * SIGMA
        out = resize(noise)
        ratios.append(out.std().item() / (SIGMA * INPUT_SIZE[0] / native))
    return sum(ratios) / len(ratios)


def main():
    resize = make_resize()
    generator = torch.Generator().manual_seed(SEED)
    for native in NATIVE_SIZES:
        kappa = measure(native, resize, generator)
        print(f"N = {native:4d}  kappa = {kappa:.3f}  "
              f"attenuation = {native / (kappa * INPUT_SIZE[0]):.2f}x")


if __name__ == "__main__":
    main()
