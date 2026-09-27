import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

PHOTOMETRIC_FACTORS = [1.0, 1.25, 1.5, 1.75, 2.0]
GAUSSIAN_STDS = [0.0, 0.05, 0.10, 0.20, 0.30]
SALT_PEPPER_AMOUNTS = [0.0, 0.02, 0.05, 0.10, 0.20]
ROTATION_DEGREES = [0, 5, 10, 20, 30]
TRANSLATION_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20]

PERTURBATIONS = {
    "brightness":  ("photometric", PHOTOMETRIC_FACTORS),
    "contrast":    ("photometric", PHOTOMETRIC_FACTORS),
    "saturation":  ("photometric", PHOTOMETRIC_FACTORS),
    "gaussian":    ("noise",       GAUSSIAN_STDS),
    "salt_pepper": ("noise",       SALT_PEPPER_AMOUNTS),
    "rotation":    ("geometric",   ROTATION_DEGREES),
    "translation": ("geometric",   TRANSLATION_FRACTIONS),
}


class _CompatTransform(v2.Transform):

    # torchvision <= 0.20 calls _transform instead of transform
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        impl = cls.__dict__.get("transform")
        if impl is not None:
            cls._transform = impl


class AddGaussianNoise(_CompatTransform):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def transform(self, inpt, params):
        if self.std <= 0 or isinstance(inpt, tv_tensors.Mask):
            return inpt
        if not isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
            return inpt
        t = inpt.float()
        # accepts images in 0..255 as well as 0..1
        is_uint = t.max() > 1.5
        scale = self.std * 255.0 if is_uint else self.std
        out = torch.clamp(t + torch.randn_like(t) * scale, 0, 255.0 if is_uint else 1.0)
        out = out.to(torch.uint8) if is_uint else out
        return tv_tensors.Image(out) if isinstance(inpt, tv_tensors.Image) else out


class AddSaltPepper(_CompatTransform):

    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def transform(self, inpt, params):
        if self.amount <= 0 or isinstance(inpt, tv_tensors.Mask):
            return inpt
        if not isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
            return inpt
        t = inpt.clone()
        hi = 255 if t.max() > 1.5 else 1.0
        sel = torch.rand(t.shape[-2:], device=t.device)
        t[..., sel < self.amount / 2] = hi
        t[..., sel > 1 - self.amount / 2] = 0
        return tv_tensors.Image(t) if isinstance(inpt, tv_tensors.Image) else t


class RotateReflect(_CompatTransform):

    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def transform(self, inpt, params):
        if not isinstance(inpt, (tv_tensors.Image, tv_tensors.Mask, torch.Tensor)):
            return inpt
        interp = (F.InterpolationMode.NEAREST if isinstance(inpt, tv_tensors.Mask)
                  else F.InterpolationMode.BILINEAR)
        rotated = F.rotate(inpt, angle=float(self.degrees), interpolation=interp)
        # Black pixels are replaced by the point-reflected image, which is black
        # in the same corners, so rotated images keep black corners.
        if not isinstance(inpt, tv_tensors.Mask) and rotated.dim() == 3:
            zero = rotated.float().sum(dim=0, keepdim=True) == 0
            flipped = torch.flip(rotated, dims=[-1, -2])
            rotated = torch.where(zero.expand_as(rotated), flipped, rotated)
        return rotated


class TranslateReflect(_CompatTransform):

    def __init__(self, frac):
        super().__init__()
        self.frac = frac

    def transform(self, inpt, params):
        if not isinstance(inpt, (tv_tensors.Image, tv_tensors.Mask, torch.Tensor)):
            return inpt
        h, w = inpt.shape[-2], inpt.shape[-1]
        # shift by frac in both axes; the exposed border is filled by reflection
        dx, dy = int(round(self.frac * w)), int(round(self.frac * h))
        if dx == 0 and dy == 0:
            return inpt
        orig_dim = inpt.dim()
        x = inpt.float()
        while x.dim() < 4:
            x = x.unsqueeze(0)
        x = torch.nn.functional.pad(x, (dx, dx, dy, dy), mode="reflect")
        x = x[..., 0:h, 0:w]
        while x.dim() > orig_dim:
            x = x.squeeze(0)
        out = x.to(inpt.dtype)
        if isinstance(inpt, tv_tensors.Image):
            return tv_tensors.Image(out)
        if isinstance(inpt, tv_tensors.Mask):
            return tv_tensors.Mask(out)
        return out


def make_transform(name, severity):
    if name == "brightness":
        return v2.ColorJitter(brightness=(severity, severity))
    if name == "contrast":
        return v2.ColorJitter(contrast=(severity, severity))
    if name == "saturation":
        return v2.ColorJitter(saturation=(severity, severity))
    if name == "gaussian":
        return AddGaussianNoise(severity)
    if name == "salt_pepper":
        return AddSaltPepper(severity)
    if name == "rotation":
        return v2.Identity() if severity == 0 else RotateReflect(severity)
    if name == "translation":
        return v2.Identity() if severity == 0 else TranslateReflect(severity)
    raise ValueError(f"unknown perturbation: {name}")
