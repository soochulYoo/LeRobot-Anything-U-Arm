"""ImageNet ResNet18 -> the npz that model.ResNet18(pretrained=True) loads.

Comp-ACT's backbone is `resnet18(pretrained=...)` with FrozenBatchNorm2d, so the
statistics never move; we fold each batch-norm into a per-channel scale and
shift and keep only the convolutions trainable.  Weights come straight from
download.pytorch.org -- torchvision is deliberately not imported, because
installing it into an environment that already has a pinned torch removes that
torch.

    python3 policy/convert_resnet18.py        # writes policy/resnet18_imagenet.npz
"""
import pathlib
import urllib.request

import numpy as np
import torch

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"   # IMAGENET1K_V1
DEST = pathlib.Path(__file__).resolve().parent / "resnet18_imagenet.npz"


def main():
    cache = pathlib.Path("/tmp") / "resnet18-f37072fd.pth"
    if not cache.exists():
        urllib.request.urlretrieve(URL, cache)
    sd = {k: v.detach() for k, v in torch.load(cache, map_location="cpu", weights_only=True).items()}

    def conv(k):                    # torch (out,in,kh,kw) -> flax (kh,kw,in,out)
        return sd[k].numpy().transpose(2, 3, 1, 0)

    def bn(p):                      # fold BN into scale/shift
        w, b = sd[f"{p}.weight"].numpy(), sd[f"{p}.bias"].numpy()
        mu, var = sd[f"{p}.running_mean"].numpy(), sd[f"{p}.running_var"].numpy()
        s = w / np.sqrt(var + 1e-5)
        return s, b - mu * s

    out = {"stem_kernel": conv("conv1.weight")}
    out["stem_scale"], out["stem_shift"] = bn("bn1")
    i = 0
    for layer in range(1, 5):
        for blk in range(2):
            p = f"layer{layer}.{blk}"
            out[f"b{i}_c1"] = conv(f"{p}.conv1.weight")
            out[f"b{i}_s1"], out[f"b{i}_h1"] = bn(f"{p}.bn1")
            out[f"b{i}_c2"] = conv(f"{p}.conv2.weight")
            out[f"b{i}_s2"], out[f"b{i}_h2"] = bn(f"{p}.bn2")
            if f"{p}.downsample.0.weight" in sd:
                out[f"b{i}_sk"] = conv(f"{p}.downsample.0.weight")
                out[f"b{i}_sks"], out[f"b{i}_skh"] = bn(f"{p}.downsample.1")
            i += 1
    np.savez_compressed(DEST, **{k: np.asarray(v, np.float32) for k, v in out.items()})
    print(f"wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB, {len(out)} arrays, {i} blocks)")


if __name__ == "__main__":
    main()
