import os
import cv2
import glob
import random
import numpy as np
import torch
import tifffile
from torchvahadane import TorchVahadaneNormalizer
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def load_tiff_with_tifffile(path, color=True, scale_factor=0.1):
    img = tifffile.imread(path)
    if img is None or (hasattr(img, 'size') and img.size == 0):
        raise ValueError(f"Empty or invalid image: {path}")

    if color:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)

def load_mask_pil(mask_path, scale_factor=0.1):
    with Image.open(mask_path) as m:
        m = m.convert('L')
        if scale_factor != 1.0:
            w, h = m.size
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            m = m.resize((new_w, new_h))
        mask = (np.array(m) > 127)
    return mask

def normalize_kfbio_vectorized(
    akoya_folder,
    akoya_masked,
    kfbio_folder,
    kfbio_masked,
    output_folder,
    scale_factor=0.1,
    num_refs=5
):
    akoya_paths = sorted(glob.glob(os.path.join(akoya_folder, "*.tiff")))
    kfbio_paths = sorted(glob.glob(os.path.join(kfbio_folder, "*.tiff")))

    os.makedirs(output_folder, exist_ok=True)

    for kfbio_path in kfbio_paths:
        kfbio_name = os.path.splitext(os.path.basename(kfbio_path))[0]
        kfbio_mask_path = os.path.join(kfbio_masked, f"{kfbio_name}_MASK.tif")

        kfbio_img = load_tiff_with_tifffile(kfbio_path, color=True, scale_factor=scale_factor)
        kfbio_mask = load_mask_pil(kfbio_mask_path, scale_factor=scale_factor)

        kfbio_vector = kfbio_img[kfbio_mask]
        print(f"Loaded and vectorized KFBio image: {kfbio_path}")

        chosen_refs = random.sample(akoya_paths, num_refs)
        ref_vectors = []

        for ref_path in chosen_refs:
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
            ref_mask_path = os.path.join(akoya_masked, f"{ref_name}_MASK.tif")

            ref_img = load_tiff_with_tifffile(ref_path, color=True, scale_factor=scale_factor)
            ref_mask = load_mask_pil(ref_mask_path, scale_factor=scale_factor)

            ref_vector = ref_img[ref_mask]
            ref_vectors.append(ref_vector)

        stacked_ref_vector = np.vstack(ref_vectors)

        normalizer = TorchVahadaneNormalizer(device=device, staintools_estimate=True)
        normalizer.fit(stacked_ref_vector.reshape(-1, 1, 3))
        print("Fitted normalizer with vectorized references.")

        kfbio_normed_vector = normalizer.transform(kfbio_vector.reshape(-1, 1, 3), return_mask=False)
        kfbio_normed_vector_np = kfbio_normed_vector.cpu().numpy().reshape(-1, 3).astype(np.uint8)

        kfbio_normed_img = np.zeros_like(kfbio_img, dtype=np.uint8)
        kfbio_normed_img[kfbio_mask] = kfbio_normed_vector_np

        ref_list_str = "_".join([os.path.splitext(os.path.basename(r))[0] for r in chosen_refs])
        out_name = f"{kfbio_name}__vectorized__{num_refs}refs__{ref_list_str}_normalized.tiff"
        out_path = os.path.join(output_folder, out_name)

        tifffile.imwrite(out_path, kfbio_normed_img)
        print(f"Saved vectorized normalized tiff: {out_path}")

        png_out = out_path.replace(".tiff", ".png")
        cv2.imwrite(png_out, cv2.cvtColor(kfbio_normed_img, cv2.COLOR_RGB2BGR))
        print(f"Also saved png: {png_out}")

if __name__ == "__main__":
    akoya_folder = "/Akoya/original/tiff"
    akoya_masked = "/Akoya/original/mask"
    kfbio_folder = "/KFBio/original/tiff"
    kfbio_masked = "/KFBio/original/mask"
    output_folder = "OUT"

    normalize_kfbio_vectorized(
        akoya_folder, akoya_masked,
        kfbio_folder, kfbio_masked,
        output_folder,
        scale_factor=0.1,
        num_refs=5
    )
