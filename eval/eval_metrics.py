import os
import glob
import csv
import numpy as np
import cv2
import tifffile
import torch
import torch.nn.functional as F
import kornia

# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Update these directory paths as needed
original_akoya_dir = "/tiff_masked/Akoya"
original_kfbio_dir = "/tiff_masked/KFBio"
akoya2kfbio_dir = "/norm_vec/Akoya2KFBio"
kfbio2akoya_dir = "/norm_vec/KFBio2Akoya"
output_csv = "/eval_gpu.csv"

# Set the target downscale dimensions (width, height) for each image.
TARGET_SIZE = (2048, 2048)  # Adjust as needed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------
# Utility: Downscale image on CPU using OpenCV
# ------------------------------------------------
def load_and_downscale_image(image_path):
    try:
        img = tifffile.imread(image_path)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

    if img is None:
        return None

    # Convert grayscale to RGB if needed.
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    # Normalize to uint8 if needed.
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)

    # Downscale on CPU using OpenCV.
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return img_resized

# ------------------------------------------------
# Utility: Convert downscaled image to GPU tensor
# ------------------------------------------------
def load_image_tensor_gpu(image_path):
    img = load_and_downscale_image(image_path)
    if img is None:
        return None
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    return img_tensor.to(device)

# ------------------------------------------------
# Utility: Find file by common id and extract common id
# ------------------------------------------------
def find_file_by_common_id(common_id, directory, ext="tiff"):
    pattern = os.path.join(directory, f"{common_id}*.{ext}")
    files = glob.glob(pattern)
    return files[0] if files else None

def extract_common_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    tokens = base.split("_")
    if len(tokens) >= 2:
        return f"{tokens[0]}_{tokens[1]}"
    else:
        return base

# ------------------------------------------------
# Utility: Compute evaluation metrics on GPU using torch and Kornia
# ------------------------------------------------
def compute_metrics_gpu(img1, img2, window_size=7):
    """
    Computes SSIM, PSNR, and MSE between two images.
    Inputs are torch tensors of shape (1, C, H, W) on the GPU.
    """
    mse_val = torch.mean((img1 - img2) ** 2)
    psnr_val = 10 * torch.log10(1.0 / mse_val) if mse_val != 0 else float('inf')
    
    # Compute SSIM map using Kornia, then take the mean to get a scalar value.
    ssim_map = kornia.metrics.ssim(img1, img2, window_size=window_size, max_val=1.0, eps=1e-12, padding='same')
    ssim_val = ssim_map.mean()
    
    return ssim_val.item(), psnr_val.item(), mse_val.item()

# ------------------------------------------------
# Main evaluation pipeline (GPU-accelerated with downscaling)
# ------------------------------------------------
if __name__ == "__main__":
    results = []

    akoya_files = glob.glob(os.path.join(original_akoya_dir, "*.tiff"))
    kfbio_files = glob.glob(os.path.join(original_kfbio_dir, "*.tiff"))
    
    common_ids_akoya = {extract_common_id(f) for f in akoya_files}
    common_ids_kfbio = {extract_common_id(f) for f in kfbio_files}
    common_ids = sorted(common_ids_akoya.intersection(common_ids_kfbio))
    print(f"Found {len(common_ids)} common image IDs in both original directories.")

    for cid in common_ids:
        akoya_orig_path = find_file_by_common_id(cid, original_akoya_dir)
        kfbio_orig_path = find_file_by_common_id(cid, original_kfbio_dir)
        if not akoya_orig_path or not kfbio_orig_path:
            print(f"Skipping {cid} as one of the originals is missing.")
            continue

        akoya_orig = load_image_tensor_gpu(akoya_orig_path)
        kfbio_orig = load_image_tensor_gpu(kfbio_orig_path)
        if akoya_orig is None or kfbio_orig is None:
            print(f"Skipping {cid} due to image loading error.")
            continue

        if akoya_orig.shape != kfbio_orig.shape:
            kfbio_orig = F.interpolate(kfbio_orig, size=akoya_orig.shape[2:], mode='bilinear', align_corners=False)

        try:
            ssim_val, psnr_val, mse_val = compute_metrics_gpu(akoya_orig, kfbio_orig, window_size=7)
        except Exception as e:
            print(f"Error computing baseline metrics for {cid}: {e}")
            continue
        results.append({
            "Comparison_Type": "Baseline_Akoya_vs_KFBio",
            "Image_Name": cid,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "MSE": mse_val
        })
        print(f"[Baseline] {cid}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, MSE={mse_val:.2f}")

        # KFBio → Akoya Transformation
        trans_kfbio2akoya_path = find_file_by_common_id(cid, kfbio2akoya_dir)
        if trans_kfbio2akoya_path:
            trans_kfbio2akoya = load_image_tensor_gpu(trans_kfbio2akoya_path)
            if trans_kfbio2akoya is None:
                print(f"Warning: Could not load transformed (KFBio2Akoya) image for {cid}")
            else:
                if akoya_orig.shape != trans_kfbio2akoya.shape:
                    trans_kfbio2akoya = F.interpolate(trans_kfbio2akoya, size=akoya_orig.shape[2:], mode='bilinear', align_corners=False)
                try:
                    ssim_val, psnr_val, mse_val = compute_metrics_gpu(akoya_orig, trans_kfbio2akoya, window_size=7)
                except Exception as e:
                    print(f"Error computing metrics for KFBio→Akoya for {cid}: {e}")
                    continue
                results.append({
                    "Comparison_Type": "Trans_KFBio2Akoya",
                    "Image_Name": cid,
                    "SSIM": ssim_val,
                    "PSNR": psnr_val,
                    "MSE": mse_val
                })
                print(f"[KFBio→Akoya] {cid}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, MSE={mse_val:.2f}")
        else:
            print(f"Warning: No transformed (KFBio2Akoya) image found for {cid}")

        # Akoya → KFBio Transformation
        trans_akoya2kfbio_path = find_file_by_common_id(cid, akoya2kfbio_dir)
        if trans_akoya2kfbio_path:
            trans_akoya2kfbio = load_image_tensor_gpu(trans_akoya2kfbio_path)
            if trans_akoya2kfbio is None:
                print(f"Warning: Could not load transformed (Akoya2KFBio) image for {cid}")
            else:
                if kfbio_orig.shape != trans_akoya2kfbio.shape:
                    trans_akoya2kfbio = F.interpolate(trans_akoya2kfbio, size=kfbio_orig.shape[2:], mode='bilinear', align_corners=False)
                try:
                    ssim_val, psnr_val, mse_val = compute_metrics_gpu(kfbio_orig, trans_akoya2kfbio, window_size=7)
                except Exception as e:
                    print(f"Error computing metrics for Akoya→KFBio for {cid}: {e}")
                    continue
                results.append({
                    "Comparison_Type": "Trans_Akoya2KFBio",
                    "Image_Name": cid,
                    "SSIM": ssim_val,
                    "PSNR": psnr_val,
                    "MSE": mse_val
                })
                print(f"[Akoya→KFBio] {cid}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, MSE={mse_val:.2f}")
        else:
            print(f"Warning: No transformed (Akoya2KFBio) image found for {cid}")

    # ------------------------------------------------
    # Save evaluation results to CSV
    # ------------------------------------------------
    fieldnames = ["Comparison_Type", "Image_Name", "SSIM", "PSNR", "MSE"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nAll evaluation results saved to {output_csv}")
