import os
import glob
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt

# ------------------------------------------------
# Configuration
# ------------------------------------------------
original_akoya_dir = "/tiff_masked/Akoya"
original_kfbio_dir = "/tiff_masked/KFBio"
akoya2kfbio_dir    = "/norm_vec/Akoya2KFBio"
kfbio2akoya_dir    = "/norm_vec/KFBio2Akoya"

hist_output_dir = "/hist"
os.makedirs(hist_output_dir, exist_ok=True)

TARGET_SIZE = (2048, 2048)  # (width, height) for downscaling on CPU

# ------------------------------------------------
# Utility: load and downscale image to RGB uint8
# ------------------------------------------------
def load_and_downscale_image(image_path):
    try:
        img = tifffile.imread(image_path)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

    if img is None:
        return None

    # If grayscale or single-channel, expand to RGB
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img]*3, axis=-1)

    # Convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)

    # Downscale (CPU)
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return img_resized

# ------------------------------------------------
# Utility: find file by a common ID prefix
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
# Plot triple histogram: Akoya, KFBio, and Transformed
# ------------------------------------------------
def save_triple_histogram_comparison(
    akoya_img, kfbio_img, trans_img,
    image_id, comparison_label, output_dir
):
    """
    Plots 3 lines per channel:
      - Akoya
      - KFBio
      - Transformed
    on the same set of subplots (R, G, B),
    while excluding black pixels (0,0,0).
    """

    # If shapes differ, resize the transformed to match one or the other
    # so that histogram bins align well:
    if trans_img.shape != akoya_img.shape:
        trans_img = cv2.resize(trans_img, (akoya_img.shape[1], akoya_img.shape[0]))

    # ------------------------------------------------------------------
    # Build non-black masks (any pixel that is purely black gets excluded)
    # ------------------------------------------------------------------
    akoya_mask  = (akoya_img.sum(axis=-1) != 0)
    kfbio_mask  = (kfbio_img.sum(axis=-1) != 0)
    trans_mask  = (trans_img.sum(axis=-1) != 0)

    plt.figure(figsize=(14, 4))
    channels = ['R', 'G', 'B']
    channel_colors = ['red', 'green', 'blue']

    for c in range(3):
        plt.subplot(1, 3, c+1)

        # Extract masked pixel values for each image, for channel c
        akoya_vals = akoya_img[..., c][akoya_mask]
        kfbio_vals = kfbio_img[..., c][kfbio_mask]
        trans_vals = trans_img[..., c][trans_mask]

        # Plot them as step-line histograms
        plt.hist(
            akoya_vals, bins=256, range=(0,256),
            histtype='step', lw=2, color='k',
            label='Akoya'
        )
        plt.hist(
            kfbio_vals, bins=256, range=(0,256),
            histtype='step', lw=2, color=channel_colors[c],
            linestyle='--', label='KFBio'
        )
        plt.hist(
            trans_vals, bins=256, range=(0,256),
            histtype='step', lw=2, color=channel_colors[c],
            label='Transformed'
        )

        plt.title(f'Channel {channels[c]}', fontsize=12)
        if c == 0:
            plt.legend(loc='upper right')

    plt.suptitle(f"{comparison_label} Histogram Comparison | Image ID: {image_id}", fontsize=14)
    save_path = os.path.join(output_dir, f"{image_id}_{comparison_label}_hist.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

# ------------------------------------------------
# Main: for each ID, generate triple hist plots
# ------------------------------------------------
if __name__ == "__main__":
    # Gather list of files
    akoya_files = glob.glob(os.path.join(original_akoya_dir, "*.tiff"))
    kfbio_files = glob.glob(os.path.join(original_kfbio_dir, "*.tiff"))

    # Common IDs
    common_ids_akoya = {extract_common_id(f) for f in akoya_files}
    common_ids_kfbio = {extract_common_id(f) for f in kfbio_files}
    common_ids = sorted(common_ids_akoya.intersection(common_ids_kfbio))
    print(f"Found {len(common_ids)} common IDs.")

    for cid in common_ids:
        akoya_orig_path = find_file_by_common_id(cid, original_akoya_dir)
        kfbio_orig_path = find_file_by_common_id(cid, original_kfbio_dir)

        if not akoya_orig_path or not kfbio_orig_path:
            continue  # missing original

        # Load original images
        akoya_img = load_and_downscale_image(akoya_orig_path)
        kfbio_img = load_and_downscale_image(kfbio_orig_path)
        if akoya_img is None or kfbio_img is None:
            continue

        # KFBio→Akoya
        trans_kfbio2akoya_path = find_file_by_common_id(cid, kfbio2akoya_dir)
        if trans_kfbio2akoya_path:
            trans_kfbio2akoya_img = load_and_downscale_image(trans_kfbio2akoya_path)
            if trans_kfbio2akoya_img is not None:
                save_triple_histogram_comparison(
                    akoya_img,
                    kfbio_img,
                    trans_kfbio2akoya_img,
                    image_id=cid,
                    comparison_label="KFBio2Akoya",
                    output_dir=hist_output_dir
                )

        # Akoya→KFBio
        trans_akoya2kfbio_path = find_file_by_common_id(cid, akoya2kfbio_dir)
        if trans_akoya2kfbio_path:
            trans_akoya2kfbio_img = load_and_downscale_image(trans_akoya2kfbio_path)
            if trans_akoya2kfbio_img is not None:
                save_triple_histogram_comparison(
                    akoya_img,
                    kfbio_img,
                    trans_akoya2kfbio_img,
                    image_id=cid,
                    comparison_label="Akoya2KFBio",
                    output_dir=hist_output_dir
                )

    print("Done. Histograms saved under:", hist_output_dir)
