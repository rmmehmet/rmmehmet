# Step 1 of the ASCII portrait pipeline: prep the raw photo once.
# A flatly-lit face converts to a dark, unreadable blob, so this:
#   1. flattens the background to pure white (distance-from-corner mask,
#      since a lightweight color-distance mask is enough for a plain
#      studio/passport-style background and needs no ML model download)
#   2. boosts local contrast with CLAHE
#   3. unsharp-masks it so fine features (eyes, brows, mustache) survive
#      the heavy downsample that make_ascii_svg.py does next
#
# Usage:
#   python3 prep_photo.py source-photo.jpg
# Writes: source-prepped.png (grayscale, ready for make_ascii_svg.py)
import sys
import cv2
import numpy as np

def prep(src_path, out_path="source-prepped.png", crop_bottom=0.82):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {src_path}")
    h, w = img.shape[:2]

    # crop out most of the shoulders/shirt so the ASCII print stays a
    # head-and-shoulders portrait instead of a solid block of dark fabric
    img = img[0:int(h * crop_bottom), :]
    h, w = img.shape[:2]

    # --- background estimate from the four corners ---
    patch = max(12, min(h, w) // 30)
    corners = np.concatenate([
        img[0:patch, 0:patch].reshape(-1, 3),
        img[0:patch, w - patch:w].reshape(-1, 3),
        img[h - patch:h, 0:patch].reshape(-1, 3),
        img[h - patch:h, w - patch:w].reshape(-1, 3),
    ])
    bg_mean = corners.mean(axis=0)

    dist = np.linalg.norm(img.astype(np.float32) - bg_mean.astype(np.float32), axis=2)
    bg_mask = (dist < 58).astype(np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    bg_mask = bg_mask.astype(bool)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # contrast-stretch the foreground only (2nd-98th percentile)
    fg_vals = gray[~bg_mask]
    lo, hi = np.percentile(fg_vals, 2), np.percentile(fg_vals, 98)
    stretched = np.clip((gray - lo) * (255.0 / max(hi - lo, 1e-5)), 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    enhanced = cv2.addWeighted(enhanced, 1.7, blurred, -0.7, 0)

    # composite onto pure white -> maps to the blank end of the ASCII ramp
    enhanced[bg_mask] = 255

    cv2.imwrite(out_path, enhanced)
    print(f"wrote {out_path} ({enhanced.shape[1]}x{enhanced.shape[0]})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 prep_photo.py <source-photo>")
    prep(sys.argv[1])
