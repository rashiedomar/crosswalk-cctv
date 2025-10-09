import cv2, json, argparse, glob, os
from src.mask.hsv_fallback import find_crosswalk_polygon

ap = argparse.ArgumentParser()
ap.add_argument("--images_glob", required=True)   # e.g., data/sample/*.jpg
ap.add_argument("--out_jsonl", required=True)     # e.g., runs/mask_seg/hsv_preds.jsonl
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
with open(args.out_jsonl, "w") as f:
    for p in sorted(glob.glob(args.images_glob)):
        img = cv2.imread(p)
        poly, conf = find_crosswalk_polygon(img)
        f.write(json.dumps({"image": p, "pred_poly": poly, "mask_conf": conf}) + "\n")
print("Saved ->", args.out_jsonl)
