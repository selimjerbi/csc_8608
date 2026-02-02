import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box, get_device


def main():
    img_candidates = list(Path("TP1/data/images").glob("*.jpeg")) + list(Path("TP1/data/images").glob("*.png"))
    if not img_candidates:
        raise FileNotFoundError("Aucune image trouvée dans TP1/data/images (jpg/png).")

    img_path = img_candidates[0]
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread a échoué pour {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


    ckpt = "TP1/models/sam_vit_b_01ec64.pth"  
    model_type = "vit_b"

    print("device:", get_device())
    print("image:", img_path.name, "shape:", rgb.shape, "dtype:", rgb.dtype)

    pred = load_sam_predictor(ckpt, model_type=model_type)

    # bbox “à la main” : adapte si besoin selon l'image
    box = np.array([50, 50, min(250, rgb.shape[1]-1), min(250, rgb.shape[0]-1)], dtype=np.int32)

    mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)
    print("mask shape:", mask.shape, "mask dtype:", mask.dtype, "score:", score, "mask_sum:", int(mask.sum()))


if __name__ == "__main__":
    main()
