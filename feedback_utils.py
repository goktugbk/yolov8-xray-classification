from __future__ import annotations

from pathlib import Path
from datetime import datetime
import uuid
from typing import Union, Optional, Tuple, List

import numpy as np
from PIL import Image


def _to_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def _sanitize_label(label: str) -> str:
    s = (label or "").strip().lower().replace(" ", "_")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    s = "".join(ch for ch in s if ch in allowed)
    if not s:
        raise ValueError("Boş veya geçersiz etiket.")
    return s


def save_feedback(
    image: Union[Image.Image, np.ndarray],
    true_label: str,
    *,
    root: Union[str, Path] = "data/feedback",
    predicted: Optional[str] = None,
    predicted_conf: Optional[float] = None,
    fmt: str = "jpg",
    allowed_labels: Optional[List[str]] = None,
) -> str:

    pil = _to_pil(image)
    label_norm = _sanitize_label(true_label)

    if allowed_labels:
        allowed_norm = {_sanitize_label(x) for x in allowed_labels}
        if label_norm not in allowed_norm:
            raise ValueError("Seçilen etiket model sınıfları arasında değil.")

    root = Path(root)
    out_dir = root / label_norm
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    ext = "png" if fmt.lower() == "png" else "jpg"
    outfile = out_dir / f"{ts}_{uid}.{ext}"

    if ext == "png":
        pil.save(outfile, format="PNG", optimize=True)
    else:
        pil.save(outfile, format="JPEG", quality=95, optimize=True)

    try:
        log_path = root / "feedback_log.csv"
        line = [
            ts,
            str(outfile),
            label_norm,
            predicted or "",
            f"{predicted_conf:.6f}" if isinstance(predicted_conf, (float, int)) else "",
        ]
        header = "timestamp,path,label,predicted,predicted_conf\n"
        if not log_path.exists():
            log_path.write_text(header, encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(",".join(line) + "\n")
    except Exception:
        pass

    return str(outfile)


def migrate_feedback_to_dataset(
    feedback_root: Union[str, Path],
    dataset_root: Union[str, Path],
    split: str = "train",
) -> Tuple[int, Path]:

    import shutil

    feedback_root = Path(feedback_root)
    dataset_root = Path(dataset_root)
    target_split_dir = dataset_root / split
    target_split_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    if not feedback_root.exists():
        return 0, target_split_dir

    for label_dir in feedback_root.iterdir():
        if not label_dir.is_dir():
            continue
        dest_label_dir = target_split_dir / label_dir.name
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in label_dir.glob("*.*"):
            dest = dest_label_dir / img_file.name
            if dest.exists():
                dest = dest_label_dir / f"{img_file.stem}_{uuid.uuid4().hex[:6]}{img_file.suffix}"
            shutil.move(str(img_file), str(dest))
            moved += 1

    return moved, target_split_dir
