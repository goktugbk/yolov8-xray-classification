from pathlib import Path
from typing import Optional
from ultralytics import YOLO


def _check_dataset_dir(data_dir: str) -> None:
    d = Path(data_dir)
    if not (d / "train").is_dir():
        raise FileNotFoundError(f"train klasörü bulunamadı: {d / 'train'}")
    if not (d / "val").is_dir():
        raise FileNotFoundError(f"val klasörü bulunamadı: {d / 'val'}")


def finetune(
    model_path: str,
    data_dir: str,
    *,
    epochs: int = 2,
    imgsz: int = 224,
    lr0: float = 1e-4,
    name: Optional[str] = None,
    project: Optional[str] = "runs/fine_tune",
) -> str:
    _check_dataset_dir(data_dir)
    

    m = YOLO(model_path)
    run_name = name or f"ft_{Path(model_path).stem}"

    m.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        lr0=lr0,
        patience=0,
        workers=0,     
        name=run_name,
        project=project,
        verbose=True,
    )

    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Eğitim sonrası best.pt bulunamadı: {best}")
    return str(best)


def get_val_accuracy(model_path: str, data_dir: str) -> float:

    _check_dataset_dir(data_dir)
    m = YOLO(model_path)
    r = m.val(data=str(data_dir), split="val", verbose=False)
    return float(r.results_dict.get("metrics/accuracy_top1", 0.0))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    model_path = str(ROOT / "models" / "best.pt")
    data_dir   = str(ROOT / "data" / "data_classifier" / "dataset_classification")

    print("[INFO] Fine-tune başlıyor...")
    new_best = finetune(model_path, data_dir, epochs=2, imgsz=224, lr0=1e-4)
    acc = get_val_accuracy(new_best, data_dir)
    print(f"[OK] Yeni model: {new_best}")
    print(f"[OK] Val Top-1 Accuracy: {acc*100:.2f}%")
