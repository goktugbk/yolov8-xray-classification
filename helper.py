import cv2
import numpy as np
from ultralytics import YOLO

def detect_disase(image, model_path, topk: int = 3):
    """
    image: PIL.Image (RGB)
    return: (annotated_rgb_np, pred_name:str, pred_prob_percent:str, topk_preds:list[(name, prob0..1)])
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # PIL -> np RGB
    image_rgb = np.asarray(image)

    # Model ve tahmin
    model = YOLO(model_path)
    results = model(image_rgb, show=False, save=False, verbose=False)[0]

    # Olasılıklar
    class_dict = results.names                   # id -> name
    probs = results.probs.data.cpu().numpy()     # shape: (num_classes,)

    # En yüksek olasılık
    top_id = int(np.argmax(probs))
    name = class_dict[top_id]
    max_prob_pct = float(probs[top_id] * 100.0)
    max_prob = str(int(max_prob_pct))            # eski arayüzle uyum için string

    # --- TOP-K adaylar [(isim, olasılık_0..1), ...] ---
    order = probs.argsort()[::-1][:topk]
    topk_preds = [(class_dict[int(i)], float(probs[int(i)])) for i in order]

    # Görsel üstüne yaz
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    text = f"{name} %{int(max_prob_pct)}"
    cv2.putText(image_bgr, text, (10, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 4 değer döndürülür (yenilik: topk_preds)
    return annotated_rgb, name, max_prob, topk_preds
