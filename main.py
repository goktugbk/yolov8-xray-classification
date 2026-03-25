import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

from helper import detect_disase
from train_update import get_val_accuracy, finetune
from feedback_utils import save_feedback, migrate_feedback_to_dataset

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(page_title="X-Ray Görüntüleme", layout="centered")

# ── Yol/konumlar ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = str(ROOT / "data" / "data_classifier" / "dataset_classification")
FEEDBACK_DIR = str(ROOT / "data" / "feedback")

def _find_latest_finetune_best(prefix: str = "web_update") -> str | None:
    base = ROOT / "runs" / "fine_tune"
    if not base.is_dir():
        return None
    candidates = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            bp = d / "weights" / "best.pt"
            if bp.is_file():
                candidates.append((bp.stat().st_mtime, bp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1])

def _resolve_default_model_path() -> str:
    latest = _find_latest_finetune_best()
    fallback = ROOT / "models" / "best.pt"
    return latest if latest else str(fallback)

DEFAULT_MODEL_PATH = _resolve_default_model_path()

# Model yolu (fine-tune sonrası güncellenecek)
if "model_path" not in st.session_state:
    st.session_state.model_path = DEFAULT_MODEL_PATH

st.title("X-Ray Görüntüleme ☠️")

@st.cache_resource
def get_class_options(model_path: str):
    m = YOLO(model_path)
    return [m.names[i] for i in sorted(m.names)]

# ── Görsel yükleme ve tahmin ─────────────────────────────────────────────────
file = st.file_uploader("Bir X-Ray görüntüsü seçin", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    st.subheader("Orijinal")
    st.image(image, use_container_width=True)

    st.subheader("Tespit Sonucu")
    # --- Değişiklik: detect_disase artık 4 değer döndürüyor ---
    annotated, pred_name, pred_prob, topk = detect_disase(
        image, st.session_state.model_path, topk=3
    )
    st.image(annotated, use_container_width=True)
    st.caption(f"Tahmin: {pred_name} %{pred_prob}")

    # Diğer olası tanılar (en iyi tahmin hariç)
    others = [(n, p) for (n, p) in topk if n != pred_name]
    if others:
        st.markdown("**Diğer olası tanılar:**")
        for n, p in others:
            st.write(f"- {n}: %{p*100:.1f}")

    # Kullanıcı düzeltmesi (yalnızca mevcut sınıflardan seçim)
    with st.expander("Tahmin yanlış mı? Doğru etiketi kaydet"):
        try:
            options = get_class_options(st.session_state.model_path)
        except Exception as e:
            st.error(f"Model yüklenemedi: {e}")
            options = []

        if options:
            idx = options.index(pred_name) if pred_name in options else 0
            true_label = st.selectbox("Doğru etiket", options, index=idx)

            if st.button("Düzeltmeyi kaydet"):
                try:
                    pred_conf = float(pred_prob) / 100.0
                except Exception:
                    pred_conf = None

                saved = save_feedback(
                    image,
                    true_label,
                    root=FEEDBACK_DIR,
                    predicted=pred_name,
                    predicted_conf=pred_conf,
                    allowed_labels=options,
                )
                st.success(f"Düzeltme kaydedildi: {saved}")

# ── Değerlendirme ve model güncelleme ─────────────────────────────────────────
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("Val doğruluğunu ölç (Top-1)"):
        try:
            acc = get_val_accuracy(st.session_state.model_path, DATA_DIR)
            st.success(f"Val Top-1 Doğruluk: {acc*100:.2f}%")
        except Exception as e:
            st.error(f"Doğruluk hesaplanamadı: {e}")

with col_b:
    st.caption("Not: **En az 5** düzeltme resmi yükledikten sonra denemeniz önerilir.")

    if st.button("Modeli kısaca güncelle"):
        try:
            # 1) Feedback → train (TAŞI)
            n, dest = migrate_feedback_to_dataset(FEEDBACK_DIR, DATA_DIR, split="train")
            if n > 0:
                st.info(f"{n} adet feedback örneği train setine eklendi: {dest}")
            else:
                st.caption("Yeni feedback bulunamadı; mevcut train ile devam ediliyor.")

            # 2) Kısa fine-tune
            new_best = finetune(
                model_path=st.session_state.model_path,
                data_dir=DATA_DIR,
                epochs=2,
                imgsz=224,
                lr0=1e-4,
                name="web_update",
            )
            st.session_state.model_path = new_best
            st.success(f"Model güncellendi: {new_best}")
        except Exception as e:
            st.error(f"Eğitim sırasında hata: {e}")

with col_c:
    if st.button("En yeni modeli kullan"):
        latest = _find_latest_finetune_best()
        if latest:
            st.session_state.model_path = latest
            st.success(f"En yeni model seçildi:\n{latest}")
        else:
            st.warning("Herhangi bir fine-tune modeli bulunamadı.")

# ── Alt not ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model hata yapabilir. Önemli bilgileri kontrol edin.")
