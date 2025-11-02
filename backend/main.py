from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np, base64, io, cv2, uvicorn, torch
from torch.nn.modules.container import Sequential
import math

app = FastAPI(title="Multi-Model Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow YOLO classes on PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
    torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, Sequential])
except Exception:
    pass

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# For vegetation, simplify all classes to "Vegetation Land"
VEG_REMAP = {
    'vegetation': 'Vegetation Land',
    'structure': 'Vegetation Land',
    'erosion': 'Vegetation Land',
}

def load_model(p: Path):
    if not p.exists():
        print(f"❌ Model not found: {p}")
        return None, None
    try:
        m = YOLO(str(p))
        print(f"✅ Loaded model: {p.name} -> {list(m.names.values())}")
        return m, str(p)
    except Exception as e:
        print(f"❌ Failed loading {p.name}: {e}")
        return None, None

soil_path = MODELS_DIR / "soil_yolov11_best.pt"
veg_path  = MODELS_DIR / "veg_yolov8_seg_best.pt"

soil_model, soil_src = load_model(soil_path)
veg_model,  veg_src  = load_model(veg_path)

# Log model tasks (detect/segment/classify)
try:
    print(f"🧭 Soil task: {getattr(soil_model, 'task', None)}")
    print(f"🧭 Veg  task: {getattr(veg_model, 'task', None)}")
except Exception:
    pass

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "soil": {
                "loaded": soil_model is not None,
                "source": soil_src,
                "classes": list(soil_model.names.values()) if soil_model else []
            },
            "vegetable": {
                "loaded": veg_model is not None,
                "source": veg_src,
                "classes": list(veg_model.names.values()) if veg_model else []
            },
        },
    }

@app.get("/models")
async def models():
    return {
        "models": [
            {
                "id": "soil",
                "name": "Soil Type Detection",
                "type": "detection",
                "available": soil_model is not None,
                "source": soil_src,
                "description": "Detects soil types: Alluvial, Black, Clay, Red",
            },
            {
                "id": "vegetable",
                "name": "Vegetation Land Detection",
                "type": "segmentation",
                "available": veg_model is not None,
                "source": veg_src,
                "description": "Detects vegetation areas",
            },
        ]
    }

# --- helpers for tiling fallback ---
def tile_image(img: np.ndarray, tile=640, overlap=0.25):
    h, w = img.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    tiles = []
    for y in range(0, max(1, h - tile + 1), step):
        for x in range(0, max(1, w - tile + 1), step):
            crop = img[y:y+tile, x:x+tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:
                pad_y = tile - crop.shape[0]
                pad_x = tile - crop.shape[1]
                crop = cv2.copyMakeBorder(crop, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
            tiles.append((x, y, crop))
    # ensure bottom-right coverage
    if (w - tile) % step != 0 or (h - tile) % step != 0:
        x = max(0, w - tile)
        y = max(0, h - tile)
        crop = img[y:y+tile, x:x+tile]
        if crop.shape[0] < tile or crop.shape[1] < tile:
            pad_y = tile - crop.shape[0]
            pad_x = tile - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
        tiles.append((x, y, crop))
    return tiles

def draw_annotations(orig: np.ndarray, preds: list):
    img = orig.copy()
    for p in preds:
        if not p.get("bbox"):
            continue
        x1, y1, x2, y2 = map(int, p["bbox"])
        cls = p["class"]
        conf = p["confidence"]
        color = (120, 90, 250)  # purple-ish
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

def mask_to_boxes(masks):
    # masks: r.masks.data (N,H,W) float tensor 0..1
    boxes = []
    if masks is None or getattr(masks, "data", None) is None:
        return boxes
    md = masks.data.cpu().numpy()
    for i in range(md.shape[0]):
        m = (md[i] > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes

def make_crops(img: np.ndarray):
    h, w = img.shape[:2]
    crops = []
    # full
    crops.append(("full", (0, 0), img))
    # center square
    s = min(h, w)
    cx, cy = w // 2, h // 2
    x1, y1 = max(0, cx - s // 2), max(0, cy - s // 2)
    crops.append(("center", (x1, y1), img[y1:y1 + s, x1:x1 + s]))
    # lower half (often soil)
    crops.append(("lower", (0, h // 2), img[h // 2:h, 0:w]))
    return crops

def draw_label(img: np.ndarray, text: str):
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 10
    cv2.rectangle(out, (10, 10), (10 + tw + pad*2, 10 + th + pad*2), (120, 90, 250), -1)
    cv2.putText(out, text, (10 + pad, 10 + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return out

def classify_soil_color(img_rgb: np.ndarray):
    # Heuristic color-based classifier (fallback when YOLO finds no boxes)
    h, w = img_rgb.shape[:2]
    roi = img_rgb[int(h*0.4):, :]  # lower 60% where soil is likely
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    R = roi[:, :, 0].astype(np.float32)
    G = roi[:, :, 1].astype(np.float32)
    B = roi[:, :, 2].astype(np.float32)

    H = hsv[:, :, 0].astype(np.float32) * 2.0      # 0..360
    S = hsv[:, :, 1].astype(np.float32) / 255.0    # 0..1
    V = hsv[:, :, 2].astype(np.float32) / 255.0    # 0..1
    L = lab[:, :, 0].astype(np.float32) / 255.0    # 0..1

    h_mean = float(np.mean(H))
    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))
    l_mean = float(np.mean(L))

    r_dom = float(np.mean(R - np.maximum(G, B)))   # red dominance
    r_ratio = float(np.mean(R / (G + B + 1e-5)))

    # Rules (tuned roughly; adjust as needed with your data)
    # 1) Black Soil: very dark, low brightness
    if v_mean < 0.28 and l_mean < 0.35:
        return "Black Soil", 0.85
    # 2) Red Soil: strong red hue and saturation or red dominance
    if (h_mean <= 25 or h_mean >= 335) and s_mean >= 0.22 and r_dom > 12:
        return "Red Soil", 0.80
    if r_ratio > 0.65 and r_dom > 8 and s_mean >= 0.18:
        return "Red Soil", 0.70
    # 3) Alluvial Soil: light, yellowish/sandy, higher V
    if 20 <= h_mean <= 70 and v_mean >= 0.55:
        return "Alluvial Soil", 0.75
    if v_mean >= 0.70 and s_mean <= 0.25:
        return "Alluvial Soil", 0.65
    # 4) Clay Soil: neutral/grayish, low saturation, mid brightness
    if s_mean <= 0.15 and 0.35 <= v_mean <= 0.65:
        return "Clay Soil", 0.70

    # Fallback by closest trait
    if v_mean < 0.35:
        return "Black Soil", 0.60
    if h_mean < 30 or h_mean > 330:
        return "Red Soil", 0.60
    if v_mean > 0.6:
        return "Alluvial Soil", 0.60
    return "Clay Soil", 0.60

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
    if model_type not in {"soil", "vegetable"}:
        raise HTTPException(400, "Invalid model_type. Use 'soil' or 'vegetable'.")

    model = soil_model if model_type == "soil" else veg_model
    if model is None:
        raise HTTPException(503, f"{model_type.capitalize()} model not available")

    # Read and prepare image
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(pil)

    # If soil model is a classification model, use classification path
    task = getattr(model, "task", None)
    if model_type == "soil" and task == "classify":
        res = model.predict(img_np, imgsz=640, verbose=False)[0]
        probs = getattr(res, "probs", None)
        if not probs:
            raise HTTPException(500, "Classification output missing")
        top1 = int(probs.top1)
        top1_name = model.names.get(top1, str(top1))
        top1_conf = float(probs.top1conf)
        # Optional: include top-5
        top5 = getattr(probs, "top5", [])
        top5conf = getattr(probs, "top5conf", [])
        predictions = []
        if top5 and top5conf:
            for i, idx in enumerate(top5):
                predictions.append({
                    "class": model.names.get(int(idx), str(int(idx))),
                    "confidence": float(top5conf[i]),
                    "bbox": None
                })
        else:
            predictions.append({"class": top1_name, "confidence": top1_conf, "bbox": None})

        annotated = draw_label(img_np, f"{top1_name} {top1_conf:.0%}")
        ok, buf = cv2.imencode(".png", annotated)
        if not ok:
            raise HTTPException(500, "Failed to encode annotated image")
        img_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "success": True,
            "model_type": model_type,
            "predictions": predictions,
            "class_counts": {top1_name: 1},
            "total_detections": len(predictions),
            "summary": f"Detected: {top1_name} ({top1_conf:.0%} confidence)",
            "dominant_soil": top1_name,
            "dominant_confidence": top1_conf,
            "annotated_image": img_b64,
        }

    # Progressive attempts (lower conf, lower IoU, larger imgsz)
    attempts = [
        {"conf": 0.20 if model_type == "soil" else 0.25, "iou": 0.45, "imgsz": 640,  "augment": False},
        {"conf": 0.05 if model_type == "soil" else 0.20, "iou": 0.30, "imgsz": 960,  "augment": True},
        {"conf": 0.01 if model_type == "soil" else 0.15, "iou": 0.20, "imgsz": 1280, "augment": True},
        {"conf": 0.001 if model_type == "soil" else 0.10, "iou": 0.15, "imgsz": 1280, "augment": True},
    ]

    predictions, class_counts, class_best = [], {}, {}
    remapped_names = None
    last_result = None
    used_tiling = False

    # 1) Multi-crop, multi-scale attempts
    for i, a in enumerate(attempts, 1):
        for tag, (ox, oy), crop_img in make_crops(img_np):
            results = model.predict(
                crop_img, conf=a["conf"], iou=a["iou"], imgsz=a["imgsz"], augment=a["augment"], verbose=False
            )
            r = results[0]
            last_result = r
            remapped_names = dict(r.names)

            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)
            probs = getattr(r, "probs", None)

            found_any = False

            # Detection path (boxes)
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cid = int(b.cls[0])
                    raw = model.names.get(cid, str(cid))
                    name = raw if model_type == "soil" else "Vegetation Land"
                    remapped_names[cid] = name

                    conf = float(b.conf[0])
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    # shift crop -> full coords
                    xyxy[0] += ox; xyxy[1] += oy; xyxy[2] += ox; xyxy[3] += oy

                    predictions.append({"class": name, "confidence": conf, "bbox": xyxy})
                    class_counts[name] = class_counts.get(name, 0) + 1
                    if name not in class_best or conf > class_best[name]:
                        class_best[name] = conf
                found_any = True

            # Segmentation path: if soil model outputs only masks, derive boxes
            elif model_type == "soil" and masks is not None and getattr(masks, "data", None) is not None and len(masks.data) > 0:
                bb = mask_to_boxes(masks)
                for xyxy in bb:
                    # shift crop -> full coords
                    xyxy[0] += ox; xyxy[1] += oy; xyxy[2] += ox; xyxy[3] += oy
                    # name fallback: first class name or generic
                    name = list(model.names.values())[0] if model.names else "Soil"
                    predictions.append({"class": name, "confidence": 0.99, "bbox": xyxy})
                    class_counts[name] = class_counts.get(name, 0) + 1
                    class_best[name] = max(class_best.get(name, 0), 0.99)
                found_any = len(bb) > 0

            # Classification path (no boxes/masks)
            elif probs is not None and getattr(probs, "top1", None) is not None:
                top1 = int(probs.top1)
                conf = float(probs.top1conf)
                name = model.names.get(top1, str(top1))
                if model_type == "vegetable":
                    name = "Vegetation Land"
                predictions.append({"class": name, "confidence": conf, "bbox": None})
                class_counts[name] = class_counts.get(name, 0) + 1
                class_best[name] = max(class_best.get(name, 0), conf)
                found_any = True

            if found_any:
                break
        if predictions:
            break

    # 2) Tiled inference fallback for soil if still nothing
    if model_type == "soil" and not predictions:
        used_tiling = True
        for (ox, oy, tile) in tile_image(img_np, tile=640, overlap=0.35):
            results = model.predict(tile, conf=0.01, iou=0.15, imgsz=640, augment=True, verbose=False)
            r = results[0]
            remapped_names = dict(r.names)
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)

            # boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cid = int(b.cls[0])
                    raw = model.names.get(cid, str(cid))
                    name = raw
                    conf = float(b.conf[0])
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    xyxy[0] += ox; xyxy[1] += oy; xyxy[2] += ox; xyxy[3] += oy
                    predictions.append({"class": name, "confidence": conf, "bbox": xyxy})
                    class_counts[name] = class_counts.get(name, 0) + 1
                    class_best[name] = max(class_best.get(name, 0), conf)
            # masks -> boxes
            elif masks is not None and getattr(masks, "data", None) is not None and len(masks.data) > 0:
                for xyxy in mask_to_boxes(masks):
                    xyxy[0] += ox; xyxy[1] += oy; xyxy[2] += ox; xyxy[3] += oy
                    name = list(model.names.values())[0] if model.names else "Soil"
                    predictions.append({"class": name, "confidence": 0.99, "bbox": xyxy})
                    class_counts[name] = class_counts.get(name, 0) + 1
                    class_best[name] = max(class_best.get(name, 0), 0.99)

    # Final fallback: color-based soil classification (ensures any image gets a result)
    if model_type == "soil" and not predictions:
        label, conf = classify_soil_color(img_np)
        predictions = [{"class": label, "confidence": conf, "bbox": None}]
        class_counts = {label: 1}
        class_best = {label: conf}
        summary = f"Detected: {label} ({conf:.0%} confidence) • heuristic"
        annotated = draw_label(img_np, f"{label} {conf:.0%}")
        ok, buf = cv2.imencode(".png", annotated)
        if not ok:
            raise HTTPException(500, "Failed to encode annotated image")
        img_b64 = base64.b64encode(buf).decode("utf-8")
        return {
            "success": True,
            "model_type": model_type,
            "predictions": predictions,
            "class_counts": class_counts,
            "total_detections": 1,
            "summary": summary,
            "dominant_soil": label,
            "dominant_confidence": conf,
            "annotated_image": img_b64,
        }

    # Summary
    summary = None
    dominant_soil, dominant_conf = None, 0.0
    if model_type == "soil":
        if class_best:
            dominant_soil = max(class_best, key=class_best.get)
            dominant_conf = class_best[dominant_soil]
            summary = f"Detected: {dominant_soil} ({dominant_conf:.0%} confidence){' • tiled' if used_tiling else ''}"
        else:
            summary = "No soil detected in image. Try a closer, clearer soil photo."
    else:
        total = sum(class_counts.values()) if class_counts else 0
        summary = "Vegetation land detected" if total > 0 else "No vegetation detected"

    # Annotate
    if model_type == "soil":
        annotated = draw_annotations(img_np, predictions)  # use our combined boxes
    else:
        # use model plot for veg segmentation
        r = last_result
        if r is None:
            raise HTTPException(500, "No results returned by model")
        try:
            if remapped_names is not None:
                r.names = remapped_names
        except Exception:
            pass
        annotated = r.plot()

    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(500, "Failed to encode annotated image")
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "model_type": model_type,
        "predictions": predictions,
        "class_counts": class_counts,
        "total_detections": len(predictions),
        "summary": summary,
        "dominant_soil": dominant_soil if model_type == "soil" else None,
        "dominant_confidence": dominant_conf if model_type == "soil" else None,
        "annotated_image": img_b64,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)