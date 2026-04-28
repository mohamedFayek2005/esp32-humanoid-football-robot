import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# YOLO model
yolo = YOLO("yolov8n.pt")

# MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(0)

# ⚙️ إعدادات حساب المسافة (عدّلها حسب الموبايل اللي بتجرب عليه)
REAL_PHONE_WIDTH_CM = 7.5      # عرض الموبايل الحقيقي بـ سم (مثلاً آيفون 13 ≈ 7.15 سم)
FOCAL_LENGTH_PIXELS = 700      # القيمة دي هتحسبها بالكاليبريشن (شوف تحت)

print("✅ Camera initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # 🎯 YOLO detection
    results = yolo(frame, verbose=False)
    boxes = results[0].boxes

    # 📏 MiDaS depth estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = depth.cpu().numpy()
    
    # 🎨 Visualization: blend frame with depth colormap
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(frame, 0.6, depth_color, 0.4, 0)

    # 📱 رسم النتائج على الموبايل بس (class 67 = 'cell phone' in COCO)
    for box in boxes:
        if int(box.cls[0]) != 67:  # 67 = cell phone
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        confidence = float(box.conf[0])
        
        # حساب المسافة بطريقة الحجم الظاهري (Known-size formula)
        pixel_width = x2 - x1  # عرض الموبايل في الصورة بالبكسل
        if pixel_width > 0:
            distance_cm = (REAL_PHONE_WIDTH_CM * FOCAL_LENGTH_PIXELS) / pixel_width
        else:
            distance_cm = None

        # جلب قيمة العمق من MiDaS (كـ backup/relative)
        if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
            relative_depth = depth_map[cy, cx]
        else:
            relative_depth = None

        # 🖼️ رسم الـ Bounding Box
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 📝 كتابة المعلومات
        cv2.putText(blended, f"Phone: {confidence:.2f}", (x1, y1-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if distance_cm is not None:
            cv2.putText(blended, f"Dist: {distance_cm:.1f} cm", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif relative_depth is not None:
            cv2.putText(blended, f"Depth: {relative_depth:.2f} (rel)", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Phone Detection + Depth", blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()