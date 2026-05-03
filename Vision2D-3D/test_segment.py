"""
Test RealSense D435 + YOLOv8 Instance Segmentation
Code đơn giản để test model v8m-seg-832.pt
Có chức năng chọn ROI để chỉ xử lý vùng đó
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

# ============ CẤU HÌNH ============
MODEL_PATH = "v8m-seg-832.pt"
CONFIDENCE = 0.5  # Ngưỡng confidence
WIDTH = 1280
HEIGHT = 720
FPS = 30
ROI_CONFIG_FILE = "roi_config.txt"  # File lưu ROI

# CROP 720x720 từ giữa (giống realsense_gui_advanced.py)
CROP_SIZE = 720
START_X = (WIDTH - CROP_SIZE) // 2  # = 280

# ============ ROI SELECTION ============
roi = None
drawing = False
start_point = None
temp_roi = None

def load_roi_from_file(filepath):
    """Đọc ROI từ file config - ROI là tọa độ trên ảnh 720x720"""
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            if line:
                values = [int(x.strip()) for x in line.split(',')]
                if len(values) == 4:
                    x1, y1, x2, y2 = values
                    # Validate ROI (trên ảnh 720x720)
                    if 0 <= x1 < x2 <= CROP_SIZE and 0 <= y1 < y2 <= CROP_SIZE:
                        print(f"✅ Loaded ROI from config (720x720): ({x1}, {y1}) -> ({x2}, {y2})")
                        return (x1, y1, x2, y2)
                    else:
                        print(f"⚠️  Invalid ROI in config file")
        return None
    except FileNotFoundError:
        print(f"⚠️  ROI config file not found: {filepath}")
        return None
    except Exception as e:
        print(f"⚠️  Error reading ROI config: {e}")
        return None

# ============ DISPLAY OPTIONS ============
show_boxes = False  # Hiển thị bounding boxes
show_binary = False  # Hiển thị binary mask window

# Màu sắc cho từng instance (BGR)
COLORS = [
    (255, 0, 0),    # Xanh dương
    (0, 255, 0),    # Xanh lá
    (0, 0, 255),    # Đỏ
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Vàng
    (128, 0, 128),  # Tím
    (255, 165, 0),  # Cam
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]

def mouse_callback(event, x, y, flags, param):
    """Callback để vẽ ROI bằng chuột - trên ảnh 720x720"""
    global roi, drawing, start_point, temp_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Bắt đầu vẽ ROI
        drawing = True
        start_point = (x, y)
        temp_roi = None
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Đang vẽ, cập nhật temp_roi
            temp_roi = (start_point[0], start_point[1], x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        # Kết thúc vẽ ROI
        drawing = False
        if start_point:
            x1, y1 = start_point
            x2, y2 = x, y
            
            # Đảm bảo x1 < x2 và y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Đảm bảo ROI trong khung hình 720x720
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(CROP_SIZE, x2)
            y2 = min(CROP_SIZE, y2)
            
            # Kiểm tra ROI có đủ lớn không
            if (x2 - x1) > 50 and (y2 - y1) > 50:
                roi = (x1, y1, x2, y2)
                temp_roi = None
                print(f"✅ ROI selected: ({x1}, {y1}) -> ({x2}, {y2})")
            else:
                print("⚠️  ROI too small, please draw a larger area")
                temp_roi = None

print("="*60)
print("🚀 TEST INSTANCE SEGMENTATION WITH ROI")
print("="*60)

# ============ LOAD ROI FROM CONFIG ============
print(f"\n📋 Loading ROI from: {ROI_CONFIG_FILE}")
roi = load_roi_from_file(ROI_CONFIG_FILE)
if roi:
    print(f"   ROI will be applied: {roi}")
else:
    print("   No ROI - will process full frame")

# ============ LOAD MODEL ============
print(f"\n📦 Loading model: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ============ KHỞI TẠO CAMERA ============
print("\n📷 Initializing RealSense D435...")
pipeline = rs.pipeline()
config = rs.config()

# Cấu hình streams
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

try:
    pipeline.start(config)
    print("✅ Camera started!")
except Exception as e:
    print(f"❌ Error starting camera: {e}")
    exit(1)

# Align depth to color
align = rs.align(rs.stream.color)

print("\n" + "="*60)
print("🎮 CONTROLS:")
print("  [MOUSE] - Drag to select ROI")
print("  [r] - Reset ROI (process full frame)")
print("  [c] - Load ROI from config file")
print("  [o] - Toggle bounding boxes ON/OFF")
print("  [b] - Toggle binary mask window ON/OFF")
print("  [q] or [ESC] - Quit")
print("  [s] - Save current frame")
print("  [SPACE] - Pause/Resume")
print("="*60)
if roi:
    print(f"💡 ROI from config: ({roi[0]}, {roi[1]}) -> ({roi[2]}, {roi[3]})")
else:
    print("💡 TIP: Draw a rectangle on the image to select ROI")
print("="*60 + "\n")

# Tạo window và gắn mouse callback
cv2.namedWindow('RealSense D435 - Instance Segmentation')
cv2.setMouseCallback('RealSense D435 - Instance Segmentation', mouse_callback)

# ============ BIẾN THEO DÕI ============
frame_count = 0
fps_time = time.time()
fps_counter = 0
fps = 0
paused = False

try:
    while True:
        if not paused:
            # Đọc frames từ camera
            frames = pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = align.process(frames)
            
            # Lấy color frame
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame:
                continue
            
            # Convert sang numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # CROP 720x720 từ giữa (giống realsense_gui_advanced.py)
            color_720 = color_image[0:CROP_SIZE, START_X:START_X+CROP_SIZE].copy()
            
            # Vẽ kết quả lên ảnh 720x720
            annotated_frame = color_720.copy()
            
            # ============ XỬ LÝ ROI ============
            process_image = color_720
            roi_offset = (0, 0)
            
            if roi is not None:
                x1, y1, x2, y2 = roi
                # Crop ảnh theo ROI (trên ảnh 720x720)
                process_image = color_720[y1:y2, x1:x2]
                roi_offset = (x1, y1)
                
                # Vẽ khung ROI lên ảnh 720x720
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated_frame, "ROI", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ============ CHẠY SEGMENTATION ============
            binary_mask = None  # Khởi tạo binary mask
            
            if process_image.size > 0:
                results = model.predict(
                    process_image,
                    conf=CONFIDENCE,
                    verbose=False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Kiểm tra có masks không
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes.data.cpu().numpy()
                        
                        # Tạo overlay cho masks
                        overlay = annotated_frame.copy()
                        
                        # Tạo binary mask (trắng đen) cho tất cả instances (720x720)
                        binary_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
                        
                        # Lấy kích thước của ảnh xử lý
                        proc_h, proc_w = process_image.shape[:2]
                        
                        # Lưu thông tin boxes để vẽ sau (không ảnh hưởng mask)
                        box_info_list = []
                        
                        for idx, (mask, box) in enumerate(zip(masks, boxes)):
                            # Lấy thông tin
                            x1, y1, x2, y2, conf_val, cls = box
                            class_id = int(cls)
                            
                            # Resize mask về kích thước ảnh đã crop
                            mask_resized = cv2.resize(mask, (proc_w, proc_h))
                            mask_bool = mask_resized > 0.5
                            
                            # Tạo mask full size (720x720)
                            full_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
                            
                            # Đặt mask vào đúng vị trí (có tính offset nếu có ROI)
                            x_off, y_off = roi_offset
                            full_mask[y_off:y_off+proc_h, x_off:x_off+proc_w] = mask_bool
                            
                            # Thêm vào binary mask (trắng = 255)
                            binary_mask[full_mask] = 255
                            
                            # Chọn màu cho instance này
                            color = COLORS[idx % len(COLORS)]
                            
                            # Vẽ mask lên overlay
                            overlay[full_mask] = overlay[full_mask] * 0.5 + np.array(color) * 0.5
                            
                            # Vẽ label ngay trên mask
                            # Tìm vị trí trung tâm của mask để đặt label
                            y_indices, x_indices = np.where(full_mask)
                            if len(y_indices) > 0:
                                center_y = int(np.mean(y_indices))
                                center_x = int(np.mean(x_indices))
                                
                                label = f"C{class_id}: {conf_val:.2f}"
                                cv2.putText(annotated_frame, label, 
                                          (center_x-30, center_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                          color, 2)
                                
                                # Lưu thông tin box để vẽ sau (tính với offset ROI)
                                box_x1 = int(x1) + x_off
                                box_y1 = int(y1) + y_off
                                box_x2 = int(x2) + x_off
                                box_y2 = int(y2) + y_off
                                box_info_list.append((box_x1, box_y1, box_x2, box_y2, color, class_id, conf_val))
                        
                        # Blend overlay với ảnh gốc
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.6, overlay, 0.4, 0)
                        
                        # VẼ BOUNDING BOXES SAU KHI ĐÃ XỬ LÝ MASK (nếu bật)
                        if show_boxes:
                            for (bx1, by1, bx2, by2, color, cls_id, conf) in box_info_list:
                                # Vẽ bounding box
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                                
                                # Vẽ label cho box
                                box_label = f"C{cls_id}"
                                (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(annotated_frame, 
                                            (bx1, by1-th-8), 
                                            (bx1+tw+6, by1), 
                                            color, -1)
                                cv2.putText(annotated_frame, box_label, 
                                          (bx1+3, by1-4), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                          (255, 255, 255), 1)
                        
                        # Hiển thị số lượng instances
                        info_text = f"Detected: {len(masks)} instances"
                        cv2.putText(annotated_frame, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # ============ TÍNH FPS ============
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Hiển thị FPS
            cv2.putText(annotated_frame, f"FPS: {fps}", (10, CROP_SIZE-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Hiển thị trạng thái ROI
            roi_status = "ROI: Active" if roi is not None else "ROI: None (Full 720x720)"
            cv2.putText(annotated_frame, roi_status, (10, CROP_SIZE-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Hiển thị trạng thái boxes và binary
            status_text = f"Boxes: {'ON' if show_boxes else 'OFF'} | Binary: {'ON' if show_binary else 'OFF'}"
            cv2.putText(annotated_frame, status_text, (10, CROP_SIZE-80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            
            frame_count += 1
            
            # ============ HIỂN THỊ BINARY MASK ============
            if show_binary and binary_mask is not None:
                cv2.imshow('Binary Mask', binary_mask)
        
        # ============ VẼ ROI ĐANG KÉO ============
        display_frame = annotated_frame.copy()
        if drawing and temp_roi is not None:
            x1, y1, x2, y2 = temp_roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display_frame, "Drawing ROI...", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ============ HIỂN THỊ ============
        cv2.imshow('RealSense D435 - Instance Segmentation', display_frame)
        
        # ============ XỬ LÝ PHÍM ============
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q hoặc ESC
            print("\n👋 Exiting...")
            break
        elif key == ord('s'):  # Save frame
            filename = f"capture_{int(time.time())}.png"
            cv2.imwrite(filename, display_frame)
            print(f"💾 Saved: {filename}")
        elif key == ord(' '):  # Pause/Resume
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"⏸️  {status}")
        elif key == ord('r'):  # Reset ROI
            roi = None
            drawing = False
            start_point = None
            temp_roi = None
            print("🔄 ROI reset - processing full frame")
        elif key == ord('o'):  # Toggle boxes
            show_boxes = not show_boxes
            status = "ON" if show_boxes else "OFF"
            print(f"📦 Bounding boxes: {status}")
        elif key == ord('b'):  # Toggle binary window
            show_binary = not show_binary
            status = "ON" if show_binary else "OFF"
            print(f"⬛ Binary mask window: {status}")
            if not show_binary:
                cv2.destroyWindow('Binary Mask')
        elif key == ord('c'):  # Load ROI from config
            loaded_roi = load_roi_from_file(ROI_CONFIG_FILE)
            if loaded_roi:
                roi = loaded_roi
                print(f"✅ Loaded ROI from config: {roi}")
            else:
                print("❌ Failed to load ROI from config")

finally:
    # ============ DỌN DẸP ============
    print("\n🧹 Cleaning up...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("✅ Done!")
    print(f"📊 Total frames processed: {frame_count}")
