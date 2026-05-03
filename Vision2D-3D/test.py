"""
test.py - YOLO v8 Instance Segmentation với làm mịn đường bo viền
Phím tắt:
  o  - Xuất ảnh từng mask riêng lẻ ra thư mục Output_image
  s  - Chuyển chế độ làm mịn viền (5 chế độ, cập nhật ngay lập tức)
  q  - Thoát
Nguồn đầu vào: truyền đường dẫn ảnh qua dòng lệnh hoặc mở hộp thoại chọn file
Ảnh được resize về 832x832 trước khi đưa vào model.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
from datetime import datetime
import scipy.interpolate as si
from collections import deque
from skimage.morphology import skeletonize

# ──────────────────────────────────────────────────────────────
# CÁC THUẬT TOÁN LÀM MỊN ĐƯỜNG BO VIỀN
# ──────────────────────────────────────────────────────────────

SMOOTH_MODES = [
    "Gốc (không làm mịn)",
    "Gaussian Blur",
    "Morphological Close+Open",
    "Spline Nội suy",
    "Polygon Approximation",
]


def smooth_gaussian(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Làm mịn mask bằng Gaussian Blur rồi ngưỡng hoá."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    return (blurred > 0.5).astype(np.uint8)


def smooth_morphological(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Làm mịn mask bằng phép toán hình thái học Close → Open."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def smooth_spline(contour: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Làm mịn contour bằng Cubic Spline nội suy (periodic)."""
    if len(contour) < 5:
        return contour
    pts = contour[:, 0, :]
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    # Đóng vòng
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_points)
    try:
        cs_x = si.CubicSpline(t, x, bc_type="periodic")
        cs_y = si.CubicSpline(t, y, bc_type="periodic")
        x_s = np.clip(cs_x(t_new), 0, 1e6).astype(int)
        y_s = np.clip(cs_y(t_new), 0, 1e6).astype(int)
        return np.array([[xi, yi] for xi, yi in zip(x_s, y_s)]).reshape(-1, 1, 2)
    except Exception:
        return contour


def smooth_approxpoly(contour: np.ndarray, epsilon_factor: float = 0.005) -> np.ndarray:
    """Làm mịn contour bằng xấp xỉ đa giác (Ramer–Douglas–Peucker)."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def apply_smooth_mask(mask: np.ndarray, smooth_mode: int) -> np.ndarray:
    """
    Trả về mask đã làm mịn (dạng filled binary) theo smooth_mode.
    Đây là hàm thống nhất dùng cho cả draw_instance lẫn render_binary
    để đảm bảo vùng tô màu và đường viền luôn khớp nhau.
    """
    if smooth_mode == 0:
        return mask
    elif smooth_mode == 1:
        return smooth_gaussian(mask)
    elif smooth_mode == 2:
        return smooth_morphological(mask)
    elif smooth_mode in (3, 4):
        # Tìm contour → làm mịn contour → rasterize lại thành filled mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        largest = max(contours, key=cv2.contourArea)
        if smooth_mode == 3:
            smooth_cnt = smooth_spline(largest)
        else:
            smooth_cnt = smooth_approxpoly(largest)
        smoothed = np.zeros_like(mask)
        cv2.fillPoly(smoothed, [smooth_cnt.astype(np.int32)], 1)
        return smoothed
    return mask


# ──────────────────────────────────────────────────────────────
# HÀM VẼ MASK VÀ ĐƯỜNG BO VIỀN
# ──────────────────────────────────────────────────────────────

# Bảng màu cho từng instance
COLORS = [
    (255, 56,  56),
    (56,  255, 56),
    (56,  56,  255),
    (255, 157, 56),
    (56,  255, 255),
    (255, 56,  255),
    (255, 255, 56),
    (128, 0,   255),
    (0,   128, 255),
    (255, 0,   128),
]


def get_color(idx: int):
    return COLORS[idx % len(COLORS)]


def draw_instance(
    canvas: np.ndarray,
    mask: np.ndarray,
    idx: int,
    label: str,
    smooth_mode: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Vẽ một instance lên canvas: tô màu vùng mask + vẽ đường bo viền đã làm mịn."""
    color = get_color(idx)
    overlay = canvas.copy()

    # Lấy mask đã làm mịn (thống nhất cho cả fill lẫn contour)
    smooth = apply_smooth_mask(mask, smooth_mode)

    # Tô màu CHỈ trong vùng mask đã smooth
    mask_bool = smooth > 0
    overlay[mask_bool] = (
        np.array(color, dtype=np.float32) * alpha
        + canvas[mask_bool].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    # Vẽ đường viền từ mask đã smooth (luôn khớp với vùng tô)
    contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return overlay
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(overlay, [largest], -1, color, 2)

    return overlay


# ──────────────────────────────────────────────────────────────
# XUẤT ẢNH TỪNG MASK
# ──────────────────────────────────────────────────────────────

def export_masks(frame: np.ndarray, masks: list, classes: list, confs: list,
                 names: dict, out_dir: str) -> int:
    """
    Lưu từng instance mask ra file ảnh trong out_dir.
    Trả về số lượng ảnh đã lưu.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0
    for i, (mask, cls_id, conf) in enumerate(zip(masks, classes, confs)):
        # Tạo ảnh chứa chỉ instance này (nền đen)
        instance_img = np.zeros_like(frame)
        instance_img[mask > 0] = frame[mask > 0]

        # Crop theo bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            cropped = instance_img[y1:y2, x1:x2]
        else:
            cropped = instance_img

        cls_name = names.get(int(cls_id), f"class{int(cls_id)}")
        filename = f"instance_{i+1}_{cls_name}_conf{conf:.2f}_{ts}.png"
        filepath = os.path.join(out_dir, filename)
        cv2.imwrite(filepath, cropped)
        saved += 1
        print(f"  [Saved] {filepath}")
    return saved


# ──────────────────────────────────────────────────────────────# TRỤC SINH TRƯỜNG 2D
# ──────────────────────────────────────────────────────────────

def _skel_neighbors(r, c, h, w):
    """Trả về các pixel lân cận 8-connectivity hợp lệ."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def ga_find_endpoints(skel_bool):
    """
    Tìm endpoint (1 lân cận) và junction (≥3 lân cận) trên skeleton.
    Trả về: endpoints_rc (N,2), junctions_rc (M,2)
    """
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    cnt = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT) * skel_u8
    endpoints_rc  = np.column_stack(np.where((cnt == 1) & skel_bool))
    junctions_rc  = np.column_stack(np.where((cnt >= 3) & skel_bool))
    return endpoints_rc, junctions_rc


def ga_bfs_dist(skel_bool, base_rc):
    """
    BFS đo kỏ geodesic dọc theo skeleton từ điểm gốc base_rc.
    Trả về dist_map: int32 (-1 = không đến được).
    """
    h, w = skel_bool.shape
    dist_map = np.full((h, w), -1, dtype=np.int32)
    if base_rc is None:
        return dist_map
    dist_map[base_rc[0], base_rc[1]] = 0
    queue = deque([base_rc])
    while queue:
        r, c = queue.popleft()
        d = dist_map[r, c]
        for nr, nc in _skel_neighbors(r, c, h, w):
            if skel_bool[nr, nc] and dist_map[nr, nc] == -1:
                dist_map[nr, nc] = d + 1
                queue.append((nr, nc))
    return dist_map


def ga_trace_path(skel_bool, dist_map, base_rc, tip_rc):
    """
    Theo dọc path từ tip → base bằng gradient descent trên dist_map.
    Trả về list[(r,c)].
    """
    h, w = skel_bool.shape
    path = []
    current = tip_rc
    visited = set()
    while True:
        r, c = current
        path.append((r, c))
        visited.add((r, c))
        if np.linalg.norm(np.array([r, c]) - np.array(base_rc)) < 2:
            break
        if dist_map[r, c] < 1:
            break
        best, best_d = None, dist_map[r, c]
        for nr, nc in _skel_neighbors(r, c, h, w):
            if skel_bool[nr, nc] and (nr, nc) not in visited and dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best = (nr, nc)
        if best is None:
            break
        current = best
        if len(path) > 1000:
            break
    return path


def compute_growth_axis_2d(masks, frame, smooth_mode):
    """
    Vẽ trục sinh trưởng 2D chỉ cho các nhánh thực sự (lọc bỏ gốc nhỏ).
    Tiêu chí lọc: diện tích mask < 15% diện tích instance lớn nhất → bỏ qua.
      - Mỗi nhánh (base → tip) một màu khác nhau, đường dày
      - Điểm base và tip là chấm tròn
    """
    BRANCH_COLORS = [
        (0, 255, 80),
        (0, 180, 255),
        (255, 60, 200),
        (80, 220, 255),
        (200, 80, 255),
        (255, 255, 60),
    ]
    LINE_THICKNESS = 4
    DOT_RADIUS     = 7

    vis = np.zeros_like(frame)

    # ── Tính diện tích từng mask, lọc instance quá nhỏ ──────────
    smoothed_masks = [apply_smooth_mask(m, smooth_mode) for m in masks]
    areas = [m.sum() for m in smoothed_masks]
    if not areas or max(areas) == 0:
        return vis
    max_area = max(areas)
    MIN_AREA_RATIO = 0.15   # bỏ instance có diện tích < 15% lớn nhất

    color_idx = 0  # chỉ số màu riêng cho các instance hợp lệ
    for inst_idx, mask in enumerate(smoothed_masks):
        if areas[inst_idx] < MIN_AREA_RATIO * max_area:
            # Instance quá nhỏ → đây là gốc nối, bỏ qua
            continue

        inst_color = get_color(inst_idx)
        if mask.sum() == 0:
            continue

        skel_bool = skeletonize(mask.astype(bool))
        if not skel_bool.any():
            continue

        endpoints_rc, _ = ga_find_endpoints(skel_bool)
        if len(endpoints_rc) == 0:
            continue

        # Base = pixel skeleton thấp nhất (row lớn nhất)
        skel_pts = np.column_stack(np.where(skel_bool))
        base_rc = tuple(skel_pts[np.argmax(skel_pts[:, 0])])

        dist_map = ga_bfs_dist(skel_bool, base_rc)

        max_dist = dist_map.max()
        # Bỏ thêm nếu skeleton tổng quá ngắn
        if max_dist < 40:
            continue

        min_len = max(30, int(0.10 * max_dist))
        tips_kept = [
            tuple(rc) for rc in endpoints_rc
            if dist_map[rc[0], rc[1]] >= min_len
        ]
        if not tips_kept:
            tips_kept = [tuple(endpoints_rc[np.argmax(
                [dist_map[r, c] for r, c in endpoints_rc]
            )])]

        # Vẽ từng nhánh base → tip
        for i, tip_rc in enumerate(tips_kept):
            path = ga_trace_path(skel_bool, dist_map, base_rc, tip_rc)
            if len(path) < 2:
                continue

            color = BRANCH_COLORS[(color_idx + i) % len(BRANCH_COLORS)]
            pts = np.array([[c, r] for r, c in path], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=False, color=color,
                          thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)

            tr, tc = tip_rc
            cv2.circle(vis, (tc, tr), DOT_RADIUS, (255, 255, 255), -1)
            cv2.circle(vis, (tc, tr), DOT_RADIUS - 2, color, -1)

        # Điểm base
        br, bc = base_rc
        cv2.circle(vis, (bc, br), DOT_RADIUS + 2, (255, 255, 255), -1)
        cv2.circle(vis, (bc, br), DOT_RADIUS, inst_color, -1)

        color_idx += len(tips_kept)

    return vis



# ──────────────────────────────────────────────────────────────# HÀM VẼ BINARY MASK
# ──────────────────────────────────────────────────────────────

def get_smoothed_masks(masks: list, smooth_mode: int) -> list:
    """Trả về danh sách mask đã làm mịn (filled) theo smooth_mode."""
    return [apply_smooth_mask(m, smooth_mode) for m in masks]


def render_binary(masks: list, size: int, smooth_mode: int) -> np.ndarray:
    """Tạo ảnh binary tổng hợp từ mask đã làm mịn: mỗi instance một mức xám, nền đen."""
    smoothed = get_smoothed_masks(masks, smooth_mode)
    binary = np.zeros((size, size), dtype=np.uint8)
    n = len(smoothed)
    for i, mask in enumerate(smoothed):
        # Chia đều mức xám 80-255 cho các instance
        gray_val = int(80 + (175 / max(n, 1)) * i) if n > 1 else 255
        binary[mask > 0] = gray_val
    # Chuyển sang BGR để ghép với imshow
    bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # HUD
    cv2.rectangle(bgr, (0, 0), (size, 26), (30, 30, 30), -1)
    cv2.putText(bgr, f"Binary Mask  |  {n} instance(s)  |  {SMOOTH_MODES[smooth_mode]}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return bgr


# ──────────────────────────────────────────────────────────────
# HÀM VẼ LẠI CANVAS TỪ DỮ LIỆU ĐÃ CACHE
# ──────────────────────────────────────────────────────────────

def render_canvas(
    frame: np.ndarray,
    masks: list,
    classes: list,
    confs: list,
    labels: list,
    smooth_mode: int,
) -> np.ndarray:
    """Vẽ toàn bộ instances lên frame với smooth_mode hiện tại."""
    canvas = frame.copy()
    for i, (mask, label) in enumerate(zip(masks, labels)):
        canvas = draw_instance(canvas, mask, i, label, smooth_mode)

    # ── HUD ────────────────────────────────────────────────────
    mode_text = f"[s] Smoothing {smooth_mode}: {SMOOTH_MODES[smooth_mode]}"
    inst_text = f"Instances: {len(masks)}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(canvas, mode_text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, inst_text, (canvas.shape[1] - 140, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 100), 1, cv2.LINE_AA)
    return canvas


# ──────────────────────────────────────────────────────────────
# HÀM CHÍNH
# ──────────────────────────────────────────────────────────────

def main():
    MODEL_SIZE  = 832          # kích thước đầu vào model
    CONF_THRESH = 0.35

    model_path = os.path.join(os.path.dirname(__file__), "v8m-seg-832.pt")
    output_dir = os.path.join(os.path.dirname(__file__), "Output_image")

    # ── Chọn ảnh ────────────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Mở hộp thoại chọn file nếu không truyền tham số
        try:
            import tkinter as tk
            from tkinter import filedialog
            root_tk = tk.Tk()
            root_tk.withdraw()
            img_path = filedialog.askopenfilename(
                title="Chọn ảnh",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
            )
            root_tk.destroy()
        except Exception:
            img_path = ""
        if not img_path:
            print("[ERROR] Không có ảnh nào được chọn.")
            return

    # ── Đọc ảnh gốc ─────────────────────────────────────────────
    original = cv2.imread(img_path)
    if original is None:
        print(f"[ERROR] Không đọc được ảnh: {img_path}")
        return
    print(f"[INFO] Ảnh gốc: {original.shape[1]}x{original.shape[0]}  →  resize về {MODEL_SIZE}x{MODEL_SIZE}")

    # Resize về 832x832 để đưa vào model
    frame_832 = cv2.resize(original, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)

    # ── Tải model và chạy inference MỘT LẦN ─────────────────────
    print(f"[INFO] Tải model: {model_path}")
    model = YOLO(model_path)
    names = model.names

    print("[INFO] Đang chạy inference ...")
    results = model(frame_832, conf=CONF_THRESH, imgsz=MODEL_SIZE, verbose=False)
    result  = results[0]
    print(f"[INFO] Phát hiện {len(result.boxes)} instance(s).")

    # ── Cache mask, class, conf ──────────────────────────────────
    cached_masks:   list[np.ndarray] = []
    cached_classes: list = []
    cached_confs:   list = []
    cached_labels:  list[str] = []

    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()   # (N, H_m, W_m)
        cls_ids    = result.boxes.cls.cpu().numpy()
        confs_data = result.boxes.conf.cpu().numpy()

        for i in range(len(masks_data)):
            # Resize mask về 832x832
            bin_mask = cv2.resize(masks_data[i], (MODEL_SIZE, MODEL_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
            bin_mask = (bin_mask > 0.5).astype(np.uint8)

            cls_id   = cls_ids[i]
            conf     = confs_data[i]
            cls_name = names.get(int(cls_id), f"class{int(cls_id)}")

            cached_masks.append(bin_mask)
            cached_classes.append(cls_id)
            cached_confs.append(conf)
            cached_labels.append(f"{cls_name} {conf:.2f}")

    # ── Vòng lặp hiển thị ────────────────────────────────────────
    smooth_mode = 0
    growth_vis = None          # cache ảnh trục sinh trưởng
    cv2.namedWindow("YOLO v8 Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)

    print("\n[PHíM TẮT]")
    print("  s  - Đổi chế độ làm mịn (cập nhật ngay)")
    print("  o  - Xuất từng mask ra Output_image")
    print("  k  - Tạo trục sinh trưởng 2D")
    print("  q  - Thoát\n")

    # Vẽ lần đầu
    canvas = render_canvas(frame_832, cached_masks, cached_classes,
                           cached_confs, cached_labels, smooth_mode)
    binary_view = render_binary(cached_masks, MODEL_SIZE, smooth_mode)
    cv2.imshow("YOLO v8 Segmentation", canvas)
    cv2.imshow("Binary Mask", binary_view)
    cv2.waitKey(1)  # Bắt buộc để window hiện ra ngay trên Windows

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("s"):
            smooth_mode = (smooth_mode + 1) % len(SMOOTH_MODES)
            print(f"[Smoothing] Chế độ {smooth_mode}: {SMOOTH_MODES[smooth_mode]}")
            # Vẽ lại cả 2 window từ cache với smooth_mode mới, KHÔNG chạy lại YOLO
            canvas = render_canvas(frame_832, cached_masks, cached_classes,
                                   cached_confs, cached_labels, smooth_mode)
            binary_view = render_binary(cached_masks, MODEL_SIZE, smooth_mode)
            cv2.imshow("YOLO v8 Segmentation", canvas)
            cv2.imshow("Binary Mask", binary_view)

        elif key == ord("k"):
            print("[Growth Axis] Đang tính trục sinh trưởng 2D ...")
            if not cached_masks:
                print("[WARN] Không có mask nào.")
            else:
                growth_vis = compute_growth_axis_2d(cached_masks, frame_832, smooth_mode)
                cv2.namedWindow("Growth Axis 2D", cv2.WINDOW_NORMAL)
                cv2.imshow("Growth Axis 2D", growth_vis)
                print(f"[Growth Axis] Xong. {len(cached_masks)} instance(s).")

        elif key == ord("o"):
            if not cached_masks:
                print("[WARN] Không có mask nào để xuất.")
            else:
                print(f"[Export] Đang lưu {len(cached_masks)} mask(s) ...")
                n = export_masks(
                    frame_832, cached_masks, cached_classes,
                    cached_confs, names, output_dir
                )
                print(f"[Export] Đã lưu {n} ảnh vào '{output_dir}'")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
