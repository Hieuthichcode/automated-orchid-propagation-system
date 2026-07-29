import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
from queue import Queue
import open3d as o3d
from ultralytics import YOLO
import os
from datetime import datetime
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from collections import deque
import json

class RealSenseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intel RealSense D435i - Advanced Point Cloud Viewer")
        self.root.geometry("750x1000")
        self.root.configure(bg='#2d2d30')
        
        # Biến trạng thái
        self.pipeline = None
        self.align = None
        self.is_running = False
        self.is_preview = True
        self.frame_queue = Queue(maxsize=2)
        
        # 🔒 Đồng bộ truy cập camera: mọi thao tác wait_for_frames()+align.process()
        # (dù từ thread preview hay từ main thread - Hybrid pipeline, Pass1/2,
        # hand-eye calib...) ĐỀU phải giữ chung camera_lock này để tránh 2 luồng
        # gọi RealSense API đồng thời. preview_event dùng để thread preview NGỦ
        # (không busy-loop) khi preview bị tắt.
        self.camera_lock = threading.Lock()
        self.preview_event = threading.Event()
        self.preview_event.set()  # preview mặc định đang BẬT (is_preview=True)
        
        # Lưu point cloud để có thể save sau
        self.last_pcd = None
        self.last_method = None  # Phương pháp vừa dùng
        
        # Frame counter cho YOLO (giảm tần suất inference)
        self.frame_count = 0
        self.yolo_skip_frames = 5  # Chỉ chạy YOLO mỗi 5 frames để ổn định
        self.last_detection_overlay = None  # Lưu overlay cuối cùng
        self.current_binary_mask = np.zeros((720, 720), dtype=np.uint8)  # Binary mask hiện tại (720x720)
        
        # 🆕 Instance Segmentation (luôn bật)
        self.export_mode = "individual"  # "individual" hoặc "combined" - default: individual
        self.instance_masks = []  # List of individual instance masks (720x720)
        self.instance_classes = []  # List of class IDs for each instance
        self.instance_confidences = []  # List of confidence scores
        
        # Transformation parameters (để apply lên individual instances)
        self.R_matrix_transform = None  # Rotation matrix từ plane detection
        self.translation_vector = None  # Translation vector [dx, dy, dz]
        
        # 🤖 Robot Cutting Point Calculation
        self.enable_cutting_point = False  # Bật/tắt tính toán điểm cắt
        self.cutting_data = None  # Lưu dữ liệu cắt: {center, cut_point, cut_angle, cut_line}
        
        # 🔍 Skeleton Tips Visualization
        self.show_skeleton_tips = True  # Bật/tắt hiển thị skeleton tips
        self.base_mode_3d = False        # False=BASE(2D), True=BASE(3D)
        self.last_tips_data_per_instance = {}  # Dict: {instance_idx: tips_data}
        self.last_preview_rgb_np = None  # Cache ảnh preview cuối để xuất file
        
        # 🤖 Picking Frame (trục tọa độ cho robot)
        self.show_picking_frame = True  # Bật/tắt tạo trục tọa độ tại thân mầm
        self.last_path_vis_per_instance = {}  # Dict: {instance_idx: path_vis_image}
        
        # 🪟 Preview Windows
        self.rgb_window = None
        self.mask_window = None
        self.roi_setting_window = None  # Cửa sổ tạm để set ROI
        
        # Vùng làm việc (ROI)
        self.roi = None
        self.is_setting_area = False
        self.roi_start = None
        self.roi_rect_id = None
        
        # Cấu hình chất lượng point cloud
        self.num_frames_avg = 2  # 2 frames đủ cho scene tĩnh (camera + vật đều đứng yên) - tiết kiệm ~60-90ms
        self.sampling_step = 4
        self.use_outlier_removal = True
        self.use_smoothing = True
        self.use_filters = False  # TẮT filters mặc định để giữ nguyên mask sắc nét
        
        # Cấu hình nâng cao cho độ chính xác
        self.use_bilateral_filter = False  # Tắt mặc định để nhanh hơn
        self.use_confidence_filter = True
        self.depth_min = 0.1  # meter
        self.depth_max = 2.5  # meter
        self.confidence_threshold = 2  # 0-3, cao hơn = chặt chẽ hơn
        
        # RealSense point cloud object
        self.pc = rs.pointcloud()
        
        # YOLOv8 Segmentation
        self.use_segmentation = True  # Luôn bật mặc định
        self.segmentation_conf = 0.5
        self.yolo_model = None
        self.last_mask = None
        self.last_detection_boxes = []  # Lưu boxes đã detect
        self.detection_results = None  # Lưu toàn bộ results
        self.show_boxes = False  # Hiển thị bounding box và class label
        self.show_binary = False  # Hiển thị binary mask window
        self.show_mask_overlay = True   # Hiển thị màu mask overlay lên preview
        self.last_clean_rgb_bgr = None  # Ảnh RGB 720x720 sạch (trưa khi vẽ mask)
        self.show_center_dot = True  # Hiển thị dấu chấm tâm class 0
        self.load_yolo_model()
        
        # Load ROI từ file nếu có
        self.load_roi()
        
        # ============================================================
        # 🆕 HYBRID 2D-3D GRASP PIPELINE - Config & state
        # (theo NOTE_Codex_Phuong_phap_MethodsX)
        # ============================================================
        self.pipeline_config = {
            'morphology': {
                'kernel_size': (3, 3),
                'erosion_iterations': 2,
                'dilation_iterations': 2,
            },
            'skeleton': {
                'connectivity': 8,
            },
            'pointcloud': {
                'sor_neighbors': 10,
                'sor_std_ratio': 2.0,
                'voxel_size_mm': 0.5,
                'ransac_threshold_mm': 3.0,
                'ransac_iterations': 500,
                'ransac_seed': 0,
                # ⚠️ 2 ngưỡng CHẤT LƯỢNG dưới đây là GIÁ TRỌ MẶC ĐỬNH,
                # CHƯA được xác nhận bằng dữ liệu ROI nền thực tế của hệ thống.
                # KHÔNG tuyên bố đây là ngưỡng đã thực nghiệm xác nhận - cần
                # người dùng hiệu chỉnh lại cho đúng điều kiện lắp đặt camera/bàn
                # trước khi dùng thực tế. Khác với min_inliers=3 (ràng buộc HìNH
                # HỌC tối thiểu để xác định 1 mặt phẳng) - đây là ngưỡng CHẤT
                # LƯỢNG (mức hỗ trợ của plane trong ROI).
                'ransac_min_inliers': 30,
                'ransac_min_inlier_ratio': 0.5,
                'normalization_rotation_only': True,
            },
            'branch': {
                'support_radius_px': 5,
                'assignment_domain': 'image_pixels',
                'assign_before_voxel_downsampling': True,
                'overlap_policy': 'nearest_path',
                'min_branch_points': 10,
            },
            'fusion': {
                'alpha': 0.6,\
                'theta_max_deg': 85.0,
                'fallback': 'longest_valid_basal_to_endpoint_path',
            },
            'grasp': {
                'basal_radius_mm': 8.0,
                'growth_axis_is_local_z': True,
                'camera_reference_axis': (0.0, 0.0, 1.0),
                # � Chỉ xử lý instance thuộc target_class_id (0 = "thân/nhánh" -
                # đối tượng cần kẹp, theo đúng quy ước dữ liệu YOLO hiện có;
                # 1 = "gốc" chỉ dùng làm mốc tham chiếu, KHÔNG kẹp).
                'target_class_id': 0,
                # 🔄 KHÔNG tự chọn instance theo mask_area (không thuộc phương
                # pháp trong bản thảo). explicit_only = chỉ dùng 1 instance duy
                # nhất khi được chỉ định rõ qua target_instance_index (ROI/người
                # dùng/lệnh ngoài), hoặc khi chỉ có đúng 1 instance thành công.
                # Nếu có nhiều instance hợp lệ mà target_instance_index=None,
                # hệ thống KHÔNG tự chọn và KHÔNG gán last_grasp_result.
                'target_selection_rule': 'explicit_only',
                'target_instance_index': None,
            },
        }
        self._intrinsics_cache = None   # cache (fx,fy,cx,cy,depth_scale,start_x)
        self.T_B_C = None                # 4x4 camera->robot-base (hand-eye calibration)
        self.handeye_note = ''
        self.load_handeye_calibration()
        
        # Cửa sổ & state cho Grasp Pose / Hand-eye Calibration
        self.grasp_result_window = None
        self.grasp_result_text = None
        self.last_grasp_results = []
        self.last_grasp_result = None
        self.handeye_calib_window = None
        self.handeye_samples = []
        
        # Tạo giao diện
        self.create_widgets()
        
        # Khởi động camera
        self.init_camera()
    
    def load_yolo_model(self):
        """Load YOLOv8 segmentation model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
            if os.path.exists(model_path):
                print(f"Loading YOLO model: {model_path}")
                self.yolo_model = YOLO(model_path)
                print("✅ YOLO model loaded successfully!")
            else:
                print(f"⚠️ YOLO model not found: {model_path}")
                self.yolo_model = None
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None
        
    def create_widgets(self):
        # Container chính với scrollbar
        container = tk.Frame(self.root, bg='#2d2d30')
        container.pack(fill=tk.BOTH, expand=True)
        
        canvas_container = tk.Canvas(container, bg='#2d2d30', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas_container.yview)
        
        main_frame = ttk.Frame(canvas_container)
        main_frame.bind(
            "<Configure>",
            lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all"))
        )
        
        canvas_container.create_window((0, 0), window=main_frame, anchor="nw")
        canvas_container.configure(yscrollcommand=scrollbar.set)
        
        canvas_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Style
        style = ttk.Style()
        style.configure('Dark.TFrame', background='#2d2d30')
        style.configure('Dark.TLabelframe', background='#2d2d30', foreground='white')
        style.configure('Dark.TLabelframe.Label', background='#2d2d30', foreground='white')
        style.configure('Dark.TLabel', background='#2d2d30', foreground='white')
        style.configure('Dark.TCheckbutton', background='#2d2d30', foreground='white')
        style.configure('Dark.TRadiobutton', background='#2d2d30', foreground='white')
        
        main_frame.configure(style='Dark.TFrame')
        
        # Tiêu đề
        title_label = ttk.Label(main_frame, text="Intel RealSense D435i - Advanced Point Cloud", 
                                font=('Arial', 16, 'bold'), style='Dark.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Button mở preview windows
        preview_btn_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        preview_btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(preview_btn_frame, text="📺 Open RGB Preview", 
                  command=self.open_rgb_window, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(preview_btn_frame, text="🎭 Open Mask Preview", 
                  command=self.open_mask_window, width=20).pack(side=tk.LEFT, padx=5)
        
        # Frame điều khiển
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding="10", style='Dark.TLabelframe')
        control_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.btn_process = ttk.Button(control_frame, text="🔬 XỬ LÝ & XUẤT POINT CLOUD", 
                                       command=self.process_and_export, width=30)
        self.btn_process.grid(row=0, column=0, padx=5, pady=5)
        
        self.preview_var = tk.BooleanVar(value=True)
        self.btn_preview = ttk.Checkbutton(control_frame, text="🎥 Preview RGB", 
                                            variable=self.preview_var,
                                            command=self.toggle_preview)
        self.btn_preview.grid(row=0, column=1, padx=5, pady=5)
        
        # Filters checkbox
        self.filters_var = tk.BooleanVar(value=False)
        self.btn_filters = ttk.Checkbutton(control_frame, text="🔧 Depth Filters", 
                                            variable=self.filters_var,
                                            command=self.toggle_filters)
        self.btn_filters.grid(row=0, column=2, padx=5, pady=5)
        
        # Show boxes checkbox
        self.boxes_var = tk.BooleanVar(value=False)
        self.btn_boxes = ttk.Checkbutton(control_frame, text="📦 Show Boxes", 
                                          variable=self.boxes_var,
                                          command=self.toggle_boxes)
        self.btn_boxes.grid(row=0, column=3, padx=5, pady=5)
        
        # Show binary mask window checkbox
        self.binary_var = tk.BooleanVar(value=False)
        self.btn_binary = ttk.Checkbutton(control_frame, text="⬛ Binary Mask", 
                                           variable=self.binary_var,
                                           command=self.toggle_binary_window)
        self.btn_binary.grid(row=0, column=4, padx=5, pady=5)
        
        # Show center dot checkbox
        self.center_var = tk.BooleanVar(value=True)
        self.btn_center = ttk.Checkbutton(control_frame, text="🎯 Center Dot", 
                                           variable=self.center_var,
                                           command=self.toggle_center_dot)
        self.btn_center.grid(row=0, column=5, padx=5, pady=5)
        
        # 🤖 Robot Cutting Point checkbox
        self.cutting_var = tk.BooleanVar(value=False)
        self.btn_cutting = ttk.Checkbutton(control_frame, text="🔪 Cutting Point", 
                                            variable=self.cutting_var,
                                            command=self.toggle_cutting_point)
        self.btn_cutting.grid(row=0, column=6, padx=5, pady=5)
        
        # 🔍 Skeleton Tips checkbox
        self.skeleton_tips_var = tk.BooleanVar(value=False)
        self.btn_skeleton_tips = ttk.Checkbutton(control_frame, text="🌿 Tips", 
                                                  variable=self.skeleton_tips_var,
                                                  command=self.toggle_skeleton_tips)
        self.btn_skeleton_tips.grid(row=0, column=7, padx=5, pady=5)
        
        # 🤖 Picking Frame checkbox
        self.picking_frame_var = tk.BooleanVar(value=True)
        self.btn_picking_frame = ttk.Checkbutton(control_frame, text="🤖 Pick Frame",
                                                   variable=self.picking_frame_var,
                                                   command=self.toggle_picking_frame)
        self.btn_picking_frame.grid(row=0, column=8, padx=5, pady=5)
        
        # 🗺️ Base mode: 2D vs 3D checkbox
        self.base_3d_var = tk.BooleanVar(value=False)
        self.btn_base_3d = ttk.Checkbutton(control_frame, text="🗺️ Base 3D",
                                            variable=self.base_3d_var, style='Dark.TCheckbutton',
                                            command=self.toggle_base_mode)
        self.btn_base_3d.grid(row=0, column=9, padx=5, pady=5)
        
        # ROI controls (Detection Area) - Row 1
        self.btn_set_area = ttk.Button(control_frame, text="🎯 Set Detection Area", 
                                        command=self.toggle_set_area, width=20)
        self.btn_set_area.grid(row=1, column=0, padx=5, pady=5)
        
        self.btn_save_area = ttk.Button(control_frame, text="💾 Save Area", 
                                         command=self.save_roi, width=15)
        self.btn_save_area.grid(row=1, column=1, padx=5, pady=5)
        
        self.btn_clear_area = ttk.Button(control_frame, text="🗑️ Clear Area", 
                                          command=self.clear_roi, width=15)
        self.btn_clear_area.grid(row=1, column=2, padx=5, pady=5)
        
        # 🆕 Clear Tips button
        self.btn_clear_tips = ttk.Button(control_frame, text="🌿 Clear Tips",
                                          command=self.clear_tips, width=15)
        self.btn_clear_tips.grid(row=1, column=6, padx=5, pady=5)
        
        # � Show Mask overlay checkbox
        self.mask_overlay_var = tk.BooleanVar(value=True)
        self.btn_mask_overlay = ttk.Checkbutton(control_frame, text="🎨 Show Mask",
                                                variable=self.mask_overlay_var, style='Dark.TCheckbutton',
                                                command=self.toggle_mask_overlay)
        self.btn_mask_overlay.grid(row=1, column=7, padx=5, pady=5)
        
        # 🆕 Export mode selection - Row 1, column 3-4
        ttk.Label(control_frame, text="📤 Export:", style='Dark.TLabel').grid(row=1, column=3, padx=(10,0), pady=5, sticky=tk.E)
        self.export_mode_var = tk.StringVar(value="individual")
        export_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        export_frame.grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(export_frame, text="Riêng (N files)", variable=self.export_mode_var, 
                       value="individual", style='Dark.TRadiobutton', command=self.update_export_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(export_frame, text="Gộp (1 file)", variable=self.export_mode_var, 
                       value="combined", style='Dark.TRadiobutton', command=self.update_export_mode).pack(side=tk.LEFT, padx=5)
        
        # Save Point Cloud button - Row 2
        self.btn_save_pc = ttk.Button(control_frame, text="💾 Save Point Cloud (.ply)", 
                                       command=self.save_point_cloud, width=25)
        self.btn_save_pc.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # 📷 Lưu ảnh mầm
        self.btn_save_all_mam = ttk.Button(control_frame, text="📷 Lưu cả mầm",
                                             command=self.save_all_mam, width=15)
        self.btn_save_all_mam.grid(row=2, column=2, padx=5, pady=5)
        
        self.btn_save_parts_mam = ttk.Button(control_frame, text="📷 Lưu từng phần",
                                              command=self.save_parts_mam, width=15)
        self.btn_save_parts_mam.grid(row=2, column=3, padx=5, pady=5)
        
        # 🆕 HYBRID 2D-3D GRASP PIPELINE + Hand-Eye Calibration - Row 3
        self.btn_hybrid_grasp = ttk.Button(control_frame, text="🤖 Tính Grasp Pose (Hybrid 2D-3D)",
                                            command=self.run_hybrid_grasp_pipeline, width=32)
        self.btn_hybrid_grasp.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        self.btn_handeye_calib = ttk.Button(control_frame, text="🛠️ Hiệu chỉnh Camera-Robot",
                                             command=self.open_handeye_calibration_window, width=26)
        self.btn_handeye_calib.grid(row=3, column=2, columnspan=2, padx=5, pady=5)
        
        self.handeye_status_label = ttk.Label(
            control_frame,
            text="🛠️ Hand-eye: " + (self.handeye_note if self.T_B_C is not None else "Chưa hiệu chỉnh"),
            style='Dark.TLabel', foreground='yellow')
        self.handeye_status_label.grid(row=3, column=4, columnspan=4, padx=5, pady=5, sticky=tk.W)
        
        # Frame cấu hình chất lượng
        quality_frame = ttk.LabelFrame(main_frame, text="⚙️ Cấu hình Chất lượng Point Cloud", padding="10", style='Dark.TLabelframe')
        quality_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Số frames để average
        ttk.Label(quality_frame, text="Số frames quét (nhiều hơn = chính xác hơn):", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.frames_scale = tk.Scale(quality_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                                      bg='#2d2d30', fg='white', highlightthickness=0,
                                      command=self.update_frames_count, length=200)
        self.frames_scale.set(5)  # Mặc định 5 frames để nhanh hơn
        self.frames_scale.grid(row=0, column=1, padx=10, pady=5)
        self.frames_label = ttk.Label(quality_frame, text="5 frames", style='Dark.TLabel', foreground='cyan')
        self.frames_label.grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # Sampling density
        ttk.Label(quality_frame, text="Độ phân giải (nhỏ = dày điểm hơn):", style='Dark.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sampling_scale = tk.Scale(quality_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                        bg='#2d2d30', fg='white', highlightthickness=0,
                                        command=self.update_sampling, length=200)
        self.sampling_scale.set(1)
        self.sampling_scale.grid(row=1, column=1, padx=10, pady=5)
        self.sampling_label = ttk.Label(quality_frame, text="Step=1 (Rất dày)", style='Dark.TLabel', foreground='cyan')
        self.sampling_label.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # Checkboxes cho các filter
        self.outlier_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(quality_frame, text="🔍 Statistical Outlier Removal (loại nhiễu)", 
                       variable=self.outlier_var, style='Dark.TCheckbutton').grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        self.smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(quality_frame, text="✨ Voxel Smoothing (làm mịn)", 
                       variable=self.smooth_var, style='Dark.TCheckbutton').grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # 🆕 Frame lọc noise (3 phương pháp)
        noise_frame = ttk.LabelFrame(main_frame, text="🧹 Lọc Nhiễu (Noise Removal)", padding="10", style='Dark.TLabelframe')
        noise_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Phương pháp 1: Mask Erosion
        self.mask_erosion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(noise_frame, text="🔹 Mask Erosion (co mask, loại biên thô)", 
                       variable=self.mask_erosion_var, style='Dark.TCheckbutton').grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(noise_frame, text="  Erosion size:", style='Dark.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2, padx=(20,0))
        self.erosion_size_scale = tk.Scale(noise_frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL,
                                           bg='#2d2d30', fg='white', highlightthickness=0, length=120)
        self.erosion_size_scale.set(2)  # 2 pixels mặc định
        self.erosion_size_scale.grid(row=1, column=1, padx=5, pady=2)
        self.erosion_label = ttk.Label(noise_frame, text="2 px", style='Dark.TLabel', foreground='cyan')
        self.erosion_label.grid(row=1, column=2, sticky=tk.W, pady=2)
        self.erosion_size_scale.config(command=lambda v: self.erosion_label.config(text=f"{int(float(v))} px"))
        
        # Phương pháp 2: Z-offset Filtering
        self.zoffset_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(noise_frame, text="🔹 Z-offset Filtering (loại điểm gần nền)", 
                       variable=self.zoffset_filter_var, style='Dark.TCheckbutton').grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(noise_frame, text="  Tolerance (mm):", style='Dark.TLabel').grid(row=3, column=0, sticky=tk.W, pady=2, padx=(20,0))
        self.zoffset_tolerance_scale = tk.Scale(noise_frame, from_=1, to=10, resolution=0.5, orient=tk.HORIZONTAL,
                                                bg='#2d2d30', fg='white', highlightthickness=0, length=120)
        self.zoffset_tolerance_scale.set(3.0)  # 3mm mặc định
        self.zoffset_tolerance_scale.grid(row=3, column=1, padx=5, pady=2)
        self.zoffset_label = ttk.Label(noise_frame, text="3.0 mm", style='Dark.TLabel', foreground='cyan')
        self.zoffset_label.grid(row=3, column=2, sticky=tk.W, pady=2)
        self.zoffset_tolerance_scale.config(command=lambda v: self.zoffset_label.config(text=f"{float(v):.1f} mm"))
        
        # Phương pháp 3: Radius Outlier Removal
        self.radius_outlier_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(noise_frame, text="🔹 Radius Outlier Removal (loại điểm cô lập)", 
                       variable=self.radius_outlier_var, style='Dark.TCheckbutton').grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(noise_frame, text="  Radius (mm):", style='Dark.TLabel').grid(row=5, column=0, sticky=tk.W, pady=2, padx=(20,0))
        self.radius_scale = tk.Scale(noise_frame, from_=1, to=10, resolution=0.5, orient=tk.HORIZONTAL,
                                     bg='#2d2d30', fg='white', highlightthickness=0, length=120)
        self.radius_scale.set(3.0)  # 3mm mặc định
        self.radius_scale.grid(row=5, column=1, padx=5, pady=2)
        self.radius_label = ttk.Label(noise_frame, text="3.0 mm", style='Dark.TLabel', foreground='cyan')
        self.radius_label.grid(row=5, column=2, sticky=tk.W, pady=2)
        self.radius_scale.config(command=lambda v: self.radius_label.config(text=f"{float(v):.1f} mm"))
        
        ttk.Label(noise_frame, text="  Min neighbors:", style='Dark.TLabel').grid(row=6, column=0, sticky=tk.W, pady=2, padx=(20,0))
        self.min_neighbors_scale = tk.Scale(noise_frame, from_=3, to=20, resolution=1, orient=tk.HORIZONTAL,
                                            bg='#2d2d30', fg='white', highlightthickness=0, length=120)
        self.min_neighbors_scale.set(10)  # 10 neighbors mặc định
        self.min_neighbors_scale.grid(row=6, column=1, padx=5, pady=2)
        self.neighbors_label = ttk.Label(noise_frame, text="10", style='Dark.TLabel', foreground='cyan')
        self.neighbors_label.grid(row=6, column=2, sticky=tk.W, pady=2)
        self.min_neighbors_scale.config(command=lambda v: self.neighbors_label.config(text=f"{int(float(v))}"))
        
        # Frame cấu hình nâng cao
        advanced_frame = ttk.LabelFrame(main_frame, text="🎯 Cấu hình Nâng cao (Độ chính xác)", padding="10", style='Dark.TLabelframe')
        advanced_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Bilateral filter
        self.bilateral_var = tk.BooleanVar(value=False)  # Tắt mặc định
        ttk.Checkbutton(advanced_frame, text="🔬 Edge-Preserving Filter (giữ cạnh sắc nét)", 
                       variable=self.bilateral_var, style='Dark.TCheckbutton').grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Confidence filter
        self.confidence_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="✅ Confidence Filtering (chỉ lấy điểm tin cậy)", 
                       variable=self.confidence_var, style='Dark.TCheckbutton').grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Depth range controls
        ttk.Label(advanced_frame, text="Khoảng cách MIN (m):", style='Dark.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.depth_min_scale = tk.Scale(advanced_frame, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                         bg='#2d2d30', fg='white', highlightthickness=0,
                                         command=self.update_depth_min, length=150)
        self.depth_min_scale.set(0.1)
        self.depth_min_scale.grid(row=2, column=1, padx=10, pady=5)
        self.depth_min_label = ttk.Label(advanced_frame, text="0.10m", style='Dark.TLabel', foreground='cyan')
        self.depth_min_label.grid(row=2, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(advanced_frame, text="Khoảng cách MAX (m):", style='Dark.TLabel').grid(row=3, column=0, sticky=tk.W, pady=5)
        self.depth_max_scale = tk.Scale(advanced_frame, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                                         bg='#2d2d30', fg='white', highlightthickness=0,
                                         command=self.update_depth_max, length=150)
        self.depth_max_scale.set(2.5)
        self.depth_max_scale.grid(row=3, column=1, padx=10, pady=5)
        self.depth_max_label = ttk.Label(advanced_frame, text="2.50m", style='Dark.TLabel', foreground='cyan')
        self.depth_max_label.grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # Confidence threshold
        ttk.Label(advanced_frame, text="Độ tin cậy tối thiểu:", style='Dark.TLabel').grid(row=4, column=0, sticky=tk.W, pady=5)
        self.confidence_scale = tk.Scale(advanced_frame, from_=0, to=3, resolution=1, orient=tk.HORIZONTAL,
                                          bg='#2d2d30', fg='white', highlightthickness=0,
                                          command=self.update_confidence, length=150)
        self.confidence_scale.set(2)
        self.confidence_scale.grid(row=4, column=1, padx=10, pady=5)
        self.confidence_label = ttk.Label(advanced_frame, text="Level 2 (Cao)", style='Dark.TLabel', foreground='cyan')
        self.confidence_label.grid(row=4, column=2, sticky=tk.W, pady=5)
        
        # Frame YOLOv8 Segmentation
        seg_frame = ttk.LabelFrame(main_frame, text="🤖 YOLOv8 Segmentation (Nhận diện & Mask)", padding="10", style='Dark.TLabelframe')
        seg_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.seg_var = tk.BooleanVar(value=True)
        seg_check = ttk.Checkbutton(seg_frame, text="✅ Segmentation: LUÔN BẬT (Auto nhận diện)", 
                       variable=self.seg_var, command=self.update_segmentation_option,
                       style='Dark.TCheckbutton')
        seg_check.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        seg_check.state(['disabled'])  # Disable checkbox
        
        # Confidence threshold cho YOLO
        ttk.Label(seg_frame, text="Confidence threshold:", style='Dark.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.seg_conf_scale = tk.Scale(seg_frame, from_=0.1, to=0.95, resolution=0.05, orient=tk.HORIZONTAL,
                                        bg='#2d2d30', fg='white', highlightthickness=0,
                                        command=self.update_seg_conf, length=150)
        self.seg_conf_scale.set(0.5)
        self.seg_conf_scale.grid(row=1, column=1, padx=10, pady=5)
        self.seg_conf_label = ttk.Label(seg_frame, text="0.50", style='Dark.TLabel', foreground='cyan')
        self.seg_conf_label.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        self.seg_status_label = ttk.Label(seg_frame, text="Model: " + ("Ready ✓" if self.yolo_model else "Not loaded ❌"), 
                                           font=('Arial', 9), foreground='lime' if self.yolo_model else 'red', style='Dark.TLabel')
        self.seg_status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Frame Plane Detection & Reorientation
        plane_frame = ttk.LabelFrame(main_frame, text="📐 Phát hiện mặt phẳng & Đặt gốc tọa độ", padding="10", style='Dark.TLabelframe')
        plane_frame.grid(row=9, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.plane_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plane_frame, text="🎯 Tự động đặt Z=0 tại mặt phẳng, XY tại tâm vật", 
                       variable=self.plane_detect_var,
                       style='Dark.TCheckbutton').grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.remove_background_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plane_frame, text="🧹 Loại bỏ nền (chỉ giữ vật thể)", 
                       variable=self.remove_background_var,
                       style='Dark.TCheckbutton').grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5, padx=20)
        
        self.plane_info_label = ttk.Label(plane_frame, text="Gốc tọa độ: Camera (chưa transform)", 
                                           font=('Arial', 9), foreground='yellow', style='Dark.TLabel')
        self.plane_info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Thông tin trạng thái
        status_frame = ttk.LabelFrame(main_frame, text="Trạng thái", padding="10", style='Dark.TLabelframe')
        status_frame.grid(row=10, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="🟢 Camera đã sẵn sàng", 
                                       font=('Arial', 10), style='Dark.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.distance_label = ttk.Label(status_frame, text="Khoảng cách tại tâm: -- m", 
                                        font=('Arial', 10), style='Dark.TLabel')
        self.distance_label.grid(row=1, column=0, sticky=tk.W)
        
        self.process_status_label = ttk.Label(status_frame, text="Point Cloud: Chưa xử lý", 
                                               font=('Arial', 10), foreground='orange', style='Dark.TLabel')
        self.process_status_label.grid(row=2, column=0, sticky=tk.W)
        
        self.roi_status_label = ttk.Label(status_frame, text="🎯 Vùng nhận diện: Toàn bộ (720x720)", 
                                           font=('Arial', 10), foreground='cyan', style='Dark.TLabel')
        self.roi_status_label.grid(row=3, column=0, sticky=tk.W)
    
    def update_frames_count(self, value):
        self.num_frames_avg = int(float(value))
        self.frames_label.config(text=f"{self.num_frames_avg} frames")
    
    def update_sampling(self, value):
        self.sampling_step = int(float(value))
        density = "Rất dày" if self.sampling_step <= 2 else "Dày" if self.sampling_step <= 4 else "Trung bình" if self.sampling_step <= 7 else "Thưa"
        self.sampling_label.config(text=f"Step={self.sampling_step} ({density})")
    
    def update_depth_min(self, value):
        self.depth_min = float(value)
        self.depth_min_label.config(text=f"{self.depth_min:.2f}m")
    
    def update_depth_max(self, value):
        self.depth_max = float(value)
        self.depth_max_label.config(text=f"{self.depth_max:.2f}m")
    
    def update_confidence(self, value):
        self.confidence_threshold = int(float(value))
        conf_text = ["Level 0 (Tất cả)", "Level 1 (Thấp)", "Level 2 (Cao)", "Level 3 (Rất cao)"][self.confidence_threshold]
        self.confidence_label.config(text=conf_text)
    
    def update_seg_conf(self, value):
        self.segmentation_conf = float(value)
        self.seg_conf_label.config(text=f"{self.segmentation_conf:.2f}")
    
    def update_segmentation_option(self):
        self.use_segmentation = self.seg_var.get()
        if self.use_segmentation:
            if self.yolo_model:
                self.seg_status_label.config(text="Segmentation: BẬT ✅", foreground='lime')
            else:
                self.seg_status_label.config(text="Lỗi: Model chưa load ❌", foreground='red')
                self.seg_var.set(False)
                self.use_segmentation = False
        else:
            self.seg_status_label.config(text="Segmentation: TắT", foreground='gray')
    
    def init_camera(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
            self.pipeline.start(config)
            
            # Align object
            self.align = rs.align(rs.stream.color)
            
            # Filters - TỐI ƯU CHO VẬT NHỎ (mầm lan)
            # KHÔNG dùng decimation để giữ full resolution
            self.decimation_filter = None  # Tắt để giữ chi tiết tối đa
            
            # Spatial filter - Cài đặt nhẹ nhàng cho vật nhỏ
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 1)  # Giảm từ 2→1
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.3)  # Giảm smoothing
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 10)  # Giữ chi tiết
            
            # Temporal filter - Nhẹ nhàng để không làm mờ chi tiết
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.2)  # Giảm từ 0.4→0.2
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 10)  # Giảm từ 20→10
            
            # Hole filling
            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 1)  # farthest from around
            
            self.is_running = True
            self.status_label.config(text="🟢 Camera đã sẵn sàng", foreground='green')
            
            # Bắt đầu threads
            self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
            self.update_thread.start()
            
            self.display_thread = threading.Thread(target=self.display_frame, daemon=True)
            self.display_thread.start()
            
        except Exception as e:
            self.status_label.config(text=f"🔴 Lỗi camera: {str(e)}", foreground='red')
            print(f"Lỗi khởi động camera: {e}")
    
    def _set_preview_active(self, active, update_checkbox=None):
        """
        Bật/tắt preview MỘT CÁCH ĐỒNG BỘ:
        - self.is_preview: cờ trạng thái (đọc ở nhiều nơi).
        - self.preview_event: threading.Event để thread preview NGỦ (event.wait())
          thay vì busy-loop (continue liên tục) khi bị tắt.
        - preview_var (checkbox GUI): mặc định chỉ tự cập nhật khi RESUME (active=True),
          giữ đúng hành vi cũ (tạm dừng theo chương trình không đổi trạng thái checkbox).
        """
        self.is_preview = active
        if active:
            self.preview_event.set()
        else:
            self.preview_event.clear()
        if update_checkbox is None:
            update_checkbox = active
        if update_checkbox:
            try:
                if hasattr(self, 'preview_var'):
                    self.preview_var.set(active)
            except Exception:
                pass

    def update_frame(self):
        while self.is_running:
            # 🔒 Ngủ (không busy-loop) khi preview bị tắt; wake dậy định kỳ
            # (timeout=0.5s) để vẫn kiểm tra được self.is_running và thoát sạch.
            if not self.preview_event.wait(timeout=0.5):
                continue
            if not self.is_running:
                break
            try:
                # 🔒 Giữ camera_lock CHỈ trong lúc lấy + align frame (nhả ngay sau
                # đó) để không chặn YOLO/point-cloud/GUI, đồng thời tránh 2 thread
                # (preview + Hybrid/Pass1-2/hand-eye calib) gọi RealSense API
                # đồng thời.
                with self.camera_lock:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Crop 720x720 - FULL DETAIL (không resize nữa)
                height, width = color_image.shape[:2]
                start_x = (width - 720) // 2
                color_720 = color_image[0:720, start_x:start_x+720].copy()
                
                # ==== CHẠY YOLO REAL-TIME trực tiếp trên 720x720 (với frame skipping) ====
                self.frame_count += 1
                should_run_yolo = (self.frame_count % self.yolo_skip_frames == 0)
                
                if self.yolo_model is not None and should_run_yolo:
                    # Xác định vùng detection (ROI hoặc toàn bộ 720x720)
                    if self.roi:
                        x1, y1, x2, y2 = self.roi
                        detection_region = color_720[y1:y2, x1:x2].copy()
                        region_offset = (x1, y1)
                    else:
                        detection_region = color_720.copy()
                        region_offset = (0, 0)
                    
                    # Inference trực tiếp trên 720x720 (chỉ khi region đủ lớn và là bội số của 32)
                    if detection_region.shape[0] > 32 and detection_region.shape[1] > 32:
                        region_size = detection_region.shape[0]  # Square, width = height
                        
                        results = self.yolo_model.predict(
                            detection_region,
                            conf=self.segmentation_conf,
                            verbose=False
                        )
                        
                        # Lưu detection results
                        self.detection_results = results
                        
                        # Tạo binary mask 720x720 (không cần resize)
                        self.current_binary_mask = np.zeros((720, 720), dtype=np.uint8)
                        
                        # Tạo mask riêng cho class 1 (để tính center)
                        self.class1_mask = np.zeros((720, 720), dtype=np.uint8)
                        
                        # 🆕 INSTANCE SEGMENTATION: Reset instance lists
                        self.instance_masks = []
                        self.instance_classes = []
                        self.instance_confidences = []
                        
                        # Vẽ lên color_720 - DÙNG THUẬT TOÁN TỪ TEST_SEGMENT.PY
                        if results and len(results) > 0 and results[0].masks is not None:
                            masks = results[0].masks.data.cpu().numpy()
                            boxes = results[0].boxes.data.cpu().numpy()
                            
                            # Màu cố định cho từng instance (giống test_segment.py)
                            instance_colors = [
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
                            
                            # TẠO OVERLAY (giống test_segment.py)
                            overlay = color_720.copy()
                            
                            # Lấy kích thước của ảnh xử lý
                            proc_h, proc_w = detection_region.shape[:2]
                            
                            # Lưu thông tin label để vẽ SAU (như test_segment.py)
                            label_info_list = []
                            
                            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                                # Lấy thông tin
                                x1, y1, x2, y2, conf_val, cls = box
                                class_id = int(cls)
                                
                                # Resize mask về kích thước ảnh đã xử lý (detection_region)
                                mask_resized = cv2.resize(mask, (proc_w, proc_h))
                                mask_bool = mask_resized > 0.5
                                
                                # Tạo mask full size (720x720)
                                full_mask = np.zeros((720, 720), dtype=bool)
                                
                                # Đặt mask vào đúng vị trí (có tính offset)
                                x_off, y_off = region_offset
                                full_mask[y_off:y_off+proc_h, x_off:x_off+proc_w] = mask_bool
                                
                                # Thêm vào binary mask (trắng = 255)
                                self.current_binary_mask[full_mask] = 255
                                
                                # Lưu mask class 1 riêng (để tính center và vẽ dấu chấm)
                                if class_id == 1:
                                    self.class1_mask[full_mask] = 255
                                
                                # 🆕 LƯU INSTANCE MASK (720x720)
                                instance_mask_full = np.zeros((720, 720), dtype=np.uint8)
                                instance_mask_full[full_mask] = 255
                                self.instance_masks.append(instance_mask_full)
                                self.instance_classes.append(class_id)
                                self.instance_confidences.append(float(conf_val))
                                
                                # Chọn màu cho instance này
                                color = instance_colors[idx % len(instance_colors)]
                                
                                # VẼ MASK LÊN OVERLAY (giống test_segment.py)
                                overlay[full_mask] = overlay[full_mask] * 0.5 + np.array(color) * 0.5
                                
                            
                            # ✅ LƯU MASK 720x720 ĐỂ DÙNG CHO PASS 1 & 2
                            self.last_mask_720 = self.current_binary_mask.copy()
                            self.last_mask = self.current_binary_mask.copy()
                            
                            # Lưu ảnh RGB sạch trước khi vẽ mask overlay
                            self.last_clean_rgb_bgr = color_720.copy()
                            
                            # BLEND OVERLAY VỚI ẢNH GỐC (có thể bật/tắt qua checkbox)
                            if self.show_mask_overlay:
                                color_720 = cv2.addWeighted(color_720, 0.6, overlay, 0.4, 0)
                            
                            # VẼ LABEL SAU KHI ĐÃ BLEND (giống test_segment.py)
                            for (cx, cy, lbl, col) in label_info_list:
                                cv2.putText(color_720, lbl, 
                                          (cx-30, cy), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                          col, 2)
                            
                            # 🤖 TÍNH TOÁN CUTTING POINT (nếu bật)
                            if self.enable_cutting_point:
                                # Tách mask class 0 và class 1
                                mask_class0 = np.zeros((720, 720), dtype=np.uint8)
                                mask_class1 = np.zeros((720, 720), dtype=np.uint8)
                                
                                for idx, (mask, box) in enumerate(zip(masks, boxes)):
                                    cls = int(box[5])
                                    mask_resized = cv2.resize(mask, (proc_w, proc_h))
                                    mask_bool = mask_resized > 0.5
                                    mask_binary = mask_bool.astype(np.uint8)
                                    
                                    # Tạo full mask
                                    full_mask_cut = np.zeros((720, 720), dtype=bool)
                                    x_off, y_off = region_offset
                                    full_mask_cut[y_off:y_off+proc_h, x_off:x_off+proc_w] = mask_bool
                                    
                                    if cls == 0:
                                        mask_class0[full_mask_cut] = 255
                                    elif cls == 1:
                                        mask_class1[full_mask_cut] = 255
                                
                                # Tính cutting points (nhiều nhánh)
                                if np.sum(mask_class0) > 0 and np.sum(mask_class1) > 0:
                                    cutting_points_list = self.calculate_cutting_points(mask_class0, mask_class1)
                                    
                                    if cutting_points_list:
                                        self.cutting_data = cutting_points_list  # Lưu list
                                        
                                        # Vẽ tất cả cutting points
                                        for cutting_data in cutting_points_list:
                                            root_c = cutting_data['root_center']
                                            cut_p = cutting_data['cut_point']
                                            cut_start = cutting_data['cut_line_start']
                                            cut_end = cutting_data['cut_line_end']
                                            
                                            # Vẽ đường nối từ gốc đến điểm cắt (màu vàng, mảnh)
                                            cv2.line(color_720, root_c, cut_p, (0, 255, 255), 1)
                                            
                                            # Vẽ đường cắt (màu đỏ, mảnh)
                                            cv2.line(color_720, cut_start, cut_end, (0, 0, 255), 1)
                                            
                                            # Vẽ điểm cắt (màu đỏ, nhỏ)
                                            cv2.circle(color_720, cut_p, 3, (0, 0, 255), 1)
                                        
                                        # Vẽ tâm gốc 1 lần (màu xanh lá, nhỏ)
                                        if cutting_points_list:
                                            root_c = cutting_points_list[0]['root_center']
                                            cv2.circle(color_720, root_c, 2, (0, 255, 0), 1)
                                    else:
                                        self.cutting_data = None
                                else:
                                    self.cutting_data = None
                            
                            # VẼ DẤU CHẤM TẠI TÂM CLASS 1 (nếu được bật)
                            if self.show_center_dot and np.sum(self.class1_mask) > 0:
                                M = cv2.moments(self.class1_mask)
                                if M['m00'] > 0:
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    
                                    # Lưu tọa độ tâm class 1 (720x720) để dùng cho trục tọa độ 3D
                                    self.class1_center_2d = (cx, cy)
                                    
                                    # Vẽ dấu chấm màu đỏ tại tâm class 1 (nhỏ)
                                    cv2.circle(color_720, (cx, cy), 4, (0, 0, 255), -1)  # Filled red circle
                                    cv2.circle(color_720, (cx, cy), 5, (255, 255, 255), 1)  # White outline mảnh
                            
                            # Lưu overlay đã xử lý
                            self.last_detection_overlay = color_720.copy()
                        else:
                            # Không có detection mới, giữ overlay cũ nếu có
                            if self.last_detection_overlay is not None:
                                color_720 = self.last_detection_overlay.copy()
                elif self.last_detection_overlay is not None:
                    # Không chạy YOLO frame này, giữ overlay cũ
                    color_720 = self.last_detection_overlay.copy()
                
                # Lấy khoảng cách tại tâm
                center_x_original = width // 2
                center_y_original = 360
                distance = depth_frame.as_depth_frame().get_distance(center_x_original, center_y_original)
                
                frame_data = {
                    'rgb': color_720,  # 720x720 - FULL DETAIL
                    'distance': distance
                }
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame_data)
                
            except Exception as e:
                print(f"Lỗi cập nhật frame: {e}")
                import traceback
                traceback.print_exc()
            
            # Sleep ngắn để giảm CPU
            import time
            time.sleep(0.01)  # 10ms
    
    def display_frame(self):
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get()
                    
                    display_image = frame_data['rgb']  # 720x720 full
                    
                    # Xác định vùng hiển thị (crop nếu có ROI)
                    if self.roi and not self.is_setting_area:
                        x1, y1, x2, y2 = self.roi
                        display_rgb_crop = display_image[y1:y2, x1:x2].copy()
                        display_mask_crop = self.current_binary_mask[y1:y2, x1:x2].copy() if hasattr(self, 'current_binary_mask') else None
                        roi_text = f"ROI: ({x1},{y1})-({x2},{y2}) | {x2-x1}x{y2-y1}px"
                    else:
                        display_rgb_crop = display_image
                        display_mask_crop = self.current_binary_mask if hasattr(self, 'current_binary_mask') else None
                        roi_text = "ROI: Full view (720x720)"
                    
                    # Resize crop để vừa canvas 720x720 (nếu crop nhỏ hơn)
                    if display_rgb_crop.shape[0] < 720 or display_rgb_crop.shape[1] < 720:
                        display_rgb_crop = cv2.resize(display_rgb_crop, (720, 720), interpolation=cv2.INTER_LINEAR)
                        if display_mask_crop is not None:
                            # SỬA: Dùng INTER_LINEAR thay vì INTER_NEAREST để tránh ô vuông
                            display_mask_crop = cv2.resize(display_mask_crop, (720, 720), interpolation=cv2.INTER_LINEAR)
                    
                    # Display vào RGB window
                    if self.rgb_window is not None:
                        try:
                            if self.rgb_window.winfo_exists():
                                display_image_rgb = cv2.cvtColor(display_rgb_crop, cv2.COLOR_BGR2RGB)
                                
                                # � Overlay path visualization if available
                                if self.show_skeleton_tips and hasattr(self, 'last_path_vis_per_instance'):
                                    for instance_idx, path_vis in self.last_path_vis_per_instance.items():
                                        if path_vis is not None:
                                            try:
                                                # Transform path_vis to match display coordinates
                                                if self.roi and not self.is_setting_area:
                                                    x1, y1, x2, y2 = self.roi
                                                    roi_w, roi_h = x2 - x1, y2 - y1
                                                    
                                                    # Crop path_vis to ROI
                                                    path_vis_cropped = path_vis[y1:y2, x1:x2]
                                                    
                                                    # Resize to 720x720
                                                    path_vis_resized = cv2.resize(path_vis_cropped, (720, 720), interpolation=cv2.INTER_NEAREST)
                                                else:
                                                    path_vis_resized = path_vis.copy()
                                                
                                                # Ensure same shape
                                                if path_vis_resized.shape[:2] != display_image_rgb.shape[:2]:
                                                    path_vis_resized = cv2.resize(path_vis_resized, 
                                                                                  (display_image_rgb.shape[1], display_image_rgb.shape[0]),
                                                                                  interpolation=cv2.INTER_NEAREST)
                                                
                                                # Convert to RGB
                                                path_vis_rgb = cv2.cvtColor(path_vis_resized, cv2.COLOR_BGR2RGB)
                                                
                                                # Blend with display image (50% opacity)
                                                # Only blend non-black pixels from path_vis
                                                mask = np.any(path_vis_rgb != [0, 0, 0], axis=2)
                                                
                                                if np.any(mask):
                                                    # Simple alpha blending
                                                    alpha = 0.5
                                                    display_image_rgb[mask] = (
                                                        display_image_rgb[mask] * (1 - alpha) + 
                                                        path_vis_rgb[mask] * alpha
                                                    ).astype(np.uint8)
                                            except Exception as e:
                                                print(f"⚠️ Error overlaying path_vis for instance {instance_idx}: {e}")
                                
                                # �🆕 Vẽ skeleton tips nếu được bật
                                if self.show_skeleton_tips and hasattr(self, 'last_tips_data_per_instance'):
                                    for instance_idx, tips_data in self.last_tips_data_per_instance.items():
                                        if tips_data is not None:
                                            # 🎯 Xử lý base coordinates dựa vào mode (checkbox)
                                            use_3d_mode = self.base_mode_3d
                                            
                                            if use_3d_mode and tips_data.get('base_3d_original') is not None:
                                                # 3D MODE: Map 3D -> 2D
                                                # Cần dùng points_pre_voxel và pixel_mapping_pre_voxel
                                                if hasattr(self, 'points_pre_voxel') and hasattr(self, 'point_to_pixel_mapping_pre_voxel'):
                                                    base_3d = tips_data['base_3d_original']
                                                    base_2d_mapped = self.map_3d_point_to_2d(
                                                        base_3d, 
                                                        self.points_pre_voxel, 
                                                        self.point_to_pixel_mapping_pre_voxel
                                                    )
                                                    if base_2d_mapped is not None:
                                                        base_u, base_v = base_2d_mapped
                                                    else:
                                                        # Fallback to original 2D
                                                        base_u, base_v = tips_data['base_2d']
                                                else:
                                                    # Fallback to original 2D
                                                    base_u, base_v = tips_data['base_2d']
                                            else:
                                                # 2D MODE: Dùng trực tiếp base_2d
                                                base_u, base_v = tips_data['base_2d']
                                            
                                            # Transform coordinates từ 720x720 gốc -> display_rgb_crop space
                                            if self.roi and not self.is_setting_area:
                                                x1, y1, x2, y2 = self.roi
                                                roi_w, roi_h = x2 - x1, y2 - y1
                                                
                                                # Check if base in ROI
                                                if not (x1 <= base_u < x2 and y1 <= base_v < y2):
                                                    continue  # Skip this instance if base outside ROI
                                                
                                                # Transform: subtract ROI offset, then scale to 720x720
                                                base_u_crop = int((base_u - x1) * 720.0 / roi_w)
                                                base_v_crop = int((base_v - y1) * 720.0 / roi_h)
                                            else:
                                                # Full view: no transform needed
                                                base_u_crop, base_v_crop = tips_data['base_2d']
                                            
                                            # Vẽ base (chấm xanh dương lớn) với label mode
                                            base_label = 'BASE(3D)' if use_3d_mode else 'BASE(2D)'
                                            cv2.circle(display_image_rgb, (base_u_crop, base_v_crop), 5, (0, 255, 255), -1)
                                            
                                            cv2.putText(display_image_rgb, base_label, (base_u_crop + 8, base_v_crop - 8),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
                                            
                                            # 🔶 Vẽ main tip (junction hoặc farthest endpoint)
                                            main_u, main_v, main_geo = tips_data['main_tip_2d']
                                            
                                            # Transform main tip
                                            if self.roi and not self.is_setting_area:
                                                x1, y1, x2, y2 = self.roi
                                                roi_w, roi_h = x2 - x1, y2 - y1
                                                
                                                if x1 <= main_u < x2 and y1 <= main_v < y2:
                                                    main_u_crop = int((main_u - x1) * 720.0 / roi_w)
                                                    main_v_crop = int((main_v - y1) * 720.0 / roi_h)
                                                else:
                                                    main_u_crop, main_v_crop = None, None
                                            else:
                                                main_u_crop, main_v_crop = main_u, main_v
                                            
                                            if main_u_crop is not None:
                                                is_junction = tips_data.get('is_junction_tip', False)
                                                
                                                if is_junction:
                                                    # Junction: Yellow/Orange diamond shape
                                                    main_color = (0, 165, 255)  # Orange (BGR)
                                                    main_label = f'JUNCTION ({main_geo}px)'
                                                    # Draw diamond (4 lines forming diamond)
                                                    pts = np.array([
                                                        [main_u_crop, main_v_crop - 10],
                                                        [main_u_crop + 8, main_v_crop],
                                                        [main_u_crop, main_v_crop + 10],
                                                        [main_u_crop - 8, main_v_crop]
                                                    ], np.int32)
                                                    cv2.fillPoly(display_image_rgb, [pts], main_color)
                                                    cv2.polylines(display_image_rgb, [pts], True, (0, 140, 220), 1)
                                                else:
                                                    # Farthest endpoint: Red circle
                                                    main_color = (255, 0, 0)  # Red
                                                    main_label = f'MAIN ({main_geo}px)'
                                                    cv2.circle(display_image_rgb, (main_u_crop, main_v_crop), 7, main_color, -1)
                                                    cv2.circle(display_image_rgb, (main_u_crop, main_v_crop), 9, main_color, 2)
                                                
                                                cv2.putText(display_image_rgb, main_label, (main_u_crop + 10, main_v_crop - 8),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.28, main_color, 1)
                                            
                                            # 🟢 Vẽ leaf tips (tất cả endpoints)
                                            for tip_idx, (tip_u, tip_v, geo_dist) in enumerate(tips_data['tips_2d']):
                                                # Transform tip coordinates
                                                if self.roi and not self.is_setting_area:
                                                    x1, y1, x2, y2 = self.roi
                                                    roi_w, roi_h = x2 - x1, y2 - y1
                                                    
                                                    # Check if tip in ROI
                                                    if not (x1 <= tip_u < x2 and y1 <= tip_v < y2):
                                                        continue  # Skip this tip if outside ROI
                                                    
                                                    tip_u_crop = int((tip_u - x1) * 720.0 / roi_w)
                                                    tip_v_crop = int((tip_v - y1) * 720.0 / roi_h)
                                                else:
                                                    tip_u_crop, tip_v_crop = tip_u, tip_v
                                                
                                                # All endpoints are green (leaves)
                                                color = (0, 255, 0)  # Green
                                                radius = 5
                                                label = f'LEAF-{tip_idx} ({geo_dist}px)'
                                                
                                                cv2.circle(display_image_rgb, (tip_u_crop, tip_v_crop), radius, color, -1)
                                                cv2.circle(display_image_rgb, (tip_u_crop, tip_v_crop), radius + 2, color, 2)
                                                cv2.putText(display_image_rgb, label, (tip_u_crop + 10, tip_v_crop - 10),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
                                
                                img_pil = Image.fromarray(display_image_rgb)
                                img_tk = ImageTk.PhotoImage(image=img_pil)
                                
                                # Cache ảnh preview để xuất file
                                self.last_preview_rgb_np = display_image_rgb.copy()
                                
                                self.rgb_canvas.create_image(360, 360, image=img_tk)
                                self.rgb_canvas.image = img_tk
                                self.rgb_info_label.config(text=roi_text)
                        except:
                            self.rgb_window = None
                    
                    # Display vào Mask window
                    if self.mask_window is not None:
                        try:
                            if self.mask_window.winfo_exists() and display_mask_crop is not None:
                                mask_rgb = cv2.cvtColor(display_mask_crop, cv2.COLOR_GRAY2RGB)
                                mask_pil = Image.fromarray(mask_rgb)
                                mask_tk = ImageTk.PhotoImage(image=mask_pil)
                                
                                self.mask_canvas.create_image(360, 360, image=mask_tk)
                                self.mask_canvas.image = mask_tk
                                self.mask_info_label.config(text=roi_text)
                        except:
                            self.mask_window = None
                    
                    # Display vào ROI setting window (full view)
                    if self.roi_setting_window is not None:
                        try:
                            if self.roi_setting_window.winfo_exists():
                                display_full_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                                full_pil = Image.fromarray(display_full_rgb)
                                full_tk = ImageTk.PhotoImage(image=full_pil)
                                
                                self.roi_canvas.create_image(360, 360, image=full_tk)
                                self.roi_canvas.image = full_tk
                        except:
                            self.roi_setting_window = None
                    
                    self.distance_label.config(text=f"Khoảng cách tại tâm: {frame_data['distance']:.3f} m")
                    
                    # ⬛ HIỂN THỊ BINARY MASK WINDOW (nếu được bật)
                    if self.show_binary and hasattr(self, 'current_binary_mask') and self.current_binary_mask is not None:
                        try:
                            binary_display = self.current_binary_mask.copy()
                            # Crop theo ROI nếu có
                            if self.roi and not self.is_setting_area:
                                x1, y1, x2, y2 = self.roi
                                binary_display = binary_display[y1:y2, x1:x2]
                                # Resize về 720x720
                                if binary_display.shape[0] > 0 and binary_display.shape[1] > 0:
                                    if binary_display.shape[0] < 720 or binary_display.shape[1] < 720:
                                        binary_display = cv2.resize(binary_display, (720, 720), interpolation=cv2.INTER_NEAREST)
                            
                            if binary_display.shape[0] > 0 and binary_display.shape[1] > 0:
                                cv2.imshow('Binary Mask', binary_display)
                                cv2.waitKey(1)  # Cần waitKey để cập nhật window
                        except Exception as e:
                            print(f"⚠️ Error displaying binary mask: {e}")
                    elif not self.show_binary:
                        # Đóng window nếu tắt
                        try:
                            cv2.destroyWindow('Binary Mask')
                        except:
                            pass
                    
            except Exception as e:
                print(f"Lỗi hiển thị frame: {e}")
                import traceback
                traceback.print_exc()
            
            # Sleep ngắn
            import time
            time.sleep(0.01)  # 10ms
    
    def toggle_preview(self):
        self._set_preview_active(self.preview_var.get())
        if self.is_preview:
            self.status_label.config(text="🟢 Preview: BẬT", foreground='green')
        else:
            self.status_label.config(text="⚪ Preview: TẮT", foreground='gray')
    
    def toggle_filters(self):
        """Bật/tắt depth filters (spatial, temporal, hole_filling)"""
        self.use_filters = self.filters_var.get()
        status = "BẬT" if self.use_filters else "TẮT"
        self.status_label.config(text=f"🔧 Depth Filters: {status}", foreground='cyan')
        print(f"\n🔧 Depth Filters: {status}")
        print("   (Spatial + Temporal + Hole Filling)")
    
    def toggle_boxes(self):
        """Bật/tắt hiển thị bounding boxes và class labels"""
        self.show_boxes = self.boxes_var.get()
        status = "BẬT" if self.show_boxes else "TẮT"
        self.status_label.config(text=f"📦 Bounding Boxes: {status}", foreground='cyan')
        print(f"\n📦 Bounding Boxes: {status}")
    
    def toggle_binary_window(self):
        """Bật/tắt cửa sổ binary mask"""
        self.show_binary = self.binary_var.get()
        status = "BẬT" if self.show_binary else "TẮT"
        self.status_label.config(text=f"⬛ Binary Mask: {status}", foreground='cyan')
        print(f"\n⬛ Binary Mask Window: {status}")
        if not self.show_binary:
            # Đóng cửa sổ binary mask nếu tắt
            cv2.destroyWindow('Binary Mask')
    
    def toggle_center_dot(self):
        """Bật/tắt hiển thị dấu chấm tâm class 1"""
        self.show_center_dot = self.center_var.get()
        status = "BẬT" if self.show_center_dot else "TẮT"
        self.status_label.config(text=f"🎯 Center Dot: {status}", foreground='cyan')
        print(f"\n🎯 Center Dot Class 1: {status}")
    
    def toggle_skeleton_tips(self):
        """Bật/tắt hiển thị skeleton base & tips"""
        self.show_skeleton_tips = self.skeleton_tips_var.get()
        status = "BẬT" if self.show_skeleton_tips else "TẮT"
        self.status_label.config(text=f"🌿 Skeleton Tips: {status}", foreground='cyan')
        print(f"\n🌿 Skeleton Tips Display: {status}")

    def toggle_picking_frame(self):
        """Bật/tắt tạo trục tọa độ (picking frame) tại vị trí thân mầm"""
        self.show_picking_frame = self.picking_frame_var.get()
        status = "BẬT" if self.show_picking_frame else "TẮT"
        self.status_label.config(text=f"🤖 Picking Frame: {status}", foreground='cyan')
        print(f"\n🤖 Picking Frame (trục tọa độ thân): {status}")

    def toggle_mask_overlay(self):
        """Bật/tắt hiển thị màu mask overlay lên ảnh preview"""
        self.show_mask_overlay = self.mask_overlay_var.get()
        status = "BẬT" if self.show_mask_overlay else "TẮT"
        self.status_label.config(text=f"🎨 Mask Overlay: {status}", foreground='cyan')
        print(f"\n🎨 Mask Overlay: {status}")
        if not self.show_mask_overlay:
            # Reset overlay cache để frame tiếp theo hiển thị ảnh gốc không có mask
            self.last_detection_overlay = None

    def toggle_base_mode(self):
        """Chuyển giữa cách hiển thị BASE: 2D (từ skeleton) hay 3D (map từ point cloud)"""
        self.base_mode_3d = self.base_3d_var.get()
        mode = "BASE(3D)" if self.base_mode_3d else "BASE(2D)"
        self.status_label.config(text=f"🗺️ Chế độ base: {mode}", foreground='cyan')
        print(f"\n🗺️ Base mode: {mode}")

    def save_preview_image(self):
        """Xuất ảnh preview hiện tại (có skeleton tips, base, bbox ...) ra Output_image."""
        try:
            if self.last_preview_rgb_np is None:
                self.status_label.config(
                    text="⚠️ Chưa có ảnh preview! Hãy mở cửa sổ Preview RGB trước.",
                    foreground='red')
                return

            output_dir = self._ensure_output_image_dir()
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"preview_{ts}.png")

            # last_preview_rgb_np là RGB, chuyển về BGR để cv2 lưu đúng màu
            bgr = cv2.cvtColor(self.last_preview_rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)

            msg = f"💾 Đã xuất preview: {os.path.basename(path)}"
            self.status_label.config(text=msg, foreground='lime')
            print(f"\n{msg}")
            print(f"   📁 {path}  |  {bgr.shape[1]}×{bgr.shape[0]} px")

        except Exception as e:
            msg = f"❌ Lỗi xuất preview: {e}"
            self.status_label.config(text=msg, foreground='red')
            print(f"\n{msg}")
            import traceback; traceback.print_exc()

    # ────────────────── Lưu ảnh mầm ──────────────────

    def _ensure_output_image_dir(self):
        """Tạo thư mục Output_image nếu chưa tồn tại."""
        output_dir = "Output_image"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Đã tạo thư mục: {output_dir}")
        return output_dir

    def _check_ready(self):
        """
        Kiểm tra điều kiện trước khi lưu.
        Trả về (clean_bgr, timestamp) hoặc None nếu chưa sẵn sàng.
        """
        if self.last_clean_rgb_bgr is None:
            self.status_label.config(
                text="⚠️ Chưa có ảnh RGB! Hãy bật Preview.", foreground='red')
            return None, None
        if len(self.instance_masks) == 0:
            self.status_label.config(
                text="⚠️ Chưa phát hiện mầm nào!", foreground='red')
            return None, None
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.last_clean_rgb_bgr.copy(), ts

    def _make_overlay_bgr(self, clean_bgr):
        """
        Tái tạo ảnh overlay (màu mask) giống như khi YOLO chạy.
        Trả về ảnh BGR với các mask đã blend màu.
        """
        instance_colors = [
            (255, 0,   0),   # Xanh dương
            (0, 255,   0),   # Xanh lá
            (0,   0, 255),   # Đỏ
            (255, 255,  0),  # Cyan
            (255,  0, 255),  # Magenta
            (0, 255, 255),   # Vàng
            (128,  0, 128),  # Tím
            (255, 165,  0),  # Cam
        ]
        overlay = clean_bgr.copy()
        for idx, (mask, cls) in enumerate(zip(self.instance_masks, self.instance_classes)):
            color = instance_colors[idx % len(instance_colors)]
            mask_bool = mask > 0
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 +
                                  np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
        return cv2.addWeighted(clean_bgr, 0.6, overlay, 0.4, 0)

    def _crop_mask(self, img_bgr, mask_bool, pad=12):
        """
        Áp mask lên ảnh (nền trắng) rồi crop theo bounding-box của mask.
        Trả về ảnh đã crop hoặc None nếu mask rỗng.
        """
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(img_bgr.shape[1] - 1, int(xs.max()) + pad)
        y2 = min(img_bgr.shape[0] - 1, int(ys.max()) + pad)
        result = np.full_like(img_bgr, 255)
        result[mask_bool] = img_bgr[mask_bool]
        return result[y1:y2+1, x1:x2+1]

    def save_all_mam(self):
        """
        Lưu mầm hoàn chỉnh (gồm gốc lẫn các nhánh, tất cả instance).
        Xuất 2 file:
          - mam_clean_TIMESTAMP.png   : ảnh RGB gốc, không có overlay
          - mam_overlay_TIMESTAMP.png : ảnh RGB + mask overlay màu
        Cả hai đều được crop sát union của tất cả instance mask.
        """
        try:
            clean_bgr, ts = self._check_ready()
            if clean_bgr is None:
                return

            output_dir = self._ensure_output_image_dir()

            # Union tất cả instance masks
            union_bool = np.zeros((720, 720), dtype=bool)
            for mask in self.instance_masks:
                union_bool |= (mask > 0)

            # Ảnh 1: clean (không overlay)
            cropped_clean = self._crop_mask(clean_bgr, union_bool)
            # Ảnh 2: overlay màu
            overlay_bgr = self._make_overlay_bgr(clean_bgr)
            cropped_overlay = self._crop_mask(overlay_bgr, union_bool)

            saved = []
            if cropped_clean is not None:
                path = os.path.join(output_dir, f"mam_clean_{ts}.png")
                cv2.imwrite(path, cropped_clean)
                saved.append(os.path.basename(path))
                print(f"   🖼️  mam_clean   : {path}  ({cropped_clean.shape[1]}×{cropped_clean.shape[0]} px)")

            if cropped_overlay is not None:
                path = os.path.join(output_dir, f"mam_overlay_{ts}.png")
                cv2.imwrite(path, cropped_overlay)
                saved.append(os.path.basename(path))
                print(f"   🖼️  mam_overlay : {path}  ({cropped_overlay.shape[1]}×{cropped_overlay.shape[0]} px)")

            msg = f"💾 Đã lưu cả mầm: {', '.join(saved)}"
            self.status_label.config(text=msg, foreground='lime')
            print(f"\n{msg}")

        except Exception as e:
            msg = f"❌ Lỗi lưu cả mầm: {e}"
            self.status_label.config(text=msg, foreground='red')
            print(f"\n{msg}")
            import traceback; traceback.print_exc()

    def save_parts_mam(self):
        """
        Lưu từng phần của mầm (mỗi instance riêng biệt).
        Tên file theo class:
          class=0 (thân/nhánh) → nhanh{n}_TIMESTAMP.png
          class=1 (gốc)       → goc_TIMESTAMP.png
        Mỗi file: nền trắng, chỉ hiện phần thuộc instance đó, crop sát mask.
        """
        try:
            clean_bgr, ts = self._check_ready()
            if clean_bgr is None:
                return

            output_dir = self._ensure_output_image_dir()

            saved = []
            nhanh_count = 0

            for idx, (mask, cls) in enumerate(zip(self.instance_masks, self.instance_classes)):
                mask_bool = mask > 0
                if cls == 0:
                    nhanh_count += 1
                    label = f"nhanh{nhanh_count}"
                else:
                    label = "goc"

                cropped = self._crop_mask(clean_bgr, mask_bool)
                if cropped is None:
                    print(f"   ⚠️ Instance {idx+1} ({label}): mask rỗng, bỏ qua")
                    continue

                path = os.path.join(output_dir, f"{label}_{ts}.png")
                cv2.imwrite(path, cropped)
                saved.append(os.path.basename(path))
                print(f"   🖼️  {label:10s}: {path}  ({cropped.shape[1]}×{cropped.shape[0]} px)")

            if saved:
                msg = f"💾 Đã lưu {len(saved)} phần: {', '.join(saved)}"
            else:
                msg = "⚠️ Không có phần nào được lưu!"
            self.status_label.config(text=msg, foreground='lime' if saved else 'red')
            print(f"\n{msg}")

        except Exception as e:
            msg = f"❌ Lỗi lưu từng phần: {e}"
            self.status_label.config(text=msg, foreground='red')
            print(f"\n{msg}")
            import traceback; traceback.print_exc()

    def open_rgb_window(self):
        """Mở cửa sổ preview RGB"""
        if self.rgb_window is None or not tk.Toplevel.winfo_exists(self.rgb_window):
            self.rgb_window = tk.Toplevel(self.root)
            self.rgb_window.title("📺 RGB Preview (Cropped)")
            self.rgb_window.configure(bg='#2d2d30')
            
            # Canvas cho RGB
            self.rgb_canvas = tk.Canvas(self.rgb_window, width=720, height=720, bg='#2d2d30', highlightthickness=0)
            self.rgb_canvas.pack(padx=10, pady=10)
            
            # Bottom bar: label + export button
            bottom_frame = tk.Frame(self.rgb_window, bg='#2d2d30')
            bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
            
            self.rgb_info_label = tk.Label(bottom_frame, text="ROI: Full view (720x720)",
                                           bg='#2d2d30', fg='cyan', font=('Arial', 10))
            self.rgb_info_label.pack(side=tk.LEFT, padx=5)
            
            ttk.Button(bottom_frame, text="💾 Xuất ảnh",
                       command=self.save_preview_image).pack(side=tk.RIGHT, padx=5)
    
    def open_mask_window(self):
        """Mở cửa sổ preview Binary Mask"""
        if self.mask_window is None or not tk.Toplevel.winfo_exists(self.mask_window):
            self.mask_window = tk.Toplevel(self.root)
            self.mask_window.title("🎭 Binary Mask Preview (Cropped)")
            self.mask_window.configure(bg='#1e1e1e')
            
            # Canvas cho mask
            self.mask_canvas = tk.Canvas(self.mask_window, width=720, height=720, bg='#1e1e1e', highlightthickness=0)
            self.mask_canvas.pack(padx=10, pady=10)
            
            # Label thông tin
            self.mask_info_label = tk.Label(self.mask_window, text="ROI: Full view (720x720)", 
                                            bg='#1e1e1e', fg='cyan', font=('Arial', 10))
            self.mask_info_label.pack(pady=5)
    
    def calculate_cutting_points(self, mask_class0, mask_class1):
        """Tính toán TẤT CẢ điểm cắt (nhiều nhánh) và hướng cắt cho robot
        
        Args:
            mask_class0: Binary mask của class 0 (thân)
            mask_class1: Binary mask của class 1 (gốc)
        
        Returns:
            list of dict: [{
                'root_center': (x, y),  # Tâm gốc
                'cut_point': (x, y),    # Điểm cắt trên thân
                'cut_angle': float,     # Góc cắt (độ)
                'cut_line_start': (x, y),
                'cut_line_end': (x, y),
                'distance': float       # Khoảng cách từ gốc đến điểm cắt
            }, ...]
        """
        try:
            # 1. Tìm tâm gốc (class 1)
            M1 = cv2.moments(mask_class1)
            if M1['m00'] == 0:
                return []
            
            root_cx = int(M1['m10'] / M1['m00'])
            root_cy = int(M1['m01'] / M1['m00'])
            
            # 2. Phân tích thân (class 0) thành các connected components (nhánh riêng)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_class0, connectivity=8)
            
            cutting_points = []
            
            # Duyệt qua từng nhánh (bỏ qua label 0 = background)
            for label_id in range(1, num_labels):
                # Lấy mask của nhánh này
                branch_mask = (labels == label_id).astype(np.uint8)
                
                # Bỏ qua nhánh quá nhỏ (< 50 pixels)
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < 50:
                    continue
                
                # 3. Tìm điểm trên nhánh này gần gốc nhất
                branch_points = np.argwhere(branch_mask > 0)  # [[y, x], [y, x], ...]
                
                if len(branch_points) == 0:
                    continue
                
                # Tính khoảng cách từ mỗi điểm đến tâm gốc
                distances = np.sqrt(
                    (branch_points[:, 1] - root_cx) ** 2 + 
                    (branch_points[:, 0] - root_cy) ** 2
                )
                
                # Tìm điểm gần nhất
                min_idx = np.argmin(distances)
                nearest_point = branch_points[min_idx]
                cut_point_x = int(nearest_point[1])
                cut_point_y = int(nearest_point[0])
                min_distance = distances[min_idx]
                
                # 4. Tính vector từ gốc đến điểm cắt
                dx = cut_point_x - root_cx
                dy = cut_point_y - root_cy
                
                # 5. Tính góc của đường nối (radian)
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                # 6. Đường cắt vuông góc với đường nối
                cut_angle_rad = angle_rad + np.pi / 2
                cut_angle_deg = angle_deg + 90
                
                # 7. Tính 2 điểm đầu cuối của đường cắt (dài 50 pixels mỗi bên)
                cut_length = 50
                cut_line_start_x = int(cut_point_x - cut_length * np.cos(cut_angle_rad))
                cut_line_start_y = int(cut_point_y - cut_length * np.sin(cut_angle_rad))
                cut_line_end_x = int(cut_point_x + cut_length * np.cos(cut_angle_rad))
                cut_line_end_y = int(cut_point_y + cut_length * np.sin(cut_angle_rad))
                
                cutting_points.append({
                    'root_center': (root_cx, root_cy),
                    'cut_point': (cut_point_x, cut_point_y),
                    'cut_angle': cut_angle_deg,
                    'cut_line_start': (cut_line_start_x, cut_line_start_y),
                    'cut_line_end': (cut_line_end_x, cut_line_end_y),
                    'distance': float(min_distance),
                    'branch_area': int(area)
                })
            
            return cutting_points
            
        except Exception as e:
            print(f"Lỗi calculate_cutting_points: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def toggle_cutting_point(self):
        """Bật/tắt tính toán điểm cắt cho robot"""
        self.enable_cutting_point = self.cutting_var.get()
        status = "BẬT" if self.enable_cutting_point else "TẮT"
        self.status_label.config(text=f"✂️ Cutting Point: {status}", foreground='cyan')
        print(f"\n✂️ Cutting Point Calculation: {status}")
        if not self.enable_cutting_point:
            self.cutting_data = None
    
    def update_export_mode(self):
        """Cập nhật chế độ xuất point cloud"""
        self.export_mode = self.export_mode_var.get()
        if self.export_mode == "individual":
            self.status_label.config(text=f"📤 Export: RIÊNG BIỆT (N files)", foreground='cyan')
            print(f"\n📤 Export Mode: INDIVIDUAL - Xuất riêng từng mầm lan")
        else:
            self.status_label.config(text=f"📤 Export: GỘP CHUNG (1 file)", foreground='cyan')
            print(f"\n📤 Export Mode: COMBINED - Xuất gộp tất cả")
    
    def stop_threads_for_viewer(self):
        """Dừng threads trước khi mở Open3D viewer"""
        print("\n⏸️ Dừng camera threads...")
        self.is_running = False
        self._set_preview_active(False)
        
        # Đợi threads dừng
        import time
        time.sleep(0.3)
    
    def restart_threads_after_viewer(self):
        """Khởi động lại threads sau khi đóng viewer"""
        print("▶️ Khởi động lại camera threads...")
        self.is_running = True
        self._set_preview_active(True)
        
        # Tạo threads mới
        self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.update_thread.start()
        
        self.display_thread = threading.Thread(target=self.display_frame, daemon=True)
        self.display_thread.start()
        
        print("✅ Threads đã khởi động lại!\n")
    
    def process_and_export(self):
        try:
            self.status_label.config(text="⏳ Đang xử lý...", foreground='orange')
            self.root.update()
            
            # 🆕 Clear old tips data trước khi process mới
            self.last_tips_data_per_instance = {}
            self.last_path_vis_per_instance = {}
            
            print(f"\n=== Bắt đầu xuất Point Cloud ===")
            print(f"Phương pháp: Manual Python (Multi-frame averaging)")
            
            self.process_with_manual_python()
                
        except Exception as e:
            self.status_label.config(text=f"🔴 Lỗi: {str(e)}", foreground='red')
            print(f"Chi tiết lỗi: {e}")
            import traceback
            traceback.print_exc()
    
    def process_with_manual_python(self):
        """Tạo point cloud theo phong cách RealSense gốc - mượt và nhanh"""
        try:
            import time
            t_start_total = time.perf_counter()
            
            print("\n" + "="*70)
            print("🐍 Sử dụng RealSense Point Cloud API (tối ưu vectorized)...")
            print("="*70)
            use_outlier = self.outlier_var.get()
            use_smooth = self.smooth_var.get()
            use_bilateral = self.bilateral_var.get()
            num_frames = self.num_frames_avg
            depth_min = self.depth_min
            depth_max = self.depth_max
            
            print(f"Cấu hình:")
            print(f"  - Frames: {num_frames}")
            print(f"  - Depth range: {depth_min:.2f}m - {depth_max:.2f}m")
            print(f"  - Bilateral filter: {use_bilateral}")
            print(f"  - Outlier removal: {use_outlier}")
            print(f"  - Smoothing: {use_smooth}")
            
            # Tạm dừng preview
            was_previewing = self.is_preview
            self._set_preview_active(False)
            self.root.update()
            
            import time
            time.sleep(0.1)
            
            # ====================================================================
            # PASS 1: QUÉT NHANH ĐỂ TÍNH NỀN VÀ Z OFFSET (CHỈ KHI CẦN)
            # ====================================================================
            z_offset_from_background = None
            plane_normal_computed = None
            R_matrix = None
            
            if self.plane_detect_var.get():  # Nếu bật plane detection
                print("\n" + "="*70)
                print("🔵 PASS 1: QUÉT NHANH ĐỂ TÍNH NỀN (1 frame, sampling thưa)")
                print("="*70)
                
                t_pass1_start = time.perf_counter()
                
                # Lấy 1 frame duy nhất
                print("\n📸 Lấy 1 frame để tính nền...")
                with self.camera_lock:
                    frames_bg = self.pipeline.wait_for_frames(timeout_ms=5000)
                    aligned_frames_bg = self.align.process(frames_bg)
                
                depth_frame_bg = aligned_frames_bg.get_depth_frame()
                color_frame_bg = aligned_frames_bg.get_color_frame()
                
                if not depth_frame_bg or not color_frame_bg:
                    print("⚠️ Không lấy được frame cho pass 1, bỏ qua")
                else:
                    # Áp dụng filters (nếu cần)
                    if self.use_filters:
                        if self.decimation_filter:
                            depth_frame_bg = self.decimation_filter.process(depth_frame_bg)
                        depth_frame_bg = self.spatial_filter.process(depth_frame_bg)
                        depth_frame_bg = self.hole_filling.process(depth_frame_bg)
                    
                    # Tạo point cloud từ frame này
                    self.pc.map_to(color_frame_bg)
                    points_bg = self.pc.calculate(depth_frame_bg)
                    
                    v_bg = points_bg.get_vertices()
                    verts_bg = np.asanyarray(v_bg).view(np.float32).reshape(-1, 3)
                    
                    # 🔥 FIX: CHỈ LẤY ĐIỂM TRONG VÙNG DETECTION (self.roi từ roi_config.txt)
                    color_image_bg = np.asanyarray(color_frame_bg.get_data())
                    h_bg, w_bg = color_image_bg.shape[:2]
                    start_x_bg = (w_bg - 720) // 2
                    
                    # Map vertices to pixel coordinates (full 1280x720)
                    t_bg_full = points_bg.get_texture_coordinates()
                    texcoords_bg_full = np.asanyarray(t_bg_full).view(np.float32).reshape(-1, 2)
                    
                    px_bg_full = (texcoords_bg_full[:, 0] * w_bg).astype(np.int32)
                    py_bg_full = (texcoords_bg_full[:, 1] * h_bg).astype(np.int32)
                    
                    # 🔥 SỬ DỤNG VÙNG DETECTION (roi_config.txt) thay vì toàn bộ 720x720
                    if self.roi is not None:
                        # ROI format: [x1, y1, x2, y2] trong tọa độ 720x720
                        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
                        
                        # Convert to full image coordinates (1280x720)
                        roi_x1_full = start_x_bg + roi_x1
                        roi_x2_full = start_x_bg + roi_x2
                        roi_y1_full = roi_y1
                        roi_y2_full = roi_y2
                        
                        # Filter points within detection area
                        roi_mask = (px_bg_full >= roi_x1_full) & (px_bg_full < roi_x2_full) & \
                                   (py_bg_full >= roi_y1_full) & (py_bg_full < roi_y2_full)
                        
                        print(f"🎯 Sử dụng vùng detection: ({roi_x1},{roi_y1})→({roi_x2},{roi_y2}) trong 720x720")
                    else:
                        # Fallback: Use entire 720x720 crop if no ROI set
                        roi_mask = (px_bg_full >= start_x_bg) & (px_bg_full < start_x_bg + 720) & \
                                   (py_bg_full >= 0) & (py_bg_full < 720)
                        print(f"⚠️ Chưa set vùng detection, dùng toàn bộ 720x720")
                    
                    verts_bg_roi = verts_bg[roi_mask]
                    texcoords_bg_roi = texcoords_bg_full[roi_mask]  # Keep texcoords aligned
                    px_bg_roi = px_bg_full[roi_mask]
                    py_bg_roi = py_bg_full[roi_mask]
                    
                    print(f"🔷 Điểm trong vùng detection: {len(verts_bg_roi):,} (từ {len(verts_bg):,} full)")
                    
                    # SAMPLING CỰC THƯA: 0.2% điểm - minimum tuyệt đối
                    sample_step = 250
                    verts_bg_sampled = verts_bg_roi[::sample_step]
                    texcoords_bg_sampled = texcoords_bg_roi[::sample_step]  # 🔥 Sample texcoords too
                    px_bg_sampled = px_bg_roi[::sample_step]
                    py_bg_sampled = py_bg_roi[::sample_step]
                    
                    print(f"🔷 Điểm sau sampling (1/{sample_step}): {len(verts_bg_sampled):,}")
                    
                    # Lọc depth range
                    z_vals_bg = verts_bg_sampled[:, 2]
                    valid_bg = (z_vals_bg > depth_min) & (z_vals_bg < depth_max) & (z_vals_bg > 0)
                    verts_bg_filtered = verts_bg_sampled[valid_bg]
                    px_bg_filtered = px_bg_sampled[valid_bg]  # 🔥 Keep pixel coords aligned
                    py_bg_filtered = py_bg_sampled[valid_bg]
                    
                    # Nếu có mask, loại bỏ các điểm trong mask (giữ lại NỀN)
                    # ✅ DÙNG LẠI MASK ĐÃ CÓ SẴN - tiết kiệm 10-20ms
                    if hasattr(self, 'last_mask_720') and self.last_mask_720 is not None:
                        print("🤖 Loại bỏ object, chỉ giữ nền (dùng mask có sẵn)...")
                        
                        # 🔥 SỬ DỤNG PIXEL COORDS ĐÃ LỌC (aligned với verts_bg_filtered)
                        px_crop_bg = px_bg_filtered - start_x_bg
                        py_crop_bg = py_bg_filtered
                        
                        # Validate coordinates (should all be valid since we filtered by ROI)
                        crop_valid_bg = (px_crop_bg >= 0) & (px_crop_bg < 720) & (py_crop_bg >= 0) & (py_crop_bg < 720)
                        
                        # Lọc: GIỮ CÁC ĐIỂM KHÔNG TRONG MASK (mask = 0 = nền)
                        background_mask = np.zeros(len(verts_bg_filtered), dtype=bool)
                        valid_idx_bg = np.where(crop_valid_bg)[0]
                        if len(valid_idx_bg) > 0:
                            # ✅ DÙNG MASK CÓ SẴN thay vì tạo mới
                            mask_vals = self.last_mask_720[py_crop_bg[valid_idx_bg], px_crop_bg[valid_idx_bg]]
                            background_mask[valid_idx_bg] = mask_vals == 0  # Nền = mask 0
                        
                        verts_bg_filtered = verts_bg_filtered[background_mask]
                        print(f"🔷 Điểm nền sau khi loại object: {len(verts_bg_filtered):,}")
                    
                    if len(verts_bg_filtered) > 100:
                        # Tạo Open3D point cloud cho nền
                        pcd_bg = o3d.geometry.PointCloud()
                        points_bg_flip = verts_bg_filtered.copy()
                        points_bg_flip[:, 1] *= -1
                        points_bg_flip[:, 2] *= -1
                        pcd_bg.points = o3d.utility.Vector3dVector(points_bg_flip)
                        
                        # RANSAC plane detection trên NỀN
                        print("\n🔷 RANSAC plane detection (chỉ nền)...")
                        plane_model, inliers = pcd_bg.segment_plane(
                            distance_threshold=0.01,   # Tăng tolerance - background phẳng
                            ransac_n=3,
                            num_iterations=10  # EXTREME MINIMAL - chỉ đủ cho 80% inliers
                        )
                        
                        [a, b, c, d] = plane_model
                        print(f"   Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
                        print(f"   Inliers: {len(inliers):,}/{len(pcd_bg.points):,}")
                        
                        # Tính rotation matrix
                        plane_normal = np.array([a, b, c])
                        plane_normal = plane_normal / np.linalg.norm(plane_normal)
                        
                        if plane_normal[2] < 0:
                            plane_normal = -plane_normal
                        
                        target_z = np.array([0, 0, 1])
                        v = np.cross(plane_normal, target_z)
                        s = np.linalg.norm(v)
                        c_val = np.dot(plane_normal, target_z)
                        
                        if s < 1e-6:
                            R_matrix = np.eye(3)
                        else:
                            vx = np.array([[0, -v[2], v[1]],
                                          [v[2], 0, -v[0]],
                                          [-v[1], v[0], 0]])
                            R_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s * s))
                        
                        # Apply rotation
                        pcd_bg_rotated = pcd_bg.rotate(R_matrix, center=(0, 0, 0))
                        points_bg_rotated = np.asarray(pcd_bg_rotated.points)
                        
                        # Tính Z offset từ nền
                        plane_inliers_rotated = points_bg_rotated[inliers]
                        bg_z_vals = plane_inliers_rotated[:, 2]
                        bg_z_p5 = np.percentile(bg_z_vals, 5)
                        z_offset_from_background = -bg_z_p5
                        
                        plane_normal_computed = plane_normal
                        
                        print(f"\n✅ Z offset từ nền: {z_offset_from_background:.4f}m")
                        print(f"✅ Plane normal: [{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]")
                        
                        # ── Xuất point cloud nền trước và sau RANSAC ──────────────────
                        try:
                            bg_out_dir = "Output_pointcloud"
                            if not os.path.exists(bg_out_dir):
                                os.makedirs(bg_out_dir)
                            
                            bg_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # 1) Trước RANSAC: pcd_bg (đã flip Y/Z, chưa xoay/dịch)
                            before_path = os.path.join(bg_out_dir, f"bg_before_ransac_{bg_ts}.ply")
                            o3d.io.write_point_cloud(before_path, pcd_bg)
                            print(f"   💾 Nền TRƯỚC RANSAC : {before_path}  ({len(pcd_bg.points):,} pts)")
                            
                            # 2) Sau RANSAC: pcd_bg_rotated + translation z_offset
                            pcd_bg_after = o3d.geometry.PointCloud()
                            pts_after = points_bg_rotated.copy()
                            pts_after[:, 2] += z_offset_from_background
                            pcd_bg_after.points = o3d.utility.Vector3dVector(pts_after)
                            
                            after_path = os.path.join(bg_out_dir, f"bg_after_ransac_{bg_ts}.ply")
                            o3d.io.write_point_cloud(after_path, pcd_bg_after)
                            print(f"   💾 Nền SAU  RANSAC : {after_path}  ({len(pcd_bg_after.points):,} pts)")
                        except Exception as _e_bg:
                            print(f"   ⚠️ Không thể xuất point cloud nền: {_e_bg}")
                        # ──────────────────────────────────────────────────────────────
                    else:
                        print("⚠️ Không đủ điểm nền, bỏ qua pass 1")
                
                t_pass1_end = time.perf_counter()
                print(f"\n⏱️  Pass 1 time: {(t_pass1_end - t_pass1_start)*1000:.1f}ms")
                print("="*70 + "\n")
            
            # ====================================================================
            # PASS 2: QUÉT CHI TIẾT ĐỂ LẤY OBJECT
            # ====================================================================
            print("\n" + "="*70)
            print("🔵 PASS 2: QUÉT CHI TIẾT (multi-frame, dày điểm)")
            print("="*70)
            
            # Thu thập nhiều frames để average depth
            print(f"\n📸 Đang thu thập {num_frames} frames...")
            t_capture_start = time.perf_counter()
            
            depth_frames_list = []
            color_frame_final = None
            depth_frame_ref = None
            
            for i in range(num_frames):
                with self.camera_lock:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Áp dụng filters NẾU được bật (tránh làm biến dạng mask)
                if self.use_filters:
                    if self.decimation_filter:  # Chỉ apply nếu không None
                        depth_frame = self.decimation_filter.process(depth_frame)
                    depth_frame = self.spatial_filter.process(depth_frame)
                    depth_frame = self.temporal_filter.process(depth_frame)
                    depth_frame = self.hole_filling.process(depth_frame)
                
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_frames_list.append(depth_image)
                
                if color_frame_final is None:
                    color_frame_final = color_frame
                if depth_frame_ref is None:
                    depth_frame_ref = depth_frame
                
                self.status_label.config(text=f"⏳ Quét {i+1}/{num_frames}...", foreground='orange')
                self.root.update()
                
                if i < num_frames - 1:
                    time.sleep(0.03)  # 30ms giữa các frames
            
            if len(depth_frames_list) == 0:
                self.status_label.config(text="🔴 Lỗi: Không lấy được frame", foreground='red')
                if was_previewing:
                    self._set_preview_active(True)
                return
            
            t_capture_end = time.perf_counter()
            print(f"⏱️  Capture time: {(t_capture_end - t_capture_start)*1000:.1f}ms")
            
            # Tính trung bình depth
            t_avg_start = time.perf_counter()
            print(f"\n🧮 Tính trung bình {len(depth_frames_list)} frames...")
            depth_avg = np.mean(depth_frames_list, axis=0).astype(np.uint16)
            t_avg_end = time.perf_counter()
            print(f"⏱️  Averaging time: {(t_avg_end - t_avg_start)*1000:.1f}ms")
            
            # Áp dụng bilateral filter nếu cần (chậm hơn)
            if use_bilateral:
                t_bilateral_start = time.perf_counter()
                print("Áp dụng bilateral filter...")
                depth_float = depth_avg.astype(np.float32)
                depth_avg = cv2.bilateralFilter(depth_float, 5, 50, 50).astype(np.uint16)
                t_bilateral_end = time.perf_counter()
                print(f"⏱️  Bilateral filter time: {(t_bilateral_end - t_bilateral_start)*1000:.1f}ms")
            
            color_image = np.asanyarray(color_frame_final.get_data())
            
            # ==== PHƯƠNG PHÁP CODE GỐC: DÙNG rs.pointcloud() ====
            print("\n🔷 Tạo point cloud bằng rs.pointcloud()...")
            t_pc_start = time.perf_counter()
            
            # Map color to depth
            self.pc.map_to(color_frame_final)
            points = self.pc.calculate(depth_frame_ref)
            
            # Lấy vertices và texture coords (ZERO-COPY - Cực nhanh!)
            v = points.get_vertices()
            t = points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
            
            t_pc_end = time.perf_counter()
            print(f"Vertices từ rs.pointcloud: {len(verts):,}")
            print(f"⏱️  rs.pointcloud time: {(t_pc_end - t_pc_start)*1000:.1f}ms")
            
            # ==== SỬ DỤNG DETECTION ĐÃ LƯU ====
            mask_720 = None
            if self.yolo_model and self.detection_results:
                print("\n🤖 Sử dụng detection đã lưu...")
                t_mask_start = time.perf_counter()
                self.status_label.config(text="🤖 Xử lý mask...", foreground='cyan')
                self.root.update()
                
                try:
                    # ✅ DÙNG LẠI MASK CÓ SẴN - tiết kiệm ~350ms
                    if hasattr(self, 'last_mask_720') and self.last_mask_720 is not None:
                        mask_720 = self.last_mask_720
                        
                        # 🆕 PHƯƠNG PHÁP 1: MASK EROSION (co mask để loại biên không chính xác)
                        if self.mask_erosion_var.get():
                            erosion_size = int(self.erosion_size_scale.get())
                            print(f"  🔹 Áp dụng mask erosion (size={erosion_size})...")
                            t_erosion_start = time.perf_counter()
                            
                            # Erosion với OpenCV (nhanh hơn scipy)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size*2+1, erosion_size*2+1))
                            mask_720 = cv2.erode(mask_720.astype(np.uint8), kernel, iterations=1)
                            
                            t_erosion_end = time.perf_counter()
                            print(f"    Mask area: {np.sum(self.last_mask_720)} → {np.sum(mask_720)} pixels (-{100*(1-np.sum(mask_720)/max(1,np.sum(self.last_mask_720))):.1f}%)")
                            print(f"  ⏱️  Erosion time: {(t_erosion_end - t_erosion_start)*1000:.1f}ms")
                        
                        t_mask_end = time.perf_counter()
                        print(f"  ✅ Sử dụng mask có sẵn (720x720)")
                        print(f"  Final mask area: {np.sum(mask_720)} pixels")
                        print(f"⏱️  Mask reuse time: {(t_mask_end - t_mask_start)*1000:.1f}ms")
                        
                        results = self.detection_results
                        if results and len(results) > 0 and results[0].masks is not None:
                            masks = results[0].masks.data.cpu().numpy()
                            print(f"  Sử dụng {len(masks)} đối tượng đã detect")
                            self.seg_status_label.config(
                                text=f"Đã xuất: {len(masks)} đối tượng ✅", 
                                foreground='lime'
                            )
                    else:
                        print("  ⚠️ Không có mask sẵn, bỏ qua")
                        mask_720 = None
                        self.seg_status_label.config(
                            text="Không có mask ⚠️", 
                            foreground='orange'
                        )
                        
                except Exception as e:
                    print(f"Lỗi xử lý mask: {e}")
                    import traceback
                    traceback.print_exc()
                    mask_720 = None
                    self.last_mask = None
            else:
                print("⚠️ Chưa có detection! Vui lòng đợi camera nhận diện vật trước.")
                mask_720 = None
            
            # Crop và ROI processing
            h, w = color_image.shape[:2]
            start_x = (w - 720) // 2
            start_y = 0
            
            # Xác định ROI bounds nếu có (ROI đã ở 720x720, không cần scale)
            roi_bounds = None
            if self.roi:
                roi_x1 = self.roi[0]
                roi_y1 = self.roi[1]
                roi_x2 = self.roi[2]
                roi_y2 = self.roi[3]
                
                roi_x1_original = start_x + roi_x1
                roi_y1_original = start_y + roi_y1
                roi_x2_original = start_x + roi_x2
                roi_y2_original = start_y + roi_y2
                
                roi_bounds = (roi_x1_original, roi_y1_original, roi_x2_original, roi_y2_original)
                print(f"ROI bounds: ({roi_x1_original},{roi_y1_original}) → ({roi_x2_original},{roi_y2_original})")
            
            # Filter vertices theo depth range và ROI - VECTORIZED (30x NHANH HƠN)
            self.status_label.config(text="⏳ Lọc điểm (Vectorized)...", foreground='orange')
            self.root.update()
            
            print("\n🔷 Filtering với OPTIMIZED PIPELINE (4-stage rejection)...")
            t_filter_start = time.perf_counter()
            
            # 1. Tính pixel coordinates một lần (vectorized)
            texcoords_scaled = texcoords * np.array([w, h], dtype=np.float32)
            px = texcoords_scaled[:, 0].astype(np.int32)
            py = texcoords_scaled[:, 1].astype(np.int32)
            z_vals = verts[:, 2]
            
            # 1.5. ✅ SAMPLING (trước khi filter để tiết kiệm tính toán)
            sampling_step = self.sampling_step
            if sampling_step > 1:
                # Tạo mask sampling (lấy mỗi N điểm)
                sample_mask = np.zeros(len(verts), dtype=bool)
                sample_mask[::sampling_step] = True
                
                # Áp dụng sampling
                original_count = len(verts)
                verts = verts[sample_mask]
                texcoords_scaled = texcoords_scaled[sample_mask]
                px = px[sample_mask]
                py = py[sample_mask]
                z_vals = z_vals[sample_mask]
                
                print(f"  Stage 0 - Sampling (1/{sampling_step}): {len(verts):,}/{original_count:,} điểm (giữ {100*len(verts)/original_count:.1f}%)")
            
            # 2. ✅ STAGE 1: BOUNDS CHECK (rất rẻ - 4 comparisons)
            valid_mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            valid_mask &= (z_vals > depth_min) & (z_vals < depth_max) & (z_vals > 0)
            print(f"  Stage 1 - Bounds check: {np.sum(valid_mask):,}/{len(verts):,} điểm (loại {100*(1-np.sum(valid_mask)/len(verts)):.1f}%)")
            
            # 3. ✅ STAGE 2: ROI FILTER (loại 70-80% nếu có ROI)
            if roi_bounds:
                rx1, ry1, rx2, ry2 = roi_bounds
                valid_mask &= (px >= rx1) & (px < rx2) & (py >= ry1) & (py < ry2)
                print(f"  Stage 2 - ROI filter: {np.sum(valid_mask):,} điểm (loại {100*(1-np.sum(valid_mask)/len(verts)):.1f}%)")
            
            # 4. ✅ STAGE 3: MASK FILTER - CHỈ XỬ LÝ ĐIỂM CÒN LẠI (~200k thay vì 921k)
            if mask_720 is not None and self.remove_background_var.get():
                # CHỈ tính toán cho điểm valid (tiết kiệm 70-80% operations!)
                px_valid = px[valid_mask]
                py_valid = py[valid_mask]
                
                px_crop = px_valid - start_x
                py_crop = py_valid
                
                # Clip và lookup - CHỈ trên valid points
                px_safe = np.clip(px_crop, 0, 719)
                py_safe = np.clip(py_crop, 0, 719)
                
                crop_valid = (px_crop >= 0) & (px_crop < 720) & (py_crop >= 0) & (py_crop < 720)
                mask_values = mask_720[py_safe, px_safe]
                
                mask_filter = crop_valid & (mask_values > 0)
                
                # Update valid_mask: chỉ giữ điểm pass mask filter
                valid_indices = np.where(valid_mask)[0]
                valid_mask[valid_indices[~mask_filter]] = False
                
                print(f"  Stage 3 - Mask filter: {np.sum(valid_mask):,} điểm (loại {100*(1-np.sum(valid_mask)/len(verts)):.1f}%)")
            
            # 5. Apply mask một lần
            if np.sum(valid_mask) == 0:
                self.status_label.config(text="🔴 Không có điểm hợp lệ!", foreground='red')
                if was_previewing:
                    self._set_preview_active(True)
                return
            
            verts_filtered = verts[valid_mask]
            px_filtered = px[valid_mask]
            py_filtered = py[valid_mask]
            
            # 6. Color lookup (vectorized - CỰC NHANH)
            colors_bgr = color_image[py_filtered, px_filtered]  # Shape: (N, 3) BGR
            valid_colors = colors_bgr[:, [2, 1, 0]] / 255.0  # Convert BGR→RGB và normalize
            
            # 7. Coordinate flip (vectorized)
            valid_points = verts_filtered.copy()
            valid_points[:, 1] *= -1  # Flip Y (Camera: Y-down → Open3D: Y-up)
            valid_points[:, 2] *= -1  # Flip Z (Camera: Z-forward → Open3D: Z-backward)
            
            # 8. Pixel mapping 720x720 (vectorized)
            px_720 = px_filtered - start_x
            py_720 = py_filtered
            point_to_pixel = np.stack([px_720, py_720], axis=1)  # Shape: (N, 2)
            
            t_filter_end = time.perf_counter()
            print(f"✅ Kết quả cuối: {len(valid_points):,} điểm")
            print(f"⏱️  Filtering time (4-STAGE OPTIMIZED + SAMPLING): {(t_filter_end - t_filter_start)*1000:.1f}ms")
            print(f"   → Từ {len(verts):,} gốc (sau sampling) → {len(valid_points):,} cuối ({100*len(valid_points)/len(verts):.1f}% giữ lại)")
            
            # Tạo Open3D point cloud (valid_points và valid_colors đã là NumPy arrays)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            
            # 💾 LƯU PRE-VOXEL DATA NGAY - Đây là dữ liệu CHÍNH XÁC nhất cho export individual
            # Pixel mapping chỉ chính xác ở bước này, sau outlier/voxel sẽ không còn sync
            self.point_to_pixel_mapping_pre_voxel = point_to_pixel.copy()  # (N, 2) - 720x720 coords
            self.points_pre_voxel = valid_points.copy()  # (N, 3) - 3D coordinates
            self.colors_pre_voxel = valid_colors.copy()  # (N, 3) - RGB colors
            print(f"💾 Đã lưu PRE-VOXEL data: {len(point_to_pixel):,} điểm với pixel mapping chính xác")
            
            # Mapping tạm cho plane detection (sẽ không chính xác sau voxel nhưng không sao)
            self.point_to_pixel_mapping = point_to_pixel  # Chỉ dùng tạm, không update sau này
            
            # 🎯 TÍNH CENTER CLASS 1 TRÊN PRE-VOXEL DATA - Chính xác nhất!
            self.class1_center_3d_before_voxel = None
            if hasattr(self, 'class1_mask') and self.class1_mask is not None and np.sum(self.class1_mask) > 0:
                print(f"\n🎯 Tính center class 1 trên PRE-VOXEL data (vectorized)...")
                t_center_start = time.perf_counter()
                
                # Dùng pre-voxel data chính xác
                points_np = self.points_pre_voxel
                px_arr = self.point_to_pixel_mapping_pre_voxel[:, 0]
                py_arr = self.point_to_pixel_mapping_pre_voxel[:, 1]
                
                # Boolean mask cho valid coordinates
                valid_coords = (px_arr >= 0) & (px_arr < 720) & (py_arr >= 0) & (py_arr < 720)
                
                # Fancy indexing để lấy mask values
                class1_filter = np.zeros(len(points_np), dtype=bool)
                if np.sum(valid_coords) > 0:
                    px_valid = px_arr[valid_coords]
                    py_valid = py_arr[valid_coords]
                    mask_values = self.class1_mask[py_valid, px_valid]
                    class1_filter[valid_coords] = mask_values > 0
                
                # Apply filter
                class1_points_pre_voxel = points_np[class1_filter]
                
                if len(class1_points_pre_voxel) > 10:
                    self.class1_center_3d_before_voxel = np.mean(class1_points_pre_voxel, axis=0)
                    t_center_end = time.perf_counter()
                    print(f"   ✓ Class 1: {len(class1_points_pre_voxel):,} điểm")
                    print(f"   ✓ Center: [{self.class1_center_3d_before_voxel[0]:.4f}, {self.class1_center_3d_before_voxel[1]:.4f}, {self.class1_center_3d_before_voxel[2]:.4f}]")
                    print(f"⏱️  Class1 center time: {(t_center_end - t_center_start)*1000:.1f}ms")
                else:
                    print(f"   ⚠️ Class 1 quá ít điểm ({len(class1_points_pre_voxel)}), bỏ qua")
            
            # 🆕 PHƯƠNG PHÁP 2: Z-OFFSET FILTERING (loại điểm gần nền)
            if self.zoffset_filter_var.get() and z_offset_from_background is not None:
                t_zoffset_start = time.perf_counter()
                self.status_label.config(text="⏳ Lọc điểm nền (Z-offset)...", foreground='orange')
                self.root.update()
                print(f"\n🔹 Z-offset filtering (loại điểm gần z_offset={z_offset_from_background:.4f}m)...")
                
                tolerance = self.zoffset_tolerance_scale.get() / 1000.0  # mm → m
                points_np = np.asarray(pcd.points)
                colors_np = np.asarray(pcd.colors)
                
                # Lọc điểm có z gần z_offset (trong khoảng tolerance)
                z_vals = points_np[:, 2]
                # Chú ý: z đã được flip (z *= -1), nên z_offset cũng cần flip
                z_offset_flipped = -z_offset_from_background
                valid_mask = np.abs(z_vals - z_offset_flipped) > tolerance
                
                original = len(points_np)
                pcd.points = o3d.utility.Vector3dVector(points_np[valid_mask])
                pcd.colors = o3d.utility.Vector3dVector(colors_np[valid_mask])
                
                # ⚠️ KHÔNG update mapping - pre_voxel data đã được lưu và không thay đổi
                
                t_zoffset_end = time.perf_counter()
                print(f"  {original:,} → {len(pcd.points):,} điểm (-{100*(1-len(pcd.points)/max(1,original)):.1f}%)")
                print(f"  Tolerance: {tolerance*1000:.1f}mm, z_offset: {z_offset_from_background:.4f}m")
                print(f"⏱️  Z-offset filtering time: {(t_zoffset_end - t_zoffset_start)*1000:.1f}ms")
            
            # 🆕 PHƯƠNG PHÁP 3: RADIUS OUTLIER REMOVAL (thay thế Statistical, nhanh hơn)
            if self.radius_outlier_var.get():
                t_radius_start = time.perf_counter()
                self.status_label.config(text="⏳ Loại nhiễu (Radius Outlier)...", foreground='orange')
                self.root.update()
                print("\n🔹 Radius outlier removal (nhanh hơn Statistical)...")
                
                radius = self.radius_scale.get() / 1000.0  # mm → m
                min_neighbors = int(self.min_neighbors_scale.get())
                
                original = len(pcd.points)
                pcd, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
                
                # ⚠️ KHÔNG update mapping - pre_voxel data vẫn giữ nguyên để export individual
                
                t_radius_end = time.perf_counter()
                print(f"  {original:,} → {len(pcd.points):,} điểm (-{100*(1-len(pcd.points)/max(1,original)):.1f}%)")
                print(f"  Radius: {radius*1000:.1f}mm, min_neighbors: {min_neighbors}")
                print(f"⏱️  Radius outlier removal time: {(t_radius_end - t_radius_start)*1000:.1f}ms")
            
            # Giữ lại Statistical Outlier nếu user bật (nhưng mặc định tắt)
            if use_outlier:
                t_outlier_start = time.perf_counter()
                self.status_label.config(text="⏳ Loại nhiễu (Statistical)...", foreground='orange')
                self.root.update()
                print("\n🔷 Statistical outlier removal (nb_neighbors=10)...")
                original = len(pcd.points)
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
                # ⚠️ KHÔNG update mapping - pre_voxel data là reference chính xác
                t_outlier_end = time.perf_counter()
                print(f"  {original:,} → {len(pcd.points):,} điểm (-{100*(1-len(pcd.points)/max(1,original)):.1f}%)")
                print(f"⏱️  Statistical outlier time: {(t_outlier_end - t_outlier_start)*1000:.1f}ms")
            
            if use_smooth:
                t_voxel_start = time.perf_counter()
                self.status_label.config(text="⏳ Làm mịn...", foreground='orange')
                self.root.update()
                print("\n🔷 Voxel downsampling (0.5mm - HIGH DETAIL)...")
                original = len(pcd.points)
                
                # ⚠️ VOXEL DOWNSAMPLING: Pixel mapping BỊ MẤT sau bước này!
                # Pre-voxel data đã được lưu → dùng cho export individual
                # PCD sau voxel chỉ dùng cho visualization/combined export
                pcd = pcd.voxel_down_sample(voxel_size=0.0005)  # 0.5mm cho mầm lan nhỏ
                t_voxel_end = time.perf_counter()
                print(f"  {original:,} → {len(pcd.points):,} điểm (-{100*(1-len(pcd.points)/original):.1f}%)")
                print(f"  ⚠️ Pixel mapping không còn sync sau voxel, dùng pre_voxel data cho export!")
                print(f"⏱️  Voxel downsampling time: {(t_voxel_end - t_voxel_start)*1000:.1f}ms")
            
            # Normal estimation (giảm max_nn 30→15 để nhanh hơn 40%)
            t_normal_start = time.perf_counter()
            self.status_label.config(text="⏳ Tính normals...", foreground='orange')
            self.root.update()
            print("\n🔷 Normal estimation (max_nn=15, tối ưu tốc độ)...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15)
            )
            pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
            t_normal_end = time.perf_counter()
            print(f"⏱️  Normal estimation time: {(t_normal_end - t_normal_start)*1000:.1f}ms")
            
            print(f"\n✅ Hoàn tất! {len(pcd.points):,} điểm\n")
            
            # ==== PLANE DETECTION & COORDINATE TRANSFORMATION ====
            if self.plane_detect_var.get():
                t_plane_start = time.perf_counter()
                # Truyền Z offset và R matrix từ pass 1 nếu có
                pcd = self.detect_plane_and_transform(
                    pcd, 
                    z_offset_precomputed=z_offset_from_background,
                    R_matrix_precomputed=R_matrix,
                    plane_normal_precomputed=plane_normal_computed
                )
                t_plane_end = time.perf_counter()
                print(f"⏱️  Plane transform time: {(t_plane_end - t_plane_start)*1000:.1f}ms\n")
            else:
                # Không transform, reset về None
                self.R_matrix_transform = None
                self.translation_vector = None
            
            # Tính tổng thời gian
            t_end_total = time.perf_counter()
            total_time = (t_end_total - t_start_total) * 1000
            
            print("="*70)
            print(f"⏱️  TỔNG THỜI GIAN XỬ LÝ: {total_time:.1f}ms ({total_time/1000:.2f}s)")
            print("="*70 + "\n")
            
            # Lưu để có thể export sau
            self.last_pcd = pcd
            self.last_method = "manual_python"
            
            # Capture RGB 720x720 for texture (FULL DETAIL - không resize)
            self.captured_rgb = color_image[start_y:start_y+720, start_x:start_x+720].copy()
            
            # Bật lại preview
            if was_previewing:
                self._set_preview_active(True)
            
            # Xuất point cloud ra file thay vì hiển thị
            self.export_point_cloud_to_file(pcd)
            
            self.process_status_label.config(text=f"Point Cloud: {len(pcd.points):,} điểm ✓", 
                                              foreground='green')
            self.status_label.config(text="🟢 Đã xuất Point Cloud thành công!", foreground='green')
            
        except Exception as e:
            self.status_label.config(text=f"🔴 Lỗi: {str(e)}", foreground='red')
            print(f"Chi tiết lỗi: {e}")
            import traceback
            traceback.print_exc()
            # Đảm bảo bật lại preview nếu có lỗi
            try:
                if was_previewing:
                    self._set_preview_active(True)
            except:
                pass
    
    def calculate_frame_size(self, geometry):
        """Tính kích thước coordinate frame tự động dựa trên geometry"""
        try:
            bbox = geometry.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()  # [width, height, depth]
            max_dim = np.max(extent)
            
            # Frame size = 8% của dimension lớn nhất (mảnh hơn, gọn hơn)
            frame_size = max_dim * 0.08
            
            # Đảm bảo frame không quá nhỏ (min 0.01m) hoặc quá lớn (max 0.3m)
            frame_size = np.clip(frame_size, 0.01, 0.3)
            
            print(f"📏 Auto frame size: {frame_size:.4f}m (geometry extent: {extent})")
            return frame_size
        except Exception as e:
            print(f"⚠️ Lỗi tính frame size: {e}, dùng default 0.03m")
            return 0.03
    
    def show_point_cloud(self, pcd):
        """Hiển thị point cloud trong Open3D viewer"""
        try:
            print(f"\nMở Open3D viewer với {len(pcd.points):,} điểm...")
            
            # DỪNG HOÀN TOÀN threads để tránh GIL conflict
            self.stop_threads_for_viewer()
            
            # Tự động tính kích thước frame
            frame_size = self.calculate_frame_size(pcd)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
            
            print("\n👁️ Điều khiển Open3D:")
            print("  - Chuột trái: Xoay")
            print("  - Chuột phải: Pan") 
            print("  - Cuộn chuột: Zoom")
            print("  - Phím Q: Thoát viewer\n")
            
            # Mở viewer (blocking call)
            o3d.visualization.draw_geometries(
                [pcd, mesh_frame],
                window_name='Point Cloud Viewer - Intel RealSense D435i [CHẤT LƯỢNG CAO]',
                width=1280,
                height=960,
                left=50,
                top=50,
                point_show_normal=False,
                mesh_show_wireframe=False,
                mesh_show_back_face=False
            )
            
            # KHỞI ĐỘNG LẠI threads sau khi đóng viewer
            self.restart_threads_after_viewer()
            
        except Exception as e:
            print(f"Lỗi hiển thị point cloud: {e}")
            import traceback
            traceback.print_exc()
            # KHỞI ĐỘNG LẠI threads nếu có lỗi
            self.restart_threads_after_viewer()
    
    def detect_plane_and_transform(self, pcd, z_offset_precomputed=None, R_matrix_precomputed=None, plane_normal_precomputed=None):
        """
        Phát hiện mặt phẳng (bàn/sàn) và chuyển đổi tọa độ:
        - Nếu có precomputed values (từ pass 1) thì dùng luôn
        - Nếu không, chạy RANSAC như cũ
        """
        import time  # Import time để dùng cho timing
        
        try:
            self.status_label.config(text="🔍 Phát hiện mặt phẳng...", foreground='cyan')
            self.root.update()
            
            print("\n" + "="*60)
            print("🎯 COORDINATE TRANSFORMATION")
            print("="*60)
            
            points_np = np.asarray(pcd.points)
            colors_np = np.asarray(pcd.colors)
            total_points = len(points_np)
            
            # Nếu đã có precomputed values từ pass 1, dùng luôn
            if z_offset_precomputed is not None and R_matrix_precomputed is not None:
                print("\n✅ Sử dụng Z offset và rotation từ PASS 1 (nền đã tính)")
                print(f"   Z offset: {z_offset_precomputed:.4f}m")
                print(f"   Plane normal: [{plane_normal_precomputed[0]:.3f}, {plane_normal_precomputed[1]:.3f}, {plane_normal_precomputed[2]:.3f}]")
                
                z_offset = z_offset_precomputed
                R = R_matrix_precomputed
                
                # Bỏ qua RANSAC, áp dụng rotation và translation luôn
                print("\n🔷 Apply rotation từ pass 1...")
                pcd_rotated = pcd.rotate(R, center=(0, 0, 0))
                
            else:
                # Không có precomputed, chạy RANSAC bình thường
                print("\n⚠️ Không có precomputed values, chạy RANSAC...")
                
                # RANSAC plane detection
                remove_bg = self.remove_background_var.get()
                num_iterations = 500 if remove_bg else 1500
                
                print(f"   RANSAC iterations: {num_iterations} ({'tối ưu' if remove_bg else 'đầy đủ'})")
                
                t_ransac_start = time.perf_counter()
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=0.003,
                    ransac_n=3,
                    num_iterations=num_iterations
                )
                t_ransac_end = time.perf_counter()
                print(f"   ⏱️  RANSAC time: {(t_ransac_end - t_ransac_start)*1000:.1f}ms")
                
                [a, b, c, d] = plane_model
                print(f"   Phương trình plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
                print(f"   Inliers: {len(inliers):,}/{total_points:,} điểm")
                
                # Tính rotation matrix
                plane_normal = np.array([a, b, c])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                
                if plane_normal[2] < 0:
                    plane_normal = -plane_normal
                
                target_z = np.array([0, 0, 1])
                v = np.cross(plane_normal, target_z)
                s = np.linalg.norm(v)
                c_val = np.dot(plane_normal, target_z)
                
                if s < 1e-6:
                    R = np.eye(3)
                else:
                    vx = np.array([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]])
                    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s * s))
                
                # Apply rotation
                pcd_rotated = pcd.rotate(R, center=(0, 0, 0))
                points_rotated = np.asarray(pcd_rotated.points)
                
                # Tính Z offset từ RANSAC plane
                plane_points_rotated = points_rotated[inliers]
                plane_z_mean = np.mean(plane_points_rotated[:, 2])
                z_offset = -plane_z_mean
                print(f"   Z offset từ RANSAC: {z_offset:.4f}m")
            
            # ==========================================================================
            # COMMON PROCESSING - DÙNG CHO CẢ 2 TRƯỜNG HỢP (precomputed hoặc RANSAC)
            # ==========================================================================
            
            # Lấy rotated points
            points_rotated = np.asarray(pcd_rotated.points)
            total_points = len(points_rotated)
            
            # Đảm bảo plane_z_mean luôn tồn tại (cần cho phân loại object/background)
            # Nếu dùng RANSAC, plane_z_mean đã được tính
            # Nếu dùng precomputed, ước lượng từ z_offset
            if 'plane_z_mean' not in locals():
                plane_z_mean = -z_offset  # Vì z_offset = -plane_z_mean
                print(f"   ℹ️ Ước lượng plane_z_mean từ precomputed z_offset: {plane_z_mean:.4f}m")
            
            # Kiểm tra các điều kiện cho tính center
            has_class1_2d = hasattr(self, 'class1_center_2d') and self.class1_center_2d is not None
            has_class1 = hasattr(self, 'class1_mask') and self.class1_mask is not None
            has_mapping = hasattr(self, 'point_to_pixel_mapping') and self.point_to_pixel_mapping is not None
            
            print(f"\n🎯 Phân loại object/background và tính tâm vật thể...")
            
            # Phân loại object/background
            is_mask_filtered = self.remove_background_var.get()

            
            if is_mask_filtered:
                # Point cloud ĐÃ được lọc theo mask → TẤT CẢ là object
                print(f"   ℹ️  Point cloud đã lọc theo mask → TẤT CẢ {total_points:,} điểm là OBJECT")
                object_indices = np.arange(total_points)
                background_indices = np.array([], dtype=int)
                print(f"   Object points: {len(object_indices):,}")
                print(f"   Background points: 0 (đã lọc bỏ trước đó)")
            else:
                # Point cloud có CẢ object và background → cần phân tách theo Z
                print(f"   ℹ️  Point cloud chứa cả object và nền → phân tách theo Z threshold")
                threshold_offset = 0.002  # 2mm để tách object khỏi nền
                object_threshold = plane_z_mean + threshold_offset
                
                object_mask_z = points_rotated[:, 2] > object_threshold
                object_indices = np.where(object_mask_z)[0]
                background_indices = np.where(~object_mask_z)[0]
                
                print(f"   Object threshold: plane_z + {threshold_offset*1000:.1f}mm = {object_threshold:.4f}m")
                print(f"   Object points (Z > threshold): {len(object_indices):,}")
                print(f"   Background points: {len(background_indices):,}")
                print(f"   Tỉ lệ object/total: {len(object_indices)/total_points*100:.1f}%")
                
                # Kiểm tra phân bố Z của object
                if len(object_indices) > 0:
                    obj_z = points_rotated[object_indices][:, 2]
                    print(f"   Object Z range: [{np.min(obj_z):.4f}, {np.max(obj_z):.4f}]")
            
            # 6. Tính center của object - ƯU TIÊN CLASS 1 CENTER ĐÃ TÍNH TRƯỚC VOXEL
            print(f"\n5️⃣ Tính tâm vật thể...")
            
            # Kiểm tra xem có class1_center_3d_before_voxel không (tính trước voxel downsampling)
            has_class1_pre_voxel = hasattr(self, 'class1_center_3d_before_voxel') and self.class1_center_3d_before_voxel is not None
            
            if has_class1_pre_voxel:
                # Dùng center đã tính TRƯỚC voxel downsampling (chính xác nhất)
                print(f"   ✅ Sử dụng CENTER CLASS 1 đã tính TRƯỚC voxel downsampling")
                
                # Center này cần được rotate theo rotation matrix R
                object_center_original = self.class1_center_3d_before_voxel
                object_center = R.dot(object_center_original)
                
                print(f"   ✓ Center gốc: [{object_center_original[0]:.4f}, {object_center_original[1]:.4f}, {object_center_original[2]:.4f}]")
                print(f"   ✓ Center sau rotate: [{object_center[0]:.4f}, {object_center[1]:.4f}, {object_center[2]:.4f}]")
                
                if hasattr(self, 'class1_center_2d') and self.class1_center_2d is not None:
                    cx_640, cy_640 = self.class1_center_2d
                    print(f"   ✓ Tọa độ XY khớp với dấu chấm đỏ tại pixel ({cx_640}, {cy_640})")
                
                # Tạo object_pcd_rotated để tính bbox (dù không dùng center của nó)
                if len(object_indices) > 0:
                    object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                    bbox = object_pcd_rotated.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()
                    min_bbox = bbox.get_min_bound()
                    max_bbox = bbox.get_max_bound()
                    
                    print(f"   ✓ Object: {len(object_indices):,} điểm")
                    print(f"   ✓ Size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
            
            elif has_class1_2d and has_class1 and has_mapping and len(object_indices) > 0:
                # Tính tâm từ TỌA ĐỘ 2D CỦA DẤU CHẤM ĐỎ (chính xác nhất)
                print(f"   ℹ️ Sử dụng TỌA ĐỘ 2D DẤU CHẤM ĐỎ + PIXEL MAPPING để tính tâm...")
                
                cx_720, cy_720 = self.class1_center_2d
                print(f"   ✓ Class 1 center 2D từ dấu chấm đỏ (720x720): [{cx_720}, {cy_720}]")
                
                # Lọc các điểm trong object_indices thuộc class1_mask
                class1_points = []
                points_rotated = np.asarray(pcd_rotated.points)
                
                # DEBUG: Kiểm tra kích thước
                print(f"   DEBUG: pcd_rotated có {len(points_rotated)} điểm")
                print(f"   DEBUG: point_to_pixel_mapping có {len(self.point_to_pixel_mapping)} entries (720x720)")
                print(f"   DEBUG: object_indices có {len(object_indices)} indices")
                print(f"   DEBUG: class1_mask có {np.sum(self.class1_mask)} pixels > 0 (720x720)")
                
                # Mapping có thể không khớp 100% sau voxel downsampling
                # Nên duyệt qua object_indices và check pixel mapping
                matched_count = 0
                for idx in object_indices:
                    if idx < len(self.point_to_pixel_mapping):
                        px_720, py_720 = self.point_to_pixel_mapping[idx]
                        # Kiểm tra pixel này có trong class1_mask không (720x720)
                        if 0 <= py_720 < 720 and 0 <= px_720 < 720:
                            if self.class1_mask[py_720, py_720] > 0:
                                class1_points.append(points_rotated[idx])
                                matched_count += 1
                
                print(f"   DEBUG: Matched {matched_count} điểm với class1_mask")
                
                if len(class1_points) > 10:
                    # Tính center từ ĐIỂM CLASS 1
                    class1_points_np = np.array(class1_points)
                    object_center = np.mean(class1_points_np, axis=0)
                    print(f"   ✓ CENTER TỪ CLASS 1: {len(class1_points):,} điểm 3D")
                    print(f"   ✓ Class 1 center 3D: [{object_center[0]:.4f}, {object_center[1]:.4f}, {object_center[2]:.4f}]")
                    print(f"   ✓ Tọa độ XY khớp với dấu chấm đỏ tại pixel ({cx_640}, {cy_640})")
                else:
                    # Fallback: dùng tất cả object points
                    object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                    object_center = np.asarray(object_pcd_rotated.get_center())
                    print(f"   ⚠️ Class 1 mask quá ít điểm 3D ({len(class1_points)}), dùng all objects")
                
            elif has_class1 and has_mapping and len(object_indices) > 0:
                # Fallback: không có class1_center_2d, tính từ moments
                print(f"   ℹ️ Không có tọa độ 2D, tính từ moments...")
                M = cv2.moments(self.class1_mask)
                if M['m00'] > 0:
                    mask_cx_720 = M['m10'] / M['m00']
                    mask_cy_720 = M['m01'] / M['m00']
                    print(f"   ✓ Class 1 mask center 2D (720x720): [{mask_cx_720:.1f}, {mask_cy_720:.1f}]")
                    
                    class1_points = []
                    points_rotated = np.asarray(pcd_rotated.points)
                    
                    for idx in object_indices:
                        if idx < len(self.point_to_pixel_mapping):
                            px_720, py_720 = self.point_to_pixel_mapping[idx]
                            if 0 <= py_720 < 720 and 0 <= px_720 < 720:
                                if self.class1_mask[py_720, py_720] > 0:
                                    class1_points.append(points_rotated[idx])
                    
                    if len(class1_points) > 10:
                        class1_points_np = np.array(class1_points)
                        object_center = np.mean(class1_points_np, axis=0)
                        print(f"   ✓ CENTER TỪ CLASS 1: {len(class1_points):,} điểm 3D")
                    else:
                        object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                        object_center = np.asarray(object_pcd_rotated.get_center())
                        print(f"   ⚠️ Class 1 mask quá ít điểm, dùng all objects")
                    object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                    object_center = np.asarray(object_pcd_rotated.get_center())
                    print(f"   ⚠️ Không tính được moment của class1_mask, dùng all objects")
                
                # Tính bbox từ tất cả object (để hiển thị size tổng thể)
                object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                bbox = object_pcd_rotated.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                min_bbox = bbox.get_min_bound()
                max_bbox = bbox.get_max_bound()
                
                print(f"   ✓ Object: {len(object_indices):,} điểm")
                print(f"   ✓ Center XYZ: [{object_center[0]:.4f}, {object_center[1]:.4f}, {object_center[2]:.4f}]")
                print(f"   ✓ BBox XY: X=[{min_bbox[0]:.4f}, {max_bbox[0]:.4f}], Y=[{min_bbox[1]:.4f}, {max_bbox[1]:.4f}]")
                print(f"   ✓ Size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
                
            elif len(object_indices) > 0:
                # Không có class1_mask, dùng tất cả object points
                object_pcd_rotated = pcd_rotated.select_by_index(object_indices.tolist())
                object_center = np.asarray(object_pcd_rotated.get_center())
                bbox = object_pcd_rotated.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                min_bbox = bbox.get_min_bound()
                max_bbox = bbox.get_max_bound()
                
                print(f"   ⚠️ Không có class1_mask, dùng all objects")
                print(f"   ✓ Object: {len(object_indices):,} điểm")
                print(f"   ✓ Center XYZ: [{object_center[0]:.4f}, {object_center[1]:.4f}, {object_center[2]:.4f}]")
                print(f"   ✓ BBox XY: X=[{min_bbox[0]:.4f}, {max_bbox[0]:.4f}], Y=[{min_bbox[1]:.4f}, {max_bbox[1]:.4f}]")
                print(f"   ✓ Size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
            else:
                object_center = np.asarray(pcd_rotated.get_center())
                print(f"   ⚠️ Không tách được object, dùng center của toàn bộ")
            
            # 7. Xử lý remove background (không cần filter vì đã filter ở Pass 2)
            remove_bg = self.remove_background_var.get()
            print(f"\n7️⃣ Xử lý nền trong transformation:")
            
            if remove_bg:
                # Background đã được lọc bỏ từ đầu (trong process_and_export)
                pcd_to_transform = pcd_rotated
                print(f"   ℹ️  Background đã lọc bỏ trước đó (trong point cloud generation)")
                print(f"   📦 Transform {len(pcd_rotated.points):,} điểm object")
            else:
                # Giữ tất cả điểm
                pcd_to_transform = pcd_rotated
                print(f"   📦 Giữ tất cả {len(pcd_rotated.points):,} điểm (object + background)")
            
            # 8. Apply translation
            translation = np.array([-object_center[0], -object_center[1], z_offset])
            pcd_transformed = pcd_to_transform.translate(translation)
            
            print(f"\n8️⃣ Transformation hoàn tất")
            print(f"   Translation XYZ: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
            print(f"   → X,Y: Đặt tại tâm vật thể")
            print(f"   → Z: Đặt tại mặt nền (z_offset={z_offset:.4f}m)")
            
            # 9. Verify
            bounds = pcd_transformed.get_axis_aligned_bounding_box()
            min_b = bounds.get_min_bound()
            max_b = bounds.get_max_bound()
            
            print(f"\n9️⃣ Kết quả cuối cùng:")
            print(f"   Số điểm: {len(pcd_transformed.points):,}")
            print(f"   X: [{min_b[0]*1000:.1f}, {max_b[0]*1000:.1f}] mm")
            print(f"   Y: [{min_b[1]*1000:.1f}, {max_b[1]*1000:.1f}] mm")
            print(f"   Z: [{min_b[2]*1000:.1f}, {max_b[2]*1000:.1f}] mm")
            print(f"   Height: {(max_b[2]-min_b[2])*1000:.1f}mm")
            
            # Kiểm tra center
            center_final = np.asarray(pcd_transformed.get_center())
            
            if remove_bg:
                # Nếu loại bỏ nền, center của point cloud = object center ≈ (0,0,height/2)
                print(f"   Object center (chỉ vật thể): [{center_final[0]*1000:.2f}, {center_final[1]*1000:.2f}, {center_final[2]*1000:.2f}] mm")
                if abs(center_final[0]) > 0.5 or abs(center_final[1]) > 0.5:
                    print(f"   ⚠️ XY center lệch khỏi (0,0)!")
            else:
                # Nếu giữ nền, center của toàn bộ point cloud sẽ khác object center
                print(f"   Center toàn bộ (object + background): [{center_final[0]*1000:.2f}, {center_final[1]*1000:.2f}, {center_final[2]*1000:.2f}] mm")
                print(f"   ℹ️  Object center đã đặt tại (0,0), nhưng có thêm background nên center toàn bộ bị lệch")
            
            # Kiểm tra Z
            if min_b[2] < -1.0:
                print(f"   ⚠️ Z min < -1mm! Có điểm nền thấp hơn nhiều so với p5")
            elif min_b[2] > 5.0:
                print(f"   ⚠️ Z min > 5mm! Nền có thể đặt quá cao")
            print("="*60 + "\n")
            
            # 🆕 LƯU TRANSFORMATION PARAMETERS để dùng cho individual export
            self.R_matrix_transform = R
            self.translation_vector = translation
            print(f"\n💾 Đã lưu transformation:")
            print(f"   R_matrix: {R.shape}")
            print(f"   Translation: {translation}")
            
            # Update GUI
            bg_status = "nền đã loại" if remove_bg else "có nền"
            height_mm = (max_b[2] - min_b[2]) * 1000
            self.plane_info_label.config(
                text=f"✅ Z=0@plane, XY@center | {len(pcd_transformed.points):,}pts | H={height_mm:.1f}mm | {bg_status}",
                foreground='lime'
            )
            
            return pcd_transformed
            
        except Exception as e:
            print(f"\n❌ Lỗi plane detection: {e}")
            import traceback
            traceback.print_exc()
            
            self.plane_info_label.config(
                text=f"⚠️ Không phát hiện được plane, giữ gốc camera",
                foreground='orange'
            )
            return pcd  # Trả về original nếu lỗi
    
    def save_point_cloud(self):
        """Lưu point cloud ra file .ply"""
        try:
            if self.last_pcd is None:
                self.status_label.config(text="⚠️ Chưa có point cloud! Xuất trước đã", foreground='orange')
                return
            
            # Tạo tên file với timestamp
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            num_points = len(self.last_pcd.points)
            
            filename = f"pointcloud_{timestamp}_{num_points}pts.ply"
            
            # Lưu file
            o3d.io.write_point_cloud(filename, self.last_pcd)
            
            self.status_label.config(text=f"💾 Đã lưu: {filename}", foreground='green')
            print(f"\n✅ Đã lưu point cloud: {filename}")
            print(f"   - Phương pháp: {self.last_method}")
            print(f"   - Số điểm: {num_points:,}")
            print(f"   - File size: {np.round(len(self.last_pcd.points) * 24 / 1024 / 1024, 2)} MB\n")
            
        except Exception as e:
            self.status_label.config(text=f"🔴 Lỗi lưu file: {str(e)}", foreground='red')
            print(f"Lỗi lưu point cloud: {e}")
    
    # ROI methods
    def toggle_set_area(self):
        self.is_setting_area = not self.is_setting_area
        if self.is_setting_area:
            # Mở cửa sổ ROI setting với full view 720x720
            if self.roi_setting_window is None or not self.roi_setting_window.winfo_exists():
                self.roi_setting_window = tk.Toplevel(self.root)
                self.roi_setting_window.title("🎯 Set Detection Area - Full View")
                self.roi_setting_window.geometry("750x800")
                self.roi_setting_window.configure(bg='#1e1e1e')
                
                # Canvas 720x720 cho full view
                self.roi_canvas = tk.Canvas(self.roi_setting_window, width=720, height=720, bg='black', highlightthickness=0)
                self.roi_canvas.pack(pady=10)
                
                # Bind mouse events
                self.roi_canvas.bind("<Button-1>", self.on_canvas_click)
                self.roi_canvas.bind("<B1-Motion>", self.on_canvas_drag)
                self.roi_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
                
                # Info label
                info = ttk.Label(self.roi_setting_window, text="🎯 Kéo chuột để chọn VÙNG NHẬN DIỆN vật (hình vuông, bội số 32px)", 
                                style='Dark.TLabel', foreground='cyan', font=('Segoe UI', 10, 'bold'))
                info.pack(pady=5)
            
            self.btn_set_area.config(text="🎯 Setting... (Active)")
            self.status_label.config(text="🎯 Kéo chuột trên cửa sổ ROI để chọn vùng", foreground='cyan')
        else:
            # Đóng cửa sổ ROI setting
            if self.roi_setting_window is not None and self.roi_setting_window.winfo_exists():
                self.roi_setting_window.destroy()
                self.roi_setting_window = None
            
            self.btn_set_area.config(text="🎯 Set Detection Area")
            self.status_label.config(text="🟢 Camera đã sẵn sàng", foreground='green')
    
    def on_canvas_click(self, event):
        if self.is_setting_area:
            self.roi_start = (event.x, event.y)
            if self.roi_rect_id:
                self.roi_canvas.delete(self.roi_rect_id)
    
    def on_canvas_drag(self, event):
        if self.is_setting_area and self.roi_start:
            x1, y1 = self.roi_start
            dx = event.x - x1
            dy = event.y - y1
            
            # Tính kích thước hình vuông, LÀM TRÒN về BỘI SỐ CỦA 32
            raw_size = min(abs(dx), abs(dy))
            size = (raw_size // 32) * 32  # Làm tròn xuống bội số 32
            if size < 32:
                size = 32  # Tối thiểu 32px
            
            x2 = x1 + (size if dx >= 0 else -size)
            y2 = y1 + (size if dy >= 0 else -size)
            
            if self.roi_rect_id:
                self.roi_canvas.delete(self.roi_rect_id)
            self.roi_canvas.delete('temp_text')
            
            self.roi_rect_id = self.roi_canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline='yellow', width=2
            )
            
            self.roi_canvas.create_text(
                x1, y1 - 10,
                text=f"{size}x{size}",
                fill='yellow',
                tags='temp_text'
            )
    
    def on_canvas_release(self, event):
        if self.is_setting_area and self.roi_start:
            x1, y1 = self.roi_start
            dx = event.x - x1
            dy = event.y - y1
            
            # Làm tròn về BỘI SỐ CỦA 32 (để khớp với YOLO training)
            raw_size = min(abs(dx), abs(dy))
            size = (raw_size // 32) * 32
            if size < 64:
                size = 64  # Tối thiểu 64px (= 32*2)
            
            x2 = x1 + (size if dx >= 0 else -size)
            y2 = y1 + (size if dy >= 0 else -size)
            
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Clamp to 720x720 canvas
            x1 = max(0, min(x1, 720))
            x2 = max(0, min(x2, 720))
            y1 = max(0, min(y1, 720))
            y2 = max(0, min(y2, 720))
            
            if (x2 - x1) < 64 or (y2 - y1) < 64:
                self.status_label.config(text="⚠️ Vùng quá nhỏ! Tối thiểu 64x64px (bội số 32)", foreground='orange')
                if self.roi_rect_id:
                    self.roi_canvas.delete(self.roi_rect_id)
                self.roi_canvas.delete('temp_text')
                return
            
            self.roi = [x1, y1, x2, y2]
            
            if self.roi_rect_id:
                self.roi_canvas.delete(self.roi_rect_id)
            self.roi_canvas.delete('temp_text')
            
            self.roi_rect_id = self.roi_canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline='#00ff00', width=2
            )
            
            size = x2 - x1
            self.roi_status_label.config(
                text=f"🎯 Vùng nhận diện: ({x1},{y1}) → ({x2},{y2}) | {size}x{size}px ■", 
                foreground='lime'
            )
            self.status_label.config(text="✓ Đã chọn vùng nhận diện!", foreground='green')
            
            self.is_setting_area = False
            self.btn_set_area.config(text="🎯 Set Detection Area")
    
    def save_roi(self):
        if self.roi:
            try:
                with open('roi_config.txt', 'w') as f:
                    f.write(','.join(map(str, self.roi)))
                self.status_label.config(text="💾 Đã lưu vùng làm việc", foreground='green')
                
                # Đóng cửa sổ ROI setting và thoát chế độ setting
                if self.roi_setting_window is not None and self.roi_setting_window.winfo_exists():
                    self.roi_setting_window.destroy()
                    self.roi_setting_window = None
                self.is_setting_area = False
                self.btn_set_area.config(text="🎯 Set Detection Area")
                
            except Exception as e:
                self.status_label.config(text=f"🔴 Lỗi lưu ROI: {str(e)}", foreground='red')
        else:
            self.status_label.config(text="⚠️ Chưa chọn vùng làm việc", foreground='orange')
    
    def load_roi(self):
        try:
            with open('roi_config.txt', 'r') as f:
                data = f.read().strip()
                self.roi = list(map(int, data.split(',')))
                print(f"✓ Đã load ROI: {self.roi}")
        except:
            pass
    
    # ==================== SKELETON & MULTIPLE TIPS DETECTION ====================
    # ⚠️ LEGACY SECTION (giữ nguyên cho tính năng "🔬 XỬ LÝ & XUẤT POINT CLOUD"
    # xuất point cloud toàn cảnh + trục sinh trưởng hiện có).
    # Toàn bộ hàm từ đây tới marker "END GROWTH AXIS COMPUTATION" KHÔNG được
    # gọi bởi HYBRID 2D-3D GRASP PIPELINE mới (xem cuối file, trước __main__).
    # Pipeline mới không dùng: branch pruning 30px, retreat 4px, tip merge
    # 40px/20mm, centerline slicing, cone filter 85° kiểu cũ, picking-frame
    # kiểu cũ, v.v. theo đúng yêu cầu NOTE_Codex_Phuong_phap_MethodsX.
    # ==============================================================================
    
    def preprocess_mask_binary(self, mask):
        """
        Preprocess mask to binary {0,1} and apply morphological closing.
        
        Args:
            mask: 720x720 uint8 array, values {0,255} or {0,1}
        
        Returns:
            mask01: uint8 array {0,1}
        """
        # Convert to {0,1}
        if mask.max() > 1:
            mask01 = (mask > 127).astype(np.uint8)
        else:
            mask01 = mask.astype(np.uint8)
        
        # Morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel)
        
        # Remove small connected components (< 100 pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask01, connectivity=8)
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                mask01[labels == i] = 0
        
        return mask01
    
    def compute_skeleton(self, mask01):
        """
        Compute morphological skeleton using skeletonize.
        
        Args:
            mask01: uint8 array {0,1}
        
        Returns:
            skel_bool: boolean array
        """
        skel_bool = skeletonize(mask01.astype(bool))
        return skel_bool
    
    def find_endpoints(self, skel_bool):
        """
        Find endpoints on skeleton (pixels with exactly 1 neighbor).
        
        Args:
            skel_bool: boolean skeleton array
        
        Returns:
            endpoints_rc: (M, 2) array of (row, col) coordinates
            junctions_rc: (K, 2) array of junction points (>=3 neighbors)
        """
        # Convert to uint8
        skel_uint8 = skel_bool.astype(np.uint8)
        
        # Count 8-neighbors using convolution
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skel_uint8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Mask to only skeleton pixels
        neighbor_count = neighbor_count * skel_uint8
        
        # Endpoints: neighbor_count == 1
        endpoints_mask = (neighbor_count == 1) & skel_bool
        endpoints_rc = np.column_stack(np.where(endpoints_mask))
        
        # Junctions: neighbor_count >= 3
        junctions_mask = (neighbor_count >= 3) & skel_bool
        junctions_rc = np.column_stack(np.where(junctions_mask))
        
        return endpoints_rc, junctions_rc
    
    def retreat_endpoints_inward(self, endpoints_rc, skel_bool, retreat_pixels=5):
        """
        Move endpoints inward along skeleton to ensure better 3D mapping.
        Endpoints at mask edge may have sparse 3D points, moving inward helps.
        
        Args:
            endpoints_rc: (M, 2) array of (row, col) endpoint coordinates
            skel_bool: boolean skeleton array
            retreat_pixels: int, number of pixels to retreat inward (default 5)
        
        Returns:
            retreated_endpoints: (M, 2) array of retreated (row, col) coordinates
        """
        retreated = []
        
        for start_r, start_c in endpoints_rc:
            # BFS to find skeleton path from endpoint inward
            from collections import deque
            
            visited = np.zeros_like(skel_bool, dtype=bool)
            queue = deque([(start_r, start_c, 0)])  # (r, c, distance)
            visited[start_r, start_c] = True
            
            retreated_rc = (start_r, start_c)  # Default: keep original
            
            # 8-connectivity offsets
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
            
            while queue:
                r, c, dist = queue.popleft()
                
                # If reached retreat distance, use this point
                if dist >= retreat_pixels:
                    retreated_rc = (r, c)
                    break
                
                # Explore neighbors
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < skel_bool.shape[0] and 
                        0 <= nc < skel_bool.shape[1] and
                        skel_bool[nr, nc] and 
                        not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc, dist + 1))
            
            retreated.append(retreated_rc)
        
        return np.array(retreated)
    
    def find_fallback_tip_on_skeleton(self, skel_bool, junction_rc, tip_rc, points_np, pixel_mapping, min_z, max_retreat_px=50):
        """
        Find a fallback tip on skeleton path when original tip mapping fails.
        Walk backward from tip toward junction until finding a valid 3D point above min_z.
        
        Args:
            skel_bool: boolean skeleton array
            junction_rc: (row, col) junction position
            tip_rc: (row, col) original tip position (failed)
            points_np: (N, 3) 3D points
            pixel_mapping: (N, 2) pixel mapping
            min_z: minimum Z threshold (junction Z)
            max_retreat_px: maximum pixels to retreat (default 50px)
        
        Returns:
            fallback_tip_3d: (3,) 3D point or None if no valid point found
            fallback_tip_2d: (u, v) 2D pixel or None
        """
        from collections import deque
        
        # BFS from tip toward junction
        visited = np.zeros_like(skel_bool, dtype=bool)
        queue = deque([(tip_rc[0], tip_rc[1], 0)])  # (r, c, distance)
        visited[tip_rc[0], tip_rc[1]] = True
        
        # 8-connectivity offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
        
        candidates = []  # List of (distance, 3D_point, 2D_pixel)
        
        while queue and len(candidates) < 10:  # Try up to 10 candidates
            r, c, dist = queue.popleft()
            
            if dist > max_retreat_px:
                break
            
            # Try to map this pixel to 3D
            u, v = c, r
            tip3d_candidate = self.map_tip_to_3d_robust(u, v, points_np, pixel_mapping, min_z=min_z)
            
            if tip3d_candidate is not None:
                # Check if this point is above min_z (valid)
                if tip3d_candidate[2] >= min_z - 0.003:  # 3mm tolerance
                    candidates.append((dist, tip3d_candidate, (u, v)))
            
            # Continue searching
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < skel_bool.shape[0] and 
                    0 <= nc < skel_bool.shape[1] and
                    skel_bool[nr, nc] and 
                    not visited[nr, nc]):
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))
        
        # Return closest valid candidate
        if len(candidates) > 0:
            candidates.sort(key=lambda x: x[0])  # Sort by distance (closest first)
            dist, tip3d, tip2d = candidates[0]
            print(f"            🔄 Found fallback tip at {dist}px from original (Z={tip3d[2]:.4f})")
            return tip3d, tip2d
        
        return None, None
    
    def find_base_pixel_on_skeleton(self, skel_bool, root_mask01):
        """
        Find base pixel on skeleton closest to root region (2D distance transform).
        
        Args:
            skel_bool: boolean skeleton array
            root_mask01: uint8 root mask {0,1}
        
        Returns:
            base_rc: (row, col) tuple or None
        """
        if root_mask01.sum() == 0:
            # No root region, use bottom-center of skeleton
            skel_points = np.column_stack(np.where(skel_bool))
            if len(skel_points) == 0:
                return None
            # Find point with max row (bottom)
            max_row_idx = np.argmax(skel_points[:, 0])
            return tuple(skel_points[max_row_idx])
        
        # Compute distance transform from root
        dist_from_root = cv2.distanceTransform((1 - root_mask01), cv2.DIST_L2, 5)
        
        # Get distances at skeleton pixels
        skel_points = np.column_stack(np.where(skel_bool))
        if len(skel_points) == 0:
            return None
        
        skel_distances = dist_from_root[skel_points[:, 0], skel_points[:, 1]]
        
        # Find skeleton pixel with minimum distance to root
        min_idx = np.argmin(skel_distances)
        base_rc = tuple(skel_points[min_idx])
        
        return base_rc
    
    def find_base_pixel_3d_based(self, skel_bool, root_mask01, points_np, pixel_mapping):
        """
        Find base pixel using TRUE 3D Euclidean distance to root center.
        More accurate when stem is curved along Z-axis.
        
        Args:
            skel_bool: boolean skeleton array
            root_mask01: uint8 root mask {0,1}
            points_np: (N, 3) numpy array of 3D points
            pixel_mapping: (N, 2) numpy array of pixel coordinates
        
        Returns:
            (base_rc, success_flag): tuple of ((row, col) or None, bool)
                success_flag=True if 3D method succeeded, False if fallback needed
        """
        # 1. Compute root center in 3D
        root_pixels = np.column_stack(np.where(root_mask01 > 0))
        if len(root_pixels) == 0:
            # No root pixels, need fallback
            return None, False
        
        # Sample up to 100 root pixels to compute center
        sample_size = min(100, len(root_pixels))
        root_sample = root_pixels[np.random.choice(len(root_pixels), sample_size, replace=False)]
        
        root_points_3d = []
        for r, c in root_sample:
            pt_3d = self.pixel_to_3d_from_mapping(c, r, points_np, pixel_mapping, search_win=3)
            if pt_3d is not None:
                root_points_3d.append(pt_3d)
        
        if len(root_points_3d) < 5:
            print(f"      ⚠️ Too few root 3D points ({len(root_points_3d)}), fallback to 2D")
            return None, False
        
        root_center_3d = np.mean(root_points_3d, axis=0)
        
        # 2. Get all skeleton pixels
        skel_pixels = np.column_stack(np.where(skel_bool))
        if len(skel_pixels) == 0:
            return None, False
        
        # 3. Map skeleton pixels to 3D and compute distances
        skel_points_3d = []
        skel_pixels_valid = []
        
        for r, c in skel_pixels:
            pt_3d = self.pixel_to_3d_from_mapping(c, r, points_np, pixel_mapping, search_win=3)
            if pt_3d is not None:
                skel_points_3d.append(pt_3d)
                skel_pixels_valid.append((r, c))
        
        if len(skel_points_3d) < 5:
            print(f"      ⚠️ Too few skeleton 3D points ({len(skel_points_3d)}), fallback to 2D")
            return None, False
        
        # 4. Find skeleton point with MINIMUM 3D DISTANCE to root center
        skel_points_3d = np.array(skel_points_3d)
        distances_3d = np.linalg.norm(skel_points_3d - root_center_3d, axis=1)
        min_idx = np.argmin(distances_3d)
        
        base_rc = skel_pixels_valid[min_idx]
        base_dist_mm = distances_3d[min_idx] * 1000
        
        print(f"      ✅ Base (3D mode): pixel=({base_rc[1]},{base_rc[0]}), distance={base_dist_mm:.1f}mm")
        
        return base_rc, True
    
    def map_3d_point_to_2d(self, point_3d, points_np, pixel_mapping):
        """
        Map a 3D point back to 2D pixel coordinates using nearest neighbor search.
        
        Args:
            point_3d: (3,) numpy array of 3D coordinates
            points_np: (N, 3) numpy array of all 3D points
            pixel_mapping: (N, 2) numpy array of pixel coordinates
        
        Returns:
            (u, v): tuple of pixel coordinates or None
        """
        if len(points_np) == 0:
            return None
        
        # Find nearest neighbor in 3D space
        distances = np.linalg.norm(points_np - point_3d, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Return corresponding 2D pixel
        if nearest_idx < len(pixel_mapping):
            px, py = pixel_mapping[nearest_idx]
            return (int(px), int(py))
        
        return None
    
    def bfs_geodesic_distance(self, skel_bool, base_rc):
        """
        Compute geodesic distance along skeleton from base using BFS.
        
        Args:
            skel_bool: boolean skeleton array
            base_rc: (row, col) starting point
        
        Returns:
            dist_map: int array, distance in pixels (-1 for unreachable)
        """
        h, w = skel_bool.shape
        dist_map = np.full((h, w), -1, dtype=np.int32)
        
        if base_rc is None:
            return dist_map
        
        # BFS
        queue = deque([base_rc])
        dist_map[base_rc[0], base_rc[1]] = 0
        
        # 8-connectivity offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
        
        while queue:
            r, c = queue.popleft()
            current_dist = dist_map[r, c]
            
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                
                # Check if skeleton pixel and not visited
                if skel_bool[nr, nc] and dist_map[nr, nc] == -1:
                    dist_map[nr, nc] = current_dist + 1
                    queue.append((nr, nc))
        
        return dist_map
    
    def prune_endpoints_by_length(self, endpoints_rc, dist_map, min_len_px=30):
        """
        Keep only endpoints with sufficient geodesic distance from base.
        
        Args:
            endpoints_rc: (M, 2) array of endpoint coordinates
            dist_map: geodesic distance map from base
            min_len_px: minimum length threshold (default 30px)
        
        Returns:
            endpoints_kept: list of (row, col) tuples
        """
        if len(endpoints_rc) == 0:
            return []
        
        max_dist = dist_map.max()
        if max_dist <= 0:
            return []
        
        # Threshold: max of min_len_px or 10% of max distance
        threshold = max(min_len_px, int(0.10 * max_dist))
        
        endpoints_kept = []
        for rc in endpoints_rc:
            r, c = rc
            dist = dist_map[r, c]
            if dist >= threshold:
                endpoints_kept.append((r, c))
        
        return endpoints_kept
    
    def pixel_to_3d_from_mapping(self, u, v, points_np, pixel_mapping, search_win=5, fallback_to_nearest=True):
        """
        Map 2D pixel (u,v) to 3D point using pre-computed pixel mapping.
        
        Args:
            u, v: pixel coordinates (col, row) in 720x720
            points_np: (N, 3) array of 3D points
            pixel_mapping: (N, 2) array of [px, py] for each point
            search_win: search window size if exact match not found
            fallback_to_nearest: if True, find nearest point in cloud if no mapping found
        
        Returns:
            tip3d: (3,) array or None
        """
        # Exact match
        match = (pixel_mapping[:, 0] == u) & (pixel_mapping[:, 1] == v)
        
        if np.sum(match) > 0:
            tip3d = np.mean(points_np[match], axis=0)
            return tip3d
        
        # Search in neighborhood
        best_dist = float('inf')
        best_points = None
        
        for du in range(-search_win//2, search_win//2 + 1):
            for dv in range(-search_win//2, search_win//2 + 1):
                nu, nv = u + du, v + dv
                if nu < 0 or nu >= 720 or nv < 0 or nv >= 720:
                    continue
                
                match_near = (pixel_mapping[:, 0] == nu) & (pixel_mapping[:, 1] == nv)
                
                if np.sum(match_near) > 0:
                    dist_sq = du*du + dv*dv
                    if dist_sq < best_dist:
                        best_dist = dist_sq
                        best_points = points_np[match_near]
        
        if best_points is not None:
            return np.mean(best_points, axis=0)
        
        # 🔥 FALLBACK: Nếu không tìm thấy trong window, tìm điểm gần nhất trong cloud
        # Giúp tránh vẽ vào vùng không có point cloud
        if fallback_to_nearest and len(points_np) > 0:
            # Ước lượng vị trí 3D từ pixel (approximate projection)
            # Lấy median Z của cloud để ước lượng
            median_z = np.median(points_np[:, 2])
            
            # Tìm các điểm có Z gần median (trong vùng object)
            z_tolerance = 0.05  # 5cm
            z_mask = np.abs(points_np[:, 2] - median_z) < z_tolerance
            
            if np.sum(z_mask) > 0:
                candidate_points = points_np[z_mask]
                candidate_mapping = pixel_mapping[z_mask]
                
                # Tìm điểm gần nhất về mặt 2D pixel
                distances_2d = np.sqrt((candidate_mapping[:, 0] - u)**2 + 
                                      (candidate_mapping[:, 1] - v)**2)
                
                if len(distances_2d) > 0:
                    nearest_idx = np.argmin(distances_2d)
                    nearest_dist = distances_2d[nearest_idx]
                    
                    # Chỉ accept nếu < 50px
                    if nearest_dist < 50:
                        return candidate_points[nearest_idx]
        
        return None
        
        return None
    
    def map_tip_to_3d_robust(self, u, v, points_np, pixel_mapping, stem_points_transformed=None, min_z=None):
        """
        Robust tip mapping from 2D pixel to 3D point cloud.
        Ensures tip is always an ACTUAL point in the cloud, not interpolated.
        
        Strategy:
        1. Check exact pixel match → use that point
        2. Check neighborhood (5px) → use closest matched point
        3. Find nearest point in cloud (fallback) → ensures tip exists
        4. Validate Z: reject points that are too low (below base/junction)
        
        Args:
            u, v: pixel coordinates (col, row)
            points_np: (N, 3) original 3D points
            pixel_mapping: (N, 2) pixel mapping
            stem_points_transformed: (M, 3) transformed stem points (optional, for final validation)
            min_z: float, minimum acceptable Z value (reject points below this)
        
        Returns:
            tip_3d: (3,) numpy array, guaranteed to be actual point in cloud
        """
        # 1. Exact match
        match = (pixel_mapping[:, 0] == u) & (pixel_mapping[:, 1] == v)
        
        if np.sum(match) > 0:
            # Found exact match! Use the actual point(s)
            matched_points = points_np[match]
            
            # 🔥 FIX: If min_z provided, filter by RELATIVE HEIGHT (not absolute Z)
            # Points should be HIGHER than min_z (in Z direction, which may be negative)
            if min_z is not None:
                # In camera space: Z-axis points AWAY from camera (negative = closer/higher)
                # So higher objects have LESS NEGATIVE Z (or more positive)
                valid_z = matched_points[:, 2] >= min_z - 0.003  # 🔥 Add 3mm tolerance
                if np.sum(valid_z) > 0:
                    matched_points = matched_points[valid_z]
                else:
                    # 🔥 RELAX: Even if all below min_z, if difference < 10mm, use it anyway
                    z_diffs = matched_points[:, 2] - min_z
                    if np.max(z_diffs) > -0.01:  # Within 10mm below min_z
                        print(f"         ⚠️ Exact match slightly below min_z (max_diff={np.max(z_diffs)*1000:.1f}mm), using anyway...")
                        # Keep original points
                    else:
                        print(f"         ⚠️ Exact match too far below min_z (max_diff={np.max(z_diffs)*1000:.1f}mm), searching neighborhood...")
                        match = np.zeros(len(points_np), dtype=bool)  # Force to neighborhood search
            
            if np.sum(match) > 0:
                # 🔥 FIX: If multiple points at same pixel, choose the HIGHEST one (closest to camera)
                # In camera space: larger Z = closer to camera (for tips, we want the foremost point)
                if len(matched_points) == 1:
                    tip_3d = matched_points[0]
                else:
                    # Choose point with MAXIMUM Z (closest to camera, likely the actual leaf tip)
                    max_z_idx = np.argmax(matched_points[:, 2])
                    tip_3d = matched_points[max_z_idx]
                    print(f"         ℹ️  Multiple points at pixel ({u},{v}), chose highest (Z={tip_3d[2]:.4f} vs mean={np.mean(matched_points[:, 2]):.4f})")
                
                print(f"         ✅ Tip mapping: Exact match at ({u},{v}) → {len(matched_points)} point(s), Z={tip_3d[2]:.4f}")
                return tip_3d
        
        # 2. Search neighborhood (5px radius)
        candidates = []
        for du in range(-5, 6):
            for dv in range(-5, 6):
                if du == 0 and dv == 0:
                    continue
                nu, nv = u + du, v + dv
                if nu < 0 or nu >= 720 or nv < 0 or nv >= 720:
                    continue
                
                match_near = (pixel_mapping[:, 0] == nu) & (pixel_mapping[:, 1] == nv)
                if np.sum(match_near) > 0:
                    dist = np.sqrt(du*du + dv*dv)
                    near_points = points_np[match_near]
                    # 🔥 If multiple points at this pixel, use the highest one (max Z)
                    if len(near_points) == 1:
                        pt = near_points[0]
                    else:
                        max_z_idx = np.argmax(near_points[:, 2])
                        pt = near_points[max_z_idx]
                    
                    # Filter by min_z if provided
                    if min_z is None or pt[2] >= min_z:
                        candidates.append((pt, dist))
        
        if len(candidates) > 0:
            # Sort by distance first (closest pixel), then by Z descending (highest point)
            candidates.sort(key=lambda x: (x[1], -x[0][2]))
            tip_3d = candidates[0][0]
            print(f"         ✅ Tip mapping: Neighborhood match at ({u},{v}) → {candidates[0][1]:.1f}px away, Z={tip_3d[2]:.4f}")
            return tip_3d
        
        # 3. Fallback: Find nearest point in cloud
        print(f"         ⚠️ Tip mapping: No direct match at ({u},{v}), finding nearest in cloud...")
        
        # Find all points with similar pixel coordinates (within 50px)
        pixel_dists = np.sqrt((pixel_mapping[:, 0] - u)**2 + (pixel_mapping[:, 1] - v)**2)
        nearby_mask = pixel_dists < 50
        
        # Apply Z filter
        if min_z is not None:
            z_filter = points_np[:, 2] >= min_z
            nearby_mask = nearby_mask & z_filter
        
        if np.sum(nearby_mask) > 0:
            nearby_points = points_np[nearby_mask]
            nearby_pixel_dists = pixel_dists[nearby_mask]
            
            # Choose point with minimum pixel distance
            min_idx = np.argmin(nearby_pixel_dists)
            tip_3d = nearby_points[min_idx]
            print(f"         ✅ Tip mapping: Nearest point found ({nearby_pixel_dists[min_idx]:.1f}px away), Z={tip_3d[2]:.4f}")
            return tip_3d
        
        # 4. Last resort: Use any point from the cloud (shouldn't happen)
        print(f"         ❌ Tip mapping: Failed completely, using centroid as fallback")
        return np.mean(points_np, axis=0)
    
    def compute_all_tips_for_instance_mask(self, stem_mask, root_mask, points_np, pixel_mapping):
        """
        Compute multiple tips for a stem instance using skeleton analysis.
        
        Args:
            stem_mask: 720x720 uint8 stem mask {0,255}
            root_mask: 720x720 uint8 root mask {0,255}
            points_np: (N, 3) 3D points array
            pixel_mapping: (N, 2) pixel mapping [px, py]
        
        Returns:
            dict with keys:
                - base_2d: (u, v) tuple
                - tips_2d: list of (u, v, geodesic_dist) tuples
                - tips_3d: list of (x, y, z) arrays
                - dist_map: geodesic distance map (optional)
                - main_tip_index: index of main tip (max geodesic)
        """
        try:
            # Preprocess masks
            stem_mask01 = self.preprocess_mask_binary(stem_mask)
            root_mask01 = self.preprocess_mask_binary(root_mask)
            
            # Compute skeleton
            skel_bool = self.compute_skeleton(stem_mask01)
            
            if not skel_bool.any():
                print("      ⚠️ Empty skeleton, cannot compute tips")
                return None
            
            # Find endpoints
            endpoints_rc, junctions_rc = self.find_endpoints(skel_bool)
            print(f"      🔍 Skeleton: {np.sum(skel_bool)} pixels, {len(endpoints_rc)} endpoints, {len(junctions_rc)} junctions")
            
            if len(endpoints_rc) == 0:
                print("      ⚠️ No endpoints found on skeleton")
                return None
            
            # 🎯 AUTO BASE CALCULATION: Try 3D first, fallback to 2D if needed
            print(f"      🎯 Trying 3D-based base calculation...")
            base_rc, use_3d_success = self.find_base_pixel_3d_based(skel_bool, root_mask01, points_np, pixel_mapping)
            
            if not use_3d_success or base_rc is None:
                print(f"      🔄 Fallback to 2D-based base calculation...")
                base_rc = self.find_base_pixel_on_skeleton(skel_bool, root_mask01)
                use_3d_mode = False
            else:
                use_3d_mode = True
            
            if base_rc is None:
                print("      ⚠️ Cannot find base pixel")
                return None
            
            # Compute geodesic distance
            dist_map = self.bfs_geodesic_distance(skel_bool, base_rc)
            
            # Prune endpoints
            endpoints_kept = self.prune_endpoints_by_length(endpoints_rc, dist_map, min_len_px=30)
            print(f"      ✂️  Pruned: {len(endpoints_kept)}/{len(endpoints_rc)} endpoints kept (>30px or >10% max)")
            
            if len(endpoints_kept) == 0:
                print("      ⚠️ No endpoints after pruning")
                return None
            
            # 🔥 RETREAT endpoints inward for better 3D mapping
            endpoints_retreated = self.retreat_endpoints_inward(endpoints_kept, skel_bool, retreat_pixels=4)
            print(f"      🔙 Retreated {len(endpoints_retreated)} endpoints by 4px inward for better 3D mapping")
            
            # 🆕 FILTER: Remove endpoints that are too close to base (< 10px)
            base_r, base_c = base_rc
            endpoints_filtered = []
            endpoints_kept_filtered = []
            for idx, rc in enumerate(endpoints_retreated):
                r, c = rc
                dist_to_base = np.sqrt((r - base_r)**2 + (c - base_c)**2)
                if dist_to_base >= 10:  # At least 10px away from base
                    endpoints_filtered.append(rc)
                    endpoints_kept_filtered.append(endpoints_kept[idx])
                else:
                    print(f"         ⚠️ Filtered out tip at {rc} (too close to base: {dist_to_base:.1f}px)")
            
            if len(endpoints_filtered) == 0:
                print(f"      ⚠️ No valid endpoints after filtering (all too close to base)")
                return None
            
            endpoints_retreated = np.array(endpoints_filtered)
            endpoints_kept = np.array(endpoints_kept_filtered)
            
            # Convert to 2D tips (u=col, v=row)
            base_2d = (base_rc[1], base_rc[0])  # (col, row) -> (u, v)
            tips_2d = []
            
            for rc in endpoints_retreated:
                r, c = rc
                u, v = c, r
                # Use ORIGINAL endpoint's geodesic distance (before retreat)
                original_rc = endpoints_kept[len(tips_2d)]  # Match by index
                geodesic_dist = dist_map[original_rc[0], original_rc[1]]
                tips_2d.append((u, v, geodesic_dist))
            
            # 🆕 Map base to 3D using ROBUST mapping (do this FIRST to get min_z reference)
            base_3d = self.map_tip_to_3d_robust(base_2d[0], base_2d[1], points_np, pixel_mapping)
            
            # 🔥 Map to 3D using ROBUST mapping (ensures tips are actual points in cloud)
            # Initially use base Z as minimum (will update after finding junction)
            tips_3d = []
            for u, v, _ in tips_2d:
                tip3d = self.map_tip_to_3d_robust(u, v, points_np, pixel_mapping, min_z=base_3d[2])
                if tip3d is not None:
                    tips_3d.append(tip3d)
                else:
                    tips_3d.append(None)
            
            # 🆕 Nếu dùng 3D mode, lưu thêm original 3D coordinates để có thể map ngược về 2D sau khi transform
            if use_3d_mode:
                # Lưu base_3d_original trước khi transform (để map lại sau)
                base_3d_original = base_3d.copy() if base_3d is not None else None
            else:
                base_3d_original = None
            
            # 🆕 Find main tip: Use junction point (if exists) instead of farthest endpoint
            # Junction = branching point in Y-structure, this is the TRUE main axis endpoint
            main_tip_2d = None
            main_tip_3d = None
            is_junction_tip = False
            
            if len(junctions_rc) > 0:
                # 🔥 NEW LOGIC: Find junction by path intersection
                # 1. Trace path from base → each tip on skeleton
                # 2. Find common points where all paths intersect
                # 3. Choose point farthest from base (highest geodesic)
                
                print(f"\n      🆕 Finding junction by path intersection method...")
                
                # Trace path from base to each tip using gradient descent on dist_map
                def trace_path_to_tip(skel_bool, dist_map, base_rc, tip_rc):
                    """Trace path from tip back to base following skeleton"""
                    path = []
                    current = tip_rc
                    visited = set()
                    
                    # Start from tip, go downhill on dist_map until reaching base
                    while True:
                        r, c = current
                        path.append((r, c))
                        visited.add((r, c))
                        
                        # Stop if reached base or very close
                        dist_to_base = np.linalg.norm(np.array([r, c]) - np.array(base_rc))
                        if dist_to_base < 2:
                            break
                        
                        # Stop if distance is very small (near base)
                        if dist_map[r, c] < 2:
                            break
                        
                        # Find neighbor with smallest distance (gradient descent)
                        best_neighbor = None
                        min_dist = dist_map[r, c]
                        
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < skel_bool.shape[0] and 0 <= nc < skel_bool.shape[1] and
                                skel_bool[nr, nc] and (nr, nc) not in visited and
                                dist_map[nr, nc] < min_dist):
                                min_dist = dist_map[nr, nc]
                                best_neighbor = (nr, nc)
                        
                        if best_neighbor is None:
                            break  # Dead end
                        
                        current = best_neighbor
                        
                        # Safety: limit path length
                        if len(path) > 500:
                            break
                    
                    return path
                
                # Get endpoint positions
                endpoint_positions = np.array(endpoints_kept)
                
                # Trace paths from base to each tip
                paths = []
                print(f"      📍 Tracing paths from base to {len(endpoints_kept)} tips:")
                for i, tip_rc in enumerate(endpoints_kept):
                    path = trace_path_to_tip(skel_bool, dist_map, base_rc, tip_rc)
                    paths.append(set(path))
                    print(f"         Tip {i+1}: {len(path)} pixels in path")
                
                # 🎨 VISUALIZATION: Draw paths on 2D image
                # Create color image for visualization
                path_vis = cv2.cvtColor((stem_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
                # Define colors for each path (BGR format)
                colors = [
                    (0, 255, 0),     # Green
                    (255, 0, 0),     # Blue
                    (0, 255, 255),   # Yellow
                    (255, 0, 255),   # Magenta
                    (255, 255, 0),   # Cyan
                    (128, 0, 255),   # Purple
                ]
                
                # Draw each path with different color
                for i, path in enumerate(paths):
                    color = colors[i % len(colors)]
                    for r, c in path:
                        path_vis[r, c] = color
                
                # Find intersection of all paths
                if len(paths) >= 2:
                    common_points = paths[0]
                    for path in paths[1:]:
                        common_points = common_points.intersection(path)
                    
                    print(f"      🔗 Found {len(common_points)} common points in all paths")
                    
                    # 🎨 Draw common points (orange)
                    for r, c in common_points:
                        path_vis[r, c] = (0, 165, 255)  # Orange
                    
                    if len(common_points) > 0:
                        # Choose common point with highest geodesic (farthest from base)
                        best_junction = None
                        max_geodesic = -1
                        
                        for r, c in common_points:
                            geodesic = dist_map[r, c]
                            if geodesic > max_geodesic:
                                max_geodesic = geodesic
                                best_junction = (r, c)
                        
                        junction_rc = best_junction
                        
                        # 🎨 Highlight junction point (small red dot)
                        cv2.circle(path_vis, (junction_rc[1], junction_rc[0]), 3, (0, 0, 255), -1)
                        
                        print(f"      ✅ JUNCTION FOUND by path intersection: rc={junction_rc}, geodesic={max_geodesic:.1f}px")
                    else:
                        # No common points - fallback to topology-based
                        print(f"      ⚠️ No common points found, using topology fallback")
                        if len(junctions_rc) > 0:
                            junction_dists = [dist_map[r, c] for r, c in junctions_rc]
                            max_junction_idx = np.argmax(junction_dists)
                            junction_rc = junctions_rc[max_junction_idx]
                            print(f"      → Using max geodesic junction: rc={junction_rc}, geodesic={junction_dists[max_junction_idx]:.1f}px")
                        else:
                            print(f"      ❌ No junctions found at all!")
                            junction_rc = None
                else:
                    # Only 1 tip - no junction needed
                    print(f"      ⚠️ Only 1 tip found, no branching")
                    junction_rc = None
                
                # Handle junction result
                if junction_rc is not None:
                    main_tip_2d = (junction_rc[1], junction_rc[0], dist_map[junction_rc[0], junction_rc[1]])  # (u, v, geodesic)
                    main_tip_3d = self.map_tip_to_3d_robust(junction_rc[1], junction_rc[0], points_np, pixel_mapping, min_z=base_3d[2])
                    is_junction_tip = True
                    
                    print(f"      🔶 Using junction as main tip: (u={main_tip_2d[0]}, v={main_tip_2d[1]}, geodesic={main_tip_2d[2]}px)")
                    
                    # 🆕 RE-MAP LEAF TIPS với junction Z làm min_z (chính xác hơn base Z)
                    # Leaf tips phải cao hơn junction, không phải base
                    if main_tip_3d is not None:
                        print(f"      🔄 Re-mapping leaf tips with junction Z as min_z (was base Z)...")
                        tips_3d_remapped = []
                        for u, v, _ in tips_2d:
                            tip3d = self.map_tip_to_3d_robust(u, v, points_np, pixel_mapping, min_z=main_tip_3d[2])
                            if tip3d is not None:
                                tips_3d_remapped.append(tip3d)
                            else:
                                tips_3d_remapped.append(None)
                        tips_3d = tips_3d_remapped
                        print(f"      ✅ Remapped {len(tips_3d)} leaf tips with junction Z={main_tip_3d[2]:.4f}")
                    
                    # 🔥 FIX: Loại bỏ TẤT CẢ leaf tips gần junction (tránh trùng lặp)
                    junction_u, junction_v = main_tip_2d[0], main_tip_2d[1]
                    tips_to_remove = []
                    
                    for i, (tip_u, tip_v, _) in enumerate(tips_2d):
                        dist = np.sqrt((tip_u - junction_u)**2 + (tip_v - junction_v)**2)
                        if dist < 15:  # Threshold 15px
                            tips_to_remove.append(i)
                            print(f"      🗑️  Marked leaf tip {i} for removal (distance to junction: {dist:.1f}px)")
                    
                    # Remove in reverse order to preserve indices
                    for idx in sorted(tips_to_remove, reverse=True):
                        tips_2d.pop(idx)
                        tips_3d.pop(idx)
                else:
                    # No junction - use tip with max geodesic as main tip
                    if len(tips_2d) > 0:
                        max_idx = np.argmax([geo for _, _, geo in tips_2d])
                        main_tip_2d = tips_2d[max_idx]
                        main_tip_3d = tips_3d[max_idx]
                        is_junction_tip = False
                        print(f"      🔷 No junction, using tip with max geodesic: (u={main_tip_2d[0]}, v={main_tip_2d[1]}, geodesic={main_tip_2d[2]}px)")
                        
                        # Remove this tip from the list
                        tips_2d.pop(max_idx)
                        tips_3d.pop(max_idx)
                    else:
                        print(f"      ❌ No tips found!")
                        main_tip_2d = None
                        main_tip_3d = None
                        is_junction_tip = False
                
                # 🔥 MERGE nearby tips on same branch (if 2 tips too close, keep only one)
                # This prevents duplicate tips from appearing on same branch
                if len(tips_2d) > 1:
                    print(f"      🔍 Checking for duplicate leaf tips (total: {len(tips_2d)})...")
                    
                    # Print all tips for debugging
                    for i, ((u, v, geo), tip_3d) in enumerate(zip(tips_2d, tips_3d)):
                        if tip_3d is not None:
                            print(f"         Tip {i}: 2D=({u},{v}), geo={geo}px, 3D=[{tip_3d[0]:.4f}, {tip_3d[1]:.4f}, {tip_3d[2]:.4f}]")
                        else:
                            print(f"         Tip {i}: 2D=({u},{v}), geo={geo}px, 3D=None")
                    
                    tips_to_merge = []
                    for i in range(len(tips_2d)):
                        for j in range(i + 1, len(tips_2d)):
                            tip_i_u, tip_i_v, geo_i = tips_2d[i]
                            tip_j_u, tip_j_v, geo_j = tips_2d[j]
                            
                            # Check 2D distance
                            dist_2d = np.sqrt((tip_i_u - tip_j_u)**2 + (tip_i_v - tip_j_v)**2)
                            
                            # Check 3D distance if both have valid 3D coordinates
                            dist_3d = None
                            if tips_3d[i] is not None and tips_3d[j] is not None:
                                dist_3d = np.linalg.norm(tips_3d[i] - tips_3d[j])
                            
                            # Print distances for debugging
                            print(f"         Distance between tip {i} and tip {j}: 2D={dist_2d:.1f}px, 3D={dist_3d*1000:.1f}mm" if dist_3d else f"         Distance between tip {i} and tip {j}: 2D={dist_2d:.1f}px, 3D=N/A")
                            
                            # Merge if either:
                            # 1. 2D distance < 40px (increased from 30px - same pixel region)
                            # 2. 3D distance < 20mm (increased from 15mm - actually close in space)
                            should_merge = False
                            reason = ""
                            
                            if dist_2d < 40:  # Increased threshold
                                should_merge = True
                                reason = f"2D close ({dist_2d:.1f}px < 40px)"
                            elif dist_3d is not None and dist_3d < 0.020:  # 20mm
                                should_merge = True
                                reason = f"3D close ({dist_3d*1000:.1f}mm < 20mm)"
                            
                            if should_merge:
                                # Keep the one with larger geodesic (further from base)
                                remove_idx = i if geo_i < geo_j else j
                                keep_idx = j if remove_idx == i else i
                                if remove_idx not in tips_to_merge:
                                    tips_to_merge.append(remove_idx)
                                    print(f"         🔀 MERGE: tip {i} and tip {j} ({reason}) → keeping tip {keep_idx} (geo={tips_2d[keep_idx][2]}px)")
                    
                    # Remove merged tips
                    if len(tips_to_merge) > 0:
                        for idx in sorted(tips_to_merge, reverse=True):
                            tips_2d.pop(idx)
                            tips_3d.pop(idx)
                        print(f"      ✅ Removed {len(tips_to_merge)} duplicate tips, {len(tips_2d)} tips remaining")
                    else:
                        print(f"      ✅ No duplicates found, all {len(tips_2d)} tips are distinct")
            else:
                # Fallback: use farthest endpoint if no junction found
                valid_tips = [(i, t) for i, t in enumerate(tips_2d) if tips_3d[i] is not None]
                if len(valid_tips) == 0:
                    print("      ⚠️ No valid 3D tips found")
                    return None
                
                main_tip_index = max(valid_tips, key=lambda x: x[1][2])[0]
                main_tip_2d = tips_2d[main_tip_index]
                main_tip_3d = tips_3d[main_tip_index]
                
                print(f"      ⚠️ No junction found, using farthest endpoint: index {main_tip_index} (geodesic={main_tip_2d[2]}px)")
            
            print(f"      ✅ Detected {len(tips_3d)} leaf tips + {'1 junction' if is_junction_tip else '0 junctions'}")
            
            # 🎨 Save path visualization if exists
            path_vis_to_save = path_vis if 'path_vis' in locals() else None
            
            return {
                'base_2d': base_2d,
                'base_3d': base_3d,
                'base_3d_original': base_3d_original,  # 🆕 For 3D mode remapping
                'tips_2d': tips_2d,  # Endpoints only (leaves)
                'tips_3d': tips_3d,
                'main_tip_2d': main_tip_2d,  # Junction or farthest endpoint
                'main_tip_3d': main_tip_3d,
                'is_junction_tip': is_junction_tip,
                'dist_map': dist_map,
                'skel_bool': skel_bool,
                'use_3d_base': use_3d_mode,  # 🆕 Flag để biết đã dùng mode nào (True=3D, False=2D)
                'path_vis': path_vis_to_save  # 🎨 Visualization of paths and junction
            }
            
        except Exception as e:
            print(f"      ❌ Error in compute_all_tips: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== GROWTH AXIS COMPUTATION ====================
    
    def load_point_cloud(self, path):
        """Load point cloud and return points as numpy array."""
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    
    def compute_base_tip(self, points, root_center):
        """
        Compute base and tip points of a stem relative to root.
        
        Args:
            points: (N, 3) numpy array of stem points
            root_center: (3,) numpy array of root centroid
        
        Returns:
            base: (3,) point closest to root_center
            tip: (3,) point farthest from root_center
        """
        # Distance from each point to root center
        distances = np.linalg.norm(points - root_center, axis=1)
        
        # Base = closest point to root
        base_idx = np.argmin(distances)
        base = points[base_idx]
        
        # Tip = farthest point from root
        tip_idx = np.argmax(distances)
        tip = points[tip_idx]
        
        return base, tip
    
    def snap_to_nearest_point(self, target, points, max_distance=0.01, base=None, growth_direction=None):
        """
        Snap target point to nearest actual point in cloud.
        Prevents drawing to interpolated/non-existent points.
        
        🔥 IMPROVED: Prioritize points along growth direction (don't snap downward)
        
        Args:
            target: (3,) target point (may not exist in cloud)
            points: (N, 3) actual point cloud
            max_distance: max distance to accept (default 10mm)
            base: (3,) base point for direction check (optional)
            growth_direction: (3,) expected growth direction (optional)
        
        Returns:
            snapped_point: (3,) nearest point in cloud, or target if too far
        """
        if len(points) == 0:
            return target
        
        # Find nearest point
        distances = np.linalg.norm(points - target, axis=1)
        
        # 🔥 FILTER: If growth_direction provided, prefer points in forward direction
        if base is not None and growth_direction is not None:
            # Normalize direction
            growth_dir_norm = growth_direction / (np.linalg.norm(growth_direction) + 1e-9)
            
            # For each candidate, check if it's in forward direction from base
            vectors_from_base = points - base
            projections = np.dot(vectors_from_base, growth_dir_norm)
            
            # Target projection (expected distance along direction)
            target_projection = np.dot(target - base, growth_dir_norm)
            
            # Prefer points with similar or greater projection (forward)
            # Penalize points that are behind target
            forward_mask = projections >= target_projection * 0.8  # Allow 20% backward tolerance
            
            print(f"         🟢 SNAP DEBUG: target_proj={target_projection:.4f}, forward_candidates={np.sum(forward_mask)}")
            
            if np.sum(forward_mask) > 0:
                # Filter to forward points only
                candidate_indices = np.where(forward_mask)[0]
                candidate_distances = distances[forward_mask]
                
                if len(candidate_distances) > 0:
                    nearest_idx_in_candidates = np.argmin(candidate_distances)
                    nearest_idx = candidate_indices[nearest_idx_in_candidates]
                    nearest_dist = candidate_distances[nearest_idx_in_candidates]
                    
                    snapped_point = points[nearest_idx]
                    snapped_projection = projections[nearest_idx]
                    print(f"         ✅ Snapped to forward point: proj={snapped_projection:.4f}, dist={nearest_dist*1000:.1f}mm")
                    
                    if nearest_dist < max_distance:
                        return snapped_point
            else:
                # 🔥 NO FORWARD CANDIDATES: Find furthest point along direction instead
                # This prevents snapping backward/downward
                print(f"         ⚠️ No forward candidates, finding furthest point along direction...")
                
                # Find point with maximum projection (furthest along direction)
                max_proj_idx = np.argmax(projections)
                furthest_point = points[max_proj_idx]
                furthest_proj = projections[max_proj_idx]
                furthest_dist = distances[max_proj_idx]
                
                print(f"         🔵 Furthest point: proj={furthest_proj:.4f} (vs target={target_projection:.4f}), dist={furthest_dist*1000:.1f}mm")
                
                # 🔥 FIX: If target is beyond point cloud (target_proj > furthest_proj * 1.2)
                # Use furthest point instead of invalid target
                if target_projection > furthest_proj * 1.2:
                    print(f"         ⚠️ Target beyond point cloud! Using furthest point as endpoint.")
                    return furthest_point
                
                # Use furthest point if it's reasonable (within 50mm) - increased tolerance
                if furthest_dist < 0.05:  # 50mm tolerance (increased from 30mm)
                    print(f"         ✅ Using furthest point (prevents backward snap)")
                    return furthest_point
                else:
                    print(f"         ⚠️ Furthest point too far ({furthest_dist*1000:.1f}mm), using it anyway (safer than nearest)")
                    # 🔥 ALWAYS use furthest, never simple nearest (which can go backward)
                    return furthest_point
        
        # Fallback: simple nearest without direction check
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        if nearest_dist < max_distance:
            return points[nearest_idx]
        else:
            # Too far, return target as-is
            return target
    
    def rotation_matrix_to_rpy(self, R):
        """
        Convert rotation matrix to Roll-Pitch-Yaw (XYZ Euler angles in radians)
        Returns: (roll, pitch, yaw) in radians
        """
        # Check for gimbal lock
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def compute_pick_frame(self, trunk_centerline, base, junction):
        """
        Compute picking coordinate frame for robot gripper
        Y-axis: Along trunk direction (growth axis)
        Z-axis: World up (orthogonalized to Y)
        X-axis: Right direction (Y × Z)
        
        Returns: (origin, rotation_matrix, roll, pitch, yaw)
        """
        if len(trunk_centerline) < 2:
            return None, None, None, None, None
        
        # Pick point: 30% along trunk from base (lower than middle for better access)
        # Calculate cumulative distances along centerline
        cumulative_distances = [0.0]
        for i in range(1, len(trunk_centerline)):
            dist = np.linalg.norm(trunk_centerline[i] - trunk_centerline[i-1])
            cumulative_distances.append(cumulative_distances[-1] + dist)
        
        total_length = cumulative_distances[-1]
        if total_length < 1e-6:
            return None, None, None, None, None
        
        # Target: 30% of trunk length from base
        target_distance = total_length * 0.30
        
        # Find closest point on centerline to target distance
        pick_idx = 0
        min_diff = abs(cumulative_distances[0] - target_distance)
        for i in range(1, len(cumulative_distances)):
            diff = abs(cumulative_distances[i] - target_distance)
            if diff < min_diff:
                min_diff = diff
                pick_idx = i
        
        origin = trunk_centerline[pick_idx]
        
        # Y-axis: trunk direction (normalized)
        y_axis = junction - base
        y_axis_len = np.linalg.norm(y_axis)
        if y_axis_len < 1e-6:
            return None, None, None, None, None
        y_axis = y_axis / y_axis_len
        
        # Z-axis: world up [0, 0, 1], but orthogonalized to Y
        world_up = np.array([0.0, 0.0, 1.0])
        
        # Project world_up onto plane perpendicular to Y
        # z_axis = world_up - (world_up · y_axis) * y_axis
        projection = np.dot(world_up, y_axis)
        z_axis = world_up - projection * y_axis
        z_axis_len = np.linalg.norm(z_axis)
        
        if z_axis_len < 1e-6:
            # Y is already pointing up/down, use arbitrary perpendicular
            if abs(y_axis[2]) > 0.9:  # Nearly vertical
                z_axis = np.array([0.0, 1.0, 0.0])  # Use Y as up
            else:
                z_axis = np.array([0.0, 0.0, 1.0])
            # Re-orthogonalize
            projection = np.dot(z_axis, y_axis)
            z_axis = z_axis - projection * y_axis
            z_axis_len = np.linalg.norm(z_axis)
        
        z_axis = z_axis / z_axis_len
        
        # X-axis: right direction = Y × Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Build rotation matrix [X | Y | Z] (column-wise)
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to RPY
        roll, pitch, yaw = self.rotation_matrix_to_rpy(R)
        
        return origin, R, roll, pitch, yaw
    
    def compute_centerline_slicing(self, points, base, direction, step=0.002, min_pts=5, smooth_window=5,
                                   adaptive_step=True, branch_filter=True, target_point=None):
        """
        Compute 3D centerline using slicing along growth direction.
        
        🆕 Improvements:
        - Adaptive step size: smaller near tips/junctions, larger at trunk
        - Branch-specific filtering: only use points near the branch direction
        
        Args:
            points: (N, 3) numpy array of stem points
            base: (3,) base point of stem
            direction: (3,) normalized growth direction vector
            step: float, base slice thickness in meters (default 2mm)
            min_pts: int, minimum points per slice to compute centroid
            smooth_window: int, moving average window size
            adaptive_step: bool, use adaptive step size (default True)
            branch_filter: bool, filter points by proximity to branch (default True)
            target_point: (3,) target endpoint (for adaptive step), optional
        
        Returns:
            centerline_pts: (M, 3) numpy array of centerline points
        """
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        
        # 🆕 IMPROVEMENT 2: Branch-Specific Filtering
        # Filter points within a cone around the branch direction
        if branch_filter and len(points) > 0:
            # Compute angle between (point - base) and direction
            vectors = points - base
            vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm[vectors_norm < 1e-9] = 1e-9
            vectors_normalized = vectors / vectors_norm
            
            # Dot product = cos(angle)
            cos_angles = np.dot(vectors_normalized, direction)
            
            # 🔥 FIX: Relax cone angle for branches
            # 60° → 85° (cos(85°) ≈ 0.087) for better branch coverage
            # Branches can be highly angled from junction, especially horizontal ones
            cone_threshold = 0.087  # 85 degrees - very permissive for curved/horizontal branches
            cone_mask = cos_angles >= cone_threshold
            
            points_filtered = points[cone_mask]
            
            # 🔥 FIX: More lenient threshold for using all points
            if len(points_filtered) < 15:  # Lowered from 20 to 15 for better cone usage
                points_filtered = points
                print(f"         ℹ️  Branch filter: Too few points after cone ({len(points[cone_mask])}), using all {len(points)} points")
        else:
            points_filtered = points
        
        # Project all points onto growth axis: t = dot(g, p - base)
        t_values = np.dot(points_filtered - base, direction)
        
        # Filter out points behind the base (t < 0)
        valid_mask = t_values >= 0
        points_valid = points_filtered[valid_mask]
        t_valid = t_values[valid_mask]
        
        if len(t_valid) == 0:
            return np.array([base])
        
        # Determine slice range
        t_min = 0
        
        # 🔥 FIX: Limit t_max to target_point distance (don't overshoot!)
        if target_point is not None:
            # Project target onto direction to get expected t_max
            t_target = np.dot(target_point - base, direction)
            
            # 🔥 CRITICAL FIX: Validate t_target against actual point cloud
            # If t_target is beyond furthest point, it's invalid (bad mapping)
            t_max_actual = np.max(t_valid)
            
            if t_target > t_max_actual * 1.2:
                # Target is WAY beyond point cloud (>20% further)
                print(f"         🔥 WARNING: target_t={t_target:.4f} beyond cloud max={t_max_actual:.4f}")
                print(f"         🔥 CLAMPING to actual point cloud extent")
                t_max = t_max_actual * 1.02  # Only 2% overshoot from actual
            else:
                # Target is reasonable, use it with small tolerance
                t_max = min(t_max_actual, t_target * 1.05)  # Allow 5% tolerance
        else:
            t_max = np.max(t_valid)
        
        # 🆕 IMPROVEMENT 1: Adaptive Step Size
        # Compute distance to target (for adaptive stepping)
        if adaptive_step and target_point is not None:
            total_length = np.linalg.norm(target_point - base)
        else:
            total_length = t_max
        
        # Compute slices
        centerline_pts = []
        t_current = t_min
        consecutive_empty_slices = 0  # 🔥 Track empty slices
        max_empty_slices = 2  # 🔥 STRICTER: Stop if 2 consecutive empty slices (was 3)
        
        while t_current <= t_max:
            # 🆕 Adaptive step: smaller near target, larger at base
            if adaptive_step and target_point is not None:
                # Progress ratio: 0 (at base) → 1 (at target)
                progress = min(t_current / total_length, 1.0)
                
                # Step size varies: 2mm at base → 1mm near target
                # Using quadratic falloff for smooth transition
                current_step = step * (1.5 - 0.5 * progress**2)
                current_step = max(current_step, step * 0.5)  # Min 1mm
            else:
                current_step = step
            
            # Find points in current slice [t_current, t_current + current_step]
            slice_mask = (t_valid >= t_current) & (t_valid < t_current + current_step)
            slice_points = points_valid[slice_mask]
            
            if len(slice_points) >= min_pts:
                # Compute centroid of slice
                centroid = np.mean(slice_points, axis=0)
                centerline_pts.append(centroid)
                consecutive_empty_slices = 0  # 🔥 Reset counter
            else:
                consecutive_empty_slices += 1  # 🔥 Increment counter
                
                # 🔥 STOP if too many consecutive empty slices (reached end of point cloud)
                if consecutive_empty_slices >= max_empty_slices:
                    break
            
            t_current += current_step
        
        if len(centerline_pts) == 0:
            return np.array([base])
        
        centerline_pts = np.array(centerline_pts)
        
        # 🔥 POST-PROCESS 1: Validate density along centerline, trim at dropout
        # Check each point has nearby points in ORIGINAL cloud (not filtered!)
        validated_pts = []
        search_radius = step * 4.0  # 🔥 RELAX MORE: 8mm search radius (was 6mm) - better coverage near junction
        min_nearby = max(min_pts, 4)  # 🔥 RELAX MORE: Need at least 4 points (was 6) - allow sparser regions near junction
        
        print(f"         🟢 POST-PROCESS 1: Validating {len(centerline_pts)} points (radius={search_radius*1000:.1f}mm, min={min_nearby})...")
        
        # 🆕 Check if we have target_point (junction) - if yes, be more lenient near it
        has_target = target_point is not None
        
        for i, center_pt in enumerate(centerline_pts):
            # 🔥 FIX: Count nearby points in ORIGINAL cloud, not filtered
            # points_filtered may be sparse due to cone filter
            distances = np.linalg.norm(points - center_pt, axis=1)  # Changed from points_filtered to points
            nearby_count = np.sum(distances < search_radius)
            
            # 🆕 SMART THRESHOLD: Be more lenient near target (junction area can be sparse)
            if has_target:
                dist_to_target = np.linalg.norm(center_pt - target_point)
                if dist_to_target < 0.010:  # Within 10mm of target (junction area)
                    # Very lenient: only need 3 points
                    threshold = 3
                else:
                    threshold = min_nearby
            else:
                threshold = min_nearby
            
            # Need at least threshold points
            if nearby_count >= threshold:
                validated_pts.append(center_pt)
            else:
                # 🔥 DROPOUT: Stop here if we're far from base AND not near target
                if i > 2:  # Allow sparse points at very beginning
                    # 🆕 If near target, be lenient (keep point even if sparse)
                    if has_target and dist_to_target < 0.010 and nearby_count >= 3:
                        validated_pts.append(center_pt)
                        print(f"         ⚠️ Point {i}: Sparse near target/junction ({nearby_count} points, dist={dist_to_target*1000:.1f}mm)")
                    else:
                        print(f"         ❌ STOPPED at point {i}: Only {nearby_count} nearby points (need {threshold})")
                        break
                else:
                    # 🔥 At start, only keep if at least 3 points
                    if nearby_count >= 3:
                        validated_pts.append(center_pt)
                        print(f"         ⚠️ Point {i}: Sparse but near base ({nearby_count} points)")
                    else:
                        print(f"         ❌ STOPPED at point {i}: Too sparse even near base ({nearby_count} < 3)")
                        break
        
        print(f"         ✅ POST-PROCESS 1: Kept {len(validated_pts)}/{len(centerline_pts)} points")
        
        if len(validated_pts) < 2:
            return np.array([base])
        
        validated_pts = np.array(validated_pts)
        
        # 🔥 POST-PROCESS 2: Remove "falling down" points (direction reversal)
        # Ensure monotonic progress along direction
        final_pts = [validated_pts[0]]  # Always keep first point
        
        print(f"         🟢 POST-PROCESS 2: Checking {len(validated_pts)} points for direction reversal...")
        
        for i in range(1, len(validated_pts)):
            prev_pt = final_pts[-1]
            curr_pt = validated_pts[i]
            
            # Check progress along direction (should increase)
            prev_t = np.dot(prev_pt - base, direction)
            curr_t = np.dot(curr_pt - base, direction)
            
            # 🔥 Must make forward progress
            if curr_t > prev_t:
                # 🔥 CHECK 1: Z doesn't drop too much (not falling down)
                z_drop = prev_pt[2] - curr_pt[2]  # Positive if dropping
                
                # 🆕 RELAXED: Allow 5mm drops for curved branches (was 2mm)
                # Curved/horizontal branches can have larger Z drops per step
                if z_drop < 0.005:  # Not dropping, or dropping < 5mm
                    # 🔥 CHECK 2: Direction doesn't reverse (angle check)
                    # Compute local direction: prev → curr
                    local_direction = curr_pt - prev_pt
                    local_dir_norm = np.linalg.norm(local_direction)
                    
                    if local_dir_norm > 1e-6:
                        local_dir_normalized = local_direction / local_dir_norm
                        
                        # Angle with expected growth direction
                        cos_angle = np.dot(local_dir_normalized, direction)
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        
                        print(f"         🟢 Point {i}: t={curr_t:.4f}, z_drop={z_drop*1000:.1f}mm, angle={angle_deg:.1f}°, cos={cos_angle:.3f}")
                        
                        # 🔥 Must be within 90 degrees (cos > 0)
                        # 🆕 RELAXED: within 80 degrees (cos > 0.17) for curved branches (was 70°/0.3)
                        if cos_angle > 0.17:  # ~80 degrees tolerance for highly curved branches
                            final_pts.append(curr_pt)
                        else:
                            # 🔥 Direction reversed (going backward), stop
                            print(f"         ❌ STOPPED: Direction reversed at point {i} (angle={angle_deg:.1f}°)")
                            break
                    else:
                        # Points too close, keep it
                        final_pts.append(curr_pt)
                else:
                    # 🔥 Large drop detected, stop here
                    print(f"         ❌ STOPPED: Large Z drop at point {i} ({z_drop*1000:.1f}mm)")
                    break
            else:
                # 🔥 No forward progress, stop
                print(f"         ❌ STOPPED: No forward progress at point {i} (t={curr_t:.4f} <= {prev_t:.4f})")
                break
        
        print(f"         ✅ POST-PROCESS 2: Kept {len(final_pts)}/{len(validated_pts)} points")
        
        if len(final_pts) < 2:
            return np.array([base])
        
        centerline_pts = np.array(final_pts)
        
        # Smooth centerline with moving average
        if len(centerline_pts) > smooth_window and smooth_window > 1:
            smoothed = []
            half_window = smooth_window // 2
            
            for i in range(len(centerline_pts)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(centerline_pts), i + half_window + 1)
                window_pts = centerline_pts[start_idx:end_idx]
                smoothed.append(np.mean(window_pts, axis=0))
            
            centerline_pts = np.array(smoothed)
        
        return centerline_pts
    
    def compute_centerline_skeleton_guided(self, points, skel_bool, base_rc, target_rc, 
                                           pixel_mapping, points_np_original, radius_mm=3.0):
        """
        🆕 IMPROVEMENT 3: Skeleton-Guided Slicing
        Follow skeleton path instead of straight line projection.
        
        Args:
            points: (N, 3) transformed 3D stem points
            skel_bool: boolean skeleton array (720x720)
            base_rc: (row, col) base pixel on skeleton
            target_rc: (row, col) target pixel on skeleton
            pixel_mapping: (N, 2) pixel mapping (pre-voxel)
            points_np_original: (N, 3) original 3D points (pre-transform)
            radius_mm: float, radius around skeleton path to collect points (default 3mm)
        
        Returns:
            centerline_pts: (M, 3) numpy array of centerline points (transformed space)
        """
        try:
            # 1. Find skeleton path from base to target using BFS
            from collections import deque
            
            h, w = skel_bool.shape
            visited = np.zeros((h, w), dtype=bool)
            parent = {}
            queue = deque([base_rc])
            visited[base_rc[0], base_rc[1]] = True
            parent[base_rc] = None
            
            # BFS to find path
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
            
            found = False
            while queue and not found:
                r, c = queue.popleft()
                
                if (r, c) == target_rc:
                    found = True
                    break
                
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < h and 0 <= nc < w and 
                        skel_bool[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        parent[(nr, nc)] = (r, c)
                        queue.append((nr, nc))
            
            if not found:
                # Fallback: return empty
                return np.array([])
            
            # 2. Reconstruct path from target to base
            path = []
            current = target_rc
            while current is not None:
                path.append(current)
                current = parent.get(current)
            
            path.reverse()  # base → target
            
            if len(path) < 2:
                return np.array([])
            
            # 3. For each skeleton point, collect nearby 3D points and compute centroid
            centerline_pts = []
            radius_m = radius_mm / 1000.0  # Convert to meters
            
            # Sample path (not every pixel, too dense)
            sample_step = max(1, len(path) // 20)  # ~20 samples
            path_sampled = path[::sample_step]
            if path[-1] not in path_sampled:
                path_sampled.append(path[-1])  # Ensure target is included
            
            for r, c in path_sampled:
                # Map skeleton pixel to 3D (using original points)
                u, v = c, r
                
                # Find 3D point at this skeleton location
                skeleton_pt_3d = self.pixel_to_3d_from_mapping(
                    u, v, points_np_original, pixel_mapping, search_win=3
                )
                
                if skeleton_pt_3d is None:
                    continue
                
                # Collect points within radius (in transformed space)
                # Need to transform skeleton_pt_3d first
                if hasattr(self, 'R_matrix_transform') and self.R_matrix_transform is not None:
                    skeleton_pt_transformed = (
                        self.R_matrix_transform @ skeleton_pt_3d + self.translation_vector
                    )
                else:
                    skeleton_pt_transformed = skeleton_pt_3d
                
                # Find nearby points
                distances = np.linalg.norm(points - skeleton_pt_transformed, axis=1)
                nearby_mask = distances < radius_m
                
                if np.sum(nearby_mask) >= 3:
                    nearby_points = points[nearby_mask]
                    centroid = np.mean(nearby_points, axis=0)
                    centerline_pts.append(centroid)
                else:
                    # Use skeleton point itself if no nearby points
                    centerline_pts.append(skeleton_pt_transformed)
            
            if len(centerline_pts) < 2:
                return np.array([])
            
            centerline_pts = np.array(centerline_pts)
            
            # 4. Light smoothing (smaller window for skeleton-guided)
            if len(centerline_pts) > 3:
                smoothed = []
                window = 3  # Smaller window
                half_window = window // 2
                
                for i in range(len(centerline_pts)):
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(centerline_pts), i + half_window + 1)
                    window_pts = centerline_pts[start_idx:end_idx]
                    smoothed.append(np.mean(window_pts, axis=0))
                
                centerline_pts = np.array(smoothed)
            
            return centerline_pts
            
        except Exception as e:
            print(f"         ⚠️ Skeleton-guided failed: {e}, using fallback")
            return np.array([])
    
    def make_lineset_from_polyline(self, centerline_pts, color=(0, 1, 0), radius=0.0008):
        """
        Create Open3D geometry from polyline points with visible thickness.
        Uses cylinders for better visibility instead of thin lines.
        
        Args:
            centerline_pts: (N, 3) numpy array of ordered centerline points
            color: (3,) RGB color tuple (default green)
            radius: float, cylinder radius in meters (default 0.8mm for visibility)
        
        Returns:
            combined_mesh: Open3D TriangleMesh with cylinders + spheres
        """
        if len(centerline_pts) < 2:
            # Return empty mesh
            return o3d.geometry.TriangleMesh()
        
        # Create combined mesh
        combined_mesh = o3d.geometry.TriangleMesh()
        
        # Create cylinders for each segment
        for i in range(len(centerline_pts) - 1):
            start = centerline_pts[i]
            end = centerline_pts[i + 1]
            
            # Calculate cylinder parameters
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length < 1e-6:
                continue
            
            # Create cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius,
                height=length
            )
            
            # Color cylinder
            cylinder.paint_uniform_color(color)
            
            # Calculate rotation to align cylinder with segment
            # Default cylinder is along Z-axis
            z_axis = np.array([0, 0, 1])
            direction_normalized = direction / length
            
            # Rotation axis (cross product)
            rotation_axis = np.cross(z_axis, direction_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                # Rotation angle
                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))
                
                # Create rotation matrix using Rodrigues' formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                
                # Apply rotation
                cylinder.rotate(R, center=[0, 0, 0])
            
            # Translate to position (center of segment)
            cylinder.translate(start + direction / 2)
            
            # Add to combined mesh
            combined_mesh += cylinder
        
        # Add spheres at each centerline point for smooth joints
        for pt in centerline_pts:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.2)
            sphere.paint_uniform_color(color)
            sphere.translate(pt)
            combined_mesh += sphere
        
        # Compute normals for better shading
        combined_mesh.compute_vertex_normals()
        
        return combined_mesh
    
    def export_stem_with_axis(self, stem_pcd, root_pcd, out_prefix, 
                              stem_mask=None, root_mask=None, 
                              points_np=None, pixel_mapping=None,
                              R_matrix=None, translation=None,
                              slice_step=0.002, min_pts=5, smooth_window=5):
        """
        Export stem point cloud and its growth axis (with multiple tips support).
        
        Args:
            stem_pcd: Open3D PointCloud of stem instance (already transformed)
            root_pcd: Open3D PointCloud of root (class1) for reference (already transformed)
            out_prefix: str, output file prefix (e.g., "output_dir/instance_1")
            stem_mask: 720x720 uint8 stem mask (for skeleton analysis)
            root_mask: 720x720 uint8 root mask (for skeleton analysis)
            points_np: (N, 3) 3D points array (ORIGINAL - before transformation)
            pixel_mapping: (N, 2) pixel mapping array
            R_matrix: (3,3) rotation matrix for transformation
            translation: (3,) translation vector for transformation
            slice_step: float, slicing step size in meters
            min_pts: int, minimum points per slice
            smooth_window: int, smoothing window size
        
        Exports:
            - <out_prefix>_pcd.ply: stem point cloud
            - <out_prefix>_axis.ply: main growth axis
            - <out_prefix>_tips.json: multiple tips data
        """
        try:
            # 1. Export stem point cloud
            pcd_path = f"{out_prefix}_pcd.ply"
            o3d.io.write_point_cloud(pcd_path, stem_pcd)
            print(f"      ✅ Saved stem PCD: {os.path.basename(pcd_path)}")
            
            # 2. Compute root center from root point cloud
            root_points = np.asarray(root_pcd.points)
            if len(root_points) == 0:
                print(f"      ⚠️ No root points, skipping axis computation")
                return
            
            root_center = np.mean(root_points, axis=0)
            
            # 3. Get stem points
            stem_points = np.asarray(stem_pcd.points)
            if len(stem_points) < 10:
                print(f"      ⚠️ Too few stem points ({len(stem_points)}), skipping axis")
                return
            
            # 🆕 4. Try multiple tips detection if masks are provided
            tips_data = None
            if stem_mask is not None and root_mask is not None and points_np is not None and pixel_mapping is not None:
                print(f"      🌿 Computing multiple tips via skeleton analysis...")
                tips_data = self.compute_all_tips_for_instance_mask(
                    stem_mask, root_mask, points_np, pixel_mapping
                )
                
                # 🔧 Apply transformation to base and tips 3D coordinates
                if tips_data is not None and R_matrix is not None and translation is not None:
                    print(f"      🔧 Applying transformation to skeleton points...")
                    
                    # Transform base_3d
                    base_3d_orig = tips_data['base_3d']
                    base_3d_rotated = R_matrix @ base_3d_orig
                    base_3d_transformed = base_3d_rotated + translation
                    tips_data['base_3d'] = base_3d_transformed
                    
                    # Transform main_tip_3d
                    if tips_data['main_tip_3d'] is not None:
                        main_tip_orig = tips_data['main_tip_3d']
                        main_tip_rotated = R_matrix @ main_tip_orig
                        main_tip_transformed = main_tip_rotated + translation
                        tips_data['main_tip_3d'] = main_tip_transformed
                    
                    # Transform all leaf tips_3d
                    for i, tip_3d in enumerate(tips_data['tips_3d']):
                        if tip_3d is not None:
                            tip_rotated = R_matrix @ tip_3d
                            tip_transformed = tip_rotated + translation
                            tips_data['tips_3d'][i] = tip_transformed
                    
                    print(f"      ✅ Transformed base, junction, and {len([t for t in tips_data['tips_3d'] if t is not None])} leaf tips")
                
                # 🆕 Lưu tips_data để hiển thị trên preview (lấy instance_idx từ out_prefix)
                if tips_data is not None:
                    try:
                        # Extract instance index from out_prefix: "path/instance_1_class0_..." -> 1
                        import re
                        match = re.search(r'instance_(\d+)_', out_prefix)
                        if match:
                            instance_idx = int(match.group(1))
                            self.last_tips_data_per_instance[instance_idx] = tips_data
                            
                            # Save path visualization
                            if 'path_vis' in tips_data and tips_data['path_vis'] is not None:
                                self.last_path_vis_per_instance[instance_idx] = tips_data['path_vis']
                            
                            print(f"      💾 Saved tips data for instance {instance_idx} (for preview)")
                    except Exception as e:
                        print(f"      ⚠️ Could not save tips data for preview: {e}")
            
            # 5. Determine base and tip(s)
            if tips_data is not None and tips_data['main_tip_3d'] is not None:
                # Use skeleton-based multiple tips
                base = tips_data['base_3d']
                main_tip = tips_data['main_tip_3d']  # Junction or farthest endpoint
                
                # Export tips JSON
                tips_json_path = f"{out_prefix}_tips.json"
                
                # Convert all numpy types to Python native types for JSON serialization
                base_2d_list = [int(tips_data['base_2d'][0]), int(tips_data['base_2d'][1])]
                
                # Main tip (junction or farthest)
                main_tip_2d_list = [
                    int(tips_data['main_tip_2d'][0]),
                    int(tips_data['main_tip_2d'][1]),
                    int(tips_data['main_tip_2d'][2])
                ]
                main_tip_3d_list = [
                    float(tips_data['main_tip_3d'][0]),
                    float(tips_data['main_tip_3d'][1]),
                    float(tips_data['main_tip_3d'][2])
                ]
                
                # All leaf tips (endpoints)
                tips_2d_list = []
                for u, v, geo in tips_data['tips_2d']:
                    tips_2d_list.append([int(u), int(v), int(geo)])
                
                tips_3d_list = []
                for xyz in tips_data['tips_3d']:
                    if xyz is not None:
                        tips_3d_list.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                    else:
                        tips_3d_list.append(None)
                
                tips_export = {
                    'base_2d': base_2d_list,
                    'base_3d': [float(base[0]), float(base[1]), float(base[2])],
                    'main_tip_2d': main_tip_2d_list,  # Junction (or farthest endpoint)
                    'main_tip_3d': main_tip_3d_list,
                    'is_junction': bool(tips_data.get('is_junction_tip', False)),
                    'leaf_tips_2d': tips_2d_list,  # All endpoints (leaves)
                    'leaf_tips_3d': tips_3d_list,
                    'use_3d_base': bool(tips_data.get('use_3d_base', False))  # 🆕 Flag mode
                }
                
                with open(tips_json_path, 'w') as f:
                    json.dump(tips_export, f, indent=2)
                print(f"      💾 Saved tips data: {os.path.basename(tips_json_path)}")
                
                # 🎯 Vẽ trục sinh trưởng dựa trên base, junction, tips từ skeleton
                print(f"      📏 Computing growth axis from skeleton structure...")
                
                combined_axis = o3d.geometry.TriangleMesh()
                
                # Case 1: Có junction (cấu trúc chữ Y)
                if tips_data.get('is_junction_tip', False):
                    junction = main_tip  # Junction point
                    print(f"         🔶 Using junction from path intersection method")
                    print(f"            Junction 3D: [{junction[0]:.4f}, {junction[1]:.4f}, {junction[2]:.4f}]")
                    
                    # 1. Vẽ main trunk: base → junction (slicing theo point cloud)
                    trunk_direction = junction - base
                    trunk_direction_norm = np.linalg.norm(trunk_direction)
                    
                    if trunk_direction_norm > 1e-6:
                        trunk_direction = trunk_direction / trunk_direction_norm
                        
                        # Use slicing for all trunk (follow point cloud)
                        trunk_centerline = self.compute_centerline_slicing(
                            stem_points, base, trunk_direction,
                            step=slice_step, min_pts=min_pts, smooth_window=smooth_window,
                            adaptive_step=True,      # Adaptive step
                            branch_filter=True,      # Filter points in cone
                            target_point=junction    # For adaptive stepping
                        )
                        
                        # Trim centerline to junction (don't overshoot)
                        if len(trunk_centerline) >= 2:
                            filtered = []
                            for pt in trunk_centerline:
                                dist_from_base = np.linalg.norm(pt - base)
                                if dist_from_base <= trunk_direction_norm * 1.05:  # 5% tolerance
                                    filtered.append(pt)
                            
                            if len(filtered) >= 2:
                                trunk_centerline = np.array(filtered)
                            else:
                                trunk_centerline = np.array([base, junction])
                        else:
                            trunk_centerline = np.array([base, junction])
                        
                        # Ensure last point is junction with smooth interpolation
                        if len(trunk_centerline) >= 2:
                            # Check if trunk endpoint is far from junction
                            gap_distance = np.linalg.norm(trunk_centerline[-1] - junction)
                            if gap_distance > 0.002:  # >2mm gap (relaxed from 3mm)
                                # 🆕 IMPROVED: Add 4 interpolated points for smoother transition
                                last_pt = trunk_centerline[-1]
                                # Create 5-point smooth curve
                                pt1 = last_pt + (junction - last_pt) * 0.20
                                pt2 = last_pt + (junction - last_pt) * 0.40
                                pt3 = last_pt + (junction - last_pt) * 0.60
                                pt4 = last_pt + (junction - last_pt) * 0.80
                                trunk_centerline = np.vstack([trunk_centerline, pt1, pt2, pt3, pt4, junction])
                                print(f"      ✅ Smoothly connected to junction (gap was {gap_distance*1000:.1f}mm, added 5 points)")
                            else:
                                # Close enough, just add junction
                                trunk_centerline = np.vstack([trunk_centerline, junction])
                                print(f"      ✓ Trunk endpoint close to junction (gap={gap_distance*1000:.1f}mm)")
                            
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to last 6 points (near junction)
                            if len(trunk_centerline) > 6:
                                smoothed_tail = []
                                window = 3
                                start_idx = len(trunk_centerline) - 6
                                for i in range(start_idx, len(trunk_centerline)):
                                    w_start = max(start_idx, i - window//2)
                                    w_end = min(len(trunk_centerline), i + window//2 + 1)
                                    smoothed_tail.append(np.mean(trunk_centerline[w_start:w_end], axis=0))
                                trunk_centerline[start_idx:] = np.array(smoothed_tail)
                            
                            trunk_mesh = self.make_lineset_from_polyline(
                                trunk_centerline,
                                color=(1.0, 0.65, 0.0),  # Orange for trunk
                                radius=0.001
                            )
                            combined_axis += trunk_mesh
                            
                            trunk_len = np.sum([np.linalg.norm(trunk_centerline[i+1] - trunk_centerline[i]) 
                                               for i in range(len(trunk_centerline)-1)])
                            print(f"      ├─ Trunk (base→junction): {len(trunk_centerline)} pts, {trunk_len*1000:.1f}mm")
                    
                    # 2. Vẽ leaf branches: junction → mỗi leaf tip (slicing ĐỘC LẬP)
                    valid_branches = 0
                    for i, leaf_tip_3d in enumerate(tips_data['tips_3d']):
                        if leaf_tip_3d is None:
                            continue
                        
                        # 🔥 SNAP tip to nearest actual point in cloud (avoid interpolated points)
                        # Pass base and expected direction for smart snapping
                        expected_direction = leaf_tip_3d - junction
                        expected_direction = expected_direction / (np.linalg.norm(expected_direction) + 1e-9)
                        
                        leaf_tip_3d_snapped = self.snap_to_nearest_point(
                            leaf_tip_3d, 
                            stem_points, 
                            max_distance=0.015,  # 🔥 Increase to 15mm
                            base=junction,
                            growth_direction=expected_direction
                        )
                        
                        # Direction: junction → leaf tip
                        branch_direction = leaf_tip_3d_snapped - junction
                        branch_direction_norm = np.linalg.norm(branch_direction)
                        
                        if branch_direction_norm < 1e-6:
                            continue
                        
                        branch_direction = branch_direction / branch_direction_norm
                        
                        # 🆕 IMPROVEMENT 3: Try skeleton-guided first, fallback to slicing
                        branch_centerline = None
                        
                        # Try skeleton-guided if we have skeleton data
                        if (tips_data and 'skel_bool' in tips_data and 
                            stem_mask is not None and pixel_mapping is not None):
                            
                            # Find target pixel (leaf tip 2D)
                            leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                            leaf_rc = (leaf_v, leaf_u)
                            
                            # Find junction pixel
                            junction_u, junction_v, _ = tips_data['main_tip_2d']
                            junction_rc = (junction_v, junction_u)
                            
                            branch_centerline = self.compute_centerline_skeleton_guided(
                                stem_points,
                                tips_data['skel_bool'],
                                junction_rc,
                                leaf_rc,
                                pixel_mapping,
                                points_np,
                                radius_mm=3.0
                            )
                        
                        # Fallback to slicing if skeleton-guided failed
                        if branch_centerline is None or len(branch_centerline) < 2:
                            # 🔥 Slicing với adaptive step và branch filter
                            branch_centerline = self.compute_centerline_slicing(
                                stem_points,          # Dùng toàn bộ stem points
                                junction,             # Start từ junction
                                branch_direction,     # Direction riêng của nhánh này
                                step=slice_step,      # Step size như trunk
                                min_pts=min_pts,      # Min points như trunk
                                smooth_window=smooth_window,
                                adaptive_step=True,   # 🆕 Adaptive step
                                branch_filter=True,   # 🆕 Branch-specific filter
                                target_point=leaf_tip_3d_snapped  # 🔥 Use snapped point
                            )
                        
                        # Trim centerline to không vượt quá leaf tip
                        if len(branch_centerline) >= 2:
                            filtered = []
                            for pt in branch_centerline:
                                dist_from_junction = np.linalg.norm(pt - junction)
                                if dist_from_junction <= branch_direction_norm * 1.05:  # 🔥 Tighten: only 5% overshoot
                                    filtered.append(pt)
                            
                            if len(filtered) >= 2:
                                branch_centerline = np.array(filtered)
                            else:
                                # Fallback nếu filter quá nhiều
                                branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                        else:
                            # Fallback: straight line nếu slicing thất bại
                            branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                        
                        # 🔥 ENSURE CENTERLINE STARTS FROM JUNCTION with smooth interpolation
                        if len(branch_centerline) >= 2:
                            first_point = branch_centerline[0]
                            gap_to_junction = np.linalg.norm(first_point - junction)
                            if gap_to_junction > 0.002:  # More than 2mm gap
                                # 🆕 IMPROVED: Add 4 interpolated points for smoother transition
                                pt1 = junction + (first_point - junction) * 0.20
                                pt2 = junction + (first_point - junction) * 0.40
                                pt3 = junction + (first_point - junction) * 0.60
                                pt4 = junction + (first_point - junction) * 0.80
                                branch_centerline = np.vstack([junction, pt1, pt2, pt3, pt4, branch_centerline])
                                print(f"      ✅ Smoothly connected from junction (gap was {gap_to_junction*1000:.1f}mm, added 5 points)")
                            else:
                                # Close enough, just prepend junction
                                branch_centerline = np.vstack([junction, branch_centerline])
                                print(f"      ✓ Branch starts close to junction (gap={gap_to_junction*1000:.1f}mm)")
                        
                        # 🔥 FINAL VALIDATION: Check if centerline endpoint is diving down
                        if len(branch_centerline) >= 2:
                            final_endpoint = branch_centerline[-1]
                            z_drop_final = final_endpoint[2] - junction[2]
                            
                            # STRICT CHECK: Reject if endpoint is BELOW junction (diving down)
                            if z_drop_final < -0.010:  # More than 10mm below junction
                                print(f"      ⚠️ Branch {i}: endpoint diving down {abs(z_drop_final)*1000:.1f}mm")
                                
                                # 🆕 FALLBACK: Try to find valid tip on skeleton path
                                if tips_data and 'skel_bool' in tips_data:
                                    leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                                    leaf_rc = (leaf_v, leaf_u)
                                    junction_u, junction_v, _ = tips_data['main_tip_2d']
                                    junction_rc = (junction_v, junction_u)
                                    
                                    print(f"      🔄 Trying fallback: searching for valid tip on skeleton path...")
                                    fallback_tip_3d, fallback_tip_2d = self.find_fallback_tip_on_skeleton(
                                        tips_data['skel_bool'],
                                        junction_rc,
                                        leaf_rc,
                                        pixel_mapping,
                                        points_np,
                                        min_z=junction[2]
                                    )
                                    
                                    if fallback_tip_3d is not None:
                                        # Retry with fallback tip
                                        print(f"      ✅ Using fallback tip, retrying branch...")
                                        leaf_tip_3d_snapped = fallback_tip_3d
                                        
                                        # Recompute branch
                                        branch_direction = leaf_tip_3d_snapped - junction
                                        branch_direction_norm = np.linalg.norm(branch_direction)
                                        if branch_direction_norm > 1e-6:
                                            branch_direction = branch_direction / branch_direction_norm
                                            
                                            branch_centerline = self.compute_centerline_slicing(
                                                stem_points, junction, branch_direction,
                                                step=slice_step * 0.75,  # 🆕 Smaller step for denser sampling
                                                min_pts=max(3, min_pts - 2),  # 🆕 Lower min_pts
                                                smooth_window=smooth_window,
                                                adaptive_step=True, branch_filter=True,
                                                target_point=leaf_tip_3d_snapped
                                            )
                                            
                                            if branch_centerline is None or len(branch_centerline) < 3:
                                                # Create interpolated line (5 points) instead of straight line
                                                branch_centerline = np.array([
                                                    junction,
                                                    junction + (leaf_tip_3d_snapped - junction) * 0.25,
                                                    junction + (leaf_tip_3d_snapped - junction) * 0.50,
                                                    junction + (leaf_tip_3d_snapped - junction) * 0.75,
                                                    leaf_tip_3d_snapped
                                                ])
                                            else:
                                                # Trim and ensure starts from junction
                                                filtered = []
                                                for pt in branch_centerline:
                                                    if np.linalg.norm(pt - junction) <= branch_direction_norm * 1.05:
                                                        filtered.append(pt)
                                                if len(filtered) >= 2:
                                                    branch_centerline = np.array(filtered)
                                                else:
                                                    branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                                                
                                                # 🆕 IMPROVED: Smooth interpolation when connecting to junction
                                                if len(branch_centerline) >= 2:
                                                    first_point = branch_centerline[0]
                                                    gap_to_junction = np.linalg.norm(first_point - junction)
                                                    if gap_to_junction > 0.002:
                                                        # Add 4 interpolated points for smoother transition
                                                        pt1 = junction + (first_point - junction) * 0.20
                                                        pt2 = junction + (first_point - junction) * 0.40
                                                        pt3 = junction + (first_point - junction) * 0.60
                                                        pt4 = junction + (first_point - junction) * 0.80
                                                        branch_centerline = np.vstack([junction, pt1, pt2, pt3, pt4, branch_centerline])
                                                    else:
                                                        branch_centerline = np.vstack([junction, branch_centerline])
                                            
                                            final_endpoint = branch_centerline[-1]
                                            z_drop_final = final_endpoint[2] - junction[2]
                                            if z_drop_final < -0.010:
                                                print(f"      ❌ Fallback still invalid, skipping")
                                                continue
                                            else:
                                                print(f"      ✅ Fallback successful!")
                                        else:
                                            continue
                                    else:
                                        print(f"      ❌ No valid fallback found, skipping")
                                        continue
                                else:
                                    continue
                        
                        if len(branch_centerline) >= 2:
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to first 6 points (near junction)
                            if len(branch_centerline) > 6:
                                smoothed_head = []
                                window = 3
                                for i in range(6):
                                    w_start = max(0, i - window//2)
                                    w_end = min(6, i + window//2 + 1)
                                    smoothed_head.append(np.mean(branch_centerline[w_start:w_end], axis=0))
                                branch_centerline[:6] = np.array(smoothed_head)
                            
                            branch_mesh = self.make_lineset_from_polyline(
                                branch_centerline,
                                color=(0, 1, 0),  # Green for branches
                                radius=0.0007
                            )
                            combined_axis += branch_mesh
                            valid_branches += 1
                            
                            branch_len = np.sum([np.linalg.norm(branch_centerline[j+1] - branch_centerline[j]) 
                                               for j in range(len(branch_centerline)-1)])
                            print(f"      ├─ Branch {i} (junction→leaf): {len(branch_centerline)} pts, {branch_len*1000:.1f}mm")
                    
                    print(f"      └─ ✅ Structure: 1 trunk + {valid_branches} branches")
                    print(f"         🎯 Junction method: Path intersection (2D skeleton analysis)")
                
                # Case 2: Không có junction (cây thẳng hoặc nhiều nhánh từ base)
                else:
                    # Vẽ từng nhánh từ base → mỗi leaf tip
                    valid_branches = 0
                    
                    for i, leaf_tip_3d in enumerate(tips_data['tips_3d']):
                        if leaf_tip_3d is None:
                            continue
                        
                        # 🔥 SNAP tip to nearest actual point in cloud
                        expected_direction = leaf_tip_3d - base
                        expected_direction = expected_direction / (np.linalg.norm(expected_direction) + 1e-9)
                        
                        leaf_tip_3d_snapped = self.snap_to_nearest_point(
                            leaf_tip_3d,
                            stem_points,
                            max_distance=0.015,  # 🔥 Increase to 15mm
                            base=base,
                            growth_direction=expected_direction
                        )
                        
                        branch_direction = leaf_tip_3d_snapped - base
                        branch_direction_norm = np.linalg.norm(branch_direction)
                        
                        if branch_direction_norm < 1e-6:
                            continue
                        
                        branch_direction = branch_direction / branch_direction_norm
                        
                        # 🆕 IMPROVEMENT 3: Try skeleton-guided first
                        branch_centerline = None
                        
                        if (tips_data and 'skel_bool' in tips_data and 
                            stem_mask is not None and pixel_mapping is not None):
                            
                            leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                            leaf_rc = (leaf_v, leaf_u)
                            base_rc = (tips_data['base_2d'][1], tips_data['base_2d'][0])
                            
                            branch_centerline = self.compute_centerline_skeleton_guided(
                                stem_points,
                                tips_data['skel_bool'],
                                base_rc,
                                leaf_rc,
                                pixel_mapping,
                                points_np,
                                radius_mm=3.0
                            )
                        
                        # Fallback to slicing
                        if branch_centerline is None or len(branch_centerline) < 2:
                            # 🆕 Slicing with improvements
                            branch_centerline = self.compute_centerline_slicing(
                                stem_points, base, branch_direction,
                                step=slice_step, min_pts=min_pts, smooth_window=smooth_window,
                                adaptive_step=True,       # 🆕 Adaptive step
                                branch_filter=True,       # 🆕 Branch filter
                                target_point=leaf_tip_3d_snapped  # 🔥 Use snapped point
                            )
                        
                        # 🔥 Trim centerline to không vượt quá leaf tip
                        if len(branch_centerline) >= 2:
                            filtered = []
                            for pt in branch_centerline:
                                dist_from_base = np.linalg.norm(pt - base)
                                if dist_from_base <= branch_direction_norm * 1.05:  # 🔥 Only 5% overshoot
                                    filtered.append(pt)
                            
                            if len(filtered) >= 2:
                                branch_centerline = np.array(filtered)
                            else:
                                # Fallback nếu filter quá nhiều
                                branch_centerline = np.array([base, leaf_tip_3d_snapped])
                        else:
                            # Fallback: straight line nếu slicing thất bại
                            branch_centerline = np.array([base, leaf_tip_3d_snapped])
                        
                        # 🔥 FINAL VALIDATION: Check if centerline endpoint is diving down
                        if len(branch_centerline) >= 2:
                            final_endpoint = branch_centerline[-1]
                            z_drop_final = final_endpoint[2] - base[2]
                            
                            # RELAXED CHECK: Reject if endpoint is significantly BELOW base
                            if z_drop_final < -0.010:  # More than 10mm below base (relaxed)
                                print(f"      ⚠️ Branch {i}: endpoint diving down {abs(z_drop_final)*1000:.1f}mm, SKIPPING")
                                continue  # Skip this branch entirely
                        
                        if len(branch_centerline) >= 2:
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to first 6 points (near base)
                            if len(branch_centerline) > 6:
                                smoothed_head = []
                                window = 3
                                for i in range(6):
                                    w_start = max(0, i - window//2)
                                    w_end = min(6, i + window//2 + 1)
                                    smoothed_head.append(np.mean(branch_centerline[w_start:w_end], axis=0))
                                branch_centerline[:6] = np.array(smoothed_head)
                            
                            branch_mesh = self.make_lineset_from_polyline(
                                branch_centerline,
                                color=(0, 1, 0),  # Green
                                radius=0.0008
                            )
                            combined_axis += branch_mesh
                            valid_branches += 1
                            
                            branch_len = np.sum([np.linalg.norm(branch_centerline[j+1] - branch_centerline[j]) 
                                               for j in range(len(branch_centerline)-1)])
                            print(f"      ├─ Branch {i} (base→leaf): {len(branch_centerline)} pts, {branch_len*1000:.1f}mm")
                    
                    print(f"      └─ ✅ Structure: {valid_branches} branches from base (no junction)")
                
                # Export combined axis
                axis_path = f"{out_prefix}_axis.ply"
                o3d.io.write_triangle_mesh(axis_path, combined_axis)
                print(f"      💾 Saved growth axis: {os.path.basename(axis_path)}")
                
            else:
                # Fallback: use old method (single tip = farthest point) + slicing
                print(f"      📌 Using fallback: single tip (farthest point)")
                base, main_tip = self.compute_base_tip(stem_points, root_center)
                
                # Compute main growth direction
                direction = main_tip - base
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    print(f"      ⚠️ Base and tip are too close, skipping axis")
                    return
                direction = direction / direction_norm
                
                # Compute centerline using slicing
                centerline_pts = self.compute_centerline_slicing(
                    stem_points, base, direction, 
                    step=slice_step, min_pts=min_pts, smooth_window=smooth_window
                )
                
                if len(centerline_pts) < 2:
                    print(f"      ⚠️ Centerline has < 2 points, skipping axis export")
                    return
                
                # Create thick mesh from centerline
                combined_axis_mesh = self.make_lineset_from_polyline(centerline_pts, color=(0, 1, 0), radius=0.0008)
                
                # Export fallback axis
                axis_path = f"{out_prefix}_axis.ply"
                o3d.io.write_triangle_mesh(axis_path, combined_axis_mesh)
                
                axis_length = np.sum([np.linalg.norm(centerline_pts[i+1] - centerline_pts[i]) 
                                      for i in range(len(centerline_pts)-1)])
                print(f"      💾 Saved fallback axis: {os.path.basename(axis_path)} (length={axis_length*1000:.1f}mm)")
                
                axis_length = np.sum([np.linalg.norm(centerline_pts[i+1] - centerline_pts[i]) 
                                      for i in range(len(centerline_pts)-1)])
                
                print(f"      ✅ Saved fallback axis: {os.path.basename(axis_path)}")
                print(f"         📏 Centerline: {len(centerline_pts)} points, length={axis_length*1000:.1f}mm")
            
        except Exception as e:
            print(f"      ❌ Error exporting stem with axis: {e}")
            import traceback
            traceback.print_exc()
    
    def export_combined_with_axis(self, combined_pcd, out_prefix, 
                                  slice_step=0.002, min_pts=5, smooth_window=5):
        """
        Export combined point cloud with multiple growth axes (one per stem branch).
        🆕 IMPROVED: Uses same skeleton/junction logic as individual mode
        
        Args:
            combined_pcd: Open3D PointCloud of all points (stems + root)
            out_prefix: str, output file prefix (e.g., "output_dir/pointcloud_20260131")
            slice_step: float, slicing step size in meters
            min_pts: int, minimum points per slice
            smooth_window: int, smoothing window size
        
        Exports:
            - <out_prefix>.ply: combined point cloud
            - <out_prefix>_axis.ply: all growth axes combined (junction-based per stem)
        """
        try:
            # 1. Export combined point cloud
            pcd_path = f"{out_prefix}.ply"
            o3d.io.write_point_cloud(pcd_path, combined_pcd)
            print(f"   ✅ Saved combined PCD: {os.path.basename(pcd_path)}")
            
            # 2. Check if we have instance masks (to detect individual stems)
            if not hasattr(self, 'instance_masks') or len(self.instance_masks) == 0:
                print(f"   ⚠️ No instance masks detected, skipping axis computation")
                return pcd_path
            
            # 3. Get all points and pixel mapping
            if not hasattr(self, 'points_pre_voxel') or self.points_pre_voxel is None:
                print(f"   ⚠️ No pre-voxel points available, skipping axis")
                return pcd_path
            
            points_np = self.points_pre_voxel
            pixel_mapping = self.point_to_pixel_mapping_pre_voxel
            
            # 4. Get root mask for base calculation
            root_mask = None
            for idx, inst_class in enumerate(self.instance_classes):
                if inst_class == 1:  # Root
                    root_mask = self.instance_masks[idx]
                    break
            
            # 4.5. 🆕 Extract root points and compute root center at z=0
            root_center_at_origin = None
            root_center_3d = None          # 🆕 Actual 3D centroid of root (explicit waypoint for growth axis)
            root_points_transformed = None
            
            if root_mask is not None:
                # Extract root points (pre-transform)
                px_arr = pixel_mapping[:, 0].astype(np.int32)
                py_arr = pixel_mapping[:, 1].astype(np.int32)
                valid_coords = (px_arr >= 0) & (px_arr < 720) & (py_arr >= 0) & (py_arr < 720)
                root_filter = np.zeros(len(px_arr), dtype=bool)
                valid_px = px_arr[valid_coords]
                valid_py = py_arr[valid_coords]
                mask_values = root_mask[valid_py, valid_px]
                root_filter[valid_coords] = mask_values > 0
                
                if np.sum(root_filter) >= 10:
                    root_points_original = points_np[root_filter].copy()
                    
                    # Apply transformation
                    root_points_transformed = root_points_original.copy()
                    if self.R_matrix_transform is not None and self.translation_vector is not None:
                        root_points_transformed = (self.R_matrix_transform @ root_points_original.T).T + self.translation_vector
                    
                    # Compute root center and set z=0
                    root_center = np.mean(root_points_transformed, axis=0)
                    root_center_3d = root_center.copy()          # 🆕 Actual 3D centroid — used as waypoint in growth axis
                    root_center_at_origin = root_center.copy()
                    root_center_at_origin[2] = 0.0  # Set z to 0 (origin point)
                    
                    print(f"   🌱 Root center (3D): [{root_center_3d[0]:.4f}, {root_center_3d[1]:.4f}, {root_center_3d[2]:.4f}]")
                    print(f"   🌱 Root center at origin (z=0): [{root_center_at_origin[0]:.4f}, {root_center_at_origin[1]:.4f}, {root_center_at_origin[2]:.4f}]")
            
            # 5. Compute axis for each stem (class0) instance
            print(f"   🌿 Computing growth axes for each stem branch (with skeleton analysis)...")
            all_axes_meshes = []
            axis_count = 0
            
            for inst_idx, (inst_mask, inst_class) in enumerate(zip(self.instance_masks, self.instance_classes), start=1):
                if inst_class != 0:  # Skip non-stem
                    continue
                
                print(f"      🌿 Stem {inst_idx}: Analyzing...")
                
                # Extract stem points
                px_arr = pixel_mapping[:, 0].astype(np.int32)
                py_arr = pixel_mapping[:, 1].astype(np.int32)
                valid_coords = (px_arr >= 0) & (px_arr < 720) & (py_arr >= 0) & (py_arr < 720)
                instance_filter = np.zeros(len(px_arr), dtype=bool)
                valid_px = px_arr[valid_coords]
                valid_py = py_arr[valid_coords]
                mask_values = inst_mask[valid_py, valid_px]
                instance_filter[valid_coords] = mask_values > 0
                
                if np.sum(instance_filter) < 10:
                    continue
                
                stem_points_original = points_np[instance_filter].copy()
                stem_pixel_mapping = pixel_mapping[instance_filter].copy()
                
                # Apply transformation
                stem_points = stem_points_original.copy()
                if self.R_matrix_transform is not None and self.translation_vector is not None:
                    stem_points = (self.R_matrix_transform @ stem_points.T).T + self.translation_vector
                
                # 🆕 USE SKELETON ANALYSIS (same as individual mode)
                tips_data = self.compute_all_tips_for_instance_mask(
                    inst_mask,
                    root_mask if root_mask is not None else np.zeros((720, 720), dtype=np.uint8),
                    points_np,
                    pixel_mapping
                )
                
                # Store original data for skeleton-guided slicing
                # 🔥 FIX: Use FILTERED data (only this instance) instead of full scene
                # to avoid mapping errors when stems overlap
                stem_pixel_mapping_original = stem_pixel_mapping.copy()  # Keep for reference
                
                # 🆕 CRITICAL FIX: Use instance-specific data for accurate skeleton mapping
                # Instead of full scene, use only points belonging to this instance
                points_np_for_skeleton = stem_points_original  # Only this instance's points (pre-transform)
                pixel_mapping_for_skeleton = stem_pixel_mapping  # Only this instance's pixel mapping
                
                print(f"         🔧 Using instance-filtered data: {len(points_np_for_skeleton)} points (vs full scene: {len(points_np)})")
                
                if tips_data is None:
                    print(f"         ⚠️ No tips detected, skipping")
                    continue
                
                # 🆕 Save tips_data for preview display
                self.last_tips_data_per_instance[inst_idx] = tips_data
                
                # Save path visualization
                if 'path_vis' in tips_data and tips_data['path_vis'] is not None:
                    self.last_path_vis_per_instance[inst_idx] = tips_data['path_vis']
                
                print(f"         💾 Saved tips data for instance {inst_idx} (for preview)")
                
                # Transform tips to world coordinates
                base = tips_data['base_3d']
                main_tip = tips_data['main_tip_3d']
                
                if base is None or main_tip is None:
                    print(f"         ⚠️ Invalid base/tip, skipping")
                    continue
                
                # Apply transformation to tips
                if self.R_matrix_transform is not None and self.translation_vector is not None:
                    base = self.R_matrix_transform @ base + self.translation_vector
                    main_tip = self.R_matrix_transform @ main_tip + self.translation_vector
                    
                    for i, tip_3d in enumerate(tips_data['tips_3d']):
                        if tip_3d is not None:
                            tips_data['tips_3d'][i] = self.R_matrix_transform @ tip_3d + self.translation_vector
                
                # 🆕 DRAW AXES (same junction-based logic as individual mode)
                combined_axis = o3d.geometry.TriangleMesh()
                
                # Case 1: Has junction (Y-structure)
                if tips_data.get('is_junction_tip', False):
                    junction = main_tip
                    print(f"         🔶 Using junction from path intersection method")
                    print(f"            Junction 3D: [{junction[0]:.4f}, {junction[1]:.4f}, {junction[2]:.4f}]")
                    
                    # 1. Trunk: base → junction
                    trunk_direction = junction - base
                    trunk_direction_norm = np.linalg.norm(trunk_direction)
                    
                    if trunk_direction_norm > 1e-6:
                        trunk_direction = trunk_direction / trunk_direction_norm
                        
                        # Use slicing for all trunk (follow point cloud)
                        trunk_centerline = self.compute_centerline_slicing(
                            stem_points, base, trunk_direction,
                            step=slice_step, min_pts=min_pts, smooth_window=smooth_window,
                            adaptive_step=True,
                            branch_filter=True,
                            target_point=junction
                        )
                        
                        # Trim to junction
                        if len(trunk_centerline) >= 2:
                            filtered = []
                            for pt in trunk_centerline:
                                if np.linalg.norm(pt - base) <= trunk_direction_norm * 1.05:
                                    filtered.append(pt)
                            if len(filtered) >= 2:
                                trunk_centerline = np.array(filtered)
                            else:
                                trunk_centerline = np.array([base, junction])
                        else:
                            trunk_centerline = np.array([base, junction])
                        
                        # 🔥 ENSURE TRUNK ENDS AT JUNCTION with smooth interpolation
                        if len(trunk_centerline) >= 2:
                            last_pt = trunk_centerline[-1]
                            gap_distance = np.linalg.norm(last_pt - junction)
                            
                            # 🔧 RELAXED: Connect if gap > 2mm with smooth transition
                            if gap_distance > 0.002:  # Gap > 2mm
                                # 🆕 IMPROVED: Add 4 interpolated points for smoother transition
                                pt1 = last_pt + (junction - last_pt) * 0.20
                                pt2 = last_pt + (junction - last_pt) * 0.40
                                pt3 = last_pt + (junction - last_pt) * 0.60
                                pt4 = last_pt + (junction - last_pt) * 0.80
                                trunk_centerline = np.vstack([trunk_centerline, pt1, pt2, pt3, pt4, junction])
                                print(f"            ✅ Smoothly connected to junction (gap was {gap_distance*1000:.1f}mm, added 5 points)")
                            else:
                                # Close enough, just add junction
                                trunk_centerline = np.vstack([trunk_centerline, junction])
                                print(f"            ✓ Trunk already reaches junction (gap={gap_distance*1000:.1f}mm)")
                            
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to last 6 points (near junction)
                            if len(trunk_centerline) > 6:
                                smoothed_tail = []
                                window = 3
                                start_idx = len(trunk_centerline) - 6
                                for i in range(start_idx, len(trunk_centerline)):
                                    w_start = max(start_idx, i - window//2)
                                    w_end = min(len(trunk_centerline), i + window//2 + 1)
                                    smoothed_tail.append(np.mean(trunk_centerline[w_start:w_end], axis=0))
                                trunk_centerline[start_idx:] = np.array(smoothed_tail)
                            
                            trunk_mesh = self.make_lineset_from_polyline(
                                trunk_centerline,
                                color=(1.0, 0.65, 0.0),  # Orange
                                radius=0.0008  # Same as branches
                            )
                            
                            # Chỉ dùng tâm segmentation của gốc làm mốc tham chiếu, không vẽ trục gốc
                            if root_points_transformed is not None and root_center_3d is not None:
                                trunk_start = trunk_centerline[0]  # Base of stem (from skeleton)
                                print(f"         🌱 Root segmentation center (reference only, no root axis)")
                                print(f"            Root center : [{root_center_3d[0]:.4f}, {root_center_3d[1]:.4f}, {root_center_3d[2]:.4f}]")
                                print(f"            Trunk start : [{trunk_start[0]:.4f}, {trunk_start[1]:.4f}, {trunk_start[2]:.4f}]")
                            
                            combined_axis += trunk_mesh
                            
                            # 🤖 ROBOT PICKING FRAME: Create coordinate frame for gripper
                            if self.show_picking_frame:
                                pick_origin, pick_R, pick_roll, pick_pitch, pick_yaw = self.compute_pick_frame(
                                    trunk_centerline, base, junction
                                )
                                
                                if pick_origin is not None:
                                    roll_deg = np.degrees(pick_roll)
                                    pitch_deg = np.degrees(pick_pitch)
                                    yaw_deg = np.degrees(pick_yaw)
                                    
                                    print(f"         🤖 PICKING FRAME (Trunk):")
                                    print(f"            Origin: [{pick_origin[0]:.4f}, {pick_origin[1]:.4f}, {pick_origin[2]:.4f}]")
                                    print(f"            RPY (deg): Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°")
                                    print(f"            RPY (rad): Roll={pick_roll:.4f}, Pitch={pick_pitch:.4f}, Yaw={pick_yaw:.4f}")
                                    
                                    pick_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.010,  # 10mm
                                        origin=pick_origin
                                    )
                                    pick_frame_mesh.rotate(pick_R, center=pick_origin)
                                    combined_axis += pick_frame_mesh
                                    print(f"            ✅ Coordinate frame added (10mm size)")
                            else:
                                print(f"         ⏭️ Picking Frame: TắT (bỏ qua trục tọa độ thân)")
                    
                    # 2. Branches: junction → each leaf tip
                    valid_branches = 0
                    for i, leaf_tip_3d in enumerate(tips_data['tips_3d']):
                        if leaf_tip_3d is None:
                            continue
                        
                        print(f"         🔍 DEBUG Branch {i}:")
                        print(f"            Junction: [{junction[0]:.4f}, {junction[1]:.4f}, {junction[2]:.4f}]")
                        print(f"            Leaf tip (original): [{leaf_tip_3d[0]:.4f}, {leaf_tip_3d[1]:.4f}, {leaf_tip_3d[2]:.4f}]")
                        
                        # Snap tip
                        expected_direction = leaf_tip_3d - junction
                        expected_direction = expected_direction / (np.linalg.norm(expected_direction) + 1e-9)
                        
                        leaf_tip_3d_snapped = self.snap_to_nearest_point(
                            leaf_tip_3d,
                            stem_points,
                            max_distance=0.015,
                            base=junction,
                            growth_direction=expected_direction
                        )
                        
                        print(f"            Leaf tip (snapped): [{leaf_tip_3d_snapped[0]:.4f}, {leaf_tip_3d_snapped[1]:.4f}, {leaf_tip_3d_snapped[2]:.4f}]")
                        print(f"            Z diff (original-junction): {(leaf_tip_3d[2] - junction[2])*1000:.1f}mm")
                        print(f"            Z diff (snapped-junction): {(leaf_tip_3d_snapped[2] - junction[2])*1000:.1f}mm")
                        
                        # 🔥 REMOVED: Pre-snap validation (was rejecting valid tips)
                        # Trust snap_to_nearest_point to find best match on actual point cloud
                        
                        branch_direction = leaf_tip_3d_snapped - junction
                        branch_direction_norm = np.linalg.norm(branch_direction)
                        
                        if branch_direction_norm < 1e-6:
                            continue
                        
                        branch_direction = branch_direction / branch_direction_norm
                        
                        # 🆕 Try skeleton-guided first (same as individual mode)
                        branch_centerline = None
                        
                        if tips_data and 'skel_bool' in tips_data:
                            # Find pixels
                            leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                            leaf_rc = (leaf_v, leaf_u)
                            
                            junction_u, junction_v, _ = tips_data['main_tip_2d']
                            junction_rc = (junction_v, junction_u)
                            
                            branch_centerline = self.compute_centerline_skeleton_guided(
                                stem_points,
                                tips_data['skel_bool'],
                                junction_rc,
                                leaf_rc,
                                pixel_mapping_for_skeleton,
                                points_np_for_skeleton,
                                radius_mm=3.0
                            )
                            
                            # 🔥 Validate skeleton-guided result (check direction)
                            if branch_centerline is not None and len(branch_centerline) >= 2:
                                # Check if last segment is going backward/downward
                                last_segment = branch_centerline[-1] - branch_centerline[-2]
                                z_drop = last_segment[2]
                                if z_drop < -0.002:  # Dropping more than 2mm
                                    print(f"         ⚠️ Skeleton-guided branch {i} diving down (Z drop: {z_drop*1000:.1f}mm), using fallback slicing")
                                    branch_centerline = None
                        
                        # Fallback to slicing if skeleton-guided failed or invalid
                        if branch_centerline is None or len(branch_centerline) < 2:
                            # 🔥 IMPROVED: Try slicing with more relaxed parameters for branches
                            branch_centerline = self.compute_centerline_slicing(
                                stem_points,
                                junction,
                                branch_direction,
                                step=slice_step * 0.75,  # 🆕 Smaller step (1.5mm) for denser sampling on branches
                                min_pts=max(3, min_pts - 2),  # 🆕 Lower min_pts (3-4) for sparse branch regions
                                smooth_window=smooth_window,
                                adaptive_step=True,
                                branch_filter=True,
                                target_point=leaf_tip_3d_snapped
                            )
                        
                        # 🔥 FINAL FALLBACK: If slicing too short, create interpolated line (not straight)
                        # 🆕 RELAXED: Changed from 3 to 2 (accept shorter slicing results for curved branches)
                        if branch_centerline is None or len(branch_centerline) < 2:
                            if branch_centerline is not None and len(branch_centerline) > 0:
                                print(f"         ⚠️ Slicing too short ({len(branch_centerline)} points), using interpolated line")
                            print(f"            🔵 INTERPOLATED LINE: junction→tip_snapped (5 points)")
                            print(f"               From: [{junction[0]:.4f}, {junction[1]:.4f}, {junction[2]:.4f}]")
                            print(f"               To:   [{leaf_tip_3d_snapped[0]:.4f}, {leaf_tip_3d_snapped[1]:.4f}, {leaf_tip_3d_snapped[2]:.4f}]")
                            print(f"               Z direction: {(leaf_tip_3d_snapped[2] - junction[2])*1000:.1f}mm")
                            # 🆕 Create 5-point interpolated line instead of 2-point straight line
                            branch_centerline = np.array([
                                junction,
                                junction + (leaf_tip_3d_snapped - junction) * 0.25,
                                junction + (leaf_tip_3d_snapped - junction) * 0.50,
                                junction + (leaf_tip_3d_snapped - junction) * 0.75,
                                leaf_tip_3d_snapped
                            ])
                        
                        # Trim
                        if len(branch_centerline) >= 2:
                            filtered = []
                            for pt in branch_centerline:
                                # 🔥 CRITICAL FIX: Use snapped_tip distance (branch_direction_norm) NOT original tip distance
                                # This ensures trimming matches actual point cloud, not estimated skeleton tip
                                if np.linalg.norm(pt - junction) <= branch_direction_norm * 1.05:
                                    filtered.append(pt)
                            if len(filtered) >= 2:
                                branch_centerline = np.array(filtered)
                            else:
                                branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                        else:
                            branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                        
                        # 🔥 ENSURE CENTERLINE STARTS FROM JUNCTION with smooth interpolation
                        if len(branch_centerline) >= 2:
                            first_point = branch_centerline[0]
                            gap_to_junction = np.linalg.norm(first_point - junction)
                            if gap_to_junction > 0.002:  # More than 2mm gap
                                # 🆕 IMPROVED: Add 4 interpolated points for smoother transition
                                pt1 = junction + (first_point - junction) * 0.20
                                pt2 = junction + (first_point - junction) * 0.40
                                pt3 = junction + (first_point - junction) * 0.60
                                pt4 = junction + (first_point - junction) * 0.80
                                branch_centerline = np.vstack([junction, pt1, pt2, pt3, pt4, branch_centerline])
                                print(f"            ✅ Smoothly connected from junction (gap was {gap_to_junction*1000:.1f}mm, added 5 points)")
                            else:
                                # Close enough, just prepend junction
                                branch_centerline = np.vstack([junction, branch_centerline])
                                print(f"            ✓ Branch starts close to junction (gap={gap_to_junction*1000:.1f}mm)")
                        
                        # 🔥 FINAL VALIDATION: Check if centerline endpoint is diving down
                        if len(branch_centerline) >= 2:
                            final_endpoint = branch_centerline[-1]
                            z_drop_final = final_endpoint[2] - junction[2]
                            print(f"            Final endpoint Z: {final_endpoint[2]:.4f} (diff from junction: {z_drop_final*1000:.1f}mm)")
                            
                            # RELAXED CHECK: Reject only if endpoint is SIGNIFICANTLY BELOW junction
                            if z_drop_final < -0.010:  # More than 10mm below junction (relaxed for curved leaves)
                                print(f"            ❌ REJECTED: Branch endpoint diving down {abs(z_drop_final)*1000:.1f}mm below junction")
                                
                                # 🆕 FALLBACK: Try to find valid tip on skeleton path
                                if tips_data and 'skel_bool' in tips_data:
                                    leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                                    leaf_rc = (leaf_v, leaf_u)
                                    junction_u, junction_v, _ = tips_data['main_tip_2d']
                                    junction_rc = (junction_v, junction_u)
                                    
                                    print(f"            🔄 Trying fallback: searching for valid tip on skeleton path...")
                                    fallback_tip_3d, fallback_tip_2d = self.find_fallback_tip_on_skeleton(
                                        tips_data['skel_bool'],
                                        junction_rc,
                                        leaf_rc,
                                        points_np_for_skeleton,
                                        pixel_mapping_for_skeleton,
                                        min_z=junction[2]
                                    )
                                    
                                    if fallback_tip_3d is not None:
                                        # Retry with fallback tip
                                        print(f"            ✅ Using fallback tip, retrying branch...")
                                        leaf_tip_3d_snapped = fallback_tip_3d
                                        
                                        # Recompute branch direction
                                        branch_direction = leaf_tip_3d_snapped - junction
                                        branch_direction_norm = np.linalg.norm(branch_direction)
                                        if branch_direction_norm > 1e-6:
                                            branch_direction = branch_direction / branch_direction_norm
                                            
                                            # Retry slicing
                                            branch_centerline = self.compute_centerline_slicing(
                                                stem_points,
                                                junction,
                                                branch_direction,
                                                step=slice_step,
                                                min_pts=min_pts,
                                                smooth_window=smooth_window,
                                                adaptive_step=True,
                                                branch_filter=True,
                                                target_point=leaf_tip_3d_snapped
                                            )
                                            
                                            # Trim and ensure starts from junction
                                            if branch_centerline is None or len(branch_centerline) < 2:
                                                branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                                            else:
                                                trimming_distance = np.linalg.norm(leaf_tip_3d_snapped - junction)
                                                filtered = []
                                                for pt in branch_centerline:
                                                    if np.linalg.norm(pt - junction) <= trimming_distance * 1.05:
                                                        filtered.append(pt)
                                                if len(filtered) >= 2:
                                                    branch_centerline = np.array(filtered)
                                                else:
                                                    branch_centerline = np.array([junction, leaf_tip_3d_snapped])
                                                
                                                # Ensure starts from junction
                                                if len(branch_centerline) >= 2:
                                                    first_point = branch_centerline[0]
                                                    gap_to_junction = np.linalg.norm(first_point - junction)
                                                    if gap_to_junction > 0.002:
                                                        branch_centerline = np.vstack([junction, branch_centerline])
                                            
                                            # Final check
                                            final_endpoint = branch_centerline[-1]
                                            z_drop_final = final_endpoint[2] - junction[2]
                                            if z_drop_final >= -0.010:  # Valid now
                                                print(f"            ✅ Fallback successful! New endpoint Z: {final_endpoint[2]:.4f}")
                                            else:
                                                print(f"            ❌ Fallback still invalid, skipping branch")
                                                continue
                                        else:
                                            continue
                                    else:
                                        print(f"            ❌ No valid fallback found, skipping branch")
                                        continue
                                else:
                                    print(f"            → Skipping this branch (invalid geometry)")
                                    continue  # Skip this branch entirely
                        
                        if len(branch_centerline) >= 2:
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to first 6 points (near junction)
                            if len(branch_centerline) > 6:
                                smoothed_head = []
                                window = 3
                                for i in range(6):
                                    w_start = max(0, i - window//2)
                                    w_end = min(6, i + window//2 + 1)
                                    smoothed_head.append(np.mean(branch_centerline[w_start:w_end], axis=0))
                                branch_centerline[:6] = np.array(smoothed_head)
                            
                            branch_mesh = self.make_lineset_from_polyline(
                                branch_centerline,
                                color=(0, 1, 0),  # Green
                                radius=0.0008  # Same as trunk and root
                            )
                            combined_axis += branch_mesh
                            valid_branches += 1
                    
                    print(f"         ✅ Structure: 1 trunk + {valid_branches} branches")
                    print(f"         🎯 Junction method: Path intersection (2D skeleton analysis)")
                
                # Case 2: No junction
                else:
                    valid_branches = 0
                    for i, leaf_tip_3d in enumerate(tips_data['tips_3d']):
                        if leaf_tip_3d is None:
                            continue
                        
                        expected_direction = leaf_tip_3d - base
                        expected_direction = expected_direction / (np.linalg.norm(expected_direction) + 1e-9)
                        
                        leaf_tip_3d_snapped = self.snap_to_nearest_point(
                            leaf_tip_3d,
                            stem_points,
                            max_distance=0.015,
                            base=base,
                            growth_direction=expected_direction
                        )
                        
                        branch_direction = leaf_tip_3d_snapped - base
                        branch_direction_norm = np.linalg.norm(branch_direction)
                        
                        if branch_direction_norm < 1e-6:
                            continue
                        
                        branch_direction = branch_direction / branch_direction_norm
                        
                        # 🆕 Try skeleton-guided first
                        branch_centerline = None
                        
                        if tips_data and 'skel_bool' in tips_data:
                            leaf_u, leaf_v, _ = tips_data['tips_2d'][i]
                            leaf_rc = (leaf_v, leaf_u)
                            base_rc = (tips_data['base_2d'][1], tips_data['base_2d'][0])
                            
                            branch_centerline = self.compute_centerline_skeleton_guided(
                                stem_points,
                                tips_data['skel_bool'],
                                base_rc,
                                leaf_rc,
                                pixel_mapping_for_skeleton,
                                points_np_for_skeleton,
                                radius_mm=3.0
                            )
                            
                            # 🔥 Validate skeleton-guided result (check direction)
                            if branch_centerline is not None and len(branch_centerline) >= 2:
                                # Check if last segment is going backward/downward
                                last_segment = branch_centerline[-1] - branch_centerline[-2]
                                z_drop = last_segment[2]
                                # 🆕 RELAXED: Allow 5mm drop for curved/horizontal leaves (was 2mm)
                                if z_drop < -0.005:  # Dropping more than 5mm
                                    print(f"         ⚠️ Skeleton-guided branch {i} diving down (Z drop: {z_drop*1000:.1f}mm), using fallback slicing")
                                    branch_centerline = None
                        
                        # Fallback to slicing if skeleton-guided failed or invalid
                        if branch_centerline is None or len(branch_centerline) < 2:
                            branch_centerline = self.compute_centerline_slicing(
                                stem_points,
                                base,
                                branch_direction,
                                step=slice_step,
                                min_pts=min_pts,
                                smooth_window=smooth_window,
                                adaptive_step=True,
                                branch_filter=True,
                                target_point=leaf_tip_3d_snapped
                            )
                        
                        # 🔥 FINAL FALLBACK: If slicing also failed (< 5 points), use straight line
                        if branch_centerline is None or len(branch_centerline) < 5:
                            if branch_centerline is not None and len(branch_centerline) > 0:
                                print(f"         ⚠️ Slicing too short ({len(branch_centerline)} points), using straight line")
                            branch_centerline = np.array([base, leaf_tip_3d_snapped])
                        
                        # Trim
                        if len(branch_centerline) >= 2:
                            filtered = []
                            for pt in branch_centerline:
                                # 🔥 CRITICAL FIX: Use snapped_tip distance (branch_direction_norm) NOT original tip distance
                                if np.linalg.norm(pt - base) <= branch_direction_norm * 1.05:
                                    filtered.append(pt)
                            if len(filtered) >= 2:
                                branch_centerline = np.array(filtered)
                            else:
                                branch_centerline = np.array([base, leaf_tip_3d_snapped])
                        else:
                            branch_centerline = np.array([base, leaf_tip_3d_snapped])
                        
                        if len(branch_centerline) >= 2:
                            # 🆕 FINAL SMOOTHING: Apply local smoothing to first 6 points (near base)
                            if len(branch_centerline) > 6:
                                smoothed_head = []
                                window = 3
                                for i in range(6):
                                    w_start = max(0, i - window//2)
                                    w_end = min(6, i + window//2 + 1)
                                    smoothed_head.append(np.mean(branch_centerline[w_start:w_end], axis=0))
                                branch_centerline[:6] = np.array(smoothed_head)
                            
                            branch_mesh = self.make_lineset_from_polyline(
                                branch_centerline,
                                color=(0, 1, 0),  # Green
                                radius=0.0008
                            )
                            combined_axis += branch_mesh
                            
                            # 🤖 ROBOT PICKING FRAME: Create coordinate frame for each branch (no junction case)
                            if self.show_picking_frame:
                                pick_origin, pick_R, pick_roll, pick_pitch, pick_yaw = self.compute_pick_frame(
                                    branch_centerline, base, leaf_tip_3d_snapped
                                )
                                
                                if pick_origin is not None:
                                    roll_deg = np.degrees(pick_roll)
                                    pitch_deg = np.degrees(pick_pitch)
                                    yaw_deg = np.degrees(pick_yaw)
                                    
                                    print(f"         🤖 PICKING FRAME (Branch {valid_branches}):")
                                    print(f"            Origin: [{pick_origin[0]:.4f}, {pick_origin[1]:.4f}, {pick_origin[2]:.4f}]")
                                    print(f"            RPY (deg): Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°")
                                    print(f"            RPY (rad): Roll={pick_roll:.4f}, Pitch={pick_pitch:.4f}, Yaw={pick_yaw:.4f}")
                                    
                                    pick_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.010,  # 10mm
                                        origin=pick_origin
                                    )
                                    pick_frame_mesh.rotate(pick_R, center=pick_origin)
                                    combined_axis += pick_frame_mesh
                                    print(f"            ✅ Coordinate frame added (10mm size)")
                            else:
                                print(f"         ⏭️ Picking Frame: TắT (bỏ qua trục tọa độ thân)")
                            
                            valid_branches += 1
                    
                    print(f"         ✅ Structure: {valid_branches} branches from base")
                    
                    # Chỉ dùng tâm segmentation của gốc làm mốc tham chiếu, không vẽ trục gốc
                    if root_points_transformed is not None and root_center_3d is not None:
                        print(f"         🌱 Root segmentation center (reference only, no root axis)")
                        print(f"            Root center : [{root_center_3d[0]:.4f}, {root_center_3d[1]:.4f}, {root_center_3d[2]:.4f}]")
                        print(f"            Stem base   : [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.4f}]")
                
                if len(np.asarray(combined_axis.vertices)) > 0:
                    all_axes_meshes.append(combined_axis)
                    axis_count += 1
            
            # 7. Combine all axes into one mesh
            if len(all_axes_meshes) == 0:
                print(f"   ⚠️ No axes computed, skipping axis export")
                return pcd_path
            
            combined_axis_mesh = all_axes_meshes[0]
            for mesh in all_axes_meshes[1:]:
                combined_axis_mesh += mesh
            
            # 8. Export combined axis mesh
            axis_path = f"{out_prefix}_axis.ply"
            o3d.io.write_triangle_mesh(axis_path, combined_axis_mesh)
            
            print(f"   ✅ Saved growth axes: {os.path.basename(axis_path)}")
            print(f"      🌿 Total stems: {axis_count} (with junction/tip analysis)")
            print(f"      🎯 Junction detection: Path intersection method (same as individual mode)")
            
            return pcd_path
            
        except Exception as e:
            print(f"   ❌ Error exporting combined with axis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== END GROWTH AXIS COMPUTATION ====================
    
    def export_point_cloud_to_file(self, pcd):
        """Xuất point cloud ra file trong thư mục Output_pointcloud
        - Individual mode: N files riêng biệt cho từng mầm lan
        - Combined mode: 1 file gộp chung tất cả
        """
        try:
            # Tạo thư mục nếu chưa tồn tại
            output_dir = "Output_pointcloud"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"✅ Đã tạo thư mục: {output_dir}")
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"\n{'='*60}")
            
            # 🆕 Kiểm tra có instance masks không
            if len(self.instance_masks) == 0:
                print(f"⚠️ Không có instance masks, xuất toàn bộ point cloud")
                self.export_mode = "combined"
            
            # 🔢 INDIVIDUAL MODE: Xuất riêng từng mầm lan
            if self.export_mode == "individual" and len(self.instance_masks) > 0:
                print(f"🔢 INDIVIDUAL MODE - Xuất {len(self.instance_masks)} mầm lan riêng biệt")
                print(f"{'='*60}")
                
                # ✅ LUÔN DÙNG PRE-VOXEL DATA để mapping chính xác 100%
                if hasattr(self, 'points_pre_voxel') and self.points_pre_voxel is not None:
                    points_np = self.points_pre_voxel
                    colors_np = self.colors_pre_voxel
                    pixel_mapping = self.point_to_pixel_mapping_pre_voxel
                    print(f"✅ Sử dụng PRE-VOXEL data: {len(points_np):,} điểm với pixel mapping CHÍNH XÁC")
                    print(f"   (Không bị ảnh hưởng bởi outlier removal và voxel downsampling)")
                else:
                    # Fallback: không nên xảy ra vì pre_voxel luôn được lưu
                    print(f"❌ LỖI: Không có pre_voxel data! Export sẽ không chính xác!")
                    points_np = np.asarray(pcd.points)
                    colors_np = np.asarray(pcd.colors)
                    pixel_mapping = None  # Không có mapping chính xác
                    print(f"⚠️ Fallback: dùng post-voxel data ({len(points_np):,} điểm) - KHÔNG KHUYẾN NGHỊ")
                
                # Crop offset để map về 720x720
                h_full, w_full = 720, 1280
                start_x = (w_full - 720) // 2
                
                exported_files = []
                total_points_exported = 0
                
                # 🌱 STEP 1: Collect all ROOT (class1) points for growth axis computation
                print(f"\n🌱 Collecting root (class1) points for growth axis reference...")
                root_points_list = []
                root_colors_list = []
                
                for inst_mask, inst_class, inst_conf in zip(
                    self.instance_masks, self.instance_classes, self.instance_confidences
                ):
                    if inst_class == 1:  # Root class
                        # Filter points for this root instance
                        if pixel_mapping is not None and len(pixel_mapping) > 0:
                            px_arr = pixel_mapping[:, 0].astype(np.int32)
                            py_arr = pixel_mapping[:, 1].astype(np.int32)
                            valid_coords = (px_arr >= 0) & (px_arr < 720) & (py_arr >= 0) & (py_arr < 720)
                            instance_filter = np.zeros(len(px_arr), dtype=bool)
                            valid_px = px_arr[valid_coords]
                            valid_py = py_arr[valid_coords]
                            mask_values = inst_mask[valid_py, valid_px]
                            instance_filter[valid_coords] = mask_values > 0
                            
                            if np.sum(instance_filter) > 0:
                                root_points_list.append(points_np[instance_filter])
                                root_colors_list.append(colors_np[instance_filter])
                
                # Create root point cloud
                root_pcd = o3d.geometry.PointCloud()
                if len(root_points_list) > 0:
                    all_root_points = np.vstack(root_points_list)
                    all_root_colors = np.vstack(root_colors_list)
                    root_pcd.points = o3d.utility.Vector3dVector(all_root_points)
                    root_pcd.colors = o3d.utility.Vector3dVector(all_root_colors)
                    
                    # Apply transformation to root
                    if self.R_matrix_transform is not None and self.translation_vector is not None:
                        root_pcd.rotate(self.R_matrix_transform, center=(0, 0, 0))
                        root_pcd.translate(self.translation_vector)
                    
                    print(f"   ✅ Root PCD: {len(root_pcd.points):,} points from {len(root_points_list)} instances")
                else:
                    print(f"   ⚠️ No root (class1) instances found - growth axis will use default reference")
                
                # 🌿 STEP 2: Export each STEM (class0) with growth axis
                for idx, (inst_mask, inst_class, inst_conf) in enumerate(
                    zip(self.instance_masks, self.instance_classes, self.instance_confidences), start=1
                ):
                    print(f"\n📦 Instance {idx}/{len(self.instance_masks)}:")
                    print(f"   Class: {inst_class}, Confidence: {inst_conf:.2f}")
                    
                    # Filter points theo instance mask
                    if pixel_mapping is not None and len(pixel_mapping) > 0:
                        # Đảm bảo kích thước khớp
                        num_points = len(points_np)
                        num_mapping = len(pixel_mapping)
                        
                        if num_points != num_mapping:
                            print(f"   ⚠️ WARNING: points={num_points}, mapping={num_mapping} - KHÔNG KHỚP!")
                        
                        # Sử dụng mapping (đã được đồng bộ với points)
                        px_arr = pixel_mapping[:, 0].astype(np.int32)
                        py_arr = pixel_mapping[:, 1].astype(np.int32)
                        
                        # Valid coordinates trong 720x720
                        valid_coords = (px_arr >= 0) & (px_arr < 720) & (py_arr >= 0) & (py_arr < 720)
                        
                        # Mask filter - VECTORIZED (nhanh hơn loop)
                        instance_filter = np.zeros(len(px_arr), dtype=bool)
                        
                        # Chỉ lấy mask values tại valid coordinates
                        valid_px = px_arr[valid_coords]
                        valid_py = py_arr[valid_coords]
                        mask_values = inst_mask[valid_py, valid_px]
                        
                        # Set filter tại valid positions
                        instance_filter[valid_coords] = mask_values > 0
                        
                        num_matched = np.sum(instance_filter)
                        print(f"   🔍 Matched: {num_matched:,}/{num_points:,} điểm với instance mask")
                        
                        # Tạo point cloud cho instance này
                        if np.sum(instance_filter) > 10:
                            pcd_instance = o3d.geometry.PointCloud()
                            pcd_instance.points = o3d.utility.Vector3dVector(points_np[instance_filter])
                            pcd_instance.colors = o3d.utility.Vector3dVector(colors_np[instance_filter])
                            
                            # 🆕 ÁP DỤNG TRANSFORMATION (nếu có)
                            if self.R_matrix_transform is not None and self.translation_vector is not None:
                                # Apply rotation
                                pcd_instance.rotate(self.R_matrix_transform, center=(0, 0, 0))
                                # Apply translation
                                pcd_instance.translate(self.translation_vector)
                                print(f"   ✅ Đã apply transform: R + T")
                            else:
                                print(f"   ⚠️ Không có transform (gốc camera)")
                            
                            # 🧹 ÁP DỤNG LỌC NHIỄU (giống như combined mode)
                            original_count = len(pcd_instance.points)
                            
                            # 1️⃣ Z-offset Filtering (nếu bật)
                            if self.zoffset_filter_var.get() and hasattr(self, 'R_matrix_transform') and self.R_matrix_transform is not None:
                                # Tính z_offset từ translation_vector (đã được lưu)
                                if hasattr(self, 'translation_vector') and self.translation_vector is not None:
                                    z_offset = self.translation_vector[2]  # Translation Z chính là z_offset
                                    tolerance = self.zoffset_tolerance_scale.get() / 1000.0
                                    
                                    points_arr = np.asarray(pcd_instance.points)
                                    colors_arr = np.asarray(pcd_instance.colors)
                                    z_vals = points_arr[:, 2]
                                    
                                    # Lọc điểm có z gần z_offset (trong khoảng tolerance)
                                    valid_mask = np.abs(z_vals - 0) > tolerance  # So với Z=0 (đã translate)
                                    
                                    pcd_instance.points = o3d.utility.Vector3dVector(points_arr[valid_mask])
                                    pcd_instance.colors = o3d.utility.Vector3dVector(colors_arr[valid_mask])
                                    
                                    removed = original_count - len(pcd_instance.points)
                                    if removed > 0:
                                        print(f"   🔹 Z-offset: loại {removed:,} điểm gần nền ({tolerance*1000:.1f}mm)")
                            
                            # 2️⃣ Radius Outlier Removal (nếu bật)
                            if self.radius_outlier_var.get():
                                radius = self.radius_scale.get() / 1000.0
                                min_neighbors = int(self.min_neighbors_scale.get())
                                
                                before = len(pcd_instance.points)
                                pcd_instance, ind = pcd_instance.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
                                removed = before - len(pcd_instance.points)
                                if removed > 0:
                                    print(f"   🔹 Radius Outlier: loại {removed:,} điểm cô lập (r={radius*1000:.1f}mm)")
                            
                            # 3️⃣ Statistical Outlier Removal (nếu bật)
                            if self.outlier_var.get():
                                before = len(pcd_instance.points)
                                pcd_instance, ind = pcd_instance.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
                                removed = before - len(pcd_instance.points)
                                if removed > 0:
                                    print(f"   🔹 Statistical: loại {removed:,} điểm nhiễu")
                            
                            # 4️⃣ Voxel Downsampling (nếu bật)
                            if self.smooth_var.get():
                                before = len(pcd_instance.points)
                                pcd_instance = pcd_instance.voxel_down_sample(voxel_size=0.0005)
                                removed = before - len(pcd_instance.points)
                                if removed > 0:
                                    print(f"   🔹 Voxel: {before:,} → {len(pcd_instance.points):,} (-{removed:,})")
                            
                            final_count = len(pcd_instance.points)
                            if final_count != original_count:
                                print(f"   ✅ Lọc nhiễu: {original_count:,} → {final_count:,} điểm (-{100*(1-final_count/original_count):.1f}%)")
                            
                            # Tính bbox
                            bbox = pcd_instance.get_axis_aligned_bounding_box()
                            extent = bbox.get_extent()
                            
                            # Lưu file - thay đổi format để có thể export axis
                            out_prefix = os.path.join(output_dir, f"instance_{idx}_class{inst_class}_{timestamp}")
                            
                            # 🌿 Export stem with growth axis (for class0 only)
                            if inst_class == 0 and len(root_pcd.points) > 0:
                                print(f"   🌿 Computing growth axis...")
                                
                                # Get root mask (combined all class1 instances)
                                root_mask_combined = np.zeros((720, 720), dtype=np.uint8)
                                for rm, rc in zip(self.instance_masks, self.instance_classes):
                                    if rc == 1:  # class1 = root
                                        root_mask_combined = np.maximum(root_mask_combined, (rm > 0).astype(np.uint8) * 255)
                                
                                # Export with multiple tips detection
                                self.export_stem_with_axis(
                                    pcd_instance, root_pcd, out_prefix,
                                    stem_mask=inst_mask,
                                    root_mask=root_mask_combined,
                                    points_np=points_np,
                                    pixel_mapping=pixel_mapping,
                                    R_matrix=self.R_matrix_transform,
                                    translation=self.translation_vector,
                                    slice_step=0.002,  # 2mm slices
                                    min_pts=5,
                                    smooth_window=5
                                )
                                filename = f"instance_{idx}_class{inst_class}_{timestamp}_pcd.ply"
                                axis_filename = f"instance_{idx}_class{inst_class}_{timestamp}_axis.ply"
                                exported_files.append(filename)
                                exported_files.append(axis_filename)
                            else:
                                # Regular export for root (class1) or if no root reference
                                filename = f"instance_{idx}_class{inst_class}_{timestamp}.ply"
                                filepath = os.path.join(output_dir, filename)
                                o3d.io.write_point_cloud(filepath, pcd_instance)
                                exported_files.append(filename)
                            
                            file_size = os.path.getsize(f"{out_prefix}_pcd.ply" if inst_class == 0 and len(root_pcd.points) > 0 
                                                       else os.path.join(output_dir, filename))
                            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                            
                            print(f"   ✅ Xuất: {filename}")
                            print(f"   🔷 Điểm: {len(pcd_instance.points):,}")
                            print(f"   📏 Size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
                            print(f"   💾 File: {size_str}")
                            
                            total_points_exported += len(pcd_instance.points)
                        else:
                            print(f"   ⚠️ Quá ít điểm ({np.sum(instance_filter)}), bỏ qua")
                
                print(f"\n{'='*60}")
                print(f"✅ Xuất hoàn tất {len(exported_files)} instances")
                print(f"   📁 Thư mục: {output_dir}")
                print(f"   🔷 Tổng: {total_points_exported:,} điểm")
                print(f"{'='*60}\n")
                
                # Hiển thị thông báo
                files_list = "\n".join([f"  • {f}" for f in exported_files])
                messagebox.showinfo(
                    "Xuất thành công", 
                    f"🔢 Individual Mode: Đã xuất {len(exported_files)} mầm lan riêng biệt\n\n"
                    f"{files_list}\n\n"
                    f"🔷 Tổng: {total_points_exported:,} điểm\n"
                    f"📁 Thư mục: {output_dir}"
                )
                
            # 🔵 COMBINED MODE: Xuất 1 file gộp chung (có trục sinh trưởng)
            else:
                print(f"💾 COMBINED MODE - Xuất point cloud gộp chung + trục sinh trưởng")
                
                # Tạo file prefix
                file_prefix = os.path.join(output_dir, f"pointcloud_{timestamp}")
                
                # Export combined với trục sinh trưởng
                print(f"🌿 Computing growth axis for combined point cloud...")
                filepath = self.export_combined_with_axis(
                    pcd, file_prefix,
                    slice_step=0.002,
                    min_pts=5,
                    smooth_window=5
                )
                
                if filepath is None:
                    # Fallback: export without axis
                    filepath = f"{file_prefix}.ply"
                    o3d.io.write_point_cloud(filepath, pcd)
                
                filename = os.path.basename(filepath)
                axis_filename = f"pointcloud_{timestamp}_axis.ply"
                
                # Kiểm tra file size
                file_size = os.path.getsize(filepath)
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                # Check if axis file exists
                axis_path = os.path.join(output_dir, axis_filename)
                has_axis = os.path.exists(axis_path)
                
                print(f"✅ Đã xuất point cloud:")
                print(f"   📁 File PCD: {filename}")
                if has_axis:
                    print(f"   🌿 File Axis: {axis_filename}")
                print(f"   📊 Kích thước: {size_str}")
                print(f"   📍 Đường dẫn: {filepath}")
                print(f"   🔷 Số điểm: {len(pcd.points):,}")
                print(f"{'='*60}\n")
                
                # Hiển thị thông báo cho user
                axis_info = f"\n🌿 {axis_filename} (trục sinh trưởng từ tâm gốc)" if has_axis else ""
                messagebox.showinfo(
                    "Xuất thành công", 
                    f"Đã lưu point cloud:\n\n"
                    f"📁 {filename}\n"
                    f"{axis_info}\n"
                    f"📊 {size_str}\n"
                    f"🔷 {len(pcd.points):,} điểm\n\n"
                    f"Sử dụng 'view_pointcloud.py' để xem file."
                )
            
        except Exception as e:
            print(f"❌ Lỗi xuất file: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Không thể xuất point cloud:\n{str(e)}")
    
    def clear_tips(self):
        """Xóa toàn bộ skeleton tips đang hiển thị"""
        self.last_tips_data_per_instance = {}
        self.last_path_vis_per_instance = {}
        self.status_label.config(text="🌿 Đã xóa Skeleton Tips", foreground='green')
        print("\n🌿 Cleared all skeleton tips data")

    def clear_roi(self):
        self.roi = None
        if self.roi_rect_id:
            if self.roi_setting_window is not None and self.roi_setting_window.winfo_exists():
                self.roi_canvas.delete(self.roi_rect_id)
            self.roi_rect_id = None
        
        # Đóng cửa sổ ROI setting nếu đang mở
        if self.roi_setting_window is not None and self.roi_setting_window.winfo_exists():
            self.roi_setting_window.destroy()
            self.roi_setting_window = None
        
        self.is_setting_area = False
        self.btn_set_area.config(text="🎯 Set Detection Area")
        
        self.roi_status_label.config(text="🎯 Vùng nhận diện: Toàn bộ (720x720)", foreground='cyan')
        self.status_label.config(text="🗑️ Đã xóa vùng nhận diện", foreground='green')
        
        # Xoá file config nếu tồn tại
        try:
            if os.path.exists('roi_config.txt'):
                os.remove('roi_config.txt')
        except Exception as e:
            print(f"⚠️ Không thể xoá roi_config.txt: {e}")
    
    def __del__(self):
        self.is_running = False
        if self.pipeline:
            self.pipeline.stop()

    # ==========================================================================
    # 🆕 HYBRID 2D-3D GRASP PIPELINE  (theo NOTE_Codex_Phuong_phap_MethodsX)
    #
    # instance mask + aligned depth + camera intrinsics
    #   -> refined mask -> 2D skeleton graph -> mask-constrained 3D point cloud
    #   -> skeleton-guided branch point clouds -> branch-wise PCA
    #   -> 2D-3D fusion -> dominant growth axis -> grasp point -> 6-DoF grasp pose
    #
    # Đây là pipeline xử lý CHÍNH cho việc tính điểm/pose kẹp mầm. Không tái sử
    # dụng các hàm "LEGACY" ở phần SKELETON & MULTIPLE TIPS DETECTION phía trên.
    # ==========================================================================

    # ---------------------- Section 1: Refine instance mask ----------------------
    def hybrid_refine_mask(self, mask):
        """Erosion(2) + Dilation(2) với kernel 3x3. Output: binary {0,1} uint8."""
        cfg = self.pipeline_config['morphology']
        mask01 = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(cfg['kernel_size']))
        refined = cv2.erode(mask01, kernel, iterations=cfg['erosion_iterations'])
        refined = cv2.dilate(refined, kernel, iterations=cfg['dilation_iterations'])
        return (refined > 0).astype(np.uint8)

    # ---------------------- Section 2: Skeleton graph ----------------------
    def hybrid_skeletonize_mask(self, mask01):
        """Skeletonize refined mask -> skeleton rộng 1 pixel (boolean array)."""
        return skeletonize(mask01.astype(bool))

    def hybrid_node_degrees(self, skel_bool):
        """Đếm số neighbor (8-connectivity) cho từng skeleton pixel = degree."""
        skel_uint8 = skel_bool.astype(np.uint8)
        kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel_uint8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neighbor_count = neighbor_count * skel_uint8
        return neighbor_count

    def hybrid_find_endpoints_junctions(self, skel_bool):
        """Endpoint: degree==1. Junction: degree>2. Trả về list các tuple (row,col) int."""
        degree_map = self.hybrid_node_degrees(skel_bool)
        endpoints_mask = (degree_map == 1) & skel_bool
        junctions_mask = (degree_map > 2) & skel_bool
        endpoints_rc = [(int(r), int(c)) for r, c in np.column_stack(np.where(endpoints_mask))]
        junctions_rc = [(int(r), int(c)) for r, c in np.column_stack(np.where(junctions_mask))]
        return endpoints_rc, junctions_rc

    def hybrid_select_basal_node(self, mask01, endpoints_rc):
        """
        Chọn basal node theo đúng 2 bước của spec:
        1) Lọc candidates = các endpoint NẰM TRONG lower mask region (20% hàng
           dưới cùng của mask) - KHÔNG xét endpoint nằm ngoài vùng này dù có thể
           gần centroid hơn (ví dụ endpoint giữ hoặc trên của một thân cong).
        2) Nếu có nhiều candidates, chọn endpoint gần centroid của lower region
           nhất trong số candidates đó (KHÔNG so với toàn bộ endpoints).
        KHÔNG dùng class/mask phụ/marker ngoài instance mask.
        """
        if len(endpoints_rc) == 0:
            return None
        ys, xs = np.where(mask01 > 0)
        if len(ys) == 0:
            return None
        y_min, y_max = float(ys.min()), float(ys.max())
        lower_threshold = y_max - 0.20 * (y_max - y_min)
        lower_sel = ys >= lower_threshold
        if np.sum(lower_sel) == 0:
            lower_ys, lower_xs = ys, xs
        else:
            lower_ys, lower_xs = ys[lower_sel], xs[lower_sel]
        centroid_r = float(np.mean(lower_ys))
        centroid_c = float(np.mean(lower_xs))
        
        # Bước 1: chỉ giữ các endpoint thực sự nằm trong lower region
        candidates = [e for e in endpoints_rc if e[0] >= lower_threshold]
        if len(candidates) == 0:
            # Không endpoint nào nằm trong lower region (mask bất thường) -> fallback
            # toàn bộ endpoints, vẫn chọn theo khoảng cách tới centroid lower region.
            candidates = endpoints_rc
        
        # Bước 2: trong số candidates, chọn endpoint gần centroid nhất
        best_idx, best_dist = None, float('inf')
        for idx, (r, c) in enumerate(candidates):
            d = (r - centroid_r) ** 2 + (c - centroid_c) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return candidates[best_idx] if best_idx is not None else None

    def hybrid_bfs_shortest_path(self, skel_bool, start_rc, end_rc):
        """BFS shortest path (8-connectivity, unweighted) từ start_rc -> end_rc trên skeleton."""
        h, w = skel_bool.shape
        if not (0 <= start_rc[0] < h and 0 <= start_rc[1] < w and skel_bool[start_rc[0], start_rc[1]]):
            return None
        if start_rc == end_rc:
            return [start_rc]
        
        visited = np.zeros_like(skel_bool, dtype=bool)
        parent = {}
        visited[start_rc[0], start_rc[1]] = True
        queue = deque([start_rc])
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        reached = False
        while queue:
            r, c = queue.popleft()
            if (r, c) == end_rc:
                reached = True
                break
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skel_bool[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        
        if not reached and not visited[end_rc[0], end_rc[1]]:
            return None
        
        path = [end_rc]
        cur = end_rc
        while cur != start_rc:
            cur = parent.get(cur)
            if cur is None:
                return None
            path.append(cur)
        path.reverse()
        return path

    # ---------------------- Camera intrinsics helper ----------------------
    def hybrid_get_intrinsics(self):
        """Lấy fx,fy,cx,cy,coeffs,depth_scale + offset crop 720x720 (cache lại)."""
        if self._intrinsics_cache is not None:
            return self._intrinsics_cache
        try:
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            start_x = (intr.width - 720) // 2
            self._intrinsics_cache = {
                'fx': intr.fx, 'fy': intr.fy, 'cx': intr.ppx, 'cy': intr.ppy,
                'coeffs': list(intr.coeffs), 'width': intr.width, 'height': intr.height,
                'depth_scale': depth_scale, 'start_x': start_x, 'start_y': 0,
            }
            return self._intrinsics_cache
        except Exception as e:
            print(f"⚠️ Không lấy được camera intrinsics: {e}")
            return None

    # ---------------------- Section 3: Back-project mask -> point cloud ----------------------
    def hybrid_backproject_mask(self, mask01, depth_image_m, intr):
        """
        Back-project MỌI foreground pixel (không subsample) có depth hợp lệ.
        z = depth(u,v); x=(u-cx)*z/fx; y=(v-cy)*z/fy
        Giữ mapping (u,v)_crop <-> point_C.
        Trả về points_C (N,3) mét, pixel_uv (N,2) int32 tọa độ crop 720x720.
        """
        start_x, start_y = intr['start_x'], intr['start_y']
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        
        vs, us = np.where(mask01 > 0)
        if len(us) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32)
        
        u_full = us + start_x
        v_full = vs + start_y
        
        h_full, w_full = depth_image_m.shape
        valid_bounds = (u_full >= 0) & (u_full < w_full) & (v_full >= 0) & (v_full < h_full)
        us, vs = us[valid_bounds], vs[valid_bounds]
        u_full, v_full = u_full[valid_bounds], v_full[valid_bounds]
        
        z = depth_image_m[v_full, u_full]
        valid_depth = (z > 0) & np.isfinite(z)
        
        us, vs = us[valid_depth], vs[valid_depth]
        u_full, v_full, z = u_full[valid_depth], v_full[valid_depth], z[valid_depth]
        
        x = (u_full.astype(np.float64) - cx) * z / fx
        y = (v_full.astype(np.float64) - cy) * z / fy
        
        points_C = np.stack([x, y, z], axis=1)
        pixel_uv = np.stack([us, vs], axis=1).astype(np.int32)
        return points_C, pixel_uv

    def hybrid_deproject_pixel(self, u, v, depth_image_m, intr, mask01=None, max_search_radius=8):
        """
        Deproject pixel (u,v) [tọa độ crop] sang 3D camera frame.
        Policy rõ ràng khi thiếu depth: quét vành ring bán kính tăng dần (1..max)
        để tìm điểm depth hợp lệ GẦN NHẤT. Không tạo tọa độ giả -> trả None nếu
        không tìm thấy trong bán kính cho phép.
        Nếu mask01 được truyền vào (refined instance mask), CHỈ chấp nhận các
        pixel nằm TRONG mask đó - tránh lấy nhầm depth của nền/bàn/vật khác.
        """
        start_x, start_y = intr['start_x'], intr['start_y']
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        h_full, w_full = depth_image_m.shape
        mask_h, mask_w = (mask01.shape if mask01 is not None else (None, None))
        
        def _try(u_c, v_c):
            if mask01 is not None:
                if not (0 <= v_c < mask_h and 0 <= u_c < mask_w) or mask01[v_c, u_c] == 0:
                    return None
            uf, vf = u_c + start_x, v_c + start_y
            if 0 <= uf < w_full and 0 <= vf < h_full:
                z = depth_image_m[vf, uf]
                if z > 0 and np.isfinite(z):
                    x = (uf - cx) * z / fx
                    y = (vf - cy) * z / fy
                    return np.array([x, y, z], dtype=np.float64)
            return None
        
        pt = _try(u, v)
        if pt is not None:
            return pt
        
        for radius in range(1, max_search_radius + 1):
            candidates = []
            for du in range(-radius, radius + 1):
                for dv in range(-radius, radius + 1):
                    if max(abs(du), abs(dv)) != radius:
                        continue
                    pt = _try(u + du, v + dv)
                    if pt is not None:
                        candidates.append((du * du + dv * dv, pt))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    # ---------------------- Section 4: Filter point cloud + supporting plane ----------------------
    def hybrid_statistical_outlier_removal(self, points_C, pixel_uv):
        """SOR (nb_neighbors=10, std_ratio=2.0). Giữ đồng bộ pixel_uv."""
        cfg = self.pipeline_config['pointcloud']
        if len(points_C) < cfg['sor_neighbors'] + 1:
            return points_C, pixel_uv
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_C)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=cfg['sor_neighbors'], std_ratio=cfg['sor_std_ratio'])
        ind = np.array(ind, dtype=np.int64)
        return points_C[ind], pixel_uv[ind]

    def hybrid_validate_plane_model(self, plane_model, inlier_points, total_points=None,
                                    min_inliers=3, eps=1e-9,
                                    quality_min_inliers=None, quality_min_ratio=None):
        """
        Kiểm tra plane_model + inlier points theo 2 lớp độc lập (tập trung TOÀN
        BỘ logic kiểm tra plane vào 1 hàm duy nhất - không lặp ở nơi khác):
        
        (A) Ràng buộc HÌNH HỌC tối thiểu (bắt buộc, không cấu hình được):
            - 4 hệ số plane đều hữu hạn (không NaN/Inf).
            - norm(normal) > eps (không suy biến về vector 0).
            - >= min_inliers=3 điểm (điều kiện TOÁN HỌC tối thiểu để xác
              định 1 mặt phẳng - KHÔNG phải ngưỡng chất lượng).
            - Inliers không suy biến thành điểm trùng nhau / đường thẳng:
              eigenvalue lớn thứ 2 của covariance phải > eps.
        
        (B) Kiểm tra CHẤT LƯỢNG supporting plane (TÙY CHỌN - chỉ áp dụng nếu
            gọi hàm kèm total_points + quality_min_inliers/quality_min_ratio).
            Một plane thỏa (A) nhưng chỉ được hỗ trợ bởi 1 phần rất nhỏ ROI
            (ví dụ 5 điểm/50000) vẫn có thể làm xoay SAI toàn bộ point cloud
            nếu không bị chặn ở đây.
        
        Trả về (is_valid, inlier_count, reason).
        """
        coeffs = np.asarray(plane_model, dtype=np.float64)
        if coeffs.shape[0] != 4 or not np.all(np.isfinite(coeffs)):
            return False, 0, 'Hệ số plane không hữu hạn (NaN/Inf) hoặc sai định dạng'
        
        normal_norm = np.linalg.norm(coeffs[:3])
        if normal_norm <= eps:
            return False, 0, 'Normal của plane suy biến (~0)'
        
        inlier_count = int(len(inlier_points))
        if inlier_count < min_inliers:
            return False, inlier_count, f'Quá ít inlier ({inlier_count} < {min_inliers}) - vi phạm ràng buộc hình học tối thiểu'
        
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        cov = (centered.T @ centered) / inlier_count
        if not np.all(np.isfinite(cov)):
            return False, inlier_count, 'Covariance của inlier không hữu hạn'
        
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[1] <= eps:
            return False, inlier_count, 'Inliers suy biến (trùng nhau hoặc thẳng hàng)'
        
        # (B) Kiểm tra chất lượng - chỉ khi caller cung cấp total_points + ngưỡng
        if total_points is not None and total_points > 0:
            inlier_ratio = inlier_count / total_points
            if quality_min_inliers is not None and inlier_count < quality_min_inliers:
                return False, inlier_count, (
                    f'Mức hỗ trợ plane quá thấp: inlier_count={inlier_count} < '
                    f'ransac_min_inliers={quality_min_inliers} (tổng ROI={total_points}, '
                    f'ratio={inlier_ratio:.3f})')
            if quality_min_ratio is not None and inlier_ratio < quality_min_ratio:
                return False, inlier_count, (
                    f'Mức hỗ trợ plane quá thấp: inlier_ratio={inlier_ratio:.3f} < '
                    f'ransac_min_inlier_ratio={quality_min_ratio} (inlier_count={inlier_count}/'
                    f'{total_points})')
        
        return True, inlier_count, 'OK'

    def hybrid_fit_supporting_plane(self, mask01, depth_image_m, intr, margin_px=60):
        """
        Fit supporting plane từ local ROI quanh bud (loại trừ chính mask).
        RANSAC threshold=3mm, iterations=500 (Section 4).
        Dùng CÙNG MỘT seed (config['ransac_seed'], mặc định 0) cho cả:
          - NumPy sampling ROI khi >8000 điểm (np.random.default_rng(seed)), và
          - RNG nội bộ của Open3D (o3d.utility.random.seed(seed)) dùng bởi
            pcd_roi.segment_plane() - đây là bộ sinh ngẫu nhiên RIÊNG, không
            chia sẻ với NumPy, nên phải cố định seed của chính nó thì kết quả
            RANSAC mới lặp lại ổn định qua các lần chạy.
        Trả về plane_model (a,b,c,d) camera frame, hoặc None nếu validation thất
        bại (KHÔNG fallback âm thầm sang identity rotation).
        """
        cfg = self.pipeline_config['pointcloud']
        ys, xs = np.where(mask01 > 0)
        if len(ys) == 0:
            return None
        y0, y1 = max(0, ys.min() - margin_px), min(mask01.shape[0], ys.max() + margin_px)
        x0, x1 = max(0, xs.min() - margin_px), min(mask01.shape[1], xs.max() + margin_px)
        
        roi_mask = np.zeros_like(mask01, dtype=bool)
        roi_mask[y0:y1, x0:x1] = True
        roi_mask &= (mask01 == 0)
        
        points_roi, _ = self.hybrid_backproject_mask(roi_mask.astype(np.uint8), depth_image_m, intr)
        
        if len(points_roi) < 50:
            print("      ⚠️ ROI nền quanh bud quá ít điểm, bỏ qua supporting plane fit")
            return None
        
        seed = int(cfg.get('ransac_seed', 0))
        
        if len(points_roi) > 8000:
            # 🔒 Sampling XÁC ĐỊNH (deterministic): dùng chung seed với RANSAC,
            # KHÔNG dùng np.random.choice() toàn cục (kết quả sẽ thay đổi giữa
            # các lần chạy).
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(points_roi), 8000, replace=False)
            points_roi_sampled = points_roi[idx]
        else:
            points_roi_sampled = points_roi
        
        pcd_roi = o3d.geometry.PointCloud()
        pcd_roi.points = o3d.utility.Vector3dVector(points_roi_sampled)
        
        try:
            # 🔒 Cố định RNG NỘI BỘ của Open3D (segment_plane dùng RANSAC ngẫu nhiên
            # riêng, không chia sẻ với numpy) - cùng seed để kết quả lặp lại ổn
            # định qua các lần chạy trong cùng môi trường Open3D.
            o3d.utility.random.seed(seed)
            plane_model, inliers = pcd_roi.segment_plane(
                distance_threshold=cfg['ransac_threshold_mm'] / 1000.0,
                ransac_n=3,
                num_iterations=cfg['ransac_iterations'],
                probability=1.0
            )
        except Exception as e:
            print(f"      ⚠️ RANSAC plane fit thất bại: {e}")
            return None
        
        inlier_points = points_roi_sampled[inliers] if len(inliers) > 0 else np.zeros((0, 3))
        is_valid, inlier_count, reason = self.hybrid_validate_plane_model(
            plane_model, inlier_points,
            total_points=len(points_roi_sampled),
            quality_min_inliers=cfg.get('ransac_min_inliers'),
            quality_min_ratio=cfg.get('ransac_min_inlier_ratio'),
        )
        inlier_ratio = inlier_count / len(points_roi_sampled) if len(points_roi_sampled) > 0 else 0.0
        
        print(f"      ℹ️ Plane fit: inlier_count={inlier_count}/{len(points_roi_sampled)} "
              f"(inlier_ratio={inlier_ratio:.3f})")
        
        if not is_valid:
            print(f"      ⚠️ Plane bị từ chối (validation thất bại): {reason}")
            return None
        
        return plane_model

    def hybrid_remove_plane_inliers(self, points_C, pixel_uv, plane_model, threshold_mm):
        """Áp dụng plane equation lên dense pixel-mapped cloud, loại bỏ điểm gần plane."""
        a, b, c, d = plane_model
        norm = np.sqrt(a * a + b * b + c * c)
        if norm < 1e-9:
            return points_C, pixel_uv
        dist = np.abs(a * points_C[:, 0] + b * points_C[:, 1] + c * points_C[:, 2] + d) / norm
        keep = dist > (threshold_mm / 1000.0)
        return points_C[keep], pixel_uv[keep]

    # ---------------------- Section 5: Chuẩn hóa mặt phẳng (rotation-only) ----------------------
    def hybrid_compute_plane_rotation(self, plane_model):
        """
        Tính R_N_C sao cho R_N_C @ plane_normal_C = [0,0,1]. Chỉ rotation, không
        translate. Đảm bảo trực chuẩn, det(R)=+1.
        """
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            return np.eye(3)
        normal = normal / norm_len
        if normal[2] < 0:
            normal = -normal
        
        target_z = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, target_z)
        s = np.linalg.norm(v)
        c_val = np.dot(normal, target_z)
        
        if s < 1e-9:
            if c_val > 0:
                R = np.eye(3)
            else:
                R = np.diag([1.0, -1.0, -1.0])  # xoay 180° quanh trục X, det=+1
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx.dot(vx) * ((1 - c_val) / (s * s))
        return R

    # ---------------------- Section 6: Gán point cho branch (pixel domain, 5px) ----------------------
    def hybrid_assign_points_to_branches(self, points_N, pixel_uv, branch_paths):
        """
        Với mỗi point còn pixel mapping (u,v), gán cho basal-to-endpoint path gần
        nhất (pixel Euclidean distance) nếu < support_radius_px (5px).
        Thực hiện TRƯỚC voxel downsampling. overlap_policy = nearest_path.
        """
        cfg = self.pipeline_config['branch']
        support_radius = cfg['support_radius_px']
        
        if len(branch_paths) == 0 or len(points_N) == 0:
            return []
        
        trees = []
        for path in branch_paths:
            path_uv = np.array([[c, r] for (r, c) in path], dtype=np.float64)  # (u,v)
            trees.append(cKDTree(path_uv))
        
        query_uv = pixel_uv.astype(np.float64)
        n_branches = len(trees)
        n_points = len(query_uv)
        dist_matrix = np.full((n_points, n_branches), np.inf)
        
        for bi, tree in enumerate(trees):
            d, _ = tree.query(query_uv, k=1)
            dist_matrix[:, bi] = d
        
        # ℹ️ TIE-BREAK: np.argmin() trả về index NHỎ NHẤT khi có nhiều branch có
        # cùng khoảng cách tối thiểu (hòa tuyệt đối). Đây CHỈ là quy tắc xác định
        # (deterministic) để mỗi point luôn được gán cho ĐÚNG MỘT branch duy nhất
        # (không nhân bản), KHÔNG phải tiêu chí chọn dominant branch - việc chọn
        # dominant branch vẫn hoàn toàn dựa trên theta<=85° + longest-path ở
        # hybrid_select_dominant_branch(). Thứ tự branch (và do đó kết quả
        # tie-break) đã được cố định bằng cách sắp xếp candidate endpoint theo
        # tọa độ pixel (row, col) trước khi tạo branch_paths.
        best_branch = np.argmin(dist_matrix, axis=1)
        best_dist = dist_matrix[np.arange(n_points), best_branch]
        assigned_mask = best_dist < support_radius
        
        branches_out = []
        for bi in range(n_branches):
            sel = assigned_mask & (best_branch == bi)
            if np.sum(sel) == 0:
                continue
            branches_out.append({
                'branch_index': bi,
                'points_N': points_N[sel],
                'pixel_uv': pixel_uv[sel],
            })
        return branches_out

    def hybrid_voxel_downsample_branch(self, points_N):
        """Voxel-downsample riêng point cloud của 1 branch (0.5mm), SAU assignment."""
        voxel_size = self.pipeline_config['pointcloud']['voxel_size_mm'] / 1000.0
        if len(points_N) == 0:
            return points_N
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_N)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(pcd_down.points)

    # ---------------------- Section 7: Branch-wise PCA ----------------------
    def hybrid_branch_pca(self, points_N):
        """PCA trên point cloud (normalized frame) của 1 branch. Reject nếu suy biến."""
        if points_N is None or len(points_N) < 3:
            return None
        centroid = np.mean(points_N, axis=0)
        centered = points_N - centroid
        cov = (centered.T @ centered) / len(centered)
        if not np.all(np.isfinite(cov)):
            return None
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return None
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # ⚠️ Suy biến: các điểm trùng nhau hoặc không có độ trải (eigenvalue lớn
        # nhất ~ 0) thì eigh() vẫn trả eigenvector đơn vị hợp lệ về mặt số học
        # nhưng huớng hoàn toàn không có ý nghĩa -> phải reject rõ ràng.
        max_eigval = eigvals[0]
        if not np.isfinite(max_eigval) or max_eigval < 1e-12:
            return None
        
        direction = eigvecs[:, 0]
        norm = np.linalg.norm(direction)
        if norm < 1e-9 or not np.all(np.isfinite(direction)):
            return None
        direction = direction / norm
        if not np.all(np.isfinite(direction)):
            return None
        return direction

    # ---------------------- Tăng độ tin cậy: lookup basal/endpoint TỪ mapping đã lọc ----------------------
    def hybrid_lookup_point_from_mapping(self, u, v, filtered_points_C, filtered_pixel_uv, mask01=None, max_search_radius=8):
        """
        Tìm point 3D (camera frame) cho pixel (u,v) TỪ MAPPING ĐÃ LỌC (sau
        back-projection + SOR + plane-inlier removal) - KHÔNG deproject lại từ
        depth thô. Raw depth có thể chứa outlier đã bị SOR/plane-removal loại
        khỏi point cloud, nên basal/endpoint dùng để tính skeleton direction và
        grasp point PHẢI lấy từ chính filtered_points_C để đồng bộ với phần còn
        lại của pipeline.
        
        1) Ưu tiên match CHÍNH XÁC pixel (u,v) trong filtered_pixel_uv.
        2) Nếu không có, tìm pixel hợp lệ GẦN NHẤT trong filtered_pixel_uv
           (trong bán kính max_search_radius px); nếu truyền mask01, candidate
           đó phải còn nằm trong refined bud mask.
        3) Trả về point từ filtered_points_C tương ứng - KHÔNG fabricate/deproject.
        4) Trả None nếu không tìm được match hợp lệ trong bán kính cho phép
           (branch gọi hàm này phải tự loại bỏ, KHÔNG fallback sang depth thô).
        """
        if filtered_pixel_uv is None or len(filtered_pixel_uv) == 0:
            return None
        
        # 1) Exact match
        exact = (filtered_pixel_uv[:, 0] == u) & (filtered_pixel_uv[:, 1] == v)
        if np.any(exact):
            return np.mean(filtered_points_C[exact], axis=0)
        
        # 2) Tìm lân cận trong bán kính cho phép
        diffs = filtered_pixel_uv.astype(np.float64) - np.array([u, v], dtype=np.float64)
        dist2 = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        within_radius = dist2 <= (max_search_radius ** 2)
        
        if mask01 is not None:
            pu = filtered_pixel_uv[:, 0]
            pv = filtered_pixel_uv[:, 1]
            h, w = mask01.shape[:2]
            valid_coords = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
            mask_ok = np.zeros(len(filtered_pixel_uv), dtype=bool)
            valid_idx = np.where(valid_coords)[0]
            mask_ok[valid_idx] = mask01[pv[valid_idx], pu[valid_idx]] > 0
            within_radius &= mask_ok
        
        if not np.any(within_radius):
            return None
        
        candidate_idx = np.where(within_radius)[0]
        nearest_idx = candidate_idx[np.argmin(dist2[candidate_idx])]
        return filtered_points_C[nearest_idx]

    # ---------------------- Section 8: Skeleton direction + align dấu PCA ----------------------
    def hybrid_compute_skeleton_direction(self, basal_rc, endpoint_rc, filtered_points_C, filtered_pixel_uv, R_N_C, mask01=None):
        """Deproject basal & endpoint pixel -> normalized frame -> direction basal->endpoint.
        🔒 Dùng mapping point cloud ĐÃ LỌC (sau SOR + plane-inlier removal) qua
        hybrid_lookup_point_from_mapping() thay vì deproject lại depth thô -
        tránh dùng nhầm điểm outlier đã bị loại khỏi point cloud.
        mask01: refined instance mask - giới hạn candidate KHÔNG rò rỉ sang nền/vật khác.
        """
        basal_u, basal_v = basal_rc[1], basal_rc[0]
        end_u, end_v = endpoint_rc[1], endpoint_rc[0]
        
        basal_C = self.hybrid_lookup_point_from_mapping(basal_u, basal_v, filtered_points_C, filtered_pixel_uv, mask01=mask01)
        end_C = self.hybrid_lookup_point_from_mapping(end_u, end_v, filtered_points_C, filtered_pixel_uv, mask01=mask01)
        
        if basal_C is None or end_C is None:
            return None, basal_C, end_C
        
        basal_N = R_N_C @ basal_C
        end_N = R_N_C @ end_C
        
        direction = end_N - basal_N
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None, basal_C, end_C
        return direction / norm, basal_C, end_C

    # ---------------------- Section 9: Fusion 2D-3D ----------------------
    def hybrid_fuse_directions(self, d_pca_N, d_skeleton_N):
        """alpha=0.6 cho PCA, 0.4 cho skeleton. Align dấu PCA theo skeleton trước khi fuse."""
        alpha = self.pipeline_config['fusion']['alpha']
        if np.dot(d_pca_N, d_skeleton_N) < 0:
            d_pca_N = -d_pca_N
        d_fused = alpha * d_pca_N + (1 - alpha) * d_skeleton_N
        norm = np.linalg.norm(d_fused)
        if norm < 1e-9:
            return None, d_pca_N
        return d_fused / norm, d_pca_N

    # ---------------------- Section 10: Consistency + dominant branch ----------------------
    def hybrid_select_dominant_branch(self, branch_records):
        """
        1) Chỉ xét branch hợp lệ (đã lọc trước).
        2) Giữ branch theta<=85°.
        3) Chọn path dài nhất trong số đó.
        4) Nếu không có branch nào <=85° -> chọn valid path dài nhất.
        5) Không có valid branch -> None (failure rõ ràng).
        """
        theta_max = self.pipeline_config['fusion']['theta_max_deg']
        if len(branch_records) == 0:
            return None
        within_theta = [b for b in branch_records if b['theta_deg'] <= theta_max]
        if within_theta:
            return max(within_theta, key=lambda b: b['path_length_px'])
        return max(branch_records, key=lambda b: b['path_length_px'])

    # ---------------------- Section 11: Grasp point ----------------------
    def hybrid_compute_grasp_point(self, basal_point_C, filtered_points_C):
        """Centroid của các filtered bud points (camera frame) trong bán kính 8mm quanh basal."""
        radius_m = self.pipeline_config['grasp']['basal_radius_mm'] / 1000.0
        if basal_point_C is None or len(filtered_points_C) == 0:
            return None
        dist = np.linalg.norm(filtered_points_C - basal_point_C, axis=1)
        neighborhood = filtered_points_C[dist <= radius_m]
        if len(neighborhood) == 0:
            return None
        return np.mean(neighborhood, axis=0)

    # ---------------------- Section 12: Local grasp frame ----------------------
    def hybrid_build_grasp_frame(self, axis_C, grasp_point_C):
        """
        Growth axis = local Z. Nếu reference [0,0,1] gần song song axis, đổi basis khác.
        Đảm bảo R_C_G trực chuẩn, det=+1.
        """
        if axis_C is None or grasp_point_C is None:
            return None, None
        z_g = axis_C / (np.linalg.norm(axis_C) + 1e-12)
        ref = np.array(self.pipeline_config['grasp']['camera_reference_axis'], dtype=np.float64)
        
        if abs(np.dot(ref, z_g)) > 0.98:
            alt = np.array([1.0, 0.0, 0.0])
            ref = alt if abs(np.dot(alt, z_g)) <= 0.98 else np.array([0.0, 1.0, 0.0])
        
        x_g = np.cross(ref, z_g)
        x_norm = np.linalg.norm(x_g)
        if x_norm < 1e-9:
            return None, None
        x_g = x_g / x_norm
        y_g = np.cross(z_g, x_g)
        y_g = y_g / (np.linalg.norm(y_g) + 1e-12)
        
        R_C_G = np.column_stack([x_g, y_g, z_g])
        if np.linalg.det(R_C_G) < 0:
            x_g = -x_g
            R_C_G = np.column_stack([x_g, y_g, z_g])
        
        T_C_G = np.eye(4)
        T_C_G[:3, :3] = R_C_G
        T_C_G[:3, 3] = grasp_point_C
        return R_C_G, T_C_G

    def hybrid_rotation_matrix_to_rotvec(self, R):
        """Rotation matrix -> rotation vector (rad), quy ước kiểu Universal Robots (x,y,z,rx,ry,rz)."""
        rotvec, _ = cv2.Rodrigues(R.astype(np.float64))
        return rotvec.flatten()

    # ---------------------- ORCHESTRATOR: toàn bộ Sections 1-12 cho 1 instance ----------------------
    def compute_grasp_pose_hybrid(self, mask_crop, depth_image_m, intr):
        """
        MAIN HYBRID 2D-3D PIPELINE cho 1 instance mask.
        mask_crop: (720,720) uint8, tọa độ crop.
        depth_image_m: depth full-resolution (mét), đã align với color.
        intr: dict từ hybrid_get_intrinsics().
        Trả về dict: success, reason, axis_C, grasp_point_C, R_C_G, T_C_G,
                     branch_count, dominant_branch_len_px, num_points_bud
        """
        result = {
            'success': False, 'reason': '', 'axis_C': None, 'grasp_point_C': None,
            'R_C_G': None, 'T_C_G': None, 'branch_count': 0,
            'dominant_branch_len_px': 0.0, 'num_points_bud': 0, 'mask_area': 0,
            'global_pca_axis_N': None, 'global_pca_axis_C': None,
        }
        
        # Section 1
        mask01 = self.hybrid_refine_mask(mask_crop)
        if np.sum(mask01) < 20:
            result['reason'] = 'Mask quá nhỏ sau khi refine'
            return result
        result['mask_area'] = int(np.sum(mask01 > 0))
        
        # Section 2
        skel_bool = self.hybrid_skeletonize_mask(mask01)
        if not skel_bool.any():
            result['reason'] = 'Skeleton rỗng'
            return result
        
        endpoints_rc, _junctions_rc = self.hybrid_find_endpoints_junctions(skel_bool)
        if len(endpoints_rc) < 2:
            result['reason'] = f'Không đủ endpoints (tìm thấy {len(endpoints_rc)}, cần >=2)'
            return result
        
        basal_rc = self.hybrid_select_basal_node(mask01, endpoints_rc)
        if basal_rc is None:
            result['reason'] = 'Không chọn được basal node'
            return result
        
        other_endpoints = [e for e in endpoints_rc if e != basal_rc]
        if len(other_endpoints) == 0:
            result['reason'] = 'Không có endpoint nào khác basal'
            return result
        
        # 🔒 SẮp xếp candidate endpoint theo tọa độ pixel (row, col) TRƯỜC khi
        # tạo các basal-to-endpoint path -> đảm bảo thứ tự/index của
        # branch_paths_all ỔN ĐỪNH giữa các lần chạy (không phụ thuộc thứ tự
        # không tường minh của np.where/skeletonize). Điều này quan trọng vì
        # np.argmin() trong hybrid_assign_points_to_branches() chọn branch đầu
        # tiên khi có nhiều branch có khoảng cách bằng nhau (tie-break).
        other_endpoints = sorted(other_endpoints, key=lambda rc: (rc[0], rc[1]))
        
        branch_paths_all = []
        for ep in other_endpoints:
            path = self.hybrid_bfs_shortest_path(skel_bool, basal_rc, ep)
            if path is not None and len(path) >= 2:
                branch_paths_all.append({'endpoint_rc': ep, 'path': path})
        
        if len(branch_paths_all) == 0:
            result['reason'] = 'Không tìm được basal-to-endpoint path nào'
            return result
        
        # Section 3
        points_C, pixel_uv = self.hybrid_backproject_mask(mask01, depth_image_m, intr)
        if len(points_C) < 30:
            result['reason'] = f'Quá ít điểm 3D hợp lệ ({len(points_C)})'
            return result
        
        # Section 4
        points_C, pixel_uv = self.hybrid_statistical_outlier_removal(points_C, pixel_uv)
        if len(points_C) < 30:
            result['reason'] = 'Quá ít điểm sau SOR'
            return result
        
        plane_model = self.hybrid_fit_supporting_plane(mask01, depth_image_m, intr)
        if plane_model is None:
            # ⚠️ KHÔNG fallback âm thầm sang R_N_C=eye(3): nếu không fit được
            # supporting plane, ta KHÔNG loại được điểm nền và KHÔNG chuẩn hóa
            # được mount phẳng theo Section 5 -> trả failure rõ ràng thay vì tiếp
            # tục với kết quả không đáng tin cậy.
            result['reason'] = 'Không fit được supporting plane (RANSAC thất bại hoặc ROI nền quá ít điểm)'
            return result
        
        ransac_thr_mm = self.pipeline_config['pointcloud']['ransac_threshold_mm']
        points_C, pixel_uv = self.hybrid_remove_plane_inliers(points_C, pixel_uv, plane_model, ransac_thr_mm)
        
        if len(points_C) < 30:
            result['reason'] = 'Quá ít điểm sau khi loại supporting plane'
            return result
        
        result['num_points_bud'] = len(points_C)
        
        # Section 5
        R_N_C = self.hybrid_compute_plane_rotation(plane_model)
        R_C_N = R_N_C.T
        points_N = (R_N_C @ points_C.T).T
        
        # 🆕 Global PCA (geometric reference của TOÀN BỘ bud, sau voxel-downsample
        # 0.5mm) - CHỈ để tham khảo/chẩn đoán, KHÔNG thay thế branch-wise PCA,
        # fused direction hay final growth axis. Dùng lại đúng logic kiểm tra suy
        # biến (eigenvalue lớn nhất phải hữu hạn & dương) như branch-wise PCA.
        global_points_N_ds = self.hybrid_voxel_downsample_branch(points_N)
        global_pca_axis_N = self.hybrid_branch_pca(global_points_N_ds)
        if global_pca_axis_N is not None:
            global_pca_axis_C = R_C_N @ global_pca_axis_N
            global_pca_axis_C = global_pca_axis_C / (np.linalg.norm(global_pca_axis_C) + 1e-12)
        else:
            global_pca_axis_C = None
        result['global_pca_axis_N'] = global_pca_axis_N
        result['global_pca_axis_C'] = global_pca_axis_C
        
        # Section 6 (assignment trước voxel downsample)
        branch_paths_px = [b['path'] for b in branch_paths_all]
        branches_assigned = self.hybrid_assign_points_to_branches(points_N, pixel_uv, branch_paths_px)
        if len(branches_assigned) == 0:
            result['reason'] = 'Không có branch nào có điểm được gán'
            return result
        
        # Sections 7-9 cho từng branch (voxel downsample -> min_branch_points check -> PCA -> skeleton dir -> fusion)
        min_branch_points = self.pipeline_config['branch']['min_branch_points']
        branch_records = []
        for ba in branches_assigned:
            bi = ba['branch_index']
            pts_branch_N = self.hybrid_voxel_downsample_branch(ba['points_N'])
            if len(pts_branch_N) < min_branch_points:
                continue
            
            d_pca_N = self.hybrid_branch_pca(pts_branch_N)
            if d_pca_N is None:
                continue
            
            endpoint_rc = branch_paths_all[bi]['endpoint_rc']
            path_px = branch_paths_all[bi]['path']
            
            d_skel_N, basal_C_pt, _end_C_pt = self.hybrid_compute_skeleton_direction(
                basal_rc, endpoint_rc, points_C, pixel_uv, R_N_C, mask01=mask01
            )
            if d_skel_N is None:
                continue
            
            d_fused_N, d_pca_aligned_N = self.hybrid_fuse_directions(d_pca_N, d_skel_N)
            if d_fused_N is None:
                continue
            
            theta = float(np.degrees(np.arccos(np.clip(np.dot(d_pca_aligned_N, d_skel_N), -1.0, 1.0))))
            
            branch_records.append({
                'branch_index': bi,
                'path_length_px': len(path_px),
                'd_fused_N': d_fused_N,
                'theta_deg': theta,
                'basal_C': basal_C_pt,
            })
        
        if len(branch_records) == 0:
            result['reason'] = 'Không có branch hợp lệ sau PCA/skeleton-direction'
            return result
        
        # Section 10
        dominant = self.hybrid_select_dominant_branch(branch_records)
        if dominant is None:
            result['reason'] = 'Không chọn được dominant branch'
            return result
        
        axis_C = R_C_N @ dominant['d_fused_N']
        axis_C = axis_C / (np.linalg.norm(axis_C) + 1e-12)
        
        # Section 11
        basal_point_C = dominant['basal_C']
        if basal_point_C is None:
            # 🔒 Dùng mapping đã lọc (KHÔNG deproject lại depth thô) - nhánh này
            # chỉ là fallback phòng thủ (về lý thuyết không xảy ra vì dominant
            # đã được chọn từ branch_records luôn có basal_C hợp lệ).
            basal_point_C = self.hybrid_lookup_point_from_mapping(basal_rc[1], basal_rc[0], points_C, pixel_uv, mask01=mask01)
        if basal_point_C is None:
            result['reason'] = 'Không deproject được basal point'
            return result
        
        grasp_point_C = self.hybrid_compute_grasp_point(basal_point_C, points_C)
        if grasp_point_C is None:
            result['reason'] = 'Vùng lân cận basal (8mm) rỗng, không tính được grasp point'
            return result
        
        # Section 12
        R_C_G, T_C_G = self.hybrid_build_grasp_frame(axis_C, grasp_point_C)
        if R_C_G is None:
            result['reason'] = 'Không xây dựng được grasp frame (axis song song reference)'
            return result
        
        result.update({
            'success': True, 'reason': 'OK', 'axis_C': axis_C, 'grasp_point_C': grasp_point_C,
            'R_C_G': R_C_G, 'T_C_G': T_C_G, 'branch_count': len(branch_records),
            'dominant_branch_len_px': dominant['path_length_px'],
        })
        return result

    # ---------------------- Capture đồng bộ mask + depth cho pipeline mới ----------------------
    def hybrid_capture_instances_and_depth(self):
        """
        Chụp 1 frame MỚI, chạy YOLO ngay trên frame đó để mask & depth luôn đồng
        bộ tuyệt đối (tránh lệch giữa 2 frame khác nhau).
        Trả về (masks01_list, classes_list, depth_image_m, intr) hoặc None nếu lỗi.
        """
        if self.yolo_model is None:
            self.status_label.config(text="⚠️ Chưa load YOLO model!", foreground='red')
            return None
        
        intr = self.hybrid_get_intrinsics()
        if intr is None:
            self.status_label.config(text="⚠️ Không lấy được camera intrinsics!", foreground='red')
            return None
        
        try:
            with self.camera_lock:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                self.status_label.config(text="⚠️ Không lấy được frame!", foreground='red')
                return None
            
            if self.use_filters:
                if self.decimation_filter:
                    depth_frame = self.decimation_filter.process(depth_frame)
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
            
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float64)
            depth_image_m = depth_raw * intr['depth_scale']
            
            color_image = np.asanyarray(color_frame.get_data())
            start_x = intr['start_x']
            color_720 = color_image[0:720, start_x:start_x + 720].copy()
            
            if self.roi:
                x1, y1, x2, y2 = self.roi
                detection_region = color_720[y1:y2, x1:x2].copy()
                region_offset = (x1, y1)
            else:
                detection_region = color_720.copy()
                region_offset = (0, 0)
            
            if detection_region.shape[0] <= 32 or detection_region.shape[1] <= 32:
                self.status_label.config(text="⚠️ Vùng detection quá nhỏ!", foreground='red')
                return None
            
            results = self.yolo_model.predict(detection_region, conf=self.segmentation_conf, verbose=False)
            
            masks01_list, classes_list = [], []
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.data.cpu().numpy()
                proc_h, proc_w = detection_region.shape[:2]
                x_off, y_off = region_offset
                
                for mask, box in zip(masks, boxes):
                    class_id = int(box[5])
                    mask_resized = cv2.resize(mask, (proc_w, proc_h))
                    mask_bool = mask_resized > 0.5
                    
                    full_mask01 = np.zeros((720, 720), dtype=np.uint8)
                    full_mask01[y_off:y_off + proc_h, x_off:x_off + proc_w] = mask_bool.astype(np.uint8)
                    
                    masks01_list.append(full_mask01)
                    classes_list.append(class_id)
            
            return masks01_list, classes_list, depth_image_m, intr
        
        except Exception as e:
            print(f"❌ Lỗi capture instance/depth cho hybrid pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _select_grasp_target(self, success_results):
        """
        Chọn 1 instance DUY NHẤT để gửi robot - KHÔNG tự chọn theo mask_area
        hay bất kỳ tiêu chí "lớn nhất" nào (không thuộc phương pháp trong bản
        thảo). Chỉ trả về 1 instance khi:
          a) `pipeline_config['grasp']['target_instance_index']` được chỉ định
             rõ ràng (bởi ROI/người dùng/lệnh ngoài), hoặc
          b) chỉ có ĐÚNG 1 instance thành công (không phải "chọn", chỉ vì đó
             là lựa chọn duy nhất).
        Nếu có nhiều instance hợp lệ mà chưa chỉ định target rõ ràng, trả về
        None (KHÔNG tự chọn).
        
        Trả về (selected_result_or_None, log_messages: list[str]).
        """
        target_instance_index = self.pipeline_config['grasp'].get('target_instance_index')
        logs = []
        
        if target_instance_index is not None:
            matches = [r for r in success_results if r['instance_index'] == target_instance_index]
            if matches:
                return matches[0], logs
            logs.append(f"⚠️ target_instance_index={target_instance_index} không nằm trong danh sách "
                         f"instance thành công -> KHÔNG gán last_grasp_result.")
            return None, logs
        
        if len(success_results) == 1:
            return success_results[0], logs
        
        logs.append(f"ℹ️ Có {len(success_results)} instance hợp lệ nhưng CHƯA chỉ định target "
                    f"(pipeline_config['grasp']['target_instance_index']) -> KHÔNG tự chọn, "
                    f"KHÔNG gán last_grasp_result. Xem cửa sổ kết quả để chọn thủ công.")
        return None, logs

    def _prompt_select_target_dialog(self, success_results):
        """
        Mở dialog MODAL cho người dùng chọn TƯỜNG MINH 1 instance (trong số các
        instance có success=True) làm target khi có NHIỀU instance thành công.
        
        Quy ước hiển thị: số thứ tự bắt đầu từ 1 (displayed = internal_index+1),
        khớp đúng với cửa sổ kết quả "Instance N". Nội bộ vẫn dùng instance_index
        0-based, không trộn lẫn 2 quy ước.
        
        Trả về internal instance_index (0-based) người dùng chọn, hoặc None nếu
        người dùng hủy/đóng cửa sổ (không chọn gì).
        """
        if not success_results:
            return None
        
        result_holder = {'selected_index': None}
        
        dialog = tk.Toplevel(self.root)
        dialog.title("🎯 Chọn Target Instance để gửi lệnh Robot")
        dialog.configure(bg='#2d2d30')
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(
            dialog,
            text=f"Có {len(success_results)} instance tính pose THÀNH CÔNG.\n"
                 f"Chọn ĐÚNG MỘT instance làm target để gửi lệnh kẹp cho robot\n"
                 f"(các instance còn lại chỉ hiển thị như candidate, KHÔNG gửi robot):",
            style='Dark.TLabel', justify=tk.LEFT
        ).pack(padx=15, pady=(15, 10))
        
        btn_frame = ttk.Frame(dialog, style='Dark.TFrame')
        btn_frame.pack(padx=15, pady=5, fill=tk.X)
        
        def _choose(idx):
            result_holder['selected_index'] = idx
            dialog.destroy()
        
        # Chỉ liệt kê instance ĐANG success=True (không cho chọn instance thất bại)
        for r in sorted(success_results, key=lambda x: x['instance_index']):
            internal_idx = r['instance_index']
            displayed_number = internal_idx + 1  # 🔢 quy ước hiển thị bắt đầu từ 1
            btn_text = (f"Instance {displayed_number}   "
                        f"(branches={r['branch_count']}, mask_area={r['mask_area']}px)")
            ttk.Button(btn_frame, text=btn_text,
                       command=lambda i=internal_idx: _choose(i)).pack(fill=tk.X, pady=3)
        
        ttk.Button(dialog, text="❌ Hủy (không chọn target nào)",
                   command=dialog.destroy).pack(padx=15, pady=(8, 15), fill=tk.X)
        
        dialog.update_idletasks()
        try:
            x = self.root.winfo_x() + max(0, (self.root.winfo_width() - dialog.winfo_width()) // 2)
            y = self.root.winfo_y() + max(0, (self.root.winfo_height() - dialog.winfo_height()) // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass
        
        self.root.wait_window(dialog)
        return result_holder['selected_index']

    # ---------------------- Nút bấm: chạy toàn bộ pipeline ----------------------
    def run_hybrid_grasp_pipeline(self):
        """Chạy HYBRID 2D-3D GRASP PIPELINE cho tất cả instance hiện tại, hiển thị
        và lưu tọa độ x,y,z,rx,ry,rz để gửi lệnh kẹp cho robot."""
        # 🔒 Xoá kết quả pose cũ NGAY Từ ĐẦU - trước bất kỳ lệnh nào có thể
        # return/raise (không lấy được frame, không có detection, không có target
        # class, exception...). Đảm bảo không bao giờ giữ lại pose của lần chạy
        # trước nếu lần chạy mới thất bại.
        self.last_grasp_results = []
        self.last_grasp_result = None
        
        self.status_label.config(text="⏳ Đang tính Grasp Pose (Hybrid)...", foreground='orange')
        self.root.update()
        
        was_previewing = self.is_preview
        self._set_preview_active(False)
        
        try:
            capture = self.hybrid_capture_instances_and_depth()
            if capture is None:
                self.status_label.config(text="🔴 Lỗi: Không lấy được dữ liệu detection", foreground='red')
                return
            
            masks01_list, classes_list, depth_image_m, intr = capture
            
            if len(masks01_list) == 0:
                self.status_label.config(text="⚠️ Không phát hiện instance nào!", foreground='orange')
                return
            
            # 🆕 Chỉ xử lý đúng instance thuộc target_class_id (mặc định class 0 =
            # "thân/nhánh" - đối tượng cần kẹp). Instance thuộc class khác (vd class 1
            # = "gốc", chỉ là mốc tham chiếu) sẽ KHÔNG được đưa qua pipeline, tránh
            # tạo pose sai cho đối tượng không phải target.
            target_class_id = self.pipeline_config['grasp']['target_class_id']
            target_indices = [i for i, c in enumerate(classes_list) if c == target_class_id]
            
            if len(target_indices) == 0:
                self.status_label.config(
                    text=f"⚠️ Không có instance nào thuộc target_class_id={target_class_id}!",
                    foreground='orange')
                return
            
            print(f"\n{'='*70}")
            print(f"🤖 HYBRID 2D-3D GRASP PIPELINE - {len(target_indices)}/{len(masks01_list)} "
                  f"instance thuộc target_class_id={target_class_id}")
            print(f"{'='*70}")
            
            all_results = []
            for idx in target_indices:
                mask01, cls = masks01_list[idx], classes_list[idx]
                print(f"\n🌿 Instance {idx+1} (class={cls}):")
                res = self.compute_grasp_pose_hybrid(mask01, depth_image_m, intr)
                res['instance_index'] = idx
                res['class_id'] = cls
                if res['success']:
                    print(f"   ✅ Thành công! branches={res['branch_count']}, "
                          f"mask_area={res['mask_area']}px, "
                          f"grasp_point_C(mm)={np.round(res['grasp_point_C']*1000, 1)}")
                else:
                    print(f"   ❌ Thất bại: {res['reason']}")
                all_results.append(res)
            
            self.last_grasp_results = all_results
            success_results = [r for r in all_results if r['success']]
            
            if len(success_results) == 0:
                self.status_label.config(text="🔴 Không có instance nào tính được grasp pose!", foreground='red')
                self.last_grasp_result = None
                self.show_grasp_result_window(all_results)
                return
            
            # 🆕 KHÔNG tự chọn instance theo mask_area (không thuộc phương pháp
            # trong bản thảo). Chỉ dùng 1 pose duy nhất khi:
            #  a) target_instance_index được chỉ định rõ (ROI/người dùng/lệnh
            #     ngoài) trong pipeline_config['grasp']['target_instance_index'], hoặc
            #  b) chỉ có ĐÚNG 1 instance thành công (không phải "chọn" theo tiêu
            #     chí nào, chỉ vì đó là lựa chọn duy nhất).
            # Nếu có NHIỀU instance hợp lệ mà CHƯA có target chỉ định sẵn, mở
            # DIALOG để NGƯỜI DÙNG chọn tường minh (chỉ áp dụng cho LẦN CHẠY
            # HIỆN TẠI - không lưu lại cho lần chạy sau, vì layout instance có
            # thể khác ở lần capture kế tiếp).
            preexisting_target = self.pipeline_config['grasp'].get('target_instance_index')
            if len(success_results) > 1 and preexisting_target is None:
                chosen_idx = self._prompt_select_target_dialog(success_results)
                if chosen_idx is not None:
                    self.pipeline_config['grasp']['target_instance_index'] = chosen_idx
                try:
                    selected, select_logs = self._select_grasp_target(success_results)
                finally:
                    # Reset lại config sau khi dùng xong - mỗi lần chạy mới PHẢI
                    # được chỉ định lại tường minh, tránh dùng nhầm target của
                    # lần chạy trước cho layout instance mới.
                    self.pipeline_config['grasp']['target_instance_index'] = preexisting_target
            else:
                selected, select_logs = self._select_grasp_target(success_results)
            
            for log_line in select_logs:
                print(log_line)
            
            self.last_grasp_result = selected
            selected_idx = selected['instance_index'] if selected is not None else None
            
            if selected is not None:
                self.status_label.config(
                    text=f"🟢 Grasp Pose OK! ({len(success_results)}/{len(all_results)} instance thành công, "
                         f"target=instance {selected_idx + 1})",
                    foreground='green'
                )
            else:
                self.status_label.config(
                    text=f"🟡 {len(success_results)}/{len(all_results)} instance thành công nhưng "
                         f"CHƯA có target được chỉ định (xem cửa sổ kết quả)",
                    foreground='orange'
                )
            
            self.show_grasp_result_window(all_results, selected_instance_index=selected_idx)
            self.save_grasp_result_json(all_results, selected_instance_index=selected_idx)
            
        except Exception as e:
            print(f"❌ Lỗi run_hybrid_grasp_pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"🔴 Lỗi: {e}", foreground='red')
        finally:
            if was_previewing:
                self._set_preview_active(True)

    def show_grasp_result_window(self, all_results, selected_instance_index=None):
        """Hiển thị bảng kết quả grasp pose (camera frame + robot base frame nếu có calib).
        selected_instance_index: instance được CHỈ ĐỊNH RÕ làm target (qua config
        target_instance_index, hoặc do chỉ có đúng 1 instance thành công) - KHÔNG
        phải kết quả của một quy tắc tự động chọn theo mask_area. Có thể là None
        nếu có nhiều instance hợp lệ nhưng chưa xác định target.
        """
        if self.grasp_result_window is None or not tk.Toplevel.winfo_exists(self.grasp_result_window):
            self.grasp_result_window = tk.Toplevel(self.root)
            self.grasp_result_window.title("🤖 Kết quả Grasp Pose (Hybrid 2D-3D)")
            self.grasp_result_window.geometry("820x600")
            self.grasp_result_window.configure(bg='#1e1e1e')
            self.grasp_result_text = tk.Text(self.grasp_result_window, width=100, height=34,
                                              bg='#1e1e1e', fg='#00ff88', font=('Consolas', 10))
            self.grasp_result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.grasp_result_text.delete('1.0', tk.END)
        
        lines = []
        lines.append("=" * 88)
        lines.append("KẾT QUẢ HYBRID 2D-3D GRASP PIPELINE")
        calib_txt = ('✅ ĐÃ LOAD (' + self.handeye_note + ')') if self.T_B_C is not None else '❌ CHƯA CÓ - chỉ hiển thị camera frame'
        lines.append(f"Hand-eye calibration: {calib_txt}")
        target_class_id = self.pipeline_config['grasp']['target_class_id']
        lines.append(f"Target class_id: {target_class_id}  |  Chế độ chọn target: "
                      f"{self.pipeline_config['grasp']['target_selection_rule']} "
                      f"(KHÔNG tự chọn theo mask_area)")
        lines.append("=" * 88 + "\n")
        
        for r in all_results:
            idx = r['instance_index']
            cls = r['class_id']
            is_selected = (selected_instance_index is not None and idx == selected_instance_index)
            marker = "  ⭐ [TARGET ĐƯỢC CHỈ ĐỊNH]" if is_selected else ""
            lines.append(f"--- Instance {idx+1} (class={cls}){marker} ---")
            if not r['success']:
                lines.append(f"  ❌ THẤT BẠI: {r['reason']}\n")
                continue
            
            pos_c = r['grasp_point_C'] * 1000.0
            rotvec_c = self.hybrid_rotation_matrix_to_rotvec(r['R_C_G'])
            
            lines.append(f"  ✅ Số nhánh hợp lệ: {r['branch_count']}, độ dài dominant path: {r['dominant_branch_len_px']} px, "
                          f"mask_area: {r['mask_area']} px")
            if not is_selected:
                lines.append(f"  ⚪ CANDIDATE POSE — KHÔNG GỬI ROBOT (chưa được chọn làm target)")
            lines.append(f"  📷 CAMERA FRAME:")
            lines.append(f"     X={pos_c[0]:8.2f}mm  Y={pos_c[1]:8.2f}mm  Z={pos_c[2]:8.2f}mm")
            lines.append(f"     Rx={rotvec_c[0]:7.4f}  Ry={rotvec_c[1]:7.4f}  Rz={rotvec_c[2]:7.4f}  (rad, rotation vector)")
            
            if self.T_B_C is not None:
                T_B_G = self.T_B_C @ r['T_C_G']
                pos_b = T_B_G[:3, 3] * 1000.0
                rotvec_b = self.hybrid_rotation_matrix_to_rotvec(T_B_G[:3, :3])
                lines.append(f"  🤖 ROBOT BASE FRAME:")
                lines.append(f"     X={pos_b[0]:8.2f}mm  Y={pos_b[1]:8.2f}mm  Z={pos_b[2]:8.2f}mm")
                lines.append(f"     Rx={rotvec_b[0]:7.4f}  Ry={rotvec_b[1]:7.4f}  Rz={rotvec_b[2]:7.4f}  (rad, rotation vector)")
                if is_selected:
                    lines.append(f"     >>> LỆNH KẸP ROBOT: X={pos_b[0]:.2f} Y={pos_b[1]:.2f} Z={pos_b[2]:.2f} "
                                  f"Rx={rotvec_b[0]:.4f} Ry={rotvec_b[1]:.4f} Rz={rotvec_b[2]:.4f}")
                else:
                    lines.append(f"     (CANDIDATE — KHÔNG gửi robot)")
            lines.append("")
        
        if selected_instance_index is None and len([r for r in all_results if r['success']]) > 1:
            lines.append("⚠️ Nhiều instance hợp lệ nhưng CHƯA có target được chỉ định rõ "
                         "(pipeline_config['grasp']['target_instance_index']). Hệ thống KHÔNG tự "
                         "chọn và KHÔNG gửi pose cho robot. Hãy chỉ định target_instance_index rõ ràng.")
        
        self.grasp_result_text.insert('1.0', '\n'.join(lines))

    def save_grasp_result_json(self, all_results, selected_instance_index=None):
        """Lưu log kết quả grasp pose ra Output_grasp/grasp_TIMESTAMP.json"""
        try:
            output_dir = "Output_grasp"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"grasp_{ts}.json")
            
            data = {
                'timestamp': ts, 'handeye_loaded': self.T_B_C is not None,
                'target_class_id': self.pipeline_config['grasp']['target_class_id'],
                'target_selection_rule': self.pipeline_config['grasp']['target_selection_rule'],
                'selected_instance_index': selected_instance_index,
                'instances': [],
            }
            
            for r in all_results:
                entry = {
                    'instance_index': r['instance_index'], 'class_id': r['class_id'],
                    'success': r['success'], 'reason': r['reason'],
                    'is_target': (selected_instance_index is not None and r['instance_index'] == selected_instance_index),
                }
                if r['success']:
                    pos_c = (r['grasp_point_C'] * 1000.0).tolist()
                    rotvec_c = self.hybrid_rotation_matrix_to_rotvec(r['R_C_G']).tolist()
                    entry['camera_frame'] = {'xyz_mm': pos_c, 'rotvec_rad': rotvec_c}
                    entry['mask_area'] = r['mask_area']
                    entry['branch_count'] = r['branch_count']
                    entry['num_points_bud'] = r['num_points_bud']
                    
                    if self.T_B_C is not None:
                        T_B_G = self.T_B_C @ r['T_C_G']
                        pos_b = (T_B_G[:3, 3] * 1000.0).tolist()
                        rotvec_b = self.hybrid_rotation_matrix_to_rotvec(T_B_G[:3, :3]).tolist()
                        entry['robot_base_frame'] = {'xyz_mm': pos_b, 'rotvec_rad': rotvec_b}
                data['instances'].append(entry)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Đã lưu kết quả grasp pose: {path}")
        except Exception as e:
            print(f"⚠️ Lỗi lưu grasp result JSON: {e}")

    # ==========================================================================
    # 🛠️ HAND-EYE CALIBRATION (Camera <-> Robot) bằng Checkerboard
    # ==========================================================================
    def load_handeye_calibration(self):
        """Load handeye_calibration.json (nếu có) -> self.T_B_C (4x4) hoặc None."""
        self.handeye_note = ''
        path = os.path.join(os.path.dirname(__file__), 'handeye_calibration.json')
        if not os.path.exists(path):
            self.T_B_C = None
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            eye_config = data.get('eye_config', 'eye_to_hand')
            if eye_config != 'eye_to_hand':
                print("⚠️ File calibration ở chế độ eye_in_hand -> không tự động áp dụng làm T_B_C")
                self.T_B_C = None
                return
            R = np.array(data['R'], dtype=np.float64)
            t = np.array(data['t'], dtype=np.float64)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            self.T_B_C = T
            self.handeye_note = f"{eye_config}, {data.get('num_samples', '?')} mẫu, {data.get('timestamp', '?')}"
            print(f"✅ Đã load hand-eye calibration: {path} ({self.handeye_note})")
        except Exception as e:
            print(f"⚠️ Lỗi load hand-eye calibration: {e}")
            self.T_B_C = None

    def euler_xyz_deg_to_matrix(self, rx_deg, ry_deg, rz_deg):
        """
        Chuyển góc Euler (độ, quy ước Rz*Ry*Rx - roll/pitch/yaw phổ biến) sang
        ma trận xoay. ⚠️ Nếu robot dùng quy ước khác, cần điều chỉnh hàm này.
        """
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def open_handeye_calibration_window(self):
        """Mở cửa sổ hiệu chỉnh Camera-Robot bằng checkerboard."""
        if self.handeye_calib_window is not None and tk.Toplevel.winfo_exists(self.handeye_calib_window):
            self.handeye_calib_window.lift()
            return
        
        win = tk.Toplevel(self.root)
        self.handeye_calib_window = win
        win.title("🛠️ Hiệu chỉnh Camera-Robot (Hand-Eye Calibration)")
        win.geometry("560x820")
        win.configure(bg='#2d2d30')
        
        self.handeye_samples = []
        
        main = ttk.Frame(win, style='Dark.TFrame')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        board_frame = ttk.LabelFrame(main, text="📐 Thông số bàn cờ Checkerboard", padding=10, style='Dark.TLabelframe')
        board_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(board_frame, text="Số góc trong theo hàng:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.he_rows_var = tk.IntVar(value=6)
        ttk.Entry(board_frame, textvariable=self.he_rows_var, width=8).grid(row=0, column=1, padx=5, pady=3)
        
        ttk.Label(board_frame, text="Số góc trong theo cột:", style='Dark.TLabel').grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.he_cols_var = tk.IntVar(value=9)
        ttk.Entry(board_frame, textvariable=self.he_cols_var, width=8).grid(row=0, column=3, padx=5, pady=3)
        
        ttk.Label(board_frame, text="Kích thước ô vuông (mm):", style='Dark.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.he_square_var = tk.DoubleVar(value=25.0)
        ttk.Entry(board_frame, textvariable=self.he_square_var, width=8).grid(row=1, column=1, padx=5, pady=3)
        
        ttk.Label(board_frame, text="Cấu hình camera:", style='Dark.TLabel').grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
        self.he_eye_config_var = tk.StringVar(value="eye_to_hand")
        eye_combo = ttk.Combobox(board_frame, textvariable=self.he_eye_config_var, state='readonly', width=16,
                                  values=["eye_to_hand", "eye_in_hand"])
        eye_combo.grid(row=1, column=3, padx=5, pady=3)
        
        ttk.Label(board_frame,
                  text="ℹ️ eye_to_hand: camera CỐ ĐỊNH, bàn cờ gắn trên gripper (khuyến nghị).\n"
                       "   eye_in_hand: camera gắn trên gripper, bàn cờ cố định trong không gian.",
                  style='Dark.TLabel', foreground='yellow', justify=tk.LEFT).grid(
            row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)
        
        pose_frame = ttk.LabelFrame(main, text="🤖 Pose Robot hiện tại (đọc từ Teach Pendant)", padding=10, style='Dark.TLabelframe')
        pose_frame.pack(fill=tk.X, pady=5)
        
        self.he_pose_vars = {}
        labels_units = [('X', 'mm'), ('Y', 'mm'), ('Z', 'mm'), ('Rx', 'deg'), ('Ry', 'deg'), ('Rz', 'deg')]
        for i, (name, unit) in enumerate(labels_units):
            ttk.Label(pose_frame, text=f"{name} ({unit}):", style='Dark.TLabel').grid(
                row=i // 3, column=(i % 3) * 2, sticky=tk.W, padx=5, pady=3)
            var = tk.DoubleVar(value=0.0)
            ttk.Entry(pose_frame, textvariable=var, width=10).grid(row=i // 3, column=(i % 3) * 2 + 1, padx=5, pady=3)
            self.he_pose_vars[name] = var
        
        ttk.Label(pose_frame, text="(Rx,Ry,Rz: góc Euler độ, quy ước R=Rz·Ry·Rx - điều chỉnh nếu robot khác quy ước)",
                  style='Dark.TLabel', foreground='cyan').grid(row=2, column=0, columnspan=6, sticky=tk.W, padx=5, pady=3)
        
        self.he_canvas = tk.Canvas(main, width=480, height=360, bg='black', highlightthickness=0)
        self.he_canvas.pack(pady=8)
        
        action_frame = ttk.Frame(main, style='Dark.TFrame')
        action_frame.pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="📸 Chụp mẫu hiệu chỉnh", command=self.capture_handeye_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Xoá mẫu cuối", command=self.remove_last_handeye_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="✅ Tính toán Calibration", command=self.compute_handeye_calibration).pack(side=tk.LEFT, padx=5)
        
        self.he_status_label = ttk.Label(main, text="Số mẫu đã chụp: 0", style='Dark.TLabel', foreground='cyan')
        self.he_status_label.pack(anchor=tk.W, pady=5)

    def capture_handeye_sample(self):
        """Chụp 1 mẫu hiệu chỉnh: phát hiện checkerboard + đọc pose robot user nhập."""
        try:
            rows = int(self.he_rows_var.get())
            cols = int(self.he_cols_var.get())
            square_mm = float(self.he_square_var.get())
            
            intr = self.hybrid_get_intrinsics()
            if intr is None:
                messagebox.showerror("Lỗi", "Không lấy được camera intrinsics!")
                return
            
            was_previewing = self.is_preview
            self._set_preview_active(False)
            try:
                with self.camera_lock:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
            finally:
                if was_previewing:
                    self._set_preview_active(True)
            
            if not color_frame:
                messagebox.showerror("Lỗi", "Không lấy được ảnh màu!")
                return
            
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            pattern_size = (cols, rows)
            found, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if not found:
                messagebox.showwarning(
                    "Không tìm thấy bàn cờ",
                    "Không phát hiện được checkerboard trong ảnh. Hãy đảm bảo bàn cờ nằm trọn trong khung hình.")
                return
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objp = np.zeros((rows * cols, 3), dtype=np.float64)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
            
            camera_matrix = np.array([
                [intr['fx'], 0, intr['cx']],
                [0, intr['fy'], intr['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.array(intr['coeffs'], dtype=np.float64).reshape(-1, 1)
            
            ok, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                messagebox.showerror("Lỗi", "solvePnP thất bại!")
                return
            
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.flatten()
            
            x_mm = self.he_pose_vars['X'].get()
            y_mm = self.he_pose_vars['Y'].get()
            z_mm = self.he_pose_vars['Z'].get()
            rx_deg = self.he_pose_vars['Rx'].get()
            ry_deg = self.he_pose_vars['Ry'].get()
            rz_deg = self.he_pose_vars['Rz'].get()
            
            R_gripper2base = self.euler_xyz_deg_to_matrix(rx_deg, ry_deg, rz_deg)
            t_gripper2base = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
            
            self.handeye_samples.append({
                'R_target2cam': R_target2cam, 't_target2cam': t_target2cam,
                'R_gripper2base': R_gripper2base, 't_gripper2base': t_gripper2base,
            })
            
            preview = color_image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners_refined, found)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            preview_resized = cv2.resize(preview_rgb, (480, 360))
            img_pil = Image.fromarray(preview_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.he_canvas.create_image(240, 180, image=img_tk)
            self.he_canvas.image = img_tk
            
            self.he_status_label.config(text=f"Số mẫu đã chụp: {len(self.handeye_samples)}", foreground='lime')
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi chụp mẫu hiệu chỉnh: {e}")
            import traceback
            traceback.print_exc()

    def remove_last_handeye_sample(self):
        if self.handeye_samples:
            self.handeye_samples.pop()
            self.he_status_label.config(text=f"Số mẫu đã chụp: {len(self.handeye_samples)}", foreground='cyan')

    def compute_handeye_calibration(self):
        """Tính toán hand-eye calibration (cv2.calibrateHandEye) từ các mẫu đã chụp."""
        n = len(self.handeye_samples)
        if n < 3:
            messagebox.showwarning("Chưa đủ mẫu", f"Cần ít nhất 3 mẫu (khuyến nghị >=10). Hiện có {n} mẫu.")
            return
        
        try:
            eye_config = self.he_eye_config_var.get()
            
            R_gripper2base_list = [s['R_gripper2base'] for s in self.handeye_samples]
            t_gripper2base_list = [s['t_gripper2base'] for s in self.handeye_samples]
            R_target2cam_list = [s['R_target2cam'] for s in self.handeye_samples]
            t_target2cam_list = [s['t_target2cam'].reshape(3, 1) for s in self.handeye_samples]
            
            if eye_config == 'eye_to_hand':
                # Camera cố định, bàn cờ gắn trên gripper: dùng nghịch đảo
                # (base2gripper) thay cho gripper2base -> kết quả trả về sẽ là
                # (R_cam2base, t_cam2base) = T_B_C trực tiếp.
                R_in, t_in = [], []
                for R_g2b, t_g2b in zip(R_gripper2base_list, t_gripper2base_list):
                    R_b2g = R_g2b.T
                    t_b2g = -R_b2g @ t_g2b
                    R_in.append(R_b2g)
                    t_in.append(t_b2g.reshape(3, 1))
            else:
                R_in = R_gripper2base_list
                t_in = [t.reshape(3, 1) for t in t_gripper2base_list]
            
            R_cam2x, t_cam2x = cv2.calibrateHandEye(
                R_in, t_in, R_target2cam_list, t_target2cam_list,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            T_B_C = np.eye(4)
            T_B_C[:3, :3] = R_cam2x
            T_B_C[:3, 3] = t_cam2x.flatten()
            
            if eye_config == 'eye_in_hand':
                messagebox.showwarning(
                    "Eye-in-hand",
                    "Chế độ eye_in_hand cần pose robot LIVE tại thời điểm chạy để quy đổi "
                    "sang robot base frame. Ứng dụng chưa có kết nối robot trực tiếp nên sẽ "
                    "lưu T_cam2gripper nhưng KHÔNG dùng làm T_B_C mặc định."
                )
                self.T_B_C = None
            else:
                self.T_B_C = T_B_C
            
            data = {
                'eye_config': eye_config,
                'R': R_cam2x.tolist(),
                't': t_cam2x.flatten().tolist(),
                'num_samples': n,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'square_size_mm': float(self.he_square_var.get()),
                'board_rows': int(self.he_rows_var.get()),
                'board_cols': int(self.he_cols_var.get()),
            }
            
            path = os.path.join(os.path.dirname(__file__), 'handeye_calibration.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.handeye_note = f"{eye_config}, {n} mẫu, {data['timestamp']}"
            if hasattr(self, 'handeye_status_label'):
                self.handeye_status_label.config(
                    text="🛠️ Hand-eye: " + (self.handeye_note if self.T_B_C is not None else "eye_in_hand (chưa áp dụng)"),
                    foreground='lime' if self.T_B_C is not None else 'orange')
            
            messagebox.showinfo("Thành công",
                                 f"Đã tính toán và lưu hand-eye calibration ({eye_config}, {n} mẫu)\nFile: {path}")
            self.he_status_label.config(text=f"✅ Đã tính xong calibration ({n} mẫu)", foreground='lime')
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi tính toán calibration: {e}")
            import traceback
            traceback.print_exc()

    # ==================== END HYBRID 2D-3D GRASP PIPELINE ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseGUI(root)
    root.mainloop()
