"""
PLY Editor - Ứng dụng chỉnh sửa file Point Cloud (.ply, .pcd)
Chức năng đầy đủ:
- Load/Save file PLY
- Xem thông tin chi tiết
- Chỉnh sửa: màu sắc, vị trí, xoay, scale
- Lọc điểm, remove outliers, downsample
- Crop theo bounding box
- Preview và Undo
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import open3d as o3d
import numpy as np
import os
import copy
from datetime import datetime


class PLYEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PLY Editor - Point Cloud Editor")
        self.root.geometry("1200x800")
        
        # Dữ liệu
        self.current_pcd = None  # Point cloud hiện tại
        self.original_pcd = None  # Bản gốc để undo
        self.history = []  # Lịch sử chỉnh sửa
        self.current_file = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        
        # ============= HEADER =============
        header_frame = tk.Frame(self.root, bg="#34495e", height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🛠️ PLY POINT CLOUD EDITOR 🛠️",
            font=("Arial", 20, "bold"),
            bg="#34495e",
            fg="white"
        )
        title_label.pack(expand=True)
        
        # ============= MAIN CONTAINER =============
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Controls
        left_panel = tk.Frame(main_container, width=400, bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Right Panel - Info
        right_panel = tk.Frame(main_container, bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # ============= LEFT PANEL - FILE OPERATIONS =============
        self.setup_file_section(left_panel)
        
        # ============= LEFT PANEL - TRANSFORM OPERATIONS =============
        self.setup_transform_section(left_panel)
        
        # ============= LEFT PANEL - COLOR OPERATIONS =============
        self.setup_color_section(left_panel)
        
        # ============= LEFT PANEL - FILTER OPERATIONS =============
        self.setup_filter_section(left_panel)
        
        # ============= RIGHT PANEL - INFO =============
        self.setup_info_section(right_panel)
        
        # ============= BOTTOM - ACTION BUTTONS =============
        self.setup_action_buttons()
    
    def setup_file_section(self, parent):
        """File operations"""
        frame = tk.LabelFrame(parent, text="📁 File Operations", font=("Arial", 11, "bold"), 
                             bg="#ecf0f1", fg="#2c3e50", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_load = tk.Button(frame, text="📂 Load PLY File", command=self.load_file,
                            bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=25, height=2)
        btn_load.pack(pady=5)
        
        self.file_label = tk.Label(frame, text="No file loaded", font=("Arial", 9),
                                   bg="#ecf0f1", fg="#7f8c8d", wraplength=350, justify=tk.LEFT)
        self.file_label.pack(pady=5)
        
        btn_frame = tk.Frame(frame, bg="#ecf0f1")
        btn_frame.pack(pady=5)
        
        btn_save = tk.Button(btn_frame, text="💾 Save", command=self.save_file,
                            bg="#27ae60", fg="white", font=("Arial", 9, "bold"), width=12)
        btn_save.pack(side=tk.LEFT, padx=2)
        
        btn_save_as = tk.Button(btn_frame, text="💾 Save As", command=self.save_file_as,
                               bg="#16a085", fg="white", font=("Arial", 9, "bold"), width=12)
        btn_save_as.pack(side=tk.LEFT, padx=2)
    
    def setup_transform_section(self, parent):
        """Transform operations"""
        frame = tk.LabelFrame(parent, text="🔄 Transform", font=("Arial", 11, "bold"),
                             bg="#ecf0f1", fg="#2c3e50", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Translate
        tk.Label(frame, text="Translate (X, Y, Z):", font=("Arial", 9, "bold"),
                bg="#ecf0f1").pack(anchor=tk.W)
        
        translate_frame = tk.Frame(frame, bg="#ecf0f1")
        translate_frame.pack(fill=tk.X, pady=5)
        
        self.translate_x = tk.Entry(translate_frame, width=8)
        self.translate_x.insert(0, "0.0")
        self.translate_x.pack(side=tk.LEFT, padx=2)
        
        self.translate_y = tk.Entry(translate_frame, width=8)
        self.translate_y.insert(0, "0.0")
        self.translate_y.pack(side=tk.LEFT, padx=2)
        
        self.translate_z = tk.Entry(translate_frame, width=8)
        self.translate_z.insert(0, "0.0")
        self.translate_z.pack(side=tk.LEFT, padx=2)
        
        tk.Button(translate_frame, text="Apply", command=self.apply_translate,
                 bg="#3498db", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Rotate
        tk.Label(frame, text="Rotate (deg) around axis:", font=("Arial", 9, "bold"),
                bg="#ecf0f1").pack(anchor=tk.W, pady=(10, 0))
        
        rotate_frame = tk.Frame(frame, bg="#ecf0f1")
        rotate_frame.pack(fill=tk.X, pady=5)
        
        self.rotate_angle = tk.Entry(rotate_frame, width=8)
        self.rotate_angle.insert(0, "0")
        rotate_frame_label = tk.Label(rotate_frame, text="Angle:", bg="#ecf0f1")
        rotate_frame_label.pack(side=tk.LEFT)
        self.rotate_angle.pack(side=tk.LEFT, padx=2)
        
        self.rotate_axis = ttk.Combobox(rotate_frame, values=["X", "Y", "Z"], width=5, state="readonly")
        self.rotate_axis.set("Z")
        self.rotate_axis.pack(side=tk.LEFT, padx=2)
        
        tk.Button(rotate_frame, text="Apply", command=self.apply_rotate,
                 bg="#3498db", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Scale
        tk.Label(frame, text="Scale (uniform):", font=("Arial", 9, "bold"),
                bg="#ecf0f1").pack(anchor=tk.W, pady=(10, 0))
        
        scale_frame = tk.Frame(frame, bg="#ecf0f1")
        scale_frame.pack(fill=tk.X, pady=5)
        
        self.scale_factor = tk.Entry(scale_frame, width=8)
        self.scale_factor.insert(0, "1.0")
        self.scale_factor.pack(side=tk.LEFT, padx=2)
        
        tk.Button(scale_frame, text="Apply Scale", command=self.apply_scale,
                 bg="#3498db", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
    
    def setup_color_section(self, parent):
        """Color operations"""
        frame = tk.LabelFrame(parent, text="🎨 Color", font=("Arial", 11, "bold"),
                             bg="#ecf0f1", fg="#2c3e50", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Uniform color
        btn_frame = tk.Frame(frame, bg="#ecf0f1")
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="🎨 Set Uniform Color", command=self.set_uniform_color,
                 bg="#9b59b6", fg="white", font=("Arial", 9, "bold"), width=20).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="🌈 Random Colors", command=self.set_random_colors,
                 bg="#e67e22", fg="white", font=("Arial", 9, "bold"), width=15).pack(side=tk.LEFT, padx=2)
        
        # Color by coordinate
        tk.Label(frame, text="Color by coordinate:", font=("Arial", 9, "bold"),
                bg="#ecf0f1").pack(anchor=tk.W, pady=(10, 0))
        
        coord_frame = tk.Frame(frame, bg="#ecf0f1")
        coord_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(coord_frame, text="X-axis", command=lambda: self.color_by_axis('x'),
                 bg="#e74c3c", fg="white", font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(coord_frame, text="Y-axis", command=lambda: self.color_by_axis('y'),
                 bg="#2ecc71", fg="white", font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(coord_frame, text="Z-axis", command=lambda: self.color_by_axis('z'),
                 bg="#3498db", fg="white", font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=2)
    
    def setup_filter_section(self, parent):
        """Filter operations"""
        frame = tk.LabelFrame(parent, text="🔍 Filter & Clean", font=("Arial", 11, "bold"),
                             bg="#ecf0f1", fg="#2c3e50", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Downsample
        downsample_frame = tk.Frame(frame, bg="#ecf0f1")
        downsample_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(downsample_frame, text="Voxel size:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)
        self.voxel_size = tk.Entry(downsample_frame, width=8)
        self.voxel_size.insert(0, "0.01")
        self.voxel_size.pack(side=tk.LEFT, padx=5)
        
        tk.Button(downsample_frame, text="Downsample", command=self.apply_downsample,
                 bg="#f39c12", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Remove outliers
        outlier_frame = tk.Frame(frame, bg="#ecf0f1")
        outlier_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(outlier_frame, text="Neighbors:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)
        self.outlier_nb = tk.Entry(outlier_frame, width=6)
        self.outlier_nb.insert(0, "20")
        self.outlier_nb.pack(side=tk.LEFT, padx=2)
        
        tk.Label(outlier_frame, text="Std:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))
        self.outlier_std = tk.Entry(outlier_frame, width=6)
        self.outlier_std.insert(0, "2.0")
        self.outlier_std.pack(side=tk.LEFT, padx=2)
        
        tk.Button(outlier_frame, text="Remove", command=self.remove_outliers,
                 bg="#e74c3c", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Crop
        tk.Label(frame, text="Crop by Z-range:", font=("Arial", 9, "bold"),
                bg="#ecf0f1").pack(anchor=tk.W, pady=(10, 0))
        
        crop_frame = tk.Frame(frame, bg="#ecf0f1")
        crop_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(crop_frame, text="Min Z:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)
        self.crop_min_z = tk.Entry(crop_frame, width=8)
        self.crop_min_z.insert(0, "0.0")
        self.crop_min_z.pack(side=tk.LEFT, padx=2)
        
        tk.Label(crop_frame, text="Max Z:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT, padx=(5, 0))
        self.crop_max_z = tk.Entry(crop_frame, width=8)
        self.crop_max_z.insert(0, "1.0")
        self.crop_max_z.pack(side=tk.LEFT, padx=2)
        
        tk.Button(crop_frame, text="Crop", command=self.apply_crop_z,
                 bg="#8e44ad", fg="white", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
    
    def setup_info_section(self, parent):
        """Info display"""
        frame = tk.LabelFrame(parent, text="ℹ️ Point Cloud Information", font=("Arial", 12, "bold"),
                             bg="#ecf0f1", fg="#2c3e50", padx=15, pady=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget với scrollbar
        text_frame = tk.Frame(frame, bg="#ecf0f1")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, font=("Courier New", 10),
                                bg="#ffffff", fg="#2c3e50", yscrollcommand=scrollbar.set,
                                relief=tk.FLAT, padx=10, pady=10)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)
        
        self.update_info_display()
    
    def setup_action_buttons(self):
        """Bottom action buttons"""
        action_frame = tk.Frame(self.root, bg="#ecf0f1", height=80)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        action_frame.pack_propagate(False)
        
        # Left side
        tk.Button(action_frame, text="↶ Undo (Reset)", command=self.undo,
                 bg="#95a5a6", fg="white", font=("Arial", 11, "bold"), 
                 width=15, height=2).pack(side=tk.LEFT, padx=10, pady=15)
        
        # Center
        tk.Button(action_frame, text="👁️ Preview", command=self.preview,
                 bg="#3498db", fg="white", font=("Arial", 12, "bold"), 
                 width=20, height=2).pack(side=tk.LEFT, padx=10, pady=15)
        
        tk.Button(action_frame, text="📊 Compute Normals", command=self.compute_normals,
                 bg="#16a085", fg="white", font=("Arial", 11, "bold"), 
                 width=18, height=2).pack(side=tk.LEFT, padx=10, pady=15)
        
        # Right side
        tk.Button(action_frame, text="🗑️ Clear", command=self.clear_all,
                 bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), 
                 width=12, height=2).pack(side=tk.RIGHT, padx=10, pady=15)
    
    # ============= CORE FUNCTIONS =============
    
    def load_file(self):
        """Load PLY/PCD file"""
        filepath = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=[("Point Cloud Files", "*.ply *.pcd"), ("PLY Files", "*.ply"), 
                      ("PCD Files", "*.pcd"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            print(f"Loading: {filepath}")
            pcd = o3d.io.read_point_cloud(filepath)
            
            if len(pcd.points) == 0:
                messagebox.showerror("Error", "Point cloud is empty!")
                return
            
            self.current_pcd = pcd
            self.original_pcd = copy.deepcopy(pcd)  # Backup
            self.current_file = filepath
            
            self.file_label.config(text=f"✅ {os.path.basename(filepath)}", fg="#27ae60")
            self.update_info_display()
            
            messagebox.showinfo("Success", f"Loaded {len(pcd.points):,} points!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def save_file(self):
        """Save to current file"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        if not self.current_file:
            self.save_file_as()
            return
        
        try:
            o3d.io.write_point_cloud(self.current_file, self.current_pcd)
            messagebox.showinfo("Success", f"Saved to:\n{self.current_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
    
    def save_file_as(self):
        """Save as new file"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Point Cloud As",
            defaultextension=".ply",
            filetypes=[("PLY Files", "*.ply"), ("PCD Files", "*.pcd"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            o3d.io.write_point_cloud(filepath, self.current_pcd)
            self.current_file = filepath
            self.file_label.config(text=f"✅ {os.path.basename(filepath)}", fg="#27ae60")
            messagebox.showinfo("Success", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{str(e)}")
    
    def undo(self):
        """Undo all changes"""
        if not self.original_pcd:
            messagebox.showwarning("Warning", "No original point cloud to restore!")
            return
        
        result = messagebox.askyesno("Confirm", "Reset to original point cloud?")
        if result:
            self.current_pcd = copy.deepcopy(self.original_pcd)
            self.update_info_display()
            messagebox.showinfo("Success", "Reset to original!")
    
    def preview(self):
        """Preview point cloud"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud to preview!")
            return
        
        try:
            print("\n" + "="*60)
            print("🔷 Opening Preview...")
            print("="*60)
            
            # Tạo coordinate frame
            bbox = self.current_pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            frame_size = np.max(extent) * 0.1
            frame_size = np.clip(frame_size, 0.01, 0.5)
            
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size, origin=[0, 0, 0]
            )
            
            o3d.visualization.draw_geometries(
                [self.current_pcd, mesh_frame],
                window_name="PLY Editor - Preview",
                width=1280,
                height=960
            )
            
            print("✅ Preview closed\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed:\n{str(e)}")
    
    def clear_all(self):
        """Clear all data"""
        result = messagebox.askyesno("Confirm", "Clear all data?")
        if result:
            self.current_pcd = None
            self.original_pcd = None
            self.current_file = None
            self.file_label.config(text="No file loaded", fg="#7f8c8d")
            self.update_info_display()
    
    # ============= TRANSFORM FUNCTIONS =============
    
    def apply_translate(self):
        """Apply translation"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            tx = float(self.translate_x.get())
            ty = float(self.translate_y.get())
            tz = float(self.translate_z.get())
            
            translation = np.array([tx, ty, tz])
            self.current_pcd.translate(translation)
            
            self.update_info_display()
            messagebox.showinfo("Success", f"Translated by ({tx}, {ty}, {tz})")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid translation values!")
    
    def apply_rotate(self):
        """Apply rotation"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            angle = float(self.rotate_angle.get())
            axis = self.rotate_axis.get()
            
            # Convert to radians
            angle_rad = np.radians(angle)
            
            # Get rotation matrix
            if axis == "X":
                R = self.current_pcd.get_rotation_matrix_from_xyz((angle_rad, 0, 0))
            elif axis == "Y":
                R = self.current_pcd.get_rotation_matrix_from_xyz((0, angle_rad, 0))
            else:  # Z
                R = self.current_pcd.get_rotation_matrix_from_xyz((0, 0, angle_rad))
            
            # Rotate around center
            center = self.current_pcd.get_center()
            self.current_pcd.rotate(R, center=center)
            
            self.update_info_display()
            messagebox.showinfo("Success", f"Rotated {angle}° around {axis}-axis")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid rotation values!")
    
    def apply_scale(self):
        """Apply scaling"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            scale = float(self.scale_factor.get())
            
            if scale <= 0:
                messagebox.showerror("Error", "Scale must be positive!")
                return
            
            center = self.current_pcd.get_center()
            self.current_pcd.scale(scale, center=center)
            
            self.update_info_display()
            messagebox.showinfo("Success", f"Scaled by {scale}x")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid scale value!")
    
    # ============= COLOR FUNCTIONS =============
    
    def set_uniform_color(self):
        """Set uniform color"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        color = colorchooser.askcolor(title="Choose Color")
        if color[0]:  # RGB tuple
            rgb = np.array(color[0]) / 255.0  # Normalize to [0, 1]
            self.current_pcd.paint_uniform_color(rgb)
            self.update_info_display()
            messagebox.showinfo("Success", f"Color set to RGB{tuple(color[0])}")
    
    def set_random_colors(self):
        """Set random colors"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        n_points = len(self.current_pcd.points)
        random_colors = np.random.rand(n_points, 3)
        self.current_pcd.colors = o3d.utility.Vector3dVector(random_colors)
        
        self.update_info_display()
        messagebox.showinfo("Success", "Random colors applied!")
    
    def color_by_axis(self, axis):
        """Color by coordinate axis"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        points = np.asarray(self.current_pcd.points)
        
        if axis == 'x':
            values = points[:, 0]
            color_name = "X (Red gradient)"
        elif axis == 'y':
            values = points[:, 1]
            color_name = "Y (Green gradient)"
        else:  # z
            values = points[:, 2]
            color_name = "Z (Blue gradient)"
        
        # Normalize to [0, 1]
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Create color gradient
        colors = np.zeros((len(points), 3))
        if axis == 'x':
            colors[:, 0] = normalized  # Red channel
        elif axis == 'y':
            colors[:, 1] = normalized  # Green channel
        else:
            colors[:, 2] = normalized  # Blue channel
        
        self.current_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.update_info_display()
        messagebox.showinfo("Success", f"Colored by {color_name}")
    
    # ============= FILTER FUNCTIONS =============
    
    def apply_downsample(self):
        """Downsample point cloud"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            voxel_size = float(self.voxel_size.get())
            
            if voxel_size <= 0:
                messagebox.showerror("Error", "Voxel size must be positive!")
                return
            
            original_count = len(self.current_pcd.points)
            self.current_pcd = self.current_pcd.voxel_down_sample(voxel_size=voxel_size)
            new_count = len(self.current_pcd.points)
            
            self.update_info_display()
            messagebox.showinfo("Success", 
                              f"Downsampled:\n{original_count:,} → {new_count:,} points\n"
                              f"({100*new_count/original_count:.1f}% retained)")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid voxel size!")
    
    def remove_outliers(self):
        """Remove statistical outliers"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            nb_neighbors = int(self.outlier_nb.get())
            std_ratio = float(self.outlier_std.get())
            
            original_count = len(self.current_pcd.points)
            
            cl, ind = self.current_pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            self.current_pcd = self.current_pcd.select_by_index(ind)
            
            new_count = len(self.current_pcd.points)
            removed = original_count - new_count
            
            self.update_info_display()
            messagebox.showinfo("Success", 
                              f"Removed {removed:,} outliers\n"
                              f"Remaining: {new_count:,} points")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid outlier parameters!")
    
    def apply_crop_z(self):
        """Crop by Z range"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            min_z = float(self.crop_min_z.get())
            max_z = float(self.crop_max_z.get())
            
            if min_z >= max_z:
                messagebox.showerror("Error", "Min Z must be less than Max Z!")
                return
            
            points = np.asarray(self.current_pcd.points)
            z_coords = points[:, 2]
            
            # Filter indices
            mask = (z_coords >= min_z) & (z_coords <= max_z)
            indices = np.where(mask)[0]
            
            original_count = len(points)
            self.current_pcd = self.current_pcd.select_by_index(indices)
            new_count = len(self.current_pcd.points)
            
            self.update_info_display()
            messagebox.showinfo("Success", 
                              f"Cropped to Z ∈ [{min_z}, {max_z}]\n"
                              f"Kept {new_count:,} / {original_count:,} points")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid crop values!")
    
    def compute_normals(self):
        """Compute surface normals"""
        if not self.current_pcd:
            messagebox.showwarning("Warning", "No point cloud loaded!")
            return
        
        try:
            self.current_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.update_info_display()
            messagebox.showinfo("Success", "Normals computed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute normals:\n{str(e)}")
    
    # ============= INFO DISPLAY =============
    
    def update_info_display(self):
        """Update information panel"""
        self.info_text.delete(1.0, tk.END)
        
        if not self.current_pcd:
            self.info_text.insert(tk.END, "No point cloud loaded.\n\n")
            self.info_text.insert(tk.END, "Click 'Load PLY File' to start editing.")
            return
        
        points = np.asarray(self.current_pcd.points)
        n_points = len(points)
        
        info = f"📌 POINT CLOUD STATISTICS\n"
        info += f"{'='*50}\n\n"
        
        # Basic info
        info += f"Number of points:  {n_points:,}\n"
        
        # Has colors?
        has_colors = self.current_pcd.has_colors()
        info += f"Has colors:        {'✅ Yes' if has_colors else '❌ No'}\n"
        
        # Has normals?
        has_normals = self.current_pcd.has_normals()
        info += f"Has normals:       {'✅ Yes' if has_normals else '❌ No'}\n"
        
        info += f"\n{'─'*50}\n"
        info += f"📐 GEOMETRY\n"
        info += f"{'─'*50}\n\n"
        
        # Bounding box
        bbox = self.current_pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        extent = bbox.get_extent()
        
        info += f"Bounding Box:\n"
        info += f"  Min: ({min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f})\n"
        info += f"  Max: ({max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f})\n"
        info += f"  Extent (W×H×D): ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})\n\n"
        
        # Center
        center = self.current_pcd.get_center()
        info += f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})\n\n"
        
        # Statistics
        info += f"{'─'*50}\n"
        info += f"📊 POINT STATISTICS\n"
        info += f"{'─'*50}\n\n"
        
        info += f"X: min={points[:, 0].min():.4f}, max={points[:, 0].max():.4f}, "
        info += f"mean={points[:, 0].mean():.4f}\n"
        
        info += f"Y: min={points[:, 1].min():.4f}, max={points[:, 1].max():.4f}, "
        info += f"mean={points[:, 1].mean():.4f}\n"
        
        info += f"Z: min={points[:, 2].min():.4f}, max={points[:, 2].max():.4f}, "
        info += f"mean={points[:, 2].mean():.4f}\n"
        
        # Color info
        if has_colors:
            colors = np.asarray(self.current_pcd.colors)
            info += f"\n{'─'*50}\n"
            info += f"🎨 COLOR STATISTICS\n"
            info += f"{'─'*50}\n\n"
            info += f"R: min={colors[:, 0].min():.3f}, max={colors[:, 0].max():.3f}, "
            info += f"mean={colors[:, 0].mean():.3f}\n"
            info += f"G: min={colors[:, 1].min():.3f}, max={colors[:, 1].max():.3f}, "
            info += f"mean={colors[:, 1].mean():.3f}\n"
            info += f"B: min={colors[:, 2].min():.3f}, max={colors[:, 2].max():.3f}, "
            info += f"mean={colors[:, 2].mean():.3f}\n"
        
        # File info
        if self.current_file:
            info += f"\n{'─'*50}\n"
            info += f"📁 FILE INFO\n"
            info += f"{'─'*50}\n\n"
            info += f"Path: {self.current_file}\n"
            info += f"Size: {os.path.getsize(self.current_file) / 1024:.1f} KB\n"
        
        self.info_text.insert(tk.END, info)


def main():
    root = tk.Tk()
    app = PLYEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
