"""
Point Cloud Viewer - Ứng dụng xem point cloud từ thư mục Output_pointcloud
Chức năng:
- Hiển thị danh sách file .pcd/.ply trong thư mục
- Sắp xếp theo thời gian (mới nhất trước)
- Xem point cloud bằng Open3D viewer
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import open3d as o3d
import os
from datetime import datetime
import glob


class PointCloudViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Viewer - DATN")
        self.root.geometry("800x600")
        
        # Thư mục chứa point cloud
        self.output_folder = "Output_pointcloud"
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Tùy chọn hiển thị trục
        self.show_axis_var = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.refresh_file_list()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        
        # Frame tiêu đề
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🔷 POINT CLOUD VIEWER 🔷",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(expand=True)
        
        # Frame thông tin
        info_frame = tk.Frame(self.root, bg="#ecf0f1", height=50)
        info_frame.pack(fill=tk.X, pady=5)
        info_frame.pack_propagate(False)
        
        self.info_label = tk.Label(
            info_frame,
            text=f"📁 Thư mục: {self.output_folder}",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # Nút chọn thư mục
        browse_btn = tk.Button(
            info_frame,
            text="📂 Chọn thư mục",
            font=("Arial", 9, "bold"),
            bg="#3498db",
            fg="white",
            command=self.browse_folder
        )
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.count_label = tk.Label(
            info_frame,
            text="Số file: 0",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        self.count_label.pack(side=tk.RIGHT, padx=20)
        
        # Frame danh sách file
        list_frame = tk.Frame(self.root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label
        tk.Label(
            list_frame, 
            text="Danh sách Point Cloud (mới nhất trước):",
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Treeview để hiển thị danh sách file
        columns = ("filename", "size", "date")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Định nghĩa cột
        self.tree.heading("filename", text="Tên file")
        self.tree.heading("size", text="Kích thước")
        self.tree.heading("date", text="Ngày tạo")
        
        self.tree.column("filename", width=400)
        self.tree.column("size", width=120, anchor=tk.CENTER)
        self.tree.column("date", width=200, anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click để xem
        self.tree.bind("<Double-1>", lambda e: self.view_selected())
        
        # Frame nút điều khiển
        button_frame = tk.Frame(self.root, bg="#ecf0f1", height=80)
        button_frame.pack(fill=tk.X, pady=10)
        button_frame.pack_propagate(False)
        
        # Checkbox để bật/tắt hiển thị trục
        axis_checkbox = tk.Checkbutton(
            button_frame,
            text="🌿 Hiển thị trục tăng trưởng (axis)",
            variable=self.show_axis_var,
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#2c3e50",
            selectcolor="#3498db"
        )
        axis_checkbox.pack(side=tk.LEFT, padx=20)
        
        # Nút Refresh
        refresh_btn = tk.Button(
            button_frame,
            text="🔄 Làm mới",
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            width=15,
            height=2,
            command=self.refresh_file_list
        )
        refresh_btn.pack(side=tk.LEFT, padx=20)
        
        # Nút Xem file được chọn
        view_btn = tk.Button(
            button_frame,
            text="👁️ Xem Point Cloud",
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            width=20,
            height=2,
            command=self.view_selected
        )
        view_btn.pack(side=tk.LEFT, padx=10)
        
        # Nút Xem file mới nhất
        view_latest_btn = tk.Button(
            button_frame,
            text="⚡ Xem file mới nhất",
            font=("Arial", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            width=20,
            height=2,
            command=self.view_latest
        )
        view_latest_btn.pack(side=tk.LEFT, padx=10)
        
        # Nút Xóa file được chọn
        delete_btn = tk.Button(
            button_frame,
            text="🗑️ Xóa file",
            font=("Arial", 11, "bold"),
            bg="#95a5a6",
            fg="white",
            width=15,
            height=2,
            command=self.delete_selected
        )
        delete_btn.pack(side=tk.RIGHT, padx=20)
    
    def browse_folder(self):
        """Chọn thư mục chứa point cloud"""
        folder_selected = filedialog.askdirectory(
            title="Chọn thư mục chứa Point Cloud",
            initialdir=self.output_folder
        )
        
        if folder_selected:
            self.output_folder = folder_selected
            self.info_label.config(text=f"📁 Thư mục: {self.output_folder}")
            self.refresh_file_list()
        
    def refresh_file_list(self):
        """Làm mới danh sách file"""
        # Xóa danh sách cũ
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Tìm tất cả file .pcd và .ply (bỏ qua file axis)
        pcd_files = glob.glob(os.path.join(self.output_folder, "*.pcd"))
        ply_files = glob.glob(os.path.join(self.output_folder, "*.ply"))
        all_files = pcd_files + ply_files
        
        # Lọc bỏ file axis (không hiển thị riêng)
        all_files = [f for f in all_files if not f.endswith("_axis.ply")]
        
        if not all_files:
            self.count_label.config(text="Số file: 0")
            return
        
        # Sắp xếp theo thời gian (mới nhất trước)
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        # Thêm vào treeview
        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            # Kích thước file
            size_bytes = os.path.getsize(filepath)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            # Ngày tạo
            mtime = os.path.getmtime(filepath)
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Thêm vào tree
            self.tree.insert("", tk.END, values=(filename, size_str, date_str), tags=(filepath,))
        
        # Cập nhật số lượng
        self.count_label.config(text=f"Số file: {len(all_files)}")
        
        # Tự động chọn file đầu tiên (mới nhất)
        if all_files:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
    
    def view_selected(self):
        """Xem point cloud được chọn"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một file để xem!")
            return
        
        # Lấy đường dẫn file từ tags
        item = selection[0]
        filepath = self.tree.item(item, "tags")[0]
        
        self.view_point_cloud(filepath)
    
    def view_latest(self):
        """Xem point cloud mới nhất"""
        if not self.tree.get_children():
            messagebox.showwarning("Cảnh báo", "Không có file nào trong thư mục!")
            return
        
        # File đầu tiên là mới nhất
        first_item = self.tree.get_children()[0]
        filepath = self.tree.item(first_item, "tags")[0]
        
        self.view_point_cloud(filepath)
    
    def view_point_cloud(self, filepath):
        """Mở Open3D viewer để xem point cloud (có thể kèm trục)"""
        try:
            print(f"\n{'='*60}")
            print(f"📂 Đang mở file: {os.path.basename(filepath)}")
            
            # Kiểm tra xem có file axis tương ứng không
            axis_filepath = None
            if filepath.endswith("_pcd.ply"):
                # File stem có format *_pcd.ply -> tìm *_axis.ply
                axis_filepath = filepath.replace("_pcd.ply", "_axis.ply")
            elif filepath.endswith(".ply") and not filepath.endswith("_axis.ply"):
                # File thường .ply -> thử tìm *_axis.ply (nếu có)
                base = filepath[:-4]  # Bỏ .ply
                axis_filepath = f"{base}_axis.ply"
            
            # Kiểm tra file axis có tồn tại không
            has_axis = axis_filepath and os.path.exists(axis_filepath)
            
            if has_axis and self.show_axis_var.get():
                print(f"   🌿 Tìm thấy file trục: {os.path.basename(axis_filepath)}")
            
            print(f"{'='*60}")
            
            # Đọc point cloud
            pcd = o3d.io.read_point_cloud(filepath)
            num_points = len(pcd.points)
            
            if num_points == 0:
                messagebox.showerror("Lỗi", "File point cloud rỗng!")
                return
            
            print(f"✅ Đã đọc {num_points:,} điểm")
            
            # Danh sách geometry để hiển thị
            geometries = [pcd]
            
            # Đọc axis nếu có và được bật
            if has_axis and self.show_axis_var.get():
                try:
                    # Try reading as mesh first (new format with cylinders)
                    try:
                        axis_mesh = o3d.io.read_triangle_mesh(axis_filepath)
                        if len(axis_mesh.vertices) > 0:
                            print(f"✅ Đã đọc trục tăng trưởng (mesh): {len(axis_mesh.vertices)} vertices")
                            geometries.append(axis_mesh)
                        else:
                            raise ValueError("Empty mesh")
                    except:
                        # Fallback: try reading as LineSet (old format)
                        axis_lineset = o3d.io.read_line_set(axis_filepath)
                        num_axis_pts = len(axis_lineset.points)
                        num_segments = len(axis_lineset.lines)
                        
                        if num_axis_pts > 0:
                            print(f"✅ Đã đọc trục tăng trưởng (lineset): {num_axis_pts} điểm, {num_segments} đoạn")
                            geometries.append(axis_lineset)
                        else:
                            print(f"⚠️ File trục rỗng")
                except Exception as e:
                    print(f"⚠️ Không thể đọc file trục: {e}")
            
            # Tính kích thước coordinate frame (giống realsense_gui_advanced.py)
            try:
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()  # [width, height, depth]
                max_dim = np.max(extent)
                
                # Frame size = 8% của dimension lớn nhất (mảnh hơn, gọn hơn)
                frame_size = max_dim * 0.08
                
                # Đảm bảo frame không quá nhỏ (min 0.01m) hoặc quá lớn (max 0.3m)
                frame_size = np.clip(frame_size, 0.01, 0.3)
                
                print(f"📏 Frame size: {frame_size:.4f}m (extent: {extent})")
            except Exception as e:
                print(f"⚠️ Lỗi tính frame size: {e}, dùng default 0.03m")
                frame_size = 0.03
            
            # Tạo coordinate frame
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size,
                origin=[0, 0, 0]
            )
            geometries.append(mesh_frame)
            
            print(f"🔷 Mở Open3D Viewer...")
            print(f"   Điều khiển:")
            print(f"   • Chuột trái: Xoay")
            print(f"   • Chuột phải: Di chuyển")
            print(f"   • Scroll: Zoom")
            print(f"   • Q: Thoát")
            if has_axis and self.show_axis_var.get():
                print(f"   • Trục tăng trưởng: XANH LÁ")
            print(f"{'='*60}\n")
            
            # Hiển thị
            window_title = f'Point Cloud Viewer - {os.path.basename(filepath)}'
            if has_axis and self.show_axis_var.get():
                window_title += ' + Growth Axis'
            
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_title,
                width=1280,
                height=960,
                left=50,
                top=50
            )
            
            print("✅ Đã đóng viewer\n")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở file:\n{str(e)}")
            print(f"❌ Lỗi: {e}")
    
    def delete_selected(self):
        """Xóa file được chọn"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một file để xóa!")
            return
        
        # Lấy thông tin file
        item = selection[0]
        filename = self.tree.item(item, "values")[0]
        filepath = self.tree.item(item, "tags")[0]
        
        # Xác nhận xóa
        result = messagebox.askyesno(
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa file này?\n\n{filename}"
        )
        
        if result:
            try:
                os.remove(filepath)
                messagebox.showinfo("Thành công", f"Đã xóa file: {filename}")
                self.refresh_file_list()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xóa file:\n{str(e)}")


def main():
    # Import numpy ở đây để dùng trong view_point_cloud
    global np
    import numpy as np
    
    root = tk.Tk()
    app = PointCloudViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
