# Enhanced evaluation plots for international-level presentation

import matplotlib.pyplot as plt
import numpy as np

# Example experimental data (replace with real measured results)
methods = ["2D", "3D", "2D–3D"]

# Mean processing time (ms) and standard deviation
time_mean = np.array([35, 120, 85])
time_std = np.array([4, 10, 6])

# Mean error (mm) and standard deviation
error_mean = np.array([8.5, 5.2, 2.8])
error_std = np.array([1.2, 0.8, 0.6])

# =========================
# Plot 1: Scatter with Error Bars
# =========================
plt.figure()
plt.errorbar(time_mean, error_mean, 
             xerr=time_std, yerr=error_std, 
             fmt='o', capsize=5)

for x, y, label in zip(time_mean, error_mean, methods):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

plt.xlabel("Thời gian xử lý trung bình (ms/frame)")
plt.ylabel("Sai số trung bình (mm)")
plt.title("So sánh tốc độ và độ chính xác (có độ lệch chuẩn)")
plt.grid(True)

scatter_path = "./errorbars.png"
plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
plt.close()

# =========================
# Plot 2: Boxplot of Error Distribution
# =========================

# Simulated distribution data (replace with real experiment measurements)
error_samples = [
    np.random.normal(8.5, 1.2, 30),
    np.random.normal(5.2, 0.8, 30),
    np.random.normal(2.8, 0.6, 30)
]

plt.figure()
plt.boxplot(error_samples, labels=methods)
plt.xlabel("Phương pháp")
plt.ylabel("Sai số trục sinh trưởng (mm)")
plt.title("Phân bố sai số giữa các phương pháp")
plt.grid(True)

boxplot_path = "./boxplot_error.png"
plt.savefig(boxplot_path, dpi=200, bbox_inches="tight")
plt.close()

scatter_path, boxplot_path
