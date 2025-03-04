import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import cv2
from skimage import feature, measure

class IGWO_ContourAnalyzer:
    def __init__(self):
        self.pop_size = 30
        self.max_iter = 100
        self.a = 2
        self.dim = 2  # 优化参数维度

    def optimize_contour(self, image):
        def fitness(canny_params):
            # 确保参数在[0,1]范围内
            canny_params = np.clip(canny_params, 0, 1)
            
            # 参数映射到实际范围
            sigma = canny_params[0] * 5 + 0.1  # [0.1, 5.1]
            high_thresh = canny_params[1] * 255  # [0, 255]
            low_thresh = 0.5 * high_thresh
            
            try:
                # 高斯滤波去噪
                blurred = cv2.GaussianBlur(image, (5, 5), sigma)
                
                # Canny边缘检测
                edges = feature.canny(blurred.astype(np.float64)/255, 
                                     low_thresh/255, 
                                     high_thresh/255)
                
                # 轮廓质量评估
                contours = measure.find_contours(edges, 0.8)
                if len(contours) == 0:
                    return float('inf')
                
                # 计算评估指标
                total_length = sum([len(c) for c in contours])
                continuity = total_length / (edges.sum() + 1e-6)
                edge_intensity = image[edges.astype(bool)].mean()
                
                return -(continuity * 0.7 + edge_intensity * 0.3)
            except:
                return float('inf')

        # 运行优化算法
        best_params, _, convergence = self.igwo_optimize(fitness)
        
        # 最终参数计算
        best_params = np.clip(best_params, 0, 1)
        sigma = best_params[0] * 5 + 0.1
        high_thresh = best_params[1] * 255
        
        return sigma, high_thresh, convergence

    def igwo_optimize(self, fitness_func):
        # 初始化种群
        positions = np.random.rand(self.pop_size, self.dim)
        alpha = beta = delta = np.zeros(self.dim)
        alpha_score = beta_score = delta_score = float('inf')
        convergence = []

        for iter in range(self.max_iter):
            for i in range(self.pop_size):
                # 边界约束
                positions[i] = np.clip(positions[i], 0, 1)
                
                fitness = fitness_func(positions[i])
                
                # 更新最优解
                if fitness < alpha_score:
                    alpha_score, beta_score, delta_score = fitness, alpha_score, beta_score
                    alpha, beta, delta = positions[i].copy(), alpha.copy(), beta.copy()
                elif fitness < beta_score:
                    beta_score, delta_score = fitness, beta_score
                    beta, delta = positions[i].copy(), beta.copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta = positions[i].copy()

            # 记录收敛情况
            convergence.append(abs(alpha_score))
            
            # 更新收敛因子
            a = self.a * (1 - iter/self.max_iter)
            
            # 位置更新
            for i in range(self.pop_size):
                # Alpha更新
                A1 = 2*a*np.random.rand() - a
                C1 = 2*np.random.rand()
                D_alpha = abs(C1*alpha - positions[i])
                X1 = alpha - A1*D_alpha
                
                # Beta更新
                A2 = 2*a*np.random.rand() - a
                C2 = 2*np.random.rand()
                D_beta = abs(C2*beta - positions[i])
                X2 = beta - A2*D_beta
                
                # Delta更新
                A3 = 2*a*np.random.rand() - a
                C3 = 2*np.random.rand()
                D_delta = abs(C3*delta - positions[i])
                X3 = delta - A3*D_delta
                
                # 更新并约束位置
                new_pos = (X1 + X2 + X3) / 3
                positions[i] = np.clip(new_pos, 0, 1)

        return alpha, alpha_score, convergence

class ContourAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IGWO轮廓优化系统")
        self.geometry("1200x800")
        self.image = None
        self.setup_ui()

    def setup_ui(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self, text="控制面板", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(control_frame, text="打开图像", command=self.load_image).pack(pady=5)
        ttk.Button(control_frame, text="优化轮廓", command=self.run_optimization).pack(pady=5)
        ttk.Button(control_frame, text="保存结果", command=self.save_results).pack(pady=5)

        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="算法参数")
        ttk.Label(param_frame, text="种群规模:").grid(row=0, column=0)
        self.pop_size_entry = ttk.Entry(param_frame)
        self.pop_size_entry.insert(0, "30")
        self.pop_size_entry.grid(row=0, column=1)
        param_frame.pack(pady=10)

        # 图像显示区域
        display_frame = ttk.LabelFrame(self, text="分析结果")
        self.original_label = ttk.Label(display_frame)
        self.result_label = ttk.Label(display_frame)
        self.convergence_label = ttk.Label(display_frame)
        
        self.original_label.grid(row=0, column=0, padx=5)
        self.result_label.grid(row=0, column=1, padx=5)
        self.convergence_label.grid(row=1, column=0, columnspan=2)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg;*.png;*.tif")])
        if path:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.show_image(self.image, self.original_label, "原始图像")

    def run_optimization(self):
        if self.image is None:
            messagebox.showerror("错误", "请先选择图像文件")
            return

        analyzer = IGWO_ContourAnalyzer()
        analyzer.pop_size = int(self.pop_size_entry.get())
        
        # 运行优化
        sigma, high_thresh, convergence = analyzer.optimize_contour(self.image)
        
        # 显示优化结果
        self.show_contour_result(sigma, high_thresh)
        self.show_convergence_curve(convergence)

    def show_contour_result(self, sigma, high_thresh):
        # 应用最优参数
        blurred = cv2.GaussianBlur(self.image, (5, 5), sigma)
        edges = cv2.Canny(blurred, 0.5*high_thresh, high_thresh)
        contours = measure.find_contours(edges.astype(float)/255, 0.8)

        # 绘制结果
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(self.image, cmap='gray')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, c='red')
        ax.set_title(f"σ={sigma:.2f}, 高阈值={high_thresh:.1f}")
        ax.axis('off')
        plt.savefig("contour_result.png", bbox_inches='tight')
        self.show_image("contour_result.png", self.result_label)

    def show_convergence_curve(self, convergence):
        plt.figure(figsize=(8,4))
        plt.plot(convergence, 'b-', linewidth=2)
        plt.title("算法收敛曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("适应度值")
        plt.grid(True)
        plt.savefig("convergence.png")
        self.show_image("convergence.png", self.convergence_label)

    def show_image(self, img_data, target_widget, title=None):
        if isinstance(img_data, str):
            img = Image.open(img_data)
        else:
            img = Image.fromarray(img_data)
        
        img = img.resize((450, 350))
        img_tk = ImageTk.PhotoImage(img)
        target_widget.configure(image=img_tk)
        target_widget.image = img_tk

    def save_results(self):
        try:
            Image.open("contour_result.png").save("优化结果.png")
            Image.open("convergence.png").save("收敛曲线.png")
            messagebox.showinfo("保存成功", "结果已保存为：\n- 优化结果.png\n- 收敛曲线.png")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存失败：{str(e)}")

if __name__ == "__main__":
    app = ContourAnalysisApp()
    app.mainloop()