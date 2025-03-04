import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

# 设置中文字体
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统字体
        except:
            print("字体设置失败，请手动安装中文字体")

set_chinese_font()

class GWO_Stability_Analyzer:
    def __init__(self, master):
        self.master = master
        master.title("GWO稳定性分析工具 v2.0")
        self.setup_ui()
        self.initialize_algorithm()

    def setup_ui(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.master, text="控制参数")
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # 测试函数选择
        self.func_var = tk.StringVar(value='ComplexTerrain')
        ttk.Label(control_frame, text="测试函数:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(control_frame, textvariable=self.func_var, 
                    values=["ComplexTerrain", "DynamicObstacle", "MultiModal"]).grid(row=0, column=1)

        # 其他参数
        params = [
            ('维度', 'dim', 30, 2, 500),
            ('种群数量', 'pop_size', 30, 10, 100),
            ('最大迭代', 'max_iter', 200, 50, 1000),
            ('试验次数', 'trials', 10, 1, 50)
        ]
        for i, (text, var, default, minv, maxv) in enumerate(params, 1):
            ttk.Label(control_frame, text=text+":").grid(row=i, column=0, sticky=tk.W)
            setattr(self, var+'_var', tk.IntVar(value=default))
            ttk.Spinbox(control_frame, from_=minv, to=maxv, 
                       textvariable=getattr(self, var+'_var')).grid(row=i, column=1)

        # 控制按钮
        ttk.Button(control_frame, text="开始分析", command=self.start_analysis).grid(row=5, column=0, columnspan=2, pady=10)

        # 可视化区域
        viz_frame = ttk.Frame(self.master)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建画布
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222, projection='3d')
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def initialize_algorithm(self):
        self.current_trial = 0
        self.results = {
            'convergence': [],
            'success_rate': [],
            'diversity': [],
            'fitness_hist': []
        }

    def complex_terrain(self, X):
        """包含多个局部最优的复杂地形函数"""
        x = X[:, 0]
        y = X[:, 1]
        return 20 + x**2 - 10*np.cos(3*np.pi*x) + y**2 - 10*np.cos(3*np.pi*y) + \
               0.5*np.sin(5*x)*np.sin(5*y)

    def dynamic_obstacle(self, X, iteration):
        """动态障碍物函数"""
        base = np.sum(X**2, axis=1)
        obstacle = 10*np.exp(-((X[:,0]-0.5*np.sin(0.1*iteration))**2 + 
                             (X[:,1]-0.5*np.cos(0.1*iteration))**2)/0.1)
        return base + obstacle

    def run_gwo(self):
        # 算法实现
        dim = self.dim_var.get()
        max_iter = self.max_iter_var.get()
        pop_size = self.pop_size_var.get()
        
        population = np.random.uniform(-5, 5, (pop_size, dim))
        fitness = self.evaluate_fitness(population, 0)
        
        alpha = beta = delta = population[np.argmin(fitness)]
        alpha_score = np.min(fitness)
        
        convergence = []
        population_history = []
        
        for iter in range(max_iter):
            a = 2 - 2 * iter / max_iter
            new_population = np.zeros_like(population)
            
            for i in range(pop_size):
                # 灰狼位置更新逻辑
                A1 = 2*a*np.random.rand(dim) - a
                C1 = 2*np.random.rand(dim)
                D_alpha = np.abs(C1*alpha - population[i])
                X1 = alpha - A1*D_alpha
                
                A2 = 2*a*np.random.rand(dim) - a
                C2 = 2*np.random.rand(dim)
                D_beta = np.abs(C2*beta - population[i])
                X2 = beta - A2*D_beta
                
                A3 = 2*a*np.random.rand(dim) - a
                C3 = 2*np.random.rand(dim)
                D_delta = np.abs(C3*delta - population[i])
                X3 = delta - A3*D_delta
                
                new_population[i] = (X1 + X2 + X3)/3
                new_population[i] = np.clip(new_population[i], -5, 5)
            
            population = new_population
            fitness = self.evaluate_fitness(population, iter)
            
            # 更新领导狼
            sorted_idx = np.argsort(fitness)
            alpha, beta, delta = population[sorted_idx[:3]]
            alpha_score = fitness[sorted_idx[0]]
            
            convergence.append(alpha_score)
            population_history.append(population.copy())
            
            # 更新可视化
            if iter % 5 == 0:
                self.update_plots(iter, population, convergence, population_history)
                self.master.update()
        
        return convergence, population_history

    def evaluate_fitness(self, population, iteration):
        func_name = self.func_var.get()
        if func_name == "ComplexTerrain":
            return self.complex_terrain(population)
        elif func_name == "DynamicObstacle":
            return self.dynamic_obstacle(population, iteration)
        elif func_name == "MultiModal":
            return 10*population.shape[1] + np.sum(population**2 - 10*np.cos(2*np.pi*population), axis=1)

    def update_plots(self, iteration, population, convergence, history):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # 收敛曲线
        self.ax1.semilogy(convergence, 'b-', lw=2)
        self.ax1.set_title(f"收敛曲线 (迭代: {iteration})")
        self.ax1.grid(True)

        # 3D地形可视化
        if self.dim_var.get() >= 2:
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            Z = self.complex_terrain(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
            
            self.ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
            self.ax2.scatter(population[:,0], population[:,1], 
                           self.evaluate_fitness(population, iteration), 
                           c='r', s=50)
            self.ax2.set_title("三维搜索空间")

        # 种群分布热力图
        if self.dim_var.get() >= 2:
            hist = np.array(history)
            self.ax3.hist2d(hist[:,:,0].flatten(), hist[:,:,1].flatten(), 
                          bins=20, cmap='hot')
            self.ax3.set_title("种群分布密度")

        # 稳定性指标
        self.ax4.text(0.5, 0.6, f"当前试验: {self.current_trial+1}/{self.trials_var.get()}\n"
                     f"成功率: {np.mean(self.results['success_rate'])*100:.1f}%\n"
                     f"平均收敛代数: {np.mean(self.results['convergence']):.1f}",
                     ha='center', va='center', fontsize=12)
        self.ax4.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def start_analysis(self):
        self.initialize_algorithm()
        total_trials = self.trials_var.get()
        
        for trial in range(total_trials):
            self.current_trial = trial
            convergence, history = self.run_gwo()
            
            # 记录结果
            success = 1 if convergence[-1] < 1e-3 else 0
            self.results['success_rate'].append(success)
            self.results['convergence'].append(len(convergence))
            self.results['diversity'].append(np.mean(np.std(history[-1], axis=0)))
            self.results['fitness_hist'].append(convergence)
            
            self.update_plots(0, np.array([]), [], [])
            self.master.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = GWO_Stability_Analyzer(root)
    root.mainloop()