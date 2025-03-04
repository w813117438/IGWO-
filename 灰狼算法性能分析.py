import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict

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

# 基准测试函数 (向量化实现)
class Benchmark:
    @staticmethod
    def sphere(x):
        return np.sum(np.square(x), axis=-1)
    
    @staticmethod
    def rastrigin(x):
        return 10*x.shape[-1] + np.sum(np.square(x) - 10*np.cos(2*np.pi*x), axis=-1)
    
    @staticmethod
    def ackley(x):
        return -20*np.exp(-0.2*np.sqrt(np.mean(np.square(x), axis=-1))) - \
               np.exp(np.mean(np.cos(2*np.pi*x), axis=-1)) + 20 + np.e

# 优化算法实现
class GWO:
    def __init__(self, obj_func, dim, pop_size, max_iter):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.convergence = []
        self.search_range = (-10, 10)  # 默认范围

    def set_search_range(self, bounds):
        self.search_range = bounds

    def optimize(self):
        # 初始化种群
        population = np.random.uniform(*self.search_range, (self.pop_size, self.dim))
        fitness = np.full(self.pop_size, np.inf)
        
        for iter in range(self.max_iter):
            # 计算适应度
            fitness = self.obj_func(population)
            
            # 排序找到三个领导者
            sorted_indices = np.argsort(fitness)
            alpha = population[sorted_indices[0]]
            beta = population[sorted_indices[1]]
            delta = population[sorted_indices[2]]
            
            # 更新收敛因子a
            a = 2.0 - iter * (2.0 / self.max_iter)
            
            # 更新每个个体的位置
            new_population = []
            for i in range(self.pop_size):
                # 计算三个领导者的影响
                X1 = self.update_position(alpha, population[i], a)
                X2 = self.update_position(beta, population[i], a)
                X3 = self.update_position(delta, population[i], a)
                
                # 综合三个方向的影响
                new_pos = (X1 + X2 + X3) / 3.0
                new_population.append(new_pos)
            
            population = np.clip(new_population, *self.search_range)
            self.convergence.append(fitness[sorted_indices[0]])
            
        return alpha, self.convergence

    def update_position(self, leader, current, a):
        A = 2 * a * np.random.rand() - a
        C = 2 * np.random.rand()
        D = np.abs(C * leader - current)
        return leader - A * D

class PSO:
    def __init__(self, obj_func, dim, pop_size, max_iter):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.convergence = []
        self.search_range = (-10, 10)
        self.max_velocity = 2.0

    def set_search_range(self, bounds):
        self.search_range = bounds
        self.max_velocity = (bounds[1] - bounds[0]) * 0.2

    def optimize(self):
        # 初始化种群
        population = np.random.uniform(*self.search_range, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        
        personal_best = population.copy()
        personal_best_fitness = np.full(self.pop_size, np.inf)
        global_best = None
        global_best_fitness = np.inf
        
        for iter in range(self.max_iter):
            # 计算适应度
            fitness = self.obj_func(population)
            
            # 更新个体最优
            improved = fitness < personal_best_fitness
            personal_best[improved] = population[improved]
            personal_best_fitness[improved] = fitness[improved]
            
            # 更新全局最优
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < global_best_fitness:
                global_best = population[min_idx]
                global_best_fitness = fitness[min_idx]
            
            # 更新速度和位置
            w = 0.729
            c1, c2 = 1.49445, 1.49445
            r1 = np.random.rand(*population.shape)
            r2 = np.random.rand(*population.shape)
            
            velocity = w * velocity + \
                      c1 * r1 * (personal_best - population) + \
                      c2 * r2 * (global_best - population)
            
            # 速度钳制
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            
            # 位置更新
            population = np.clip(population + velocity, *self.search_range)
            self.convergence.append(global_best_fitness)
            
        return global_best, self.convergence

# 图形用户界面
class OptimizationApp:
    def __init__(self, master):
        self.master = master
        master.title("优化算法对比分析系统 v2.0")
        self.setup_ui()
        self.runs = 5  # 默认运行次数
        
    def setup_ui(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.master, text="参数设置")
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        # 参数控件
        ttk.Label(control_frame, text="测试函数:").grid(row=0, column=0, sticky=tk.W)
        self.func_var = tk.StringVar()
        self.func_combobox = ttk.Combobox(control_frame, textvariable=self.func_var, 
                                        values=["Sphere", "Rastrigin", "Ackley"])
        self.func_combobox.grid(row=0, column=1)
        self.func_combobox.current(0)
        
        ttk.Label(control_frame, text="维度:").grid(row=1, column=0, sticky=tk.W)
        self.dim_entry = ttk.Entry(control_frame)
        self.dim_entry.insert(0, "10")
        self.dim_entry.grid(row=1, column=1)
        
        ttk.Label(control_frame, text="种群大小:").grid(row=2, column=0, sticky=tk.W)
        self.pop_entry = ttk.Entry(control_frame)
        self.pop_entry.insert(0, "30")
        self.pop_entry.grid(row=2, column=1)
        
        ttk.Label(control_frame, text="最大迭代:").grid(row=3, column=0, sticky=tk.W)
        self.iter_entry = ttk.Entry(control_frame)
        self.iter_entry.insert(0, "100")
        self.iter_entry.grid(row=3, column=1)
        
        ttk.Label(control_frame, text="运行次数:").grid(row=4, column=0, sticky=tk.W)
        self.runs_entry = ttk.Entry(control_frame)
        self.runs_entry.insert(0, "5")
        self.runs_entry.grid(row=4, column=1)
        
        self.run_button = ttk.Button(control_frame, text="开始优化", command=self.run_optimization)
        self.run_button.grid(row=5, columnspan=2, pady=10)
        
        # 结果展示面板
        result_frame = ttk.Frame(self.master)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 收敛曲线图
        self.figure = plt.Figure(figsize=(8,5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 结果表格
        self.result_tree = ttk.Treeview(result_frame, columns=('Algorithm', 'Best', 'Mean', 'Std'), show='headings')
        self.result_tree.heading('Algorithm', text='算法')
        self.result_tree.heading('Best', text='最优值')
        self.result_tree.heading('Mean', text='平均值')
        self.result_tree.heading('Std', text='标准差')
        self.result_tree.column('Algorithm', width=100)
        self.result_tree.column('Best', width=120)
        self.result_tree.column('Mean', width=120)
        self.result_tree.column('Std', width=120)
        self.result_tree.pack(side=tk.BOTTOM, fill=tk.X)
        
    def get_search_range(self, func_name):
        ranges = {
            'sphere': (-100, 100),
            'rastrigin': (-5.12, 5.12),
            'ackley': (-32.768, 32.768)
        }
        return ranges.get(func_name.lower(), (-10, 10))
        
    def run_optimization(self):
        # 获取参数
        func_name = self.func_var.get().lower()
        dim = int(self.dim_entry.get())
        pop_size = int(self.pop_entry.get())
        max_iter = int(self.iter_entry.get())
        runs = int(self.runs_entry.get())
        search_range = self.get_search_range(func_name)
        
        # 初始化算法
        algorithms = {
            'GWO': GWO(getattr(Benchmark, func_name), dim, pop_size, max_iter),
            'PSO': PSO(getattr(Benchmark, func_name), dim, pop_size, max_iter)
        }
        
        # 设置搜索范围
        for algo in algorithms.values():
            algo.set_search_range(search_range)
            
        # 运行优化
        results = defaultdict(list)
        self.ax.clear()
        
        for run in range(runs):
            for name, algo in algorithms.items():
                best, convergence = algo.optimize()
                results[name].append(convergence)
                
        # 绘制收敛曲线
        for name, convs in results.items():
            mean_conv = np.mean(convs, axis=0)
            self.ax.semilogy(mean_conv, label=name)
            
        self.ax.set_title(f"{func_name.capitalize()} 函数收敛曲线")
        self.ax.set_xlabel("迭代次数")
        self.ax.set_ylabel("适应度值 (对数)")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()
        
        # 更新结果表格
        self.result_tree.delete(*self.result_tree.get_children())
        for name, convs in results.items():
            final_values = [c[-1] for c in convs]
            best = np.min(final_values)
            mean = np.mean(final_values)
            std = np.std(final_values)
            self.result_tree.insert('', 'end', values=(
                name,
                f"{best:.3e}",
                f"{mean:.3e} ± {std:.1e}"
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()