import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GWO:
    """标准灰狼优化算法"""
    def __init__(self, obj_func, dim=2, lb=-10, ub=10, pop_size=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # 初始化种群
        self.positions = np.random.uniform(lb, ub, (pop_size, dim))
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
        self.convergence = []

    def optimize(self):
        for iter in range(self.max_iter):
            # 计算适应度
            fitness = np.array([self.obj_func(ind) for ind in self.positions])
            
            # 更新alpha, beta, delta
            sorted_indices = np.argsort(fitness)
            self.alpha_pos = self.positions[sorted_indices[0]]
            self.alpha_score = fitness[sorted_indices[0]]
            self.beta_pos = self.positions[sorted_indices[1]]
            self.beta_score = fitness[sorted_indices[1]]
            self.delta_pos = self.positions[sorted_indices[2]]
            self.delta_score = fitness[sorted_indices[2]]
            
            a = 2 - 2 * iter / self.max_iter  # 线性递减
            
            # 更新位置
            new_positions = []
            for i in range(self.pop_size):
                A1 = 2 * a * np.random.rand() - a
                C1 = 2 * np.random.rand()
                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A1 * D_alpha

                A2 = 2 * a * np.random.rand() - a
                C2 = 2 * np.random.rand()
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                X2 = self.beta_pos - A2 * D_beta

                A3 = 2 * a * np.random.rand() - a
                C3 = 2 * np.random.rand()
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])
                X3 = self.delta_pos - A3 * D_delta

                new_pos = (X1 + X2 + X3) / 3
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_positions.append(new_pos)
            
            self.positions = np.array(new_positions)
            self.convergence.append(self.alpha_score)

class IGWO(GWO):
    """改进灰狼优化算法"""
    def __init__(self, obj_func, dim=2, lb=-10, ub=10, pop_size=30, max_iter=100):
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter)
        
        # 改进的Logistic映射初始化
        self.positions = self.chaotic_initialization()
        
    def chaotic_initialization(self):
        positions = []
        for _ in range(self.pop_size):
            x = np.zeros(self.dim)
            x[0] = np.random.rand()
            for j in range(1, self.dim):
                x[j] = 3.9 * x[j-1] * (1 - x[j-1])  # 改进的Logistic映射参数
            positions.append(self.lb + x * (self.ub - self.lb))
        return np.array(positions)
    
    def optimize(self):
        for iter in range(self.max_iter):
            fitness = np.array([self.obj_func(ind) for ind in self.positions])
            
            sorted_indices = np.argsort(fitness)
            self.alpha_pos = self.positions[sorted_indices[0]]
            self.alpha_score = fitness[sorted_indices[0]]
            self.beta_pos = self.positions[sorted_indices[1]]
            self.beta_score = fitness[sorted_indices[1]]
            self.delta_pos = self.positions[sorted_indices[2]]
            self.delta_score = fitness[sorted_indices[2]]
            
            # 非线性收敛因子
            a = 2 * np.exp(-5 * iter / self.max_iter)
            
            # 双模态步长调节
            base_step = 0.5
            step_size = base_step * (1 / (1 + np.exp((iter - self.max_iter/2)/10)))
            
            # 种群密度计算
            avg_distance = np.mean([np.linalg.norm(ind - self.alpha_pos) for ind in self.positions])
            explore_weight = 0.2 * np.sqrt(1 - iter/self.max_iter) if avg_distance < 0.1*(self.ub-self.lb) else 0.5
            
            new_positions = []
            for i in range(self.pop_size):
                if np.random.rand() < explore_weight:  # 探索增强
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    leader = self.alpha_pos if np.random.rand() < 0.5 else self.beta_pos
                    D = np.abs(C * leader - self.positions[i])
                    new_pos = leader - A * D * step_size
                else:  # 标准更新
                    A1 = 2 * a * np.random.rand() - a
                    C1 = 2 * np.random.rand()
                    D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                    X1 = self.alpha_pos - A1 * D_alpha

                    A2 = 2 * a * np.random.rand() - a
                    C2 = 2 * np.random.rand()
                    D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                    X2 = self.beta_pos - A2 * D_beta

                    A3 = 2 * a * np.random.rand() - a
                    C3 = 2 * np.random.rand()
                    D_delta = np.abs(C3 * self.delta_pos - self.positions[i])
                    X3 = self.delta_pos - A3 * D_delta

                    new_pos = (X1 + X2 + X3) / 3
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_positions.append(new_pos)
            
            self.positions = np.array(new_positions)
            self.convergence.append(self.alpha_score)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("灰狼优化算法验证程序")
        self.geometry("1000x800")
        
        # 输入参数
        self.frame_input = tk.Frame(self)
        self.frame_input.pack(side=tk.TOP, pady=10)
        
        tk.Label(self.frame_input, text="种群大小:").grid(row=0, column=0)
        self.pop_size = tk.Entry(self.frame_input, width=10)
        self.pop_size.insert(0, "30")
        self.pop_size.grid(row=0, column=1)
        
        tk.Label(self.frame_input, text="迭代次数:").grid(row=0, column=2)
        self.max_iter = tk.Entry(self.frame_input, width=10)
        self.max_iter.insert(0, "100")
        self.max_iter.grid(row=0, column=3)
        
        tk.Label(self.frame_input, text="问题维度:").grid(row=0, column=4)
        self.dim = tk.Entry(self.frame_input, width=10)
        self.dim.insert(0, "2")
        self.dim.grid(row=0, column=5)
        
        # 运行按钮
        self.btn_run = tk.Button(self.frame_input, text="运行优化", command=self.run_optimization)
        self.btn_run.grid(row=0, column=6, padx=10)
        
        # 绘图区域
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def run_optimization(self):
        # 获取参数
        pop_size = int(self.pop_size.get())
        max_iter = int(self.max_iter.get())
        dim = int(self.dim.get())
        
        # 定义测试函数
        def sphere(x):
            return sum(x**2)
        
        # 运行算法
        gwo = GWO(sphere, dim=dim, pop_size=pop_size, max_iter=max_iter)
        gwo.optimize()
        
        igwo = IGWO(sphere, dim=dim, pop_size=pop_size, max_iter=max_iter)
        igwo.optimize()
        
        # 绘制结果
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax1.plot(gwo.convergence, label='标准GWO')
        ax1.plot(igwo.convergence, label='改进IGWO')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('适应度值')
        ax1.set_title('收敛曲线对比')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制种群分布（取前两维）
        if dim >= 2:
            ax2 = self.figure.add_subplot(212)
            ax2.scatter(gwo.positions[:,0], gwo.positions[:,1], c='r', label='标准GWO')
            ax2.scatter(igwo.positions[:,0], igwo.positions[:,1], c='b', label='改进IGWO')
            ax2.scatter(0, 0, marker='*', s=200, c='gold', label='全局最优')
            ax2.set_xlabel('X1')
            ax2.set_ylabel('X2')
            ax2.set_title('最终种群分布')
            ax2.legend()
            ax2.grid(True)
        
        self.canvas.draw()

if __name__ == "__main__":
    app = Application()
    app.mainloop()