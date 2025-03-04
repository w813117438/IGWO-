# -*- coding: utf-8 -*-
"""
灰狼算法收敛性分析可视化平台
功能：参数交互设置、实时收敛曲线、多函数测试支持
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from threading import Thread
from queue import Queue

class GWOVisualizer:
    def __init__(self, master):
        self.master = master
        self.setup_ui()
        self.setup_algorithm()
        self.data_queue = Queue()
        
    def setup_ui(self):
        """构建图形用户界面"""
        self.master.title("灰狼算法收敛分析平台")
        self.master.geometry("1200x800")
        
        # 主布局容器
        main_frame = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=300)
        main_frame.add(control_frame)
        
        # 右侧可视化面板
        viz_frame = ttk.Frame(main_frame)
        main_frame.add(viz_frame)
        
        # ========== 控制面板组件 ==========
        ttk.Label(control_frame, text="算法参数设置", font=('微软雅黑', 12)).pack(pady=10)
        
        # 参数输入区
        self.create_input(control_frame, "种群规模:", "50", "pop_size")
        self.create_input(control_frame, "最大迭代:", "500", "max_iter")
        self.create_input(control_frame, "问题维度:", "30", "dim")
        
        # 测试函数选择
        self.func_var = tk.StringVar(value="sphere")
        ttk.Label(control_frame, text="测试函数:").pack(pady=5)
        funcs = [("Sphere函数", "sphere"), ("Rastrigin函数", "rastrigin")]
        for text, value in funcs:
            ttk.Radiobutton(control_frame, text=text, variable=self.func_var, 
                          value=value).pack(anchor=tk.W)
        
        # 控制按钮
        ttk.Button(control_frame, text="开始优化", 
                 command=self.start_optimization).pack(pady=10)
        ttk.Button(control_frame, text="重置系统", 
                 command=self.reset_system).pack(pady=5)
        
        # 实时数据监视
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(control_frame, textvariable=self.status_var, 
                foreground="blue").pack(pady=10)
        
        # ========== 可视化面板组件 ==========
        # 收敛曲线图
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("收敛曲线分析", fontsize=12)
        self.ax.set_xlabel("迭代次数")
        self.ax.set_ylabel("适应度值(log)")
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_input(self, parent, label, default, var_name):
        """创建带标签的输入框"""
        frame = ttk.Frame(parent)
        frame.pack(pady=5, fill=tk.X)
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        entry = ttk.Entry(frame)
        entry.insert(0, default)
        entry.pack(side=tk.RIGHT)
        setattr(self, var_name, entry)
        
    def setup_algorithm(self):
        """初始化算法参数"""
        self.optimizer = None
        self.running = False
        self.thread = None
        
    def start_optimization(self):
        """启动优化线程"""
        if self.running: return
        
        # 获取参数
        try:
            params = {
                'pop_size': int(self.pop_size.get()),
                'max_iter': int(self.max_iter.get()),
                'dim': int(self.dim.get()),
                'func': self.func_var.get()
            }
        except ValueError:
            self.status_var.set("参数输入错误!")
            return
        
        # 定义目标函数
        func_map = {
            'sphere': lambda x: np.sum(x**2),
            'rastrigin': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        }
        obj_func = func_map[params['func']]
        
        # 初始化优化器
        self.optimizer = EnhancedGWO(
            obj_func=obj_func,
            dim=params['dim'],
            pop_size=params['pop_size'],
            max_iter=params['max_iter']
        )
        
        # 启动后台线程
        self.running = True
        self.thread = Thread(target=self.run_optimization)
        self.thread.start()
        self.master.after(100, self.update_plot)
        
    def run_optimization(self):
        """在后台线程中执行优化"""
        self.optimizer.optimize()
        self.data_queue.put({'done': True})
        
    def update_plot(self):
        """更新收敛曲线"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                if 'done' in data:
                    self.running = False
                    self.status_var.set(f"优化完成! 最优值: {self.optimizer.alpha_score:.2e}")
                    break
                
            if self.running and self.optimizer:
                # 绘制最新数据
                self.ax.clear()
                iterations = np.arange(len(self.optimizer.best_fitness))
                self.ax.semilogy(iterations, self.optimizer.best_fitness, 'b-', lw=2)
                self.ax.set_title(f"{self.func_var.get().capitalize()}函数收敛曲线")
                self.ax.set_xlabel("迭代次数")
                self.ax.set_ylabel("适应度值(log)")
                self.ax.grid(True)
                self.canvas.draw()
                
                # 更新状态
                current_iter = len(self.optimizer.best_fitness)
                self.status_var.set(
                    f"运行中... 迭代: {current_iter}/{self.max_iter.get()} | "
                    f"当前最优: {self.optimizer.alpha_score:.2e}"
                )
                self.master.after(100, self.update_plot)
        except Exception as e:
            print(f"更新异常: {str(e)}")
            self.running = False
            
    def reset_system(self):
        """重置系统状态"""
        self.running = False
        self.ax.clear()
        self.canvas.draw()
        self.status_var.set("准备就绪")
        
class EnhancedGWO:
    def __init__(self, obj_func, dim=30, pop_size=50, max_iter=500):
        """改进型灰狼优化算法"""
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.best_fitness = []
        
        # 初始化种群
        self.lb = -100 * np.ones(dim)
        self.ub = 100 * np.ones(dim)
        self.positions = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.fitness = np.apply_along_axis(obj_func, 1, self.positions)
        
        # 领导狼初始化
        self.alpha_score = np.min(self.fitness)
        self.alpha_pos = self.positions[np.argmin(self.fitness)]
        
    def update_convergence_factor(self, t):
        """指数衰减收敛因子"""
        return 2 * np.exp(-5 * t / self.max_iter)
    
    def cauchy_mutation(self, position):
        """柯西变异增强探索"""
        return position + 0.1 * np.random.standard_cauchy(self.dim)
    
    def optimize(self):
        """执行优化主循环"""
        for t in range(self.max_iter):
            a = self.update_convergence_factor(t)
            
            # 更新每个个体
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                
                D_alpha = np.abs(C * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A * D_alpha
                
                # 后期增强变异
                if t > 0.8 * self.max_iter:
                    X1 = self.cauchy_mutation(X1)
                
                new_pos = np.clip(X1, self.lb, self.ub)
                new_fitness = self.obj_func(new_pos)
                
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness
            
            # 更新全局最优
            current_min = np.min(self.fitness)
            if current_min < self.alpha_score:
                self.alpha_score = current_min
                self.alpha_pos = self.positions[np.argmin(self.fitness)]
            
            self.best_fitness.append(self.alpha_score)
            
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TButton', font=('微软雅黑', 10))
    style.configure('TRadiobutton', font=('微软雅黑', 9))
    app = GWOVisualizer(root)
    root.mainloop()