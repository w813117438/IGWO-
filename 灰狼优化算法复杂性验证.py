# -*- coding: utf-8 -*-
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import psutil
import os
import gc

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
        self.positions = np.random.uniform(lb, ub, (pop_size, dim))
        self.convergence = []

    def optimize(self):
        alpha_pos = np.zeros(self.dim)
        alpha_score = float('inf')
        
        for iter in range(self.max_iter):
            fitness = np.array([self.obj_func(ind) for ind in self.positions])
            sorted_indices = np.argsort(fitness)
            current_alpha = self.positions[sorted_indices[0]]
            
            a = 2.0 - 2.0 * iter / self.max_iter
            new_positions = []
            for i in range(self.pop_size):
                A1 = 2 * a * np.random.rand() - a
                C1 = 2 * np.random.rand()
                D_alpha = np.abs(C1 * current_alpha - self.positions[i])
                new_pos = current_alpha - A1 * D_alpha
                new_positions.append(np.clip(new_pos, self.lb, self.ub))
            
            self.positions = np.array(new_positions)
            self.convergence.append(fitness[sorted_indices[0]])

class IGWO(GWO):
    """改进灰狼优化算法"""
    def __init__(self, obj_func, dim=2, lb=-10, ub=10, pop_size=30, max_iter=100):
        super().__init__(obj_func, dim, lb, ub, pop_size, max_iter)
        self.positions = self._chaotic_initialization()
        
    def _chaotic_initialization(self):
        positions = []
        for _ in range(self.pop_size):
            x = np.zeros(self.dim)
            x[0] = np.random.rand()
            for j in range(1, self.dim):
                x[j] = 3.9 * x[j-1] * (1 - x[j-1])
            positions.append(self.lb + x * (self.ub - self.lb))
        return np.array(positions)
    
    def optimize(self):
        alpha_pos = np.zeros(self.dim)
        alpha_score = float('inf')
        
        for iter in range(self.max_iter):
            fitness = np.array([self.obj_func(ind) for ind in self.positions])
            sorted_indices = np.argsort(fitness)
            current_alpha = self.positions[sorted_indices[0]]
            
            a = 2 * np.exp(-5 * iter / self.max_iter)
            avg_distance = np.mean([np.linalg.norm(ind - current_alpha) for ind in self.positions])
            explore_weight = 0.2 * np.sqrt(1 - iter/self.max_iter) if avg_distance < 0.1*(self.ub-self.lb) else 0.5
            
            new_positions = []
            for i in range(self.pop_size):
                if np.random.rand() < explore_weight:
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    step_size = 0.5 / (1 + np.exp((iter - self.max_iter/2)/10))
                    D = np.abs(C * current_alpha - self.positions[i])
                    new_pos = current_alpha - A * D * step_size
                else:
                    A1 = 2 * a * np.random.rand() - a
                    C1 = 2 * np.random.rand()
                    D_alpha = np.abs(C1 * current_alpha - self.positions[i])
                    new_pos = current_alpha - A1 * D_alpha
                
                new_positions.append(np.clip(new_pos, self.lb, self.ub))
            
            self.positions = np.array(new_positions)
            self.convergence.append(fitness[sorted_indices[0]])

class ComplexityAnalyzer:
    """复杂度分析工具"""
    def __init__(self):
        self.time_data = {'GWO': [], 'IGWO': []}
        self.mem_data = {'GWO': [], 'IGWO': []}
    
    def measure(self, algo_class, dim_range, max_iter=50):
        process = psutil.Process(os.getpid())
        time_results = []
        mem_results = []
        
        for d in dim_range:
            try:
                # 内存基准测量
                gc.collect()
                start_mem = process.memory_info().rss
                
                # 显式初始化种群数据
                algo = algo_class(lambda x: sum(x**2), dim=d, max_iter=max_iter)
                algo.positions = np.random.uniform(algo.lb, algo.ub, (algo.pop_size, algo.dim))
                
                # 精确内存测量
                current_mem = process.memory_info().rss
                mem_usage = max((current_mem - start_mem) / 1024**2, 0.0)
                mem_results.append(round(mem_usage, 2))
                
                # 时间性能测试
                start_time = time.perf_counter()
                algo.optimize()
                elapsed = (time.perf_counter() - start_time) * 1000
                time_results.append(round(elapsed, 2))
                
                # 清理资源
                del algo
                gc.collect()
                
            except Exception as e:
                print(f"维度{d}测试失败: {str(e)}")
                time_results.append(0)
                mem_results.append(0)
        
        return time_results, mem_results

class AnalysisApp(tk.Tk):
    """主应用程序"""
    def __init__(self):
        super().__init__()
        self.title("灰狼算法复杂度分析 v3.0")
        self.geometry("1200x800")
        self.analyzer = ComplexityAnalyzer()
        self._init_ui()
    
    def _init_ui(self):
        # 控制面板
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill=tk.X)
        
        # 维度设置组件
        ttk.Label(control_frame, text="维度范围:").grid(row=0, column=0, padx=5)
        self.dim_start = ttk.Entry(control_frame, width=8)
        self.dim_start.insert(0, "10")
        self.dim_start.grid(row=0, column=1, padx=5)
        
        ttk.Label(control_frame, text="-").grid(row=0, column=2)
        self.dim_end = ttk.Entry(control_frame, width=8)
        self.dim_end.insert(0, "100")
        self.dim_end.grid(row=0, column=3, padx=5)
        
        ttk.Label(control_frame, text="步长:").grid(row=0, column=4, padx=5)
        self.dim_step = ttk.Entry(control_frame, width=8)
        self.dim_step.insert(0, "10")
        self.dim_step.grid(row=0, column=5, padx=5)
        
        # 操作按钮
        self.analyze_btn = ttk.Button(control_frame, text="开始分析", command=self.start_analysis)
        self.analyze_btn.grid(row=0, column=6, padx=10)
        
        # 可视化画布
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def start_analysis(self):
        """启动分析过程"""
        try:
            dim_start = int(self.dim_start.get())
            dim_end = int(self.dim_end.get())
            dim_step = int(self.dim_step.get())
            dim_range = list(range(dim_start, dim_end + 1, dim_step))
            
            if not dim_range or dim_start >= dim_end or dim_step <= 0:
                raise ValueError("参数错误")
            
            # 执行测量
            self.analyzer.time_data['GWO'], self.analyzer.mem_data['GWO'] = self.analyzer.measure(GWO, dim_range)
            self.analyzer.time_data['IGWO'], self.analyzer.mem_data['IGWO'] = self.analyzer.measure(IGWO, dim_range)
            
            # 更新图表
            self.update_plots(dim_range)
            
        except ValueError as e:
            print(f"输入错误: {str(e)}")
    
    def update_plots(self, dim_range):
        """更新可视化图表"""
        self.figure.clear()
        
        # 时间复杂度图表
        ax1 = self.figure.add_subplot(211)
        ax1.plot(dim_range, self.analyzer.time_data['GWO'], 'ro-', markersize=6, label='标准GWO')
        ax1.plot(dim_range, self.analyzer.time_data['IGWO'], 'bs--', markersize=6, label='改进IGWO')
        ax1.set_xlabel('问题维度', fontsize=10)
        ax1.set_ylabel('运行时间 (ms)', fontsize=10)
        ax1.set_title('时间复杂度对比 (迭代次数=50)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 空间复杂度图表
        ax2 = self.figure.add_subplot(212)
        width = 0.35
        x = np.arange(len(dim_range))
        
        # 处理无效数据
        gwo_mem = [max(m, 0.1) for m in self.analyzer.mem_data['GWO']]
        igwo_mem = [max(m, 0.1) for m in self.analyzer.mem_data['IGWO']]
        
        rects1 = ax2.bar(x - width/2, gwo_mem, width, label='标准GWO', 
                        color='lightcoral', edgecolor='darkred', alpha=0.8)
        rects2 = ax2.bar(x + width/2, igwo_mem, width, label='改进IGWO',
                        color='skyblue', edgecolor='navy', alpha=0.8)
        
        # 添加数据标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.text(rect.get_x() + rect.get_width()/2., height + 0.05,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8, rotation=45)
        autolabel(rects1)
        autolabel(rects2)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(dim_range, rotation=45, fontsize=8)
        ax2.set_xlabel('问题维度', fontsize=10)
        ax2.set_ylabel('内存占用 (MB)', fontsize=10)
        ax2.set_title('空间复杂度对比', fontsize=12)
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_ylim(0, max(gwo_mem + igwo_mem)*1.2)
        
        plt.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = AnalysisApp()
    app.mainloop()