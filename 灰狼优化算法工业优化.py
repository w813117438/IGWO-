import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import networkx as nx
import threading
from queue import Queue

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

class IndustrialGWO:
    """工业优化灰狼算法核心"""
    def __init__(self, problem_type, dim=10, pop_size=30, max_iter=100, callback=None):
        self.problem_type = problem_type  # 'scheduling' 或 'routing'
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.callback = callback
        self._stop_event = threading.Event()
        self._init_problem_data()

    def _init_problem_data(self):
        """初始化问题相关数据"""
        np.random.seed()
        if self.problem_type == 'scheduling':
            # 生产调度参数
            self.machines = np.random.randint(3, 8)  # 3-7台机器
            self.processing_times = np.random.randint(5, 30, (self.machines, self.dim))
        else:
            # 路径规划参数
            self.locations = np.random.rand(self.dim, 2) * 100  # 节点坐标
            self.vehicle_num = np.random.randint(2, 5)  # 2-4辆车辆

    def _calculate_fitness(self, solution):
        """计算适应度"""
        if self.problem_type == 'scheduling':
            return self._calculate_makespan(solution)
        return self._calculate_route_distance(solution)

    def _calculate_makespan(self, schedule):
        """计算生产周期"""
        machine_load = np.zeros(self.machines)
        for job in schedule:
            machine = job % self.machines
            machine_load[machine] += self.processing_times[machine, job]
        return np.max(machine_load)

    def _calculate_route_distance(self, route):
        """计算路径距离"""
        return np.sum(np.linalg.norm(np.diff(self.locations[route], axis=0), axis=1))

    def optimize(self):
        """执行优化主流程"""
        best_solution = None
        best_score = float('inf')
        population = self._initialize_population()
        
        for iter in range(self.max_iter):
            if self._stop_event.is_set():
                break
            
            # 评估种群
            scores = [self._calculate_fitness(ind) for ind in population]
            
            # 更新最优解
            min_idx = np.argmin(scores)
            current_score = scores[min_idx]
            if current_score < best_score:
                best_score = current_score
                best_solution = population[min_idx].copy()
                self._report_progress(iter, best_score, best_solution)
            
            # 进化操作
            population = self._evolve_population(population, best_solution, iter)
        
        return best_solution, best_score

    def _initialize_population(self):
        """初始化种群"""
        return [np.random.permutation(self.dim) for _ in range(self.pop_size)]

    def _evolve_population(self, population, best_solution, iter):
        """执行种群进化"""
        a = 2 * np.exp(-5 * iter / self.max_iter)
        new_population = []
        
        for ind in population:
            if np.random.rand() < 0.3:
                new_ind = self._mutate(ind)
            else:
                new_ind = self._crossover(ind, best_solution)
            new_population.append(new_ind)
        
        return new_population

    def _mutate(self, individual):
        """变异操作"""
        if self.problem_type == 'scheduling':
            idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
            mutated = individual.copy()
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
            return mutated
        else:
            start, end = sorted(np.random.choice(len(individual), 2, replace=False))
            return np.concatenate([individual[:start], individual[start:end][::-1], individual[end:]])

    def _crossover(self, parent, best_solution):
        """交叉操作"""
        size = len(parent)
        cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
        
        if self.problem_type == 'scheduling':
            # 部分匹配交叉
            child = -np.ones(size, dtype=int)
            child[cx1:cx2] = parent[cx1:cx2]
            for i in range(cx1, cx2):
                if best_solution[i] not in child:
                    j = i
                    while best_solution[j] in child[cx1:cx2]:
                        j = np.where(parent == best_solution[j])[0][0]
                    child[j] = best_solution[i]
            child[child == -1] = [x for x in best_solution if x not in child]
        else:
            # 顺序交叉
            child = -np.ones(size, dtype=int)
            child[cx1:cx2] = parent[cx1:cx2]
            ptr = 0
            for gene in best_solution:
                if gene not in child:
                    while cx1 <= ptr < cx2:
                        ptr += 1
                    if ptr < size:
                        child[ptr] = gene
                        ptr += 1
        return child

    def _report_progress(self, iteration, score, solution):
        """报告优化进度"""
        if self.callback:
            self.callback({
                'iteration': iteration,
                'score': score,
                'solution': solution.copy()
            })

class OptimizationApp(tk.Tk):
    """工业优化可视化平台"""
    def __init__(self):
        super().__init__()
        self.title("工业优化智能平台 v7.0")
        self.geometry("1000x800")
        self._init_ui()
        self.optimizer = None
        self.update_queue = Queue()
        self.running = False

    def _init_ui(self):
        """初始化用户界面"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 问题类型选择
        self.problem_type = tk.StringVar(value='scheduling')
        ttk.Radiobutton(control_frame, text="生产调度", variable=self.problem_type, 
                      value='scheduling').pack(pady=5, anchor='w')
        ttk.Radiobutton(control_frame, text="路径规划", variable=self.problem_type,
                      value='routing').pack(pady=5, anchor='w')

        # 参数设置
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(pady=10, fill=tk.X)
        ttk.Label(param_frame, text="任务维度:").pack()
        self.dim_entry = ttk.Entry(param_frame)
        self.dim_entry.insert(0, "20")
        self.dim_entry.pack(pady=5)

        # 操作按钮
        ttk.Button(control_frame, text="生成数据", command=self.generate_data).pack(pady=5)
        self.start_btn = ttk.Button(control_frame, text="开始优化", command=self.toggle_optimization)
        self.start_btn.pack(pady=5)
        ttk.Button(control_frame, text="导出结果", command=self.export_result).pack(pady=5)

        # 可视化区域
        self.figure = plt.figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def generate_data(self):
        """生成新的问题数据"""
        if self.running:
            messagebox.showwarning("警告", "请先停止当前优化")
            return

        try:
            dim = int(self.dim_entry.get())
            self.optimizer = IndustrialGWO(
                problem_type=self.problem_type.get(),
                dim=dim,
                callback=lambda data: self.update_queue.put(data)
            )
            self._draw_initial_data()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的任务维度")

    def _draw_initial_data(self):
        """绘制初始数据视图"""
        self.ax.clear()
        if self.optimizer.problem_type == 'scheduling':
            im = self.ax.imshow(self.optimizer.processing_times.T, cmap='YlGnBu', aspect='auto')
            plt.colorbar(im, ax=self.ax, label='处理时间 (分钟)')
            self.ax.set_title(f"机器配置（共{self.optimizer.machines}台机器）")
        else:
            locs = self.optimizer.locations
            self.ax.scatter(locs[:,0], locs[:,1], c='steelblue', s=80, edgecolor='white')
            for i, (x, y) in enumerate(locs):
                self.ax.text(x+2, y+2, str(i), ha='center', va='center', fontsize=8, color='darkred')
            self.ax.set_title("配送节点分布图")
        self._refresh_canvas()

    def toggle_optimization(self):
        """启动/停止优化过程"""
        if not self.optimizer:
            messagebox.showwarning("警告", "请先生成数据")
            return
        
        if self.running:
            self._stop_optimization()
        else:
            self._start_optimization()

    def _start_optimization(self):
        """启动优化线程"""
        self.running = True
        self.start_btn.config(text="停止优化")
        self.optimizer._stop_event.clear()
        threading.Thread(target=self._run_optimization, daemon=True).start()
        self.after(100, self._process_updates)

    def _stop_optimization(self):
        """停止优化过程"""
        self.running = False
        self.start_btn.config(text="开始优化")
        self.optimizer._stop_event.set()

    def _run_optimization(self):
        """在后台线程执行优化"""
        best_solution, best_score = self.optimizer.optimize()
        self.update_queue.put({'solution': best_solution, 'score': best_score, 'final': True})

    def _process_updates(self):
        """处理更新队列"""
        try:
            while True:
                data = self.update_queue.get_nowait()
                if data.get('final'):
                    self._show_final_result(data)
                    break
                self._update_visualization(data)
        except:
            pass
        
        if self.running:
            self.after(100, self._process_updates)

    def _update_visualization(self, data):
        """更新可视化显示"""
        self.ax.clear()
        if self.optimizer.problem_type == 'scheduling':
            self._draw_realtime_gantt(data['solution'])
        else:
            self._draw_realtime_route(data['solution'])
        self._refresh_canvas()

    def _show_final_result(self, data):
        """显示最终结果"""
        self.ax.clear()
        if self.optimizer.problem_type == 'scheduling':
            self._draw_final_gantt(data['solution'], data['score'])
        else:
            self._draw_final_route(data['solution'], data['score'])
        self._refresh_canvas()
        self.running = False
        self.start_btn.config(text="开始优化")

    def _draw_realtime_gantt(self, schedule):
        """实时绘制甘特图框架"""
        self.ax.set_title("优化进行中...")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")

    def _draw_realtime_route(self, route):
        """实时绘制路径框架"""
        self.ax.set_title("路径优化中...")
        self.ax.scatter(self.optimizer.locations[:,0], self.optimizer.locations[:,1], c='lightgray', s=60)

    def _draw_final_gantt(self, schedule, score):
        """绘制最终甘特图"""
        machine_load = [[] for _ in range(self.optimizer.machines)]
        current_time = np.zeros(self.optimizer.machines)
        
        for job in schedule:
            machine = job % self.optimizer.machines
            duration = self.optimizer.processing_times[machine, job]
            machine_load[machine].append((current_time[machine], duration))
            current_time[machine] += duration

        colors = plt.cm.tab20.colors
        for i, jobs in enumerate(machine_load):
            for start, dur in jobs:
                self.ax.add_patch(Rectangle(
                    (start, i-0.4), dur, 0.8,
                    edgecolor='black', facecolor=colors[i%20], alpha=0.7
                ))

        self.ax.set_xlim(0, np.max(current_time)*1.1)
        self.ax.set_ylim(-0.5, self.optimizer.machines-0.5)
        self.ax.set_title(f"最优生产调度方案（总耗时: {score:.1f} 分钟）")
        self.ax.set_xlabel("时间 (分钟)")
        self.ax.set_ylabel("机器编号")
        self.ax.grid(True, axis='x')

    def _draw_final_route(self, route, score):
        """绘制最终路径图"""
        locs = self.optimizer.locations
        
        # 绘制基础节点
        self.ax.scatter(locs[:,0], locs[:,1], c='lightgray', s=60, edgecolor='black', zorder=1)
        
        # 绘制最优路径
        path_locs = locs[route]
        self.ax.plot(path_locs[:,0], path_locs[:,1], 'b-', alpha=0.5, zorder=2)
        self.ax.scatter(path_locs[:,0], path_locs[:,1], c='red', s=80, edgecolor='black', zorder=3)
        
        # 添加路径箭头
        for i in range(len(route)-1):
            start = locs[route[i]]
            end = locs[route[i+1]]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            self.ax.arrow(start[0], start[1], dx*0.9, dy*0.9, 
                        head_width=3, head_length=5, fc='blue', ec='blue', zorder=4)
        
        self.ax.set_title(f"最优配送路径（总距离: {score:.1f} km）")
        self.ax.grid(True)

    def _refresh_canvas(self):
        """刷新画布"""
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def export_result(self):
        """导出可视化结果为图片"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")]
        )
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("导出成功", f"结果已保存至:\n{file_path}")

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()