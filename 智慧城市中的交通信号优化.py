"""
智慧城市交通信号优化系统 - 增强版
修改内容：
1. 始终显示生成的路口
2. 添加算法选择功能（灰狼优化算法/灰狼算法）
3. 优化界面布局和可视化效果
"""
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, Wedge
from scipy.spatial import cKDTree
import random
import time
import threading
from queue import Queue

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 算法配置类 ====================== #
class Config:
    """动态可配置参数"""
    def __init__(self, n_cross=10, n_phase=4, pop_size=20, max_iter=50):
        self.update_config(n_cross, n_phase, pop_size, max_iter)
        
    def update_config(self, n_cross, n_phase, pop_size, max_iter):
        self.n_cross = n_cross      # 交叉口数量
        self.n_phase = n_phase      # 每交叉口相位数
        self.pop_size = pop_size    
        self.max_iter = max_iter    
        
        # 信号参数范围
        self.g_min, self.g_max = 10, 60     # 绿灯时间范围
        self.offset_min, self.offset_max = -10, 10  # 相位差范围
        self.T_min, self.T_max = 60, 180    # 周期长度范围
        
        # 优化权重
        self.alpha = 0.6   # 等待时间权重
        self.beta = 0.3    # 通行效率权重
        self.gamma = 0.1   # 应急车辆权重

# ====================== 交通仿真模块 ====================== #
class RoadNetwork:
    """随机路网生成器"""
    def __init__(self, n_cross):
        self.n_cross = n_cross
        self.intersections = []  # 交叉口坐标列表
        self.roads = []         # 道路连接关系
        self.vehicles = []      # 车辆列表
        self.generate_random_network()
        
    def generate_random_network(self):
        """生成随机路网"""
        np.random.seed(int(time.time()))
        self.intersections = np.random.rand(self.n_cross, 2) * 700 + 50
        
        # 构建道路连接
        kdtree = cKDTree(self.intersections)
        self.roads = [[] for _ in range(self.n_cross)]
        
        for i in range(self.n_cross):
            distances, indices = kdtree.query(self.intersections[i], k=random.randint(3,5))
            for idx in indices[1:]:
                if idx not in self.roads[i]:
                    self.roads[i].append(int(idx))
                if i not in self.roads[idx]:
                    self.roads[idx].append(int(i))
        
        self.ensure_connectivity()
    
    def ensure_connectivity(self):
        """确保所有节点连通"""
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend([n for n in self.roads[node] if n not in visited])
        
        if len(visited) < self.n_cross:
            unconnected = set(range(self.n_cross)) - visited
            for node in unconnected:
                nearest = min(visited, key=lambda x: np.linalg.norm(
                    self.intersections[node] - self.intersections[x]))
                self.roads[node].append(nearest)
                self.roads[nearest].append(node)
                visited.add(node)
    
    def add_vehicle(self):
        """添加随机车辆"""
        if len(self.intersections) < 2:
            return
        
        start = random.randint(0, len(self.intersections)-1)
        end = random.choice([x for x in range(len(self.intersections)) if x != start])
        
        path = self.a_star_path(start, end)
        if not path:
            return
        
        speed = random.uniform(3, 6)
        self.vehicles.append({
            'pos': self.intersections[start].copy(),
            'path': path,
            'current_node': 0,
            'speed': speed,
            'color': (random.random(), random.random(), random.random()),
            'progress': 0.0,
            'waiting': 0
        })
    
    def a_star_path(self, start, end):
        """A*路径规划"""
        open_set = set([start])
        came_from = {}
        g_score = {n: float('inf') for n in range(self.n_cross)}
        g_score[start] = 0
        f_score = {n: float('inf') for n in range(self.n_cross)}
        f_score[start] = np.linalg.norm(self.intersections[end] - self.intersections[start])
        
        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            open_set.remove(current)
            for neighbor in self.roads[current]:
                tentative_g = g_score[current] + np.linalg.norm(
                    self.intersections[neighbor] - self.intersections[current])
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(
                        self.intersections[end] - self.intersections[neighbor])
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        return None
    
    def update_vehicles(self, signal_params):
        """更新车辆位置"""
        to_remove = []
        for idx, v in enumerate(self.vehicles):
            if v['current_node'] >= len(v['path'])-1:
                to_remove.append(idx)
                continue
                
            current = v['path'][v['current_node']]
            next_node = v['path'][v['current_node'] + 1]
            
            phase_index = current * len(self.roads[current])
            phase_status = signal_params[phase_index % len(signal_params)]
            
            start_pos = self.intersections[current]
            end_pos = self.intersections[next_node]
            distance = np.linalg.norm(end_pos - start_pos)
            
            if phase_status > 30:
                v['progress'] += v['speed'] * 2
                v['waiting'] = 0
            else:
                v['progress'] += v['speed'] * 0.3
                v['waiting'] += 1
                
            direction = (end_pos - start_pos) / distance
            v['pos'] = start_pos + direction * v['progress']
            
            if v['progress'] >= distance:
                v['current_node'] += 1
                v['progress'] = 0
        
        for idx in reversed(to_remove):
            del self.vehicles[idx]
        
        if random.random() < 0.2:
            self.add_vehicle()

# ====================== 优化算法类 ====================== #
class IGWO_TrafficOptimizer:
    """改进灰狼优化器"""
    def __init__(self, config, algorithm_type='igwo'):
        self.config = config
        self.algorithm_type = algorithm_type
        self.road_net = RoadNetwork(config.n_cross)
        self.min_bounds = []
        self.max_bounds = []
        self.calculate_bounds()
        self.running = True
        self.queue = Queue()
        self.analysis_data = {
            'wait_times': [],
            'throughputs': [],
            'fitness_history': []
        }
        
    def calculate_bounds(self):
        """计算参数边界"""
        self.min_bounds.clear()
        self.max_bounds.clear()
        for _ in range(self.config.n_cross):
            self.min_bounds.extend([self.config.g_min] * self.config.n_phase)
            self.max_bounds.extend([self.config.g_max] * self.config.n_phase)
            self.min_bounds.append(self.config.offset_min)
            self.max_bounds.append(self.config.offset_max)
            self.min_bounds.append(self.config.T_min)
            self.max_bounds.append(self.config.T_max)
    
    def chaotic_initialization(self):
        """混沌初始化种群"""
        population = []
        mu = 3.9
        dim = len(self.min_bounds)
        
        for _ in range(self.config.pop_size):
            chaos_seq = []
            x = np.random.rand()
            for _ in range(dim):
                x = mu * x * (1 - x)
                chaos_seq.append(x)
            
            individual = np.array([
                min_val + chaos * (max_val - min_val)
                for chaos, min_val, max_val in zip(chaos_seq, self.min_bounds, self.max_bounds)
            ])
            individual = np.clip(individual, self.min_bounds, self.max_bounds)
            population.append(individual)
        
        return np.array(population)
    
    def random_initialization(self):
        """随机初始化种群"""
        population = []
        dim = len(self.min_bounds)
        for _ in range(self.config.pop_size):
            individual = np.array([
                np.random.uniform(min_val, max_val)
                for min_val, max_val in zip(self.min_bounds, self.max_bounds)
            ])
            population.append(individual)
        return np.array(population)
    
    def evaluate_fitness(self, X):
        """适应度计算"""
        self.road_net.update_vehicles(X)
        total_wait = sum(v['waiting'] for v in self.road_net.vehicles)
        throughput = len(self.road_net.vehicles)
        penalty = 0.1 * np.sum(X < self.min_bounds) + 0.1 * np.sum(X > self.max_bounds)
        
        # 记录分析数据
        self.analysis_data['wait_times'].append(total_wait)
        self.analysis_data['throughputs'].append(throughput)
        
        return self.config.alpha * total_wait - self.config.beta * throughput + penalty
    
    def optimize(self):
        """主优化循环"""
        # 根据算法类型选择初始化方式
        if self.algorithm_type == 'igwo':
            population = self.chaotic_initialization()
        else:
            population = self.random_initialization()
            
        fitness = np.array([self.evaluate_fitness(ind) for ind in population])
        self.analysis_data['fitness_history'] = [np.min(fitness)]
        
        alpha = population[np.argmin(fitness)]
        beta = population[np.argsort(fitness)[1]]
        delta = population[np.argsort(fitness)[2]]
        
        for t in range(self.config.max_iter):
            if not self.running:
                break
            
            # 根据算法类型计算参数a
            if self.algorithm_type == 'igwo':
                a = 2 * np.exp(-5 * t / self.config.max_iter)
            else:
                a = 2 - 2 * t / self.config.max_iter
            
            for i in range(self.config.pop_size):
                A1 = 2*a*np.random.rand() - a
                A2 = 2*a*np.random.rand() - a
                A3 = 2*a*np.random.rand() - a
                C1 = 2*np.random.rand()
                C2 = 2*np.random.rand()
                C3 = 2*np.random.rand()
                
                D_alpha = np.abs(C1*alpha - population[i])
                X1 = alpha - A1*D_alpha
                D_beta = np.abs(C2*beta - population[i])
                X2 = beta - A2*D_beta
                D_delta = np.abs(C3*delta - population[i])
                X3 = delta - A3*D_delta
                
                new_pos = (X1 + X2 + X3) / 3
                new_pos = np.clip(new_pos, self.min_bounds, self.max_bounds)
                
                new_fitness = self.evaluate_fitness(new_pos)
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
            
            sorted_indices = np.argsort(fitness)
            alpha = population[sorted_indices[0]]
            beta = population[sorted_indices[1]]
            delta = population[sorted_indices[2]]
            self.analysis_data['fitness_history'].append(np.min(fitness))
            
            self.queue.put((self.road_net.vehicles.copy(), alpha))
            time.sleep(0.5)
        
        self.queue.put(None)
        return alpha

# ====================== 图形用户界面 ====================== #
class TrafficOptimizationGUI:
    """改进的交互界面"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("智慧交通优化系统 v10.0")
        self.config = Config()
        self.optimizer = None
        self.sim_thread = None
        self.update_interval = 500
        self.default_road_net = RoadNetwork(self.config.n_cross)  # 默认路网
        
        self.create_controls()
        self.create_visualization()
        self.create_analysis_panel()
        self.root.after(1000, self.add_random_vehicle)
        self.draw_roadnet([])  # 初始绘制路网
        
    def create_controls(self):
        """控制面板"""
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')
        
        ttk.Label(control_frame, text="交叉口数量:").grid(row=0, column=0)
        self.n_cross_entry = ttk.Entry(control_frame)
        self.n_cross_entry.insert(0, "10")
        self.n_cross_entry.grid(row=0, column=1)
        
        ttk.Label(control_frame, text="相位数量:").grid(row=1, column=0)
        self.n_phase_entry = ttk.Entry(control_frame)
        self.n_phase_entry.insert(0, "4")
        self.n_phase_entry.grid(row=1, column=1)
        
        # 算法选择
        ttk.Label(control_frame, text="算法选择:").grid(row=2, column=0)
        self.algorithm_var = tk.StringVar(value='灰狼优化算法')
        self.algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var,
                                           values=['灰狼优化算法', '灰狼算法'])
        self.algorithm_combo.grid(row=2, column=1)
        
        self.start_btn = ttk.Button(control_frame, text="开始优化", command=self.start_optimization)
        self.start_btn.grid(row=4, column=0, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="停止", command=self.stop_optimization)
        self.stop_btn.grid(row=4, column=1, pady=5)
        
    def create_visualization(self):
        """可视化区域"""
        self.fig = plt.figure(figsize=(12, 6))
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)
        
        # 主地图
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_map.set_title("实时交通仿真")
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        
        # 图例区域
        self.ax_legend = self.fig.add_subplot(gs[1])
        self.ax_legend.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10)
        
    def create_analysis_panel(self):
        """结果分析面板"""
        analysis_frame = ttk.LabelFrame(self.root, text="优化结果分析")
        analysis_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        # 创建分析指标显示
        self.analysis_text = tk.Text(analysis_frame, height=8, width=80)
        self.analysis_text.pack(padx=5, pady=5)
        
        # 初始提示信息
        self.analysis_text.insert(tk.END, "优化结果将在此显示...\n")
        self.analysis_text.configure(state='disabled')
        
    def update_analysis(self):
        """更新分析面板"""
        if not self.optimizer:
            return
        
        analysis_data = self.optimizer.analysis_data
        best_params = self.optimizer.queue.get() if not self.optimizer.queue.empty() else None
        
        self.analysis_text.configure(state='normal')
        self.analysis_text.delete(1.0, tk.END)
        
        # 显示关键指标
        self.analysis_text.insert(tk.END, "=== 最终优化结果 ===\n")
        self.analysis_text.insert(tk.END, f"最佳适应度值: {min(analysis_data['fitness_history']):.2f}\n")
        self.analysis_text.insert(tk.END, f"平均等待时间: {np.mean(analysis_data['wait_times']):.2f} 秒\n")
        self.analysis_text.insert(tk.END, f"平均通行量: {np.mean(analysis_data['throughputs']):.1f} 辆/分钟\n\n")
        
        # 显示关键参数
        self.analysis_text.insert(tk.END, "=== 最优信号参数 ===\n")
        params_per_cross = len(self.optimizer.min_bounds) // self.config.n_cross
        for i in range(min(3, self.config.n_cross)):  # 显示前3个路口
            start = i * params_per_cross
            end = start + params_per_cross
            cross_params = self.optimizer.alpha[start:end]
            self.analysis_text.insert(tk.END, f"路口{i+1}: 相位时间{cross_params[:self.config.n_phase]} | "
                                           f"相位差{cross_params[self.config.n_phase]:.1f} | "
                                           f"周期{cross_params[self.config.n_phase+1]:.1f}\n")
        
        self.analysis_text.configure(state='disabled')
        
    def add_random_vehicle(self):
        """定期添加车辆"""
        if self.optimizer and self.optimizer.running:
            try:
                self.optimizer.road_net.add_vehicle()
            except:
                pass
            self.root.after(3000, self.add_random_vehicle)
        
    def draw_roadnet(self, vehicles):
        """绘制随机路网"""
        self.ax_map.clear()
        
        # 使用当前路网或默认路网
        road_net = self.optimizer.road_net if self.optimizer else self.default_road_net
        
        # 绘制交叉口和道路
        if road_net:
            # 绘制交叉口
            for (x, y) in road_net.intersections:
                self.ax_map.add_patch(Circle((x, y), 10, color='darkred', zorder=3))
                
            # 绘制道路
            for i, neighbors in enumerate(road_net.roads):
                x1, y1 = road_net.intersections[i]
                for n in neighbors:
                    if n > i:
                        x2, y2 = road_net.intersections[n]
                        self.ax_map.plot([x1, x2], [y1, y2], 'gray', linewidth=4, zorder=1)
        
        # 绘制车辆
        if self.optimizer and vehicles:
            for v in vehicles:
                x, y = v['pos']
                self.ax_map.add_patch(Rectangle((x-8, y-8), 16, 16, 
                                     color=v['color'], alpha=0.8, zorder=2))
        
        self.ax_map.set_xlim(0, 800)
        self.ax_map.set_ylim(0, 600)
        self.canvas.draw()
        
    def draw_visual_legend(self, best_params):
        """可视化图例"""
        self.ax_legend.clear()
        self.ax_legend.axis('off')
        
        if best_params is None:
            return
        
        # 参数解析
        cross_params = []
        for i in range(0, len(best_params), self.config.n_phase + 2):
            phase_times = best_params[i:i+self.config.n_phase]
            offset = best_params[i+self.config.n_phase]
            cycle = best_params[i+self.config.n_phase +1]
            cross_params.append({
                'phases': phase_times,
                'offset': offset,
                'cycle': cycle
            })
        
        # 主标题
        self.ax_legend.text(0.5, 0.95, "信号参数可视化", 
                          ha='center', va='top', fontsize=12, color='navy')
        
        # 绘制前3个路口
        for idx, param in enumerate(cross_params[:3]):
            y_base = 0.85 - idx*0.25
            
            # 路口标题
            self.ax_legend.text(0.1, y_base + 0.08, f"路口{idx+1}", 
                               ha='left', va='top', fontsize=10)
            
            # 相位时间比例条
            total_time = sum(param['phases'])
            colors = plt.cm.viridis(np.array(param['phases'])/self.config.g_max)
            left = 0.1
            for p_time, color in zip(param['phases'], colors):
                width = p_time / total_time * 0.8
                self.ax_legend.barh(y_base, width=width, height=0.1,
                                   left=left, color=color, edgecolor='white')
                left += width
            
            # 相位差指示
            offset_ratio = (param['offset'] - self.config.offset_min) / (self.config.offset_max - self.config.offset_min)
            arrow_x = 0.1 + offset_ratio * 0.8
            self.ax_legend.arrow(arrow_x, y_base - 0.05, 0, -0.03,
                               head_width=0.03, head_length=0.02, fc='red')
            
            # 周期指示
            cycle_ratio = (param['cycle'] - self.config.T_min) / (self.config.T_max - self.config.T_min)
            self.ax_legend.add_patch(Wedge((0.8, y_base - 0.07), 0.04, 0, 360*cycle_ratio,
                                          color=plt.cm.plasma(cycle_ratio)))
        
        # 全局图例
        self.ax_legend.text(0.1, 0.15, "相位时长比例", color='dimgray', fontsize=9)
        self.ax_legend.text(0.1, 0.10, "←相位差→", color='red', fontsize=9)
        self.ax_legend.text(0.1, 0.05, "周期强度", color='purple', fontsize=9)
        
        self.canvas.draw()
        
    def update_gui(self):
        """定时更新界面"""
        try:
            if self.optimizer.queue.empty():
                self.root.after(self.update_interval, self.update_gui)
                return
            
            data = self.optimizer.queue.get()
            if data is None:
                self.update_analysis()
                return
            
            vehicles, best_params = data
            self.draw_roadnet(vehicles)
            self.draw_visual_legend(best_params)
            self.canvas.draw()
            
        except Exception as e:
            print(f"更新错误: {e}")
        
        self.root.after(self.update_interval, self.update_gui)
        
    def start_optimization(self):
        """启动优化线程"""
        try:
            self.config.update_config(
                n_cross=int(self.n_cross_entry.get()),
                n_phase=int(self.n_phase_entry.get()),
                pop_size=20,
                max_iter=50
            )
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字参数")
            return
            
        if self.optimizer and self.optimizer.running:
            return
            
        # 获取算法类型
        algorithm_type = 'igwo' if self.algorithm_var.get() == '灰狼优化算法' else 'gwo'
        self.optimizer = IGWO_TrafficOptimizer(self.config, algorithm_type)
        self.optimizer.running = True
        
        # 初始生成车辆
        for _ in range(8):
            self.optimizer.road_net.add_vehicle()
        
        self.sim_thread = threading.Thread(target=self.optimizer.optimize)
        self.sim_thread.start()
        
        self.root.after(self.update_interval, self.update_gui)
        
    def stop_optimization(self):
        """停止优化"""
        if self.optimizer:
            self.optimizer.running = False

# ====================== 主程序入口 ====================== #
if __name__ == "__main__":
    app = TrafficOptimizationGUI()
    app.root.mainloop()