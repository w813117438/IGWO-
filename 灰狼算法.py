import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import matplotlib.pyplot as plt
import os

# ==================== 中文字体配置 ====================
try:
    if os.name == 'nt':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：中文字体配置失败")

# ==================== 主程序类 ====================
class GWOAnalysisSystem:
    def __init__(self, master):
        self.master = master
        self.setup_ui()
        self.init_algorithm()

    # ==================== 界面构建 ====================
    def setup_ui(self):
        """构建图形用户界面（关键修复点）"""
        self.master.title("灰狼算法分析平台")
        self.master.geometry("1200x600")
        
        # 主容器（左右分割布局）
        main_panel = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板（算法控制与主可视化）
        left_panel = ttk.Frame(main_panel)
        main_panel.add(left_panel)
        
        # 右侧面板（实时分析曲线）
        right_panel = ttk.Frame(main_panel)
        main_panel.add(right_panel)

        # ---------- 左侧控制面板 ----------
        control_frame = ttk.LabelFrame(left_panel, text="算法控制", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 参数输入组件（关键修复：保存为实例变量）
        self.pop_size = self.create_input_field(control_frame, "狼群数量:", "30", 0)
        self.max_iter = self.create_input_field(control_frame, "最大迭代:", "100", 1)

        # 目标选择单选按钮
        self.target_mode = tk.StringVar(value="random")
        self.create_radiobtn(control_frame, "随机目标", "random", 2)
        self.create_radiobtn(control_frame, "点击设置", "click", 3)

        # 控制按钮
        self.create_button(control_frame, "开始优化", self.start_optimization, 4)
        self.create_button(control_frame, "重置系统", self.reset_system, 5)

        # 搜索模式选择
        self.search_mode = tk.StringVar(value="auto")
        ttk.Label(control_frame, text="搜索模式:").grid(row=6, column=0, sticky=tk.W)
        modes = [("自适应", "auto"), ("全局搜索", "global"), ("局部搜索", "local")]
        for idx, (text, val) in enumerate(modes, 7):
            self.create_radiobtn(control_frame, text, val, idx, self.search_mode)

        # 主可视化画布
        main_fig = Figure(figsize=(5, 5), dpi=100)
        self.ax_main = main_fig.add_subplot(111)
        self.ax_main.set_title("灰狼群体动态分布", fontsize=12)
        self.canvas_main = FigureCanvasTkAgg(main_fig, master=left_panel)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_main.mpl_connect('button_press_event', self.set_target)

        # ---------- 右侧分析面板 ----------
        analysis_frame = ttk.LabelFrame(right_panel, text="实时分析", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 多样性曲线画布
        analysis_fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_analysis = analysis_fig.add_subplot(111)
        self.ax_analysis.set_title("种群多样性变化曲线", fontsize=12)
        self.ax_analysis.set_xlabel("迭代次数")
        self.ax_analysis.set_ylabel("多样性指数")
        self.canvas_analysis = FigureCanvasTkAgg(analysis_fig, master=analysis_frame)
        self.canvas_analysis.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_input_field(self, frame, label_text, default_val, row):
        """创建带标签的输入框（返回Entry对象）"""
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W)
        entry = ttk.Entry(frame, width=12)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, pady=2)
        return entry  # 必须返回Entry对象

        
    # ==================== 界面组件生成方法 ====================
    def create_input_field(self, frame, label_text, default_val, row):
        """创建带标签的输入框"""
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W)
        entry = ttk.Entry(frame, width=12)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, pady=2)
        return entry

    def create_radiobtn(self, frame, text, value, row, variable=None):
        """创建单选按钮"""
        variable = variable if variable else self.target_mode
        rb = ttk.Radiobutton(frame, text=text, variable=variable, value=value)
        rb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

    def create_button(self, frame, text, command, row):
        """创建功能按钮"""
        btn = ttk.Button(frame, text=text, command=command)
        btn.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.EW)

    # ==================== 算法核心方法 ====================
    def init_algorithm(self):
        """初始化算法状态"""
        self.target = None        # 目标位置坐标
        self.wolves = None        # 狼群位置数据
        self.alpha = None         # α狼（最优解）
        self.beta = None          # β狼（次优解）
        self.delta = None         # δ狼（第三优解）
        self.iteration = 0        # 当前迭代次数
        self.running = False      # 算法运行状态
        self.start_time = 0       # 运行开始时间
        self.history = []         # 历史数据记录
        self.current_a = 0.0      # 当前收敛因子值
        self.diversity_data = []  # 多样性数据集

    def objective_function(self, position):
        """目标函数（二维欧氏距离）"""
        return np.sqrt((position[0]-self.target[0])**2 + (position[1]-self.target[1])**2)

    def set_target(self, event):
        """鼠标点击设置目标位置"""
        if self.target_mode.get() == "click" and not self.running:
            self.target = np.array([event.xdata, event.ydata])
            self.update_visualization()

    # ==================== 可视化更新方法 ====================
    def update_visualization(self):
        """更新主可视化画面"""
        self.ax_main.clear()
        self.ax_main.set_xlim(0, 100)
        self.ax_main.set_ylim(0, 100)
        
        # 绘制目标位置
        if self.target is not None:
            self.ax_main.scatter(*self.target, c='red', s=200, marker='*', label='目标位置')
            
        # 绘制狼群分布
        if self.wolves is not None:
            positions = np.array([w['position'] for w in self.wolves])
            self.ax_main.scatter(positions[:,0], positions[:,1], c='blue', alpha=0.5, label='狼群')
            
            # 绘制领导狼
            leaders = [
                (self.alpha, 'gold', 'α狼'),
                (self.beta, 'silver', 'β狼'),
                (self.delta, 'brown', 'δ狼')
            ]
            for wolf, color, label in leaders:
                if wolf is not None:
                    self.ax_main.scatter(*wolf['position'], c=color, s=100, marker='s', label=label)
        
        # 添加图例和状态信息
        mode_map = {"auto": "自适应", "global": "全局", "local": "局部"}
        title = f"灰狼算法 - {mode_map[self.search_mode.get()]}模式 | a={self.current_a:.2f}"
        self.ax_main.set_title(title, fontsize=12)
        self.ax_main.legend(loc='upper right')
        self.canvas_main.draw()

    def update_analysis(self):
        """更新分析曲线"""
        self.ax_analysis.clear()
        iterations = [d["iteration"] for d in self.history]
        diversities = [d["diversity"] for d in self.history]
        
        # 绘制多样性曲线
        self.ax_analysis.plot(iterations, diversities, 'b-', lw=2, label='多样性')
        self.ax_analysis.set_title("种群多样性变化趋势")
        self.ax_analysis.set_xlabel("迭代次数")
        self.ax_analysis.set_ylabel("多样性指数")
        self.ax_analysis.grid(True)
        self.ax_analysis.set_xlim(0, int(self.max_iter.get()))
        self.ax_analysis.legend()
        self.canvas_analysis.draw()

    # ==================== 算法控制方法 ====================
    def start_optimization(self):
        """启动优化流程"""
        if self.target is None:
            if self.target_mode.get() == "random":
                self.target = np.random.uniform(0, 100, 2)
            else:
                return
        
        self.init_wolf_population()  # 初始化狼群
        self.running = True          # 设置运行标志
        self.start_time = time.time()
        self.optimization_loop()     # 进入优化循环

    def init_wolf_population(self):
        """初始化狼群位置"""
        pop_size = int(self.pop_size.get())  # 正确访问输入框
        self.wolves = []
        for _ in range(pop_size):
            pos = np.random.uniform(0, 100, 2)
            fitness = self.objective_function(pos)
            self.wolves.append({'position': pos, 'fitness': fitness})
        self.update_leaders()

    def update_leaders(self):
        """更新领导狼信息"""
        sorted_wolves = sorted(self.wolves, key=lambda x: x['fitness'])
        self.alpha = sorted_wolves[0]
        self.beta = sorted_wolves[1]
        self.delta = sorted_wolves[2]

    def optimization_loop(self):
        """优化过程主循环"""
        if not self.running or self.iteration >= int(self.max_iter.get()):
            self.running = False
            print(f"优化完成 | 耗时: {time.time()-self.start_time:.2f}s")
            return
        
        # 计算当前收敛因子
        self.current_a = self.calc_convergence_factor()
        
        # 更新每个狼的位置
        for i in range(len(self.wolves)):
            r1, r2 = np.random.rand(2), np.random.rand(2)
            A = 2 * self.current_a * r1 - self.current_a
            C = 2 * r2
            
            # 计算三个领导狼的影响
            D_alpha = abs(C * self.alpha['position'] - self.wolves[i]['position'])
            X1 = self.alpha['position'] - A * D_alpha
            
            D_beta = abs(C * self.beta['position'] - self.wolves[i]['position'])
            X2 = self.beta['position'] - A * D_beta
            
            D_delta = abs(C * self.delta['position'] - self.wolves[i]['position'])
            X3 = self.delta['position'] - A * D_delta
            
            # 位置更新与边界处理
            new_pos = np.clip((X1 + X2 + X3) / 3, 0, 100)
            
            # 柯西变异增强探索能力
            if np.random.rand() < 0.1:
                new_pos += 0.1 * np.random.standard_cauchy(size=2)
            new_pos = np.clip(new_pos, 0, 100)
            
            self.wolves[i]['position'] = new_pos
            self.wolves[i]['fitness'] = self.objective_function(new_pos)
        
        # 记录运行数据
        positions = np.array([w['position'] for w in self.wolves])
        diversity = np.mean(np.std(positions, axis=0))
        self.history.append({
            "iteration": self.iteration,
            "a": self.current_a,
            "diversity": diversity,
            "best_fitness": self.alpha['fitness']
        })
        
        # 更新界面显示
        self.update_leaders()
        self.update_visualization()
        self.update_analysis()
        
        # 迭代计数更新
        self.iteration += 1
        self.master.after(50, self.optimization_loop)

    def calc_convergence_factor(self):
        """计算收敛因子"""
        total_iter = int(self.max_iter.get())
        current_iter = self.iteration
        
        if self.search_mode.get() == "global":
            return 2.0
        elif self.search_mode.get() == "local":
            return 0.0
        else:  # 自适应模式
            return 2 * np.exp(-5 * current_iter / total_iter)

    def reset_system(self):
        """重置系统状态"""
        self.running = False
        self.init_algorithm()
        self.update_visualization()
        self.ax_analysis.clear()
        self.canvas_analysis.draw()

# ==================== 程序入口 ====================
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TButton', font=('Microsoft YaHei', 10))  # 按钮样式
    style.configure('TRadiobutton', font=('Microsoft YaHei', 9))  # 单选按钮样式
    app = GWOAnalysisSystem(root)
    root.mainloop()