import pygame
import numpy as np
import sys

# Configuration constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DEFAULT_GRID_SIZE = 15  # 默认格子大小
GRID_COLS = 300 # 逻辑网格宽度 (更大范围以模拟无限)
GRID_ROWS = 300 # 逻辑网格高度
UPDATE_INTERVAL = 200 # 演化间隔 (ms)，实现慢放效果

# Colors
COLOR_BLACK = (20, 20, 20)      # 背景色
COLOR_WHITE = (230, 230, 230)    # 存活细胞颜色
COLOR_GRID = (40, 40, 40)       # 网格线颜色
COLOR_TEXT = (200, 200, 200)    # 文本颜色
COLOR_STABLE = (100, 255, 100)  # 稳定状态提示色

class EvolutionDemo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Evolution Demo - 演化计时与观察")
        
        # 逻辑网格尺寸 (比窗口大，支持拖动查看)
        self.cols = GRID_COLS
        self.rows = GRID_ROWS
        
        # 缩放相关
        self.cell_size = float(DEFAULT_GRID_SIZE) # 使用浮点数以获得更高精度
        self.min_cell_size = 2.0
        self.max_cell_size = 100.0
        
        # 视口偏移量 (像素)
        # 初始居中
        initial_size = int(self.cell_size)
        self.offset_x = -(self.cols * initial_size - WINDOW_WIDTH) // 2
        self.offset_y = -(self.rows * initial_size - WINDOW_HEIGHT) // 2
        
        # 初始化网格
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        
        # 状态控制
        self.paused = True
        self.is_stable = False
        self.font = pygame.font.SysFont("SimHei", 20) # 使用支持中文的字体如果系统有，否则回退
        
        # 计时器相关
        self.start_ticks = 0
        self.total_elapsed_ticks = 0
        self.last_update_ticks = 0
        self.generation = 0
        
        # 鼠标拖动状态
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        # 初始密度设置 (None 表示随机 0.3-0.4)
        self.target_density = None 
        self.last_density_val = 0.0 # 用于显示上一次生成的实际密度
        
        # 输入模式状态
        self.input_mode = False
        self.input_buffer = ""
        
        self.ui_visible = True
        self.speed_multiplier = 1.0
        
        # 初始随机状态
        self.randomize_grid()

    def randomize_grid(self):
        """随机初始化网格并重置状态"""
        # 确定密度
        if self.target_density is None:
            density = np.random.uniform(0.3, 0.4)
            self.last_density_val = density
        else:
            density = self.target_density
            self.last_density_val = density

        # 遍布整个可运动到的地方 (整个逻辑网格)
        self.grid = np.random.choice([0, 1], size=(self.rows, self.cols), p=[1-density, density])
        
        self.reset_timer()

    def reset_timer(self):
        self.paused = True
        self.is_stable = False
        self.generation = 0
        self.total_elapsed_ticks = 0
        self.start_ticks = pygame.time.get_ticks()

    def update_grid(self):
        """应用演化规则并检测稳定状态"""
        if self.is_stable:
            return

        # 周期性边界条件 (Wrap-around)
        neighbors_count = (
            np.roll(np.roll(self.grid, 1, axis=0), 1, axis=1) +
            np.roll(self.grid, 1, axis=0) +
            np.roll(np.roll(self.grid, 1, axis=0), -1, axis=1) +
            np.roll(self.grid, 1, axis=1) +
            np.roll(self.grid, -1, axis=1) +
            np.roll(np.roll(self.grid, -1, axis=0), 1, axis=1) +
            np.roll(self.grid, -1, axis=0) +
            np.roll(np.roll(self.grid, -1, axis=0), -1, axis=1)
        )

        birth_mask = (neighbors_count == 3)
        survival_mask = (self.grid == 1) & (neighbors_count == 2)
        new_grid = np.where(birth_mask | survival_mask, 1, 0)
        
        # 检测是否达到稳定状态 (完全静止)
        if np.array_equal(self.grid, new_grid):
            self.is_stable = True
            # print(f"Stable state reached at Generation {self.generation}")
        
        self.grid = new_grid
        self.generation += 1

    def draw(self):
        self.screen.fill(COLOR_BLACK)
        
        # 使用当前缩放级别计算绘制参数
        current_cell_size = self.cell_size
        
        # 计算可见区域以优化绘制
        # 视口左上角在世界坐标中的位置: -offset_x, -offset_y
        view_x = -self.offset_x
        view_y = -self.offset_y
        
        start_col = max(0, int(view_x / current_cell_size))
        end_col = min(self.cols, int((view_x + WINDOW_WIDTH) / current_cell_size) + 1)
        start_row = max(0, int(view_y / current_cell_size))
        end_row = min(self.rows, int((view_y + WINDOW_HEIGHT) / current_cell_size) + 1)
        
        # 绘制可见的存活细胞
        # 获取可见区域内的切片
        if start_row < end_row and start_col < end_col:
            visible_grid = self.grid[start_row:end_row, start_col:end_col]
            local_y, local_x = np.where(visible_grid == 1)
            
            # 预计算间隙大小
            gap = 1 if current_cell_size > 4 else 0
            block_size = max(1, current_cell_size - gap)
            
            # 转换为 int 以绘制，使用 ceil 确保覆盖
            # 注意：pygame.Rect 接受 float，但为了避免缝隙，显式控制更好
            
            for y, x in zip(local_y, local_x):
                # 转换回世界坐标 -> 屏幕坐标
                world_row = start_row + y
                world_col = start_col + x
                
                screen_x = world_col * current_cell_size + self.offset_x
                screen_y = world_row * current_cell_size + self.offset_y
                
                # 使用 pygame 的 float 坐标支持
                pygame.draw.rect(
                    self.screen, 
                    COLOR_WHITE, 
                    (screen_x, screen_y, block_size, block_size)
                )

        # 绘制网格边界 (可选，提示有限的世界范围)
        border_rect = pygame.Rect(
            self.offset_x, 
            self.offset_y, 
            self.cols * current_cell_size, 
            self.rows * current_cell_size
        )
        pygame.draw.rect(self.screen, COLOR_GRID, border_rect, 2)

        if self.ui_visible:
            self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):
        # 状态文本
        status = "已暂停 (按空格开始)" if self.paused else "演化中..."
        if self.is_stable:
            status = "已达到稳定状态 (静止)"
            color = COLOR_STABLE
        else:
            color = COLOR_TEXT

        # 时间计算
        if not self.paused and not self.is_stable:
            current_ticks = pygame.time.get_ticks()
            # 简单处理：仅在运行时累加时间不太准确，这里直接显示运行时长
            # 更精确的做法是每帧累加 delta_time
            pass 
        
        # 显示时间：秒.毫秒
        seconds = self.total_elapsed_ticks / 1000.0
        
        # 显示当前密度信息
        if self.input_mode:
            density_str = f"输入中: {self.input_buffer}% (按 Enter 确认)"
            color_density = COLOR_STABLE
        elif self.target_density is None:
            density_str = f"随机 (30%-40%) [当前: {self.last_density_val:.1%}]"
            color_density = COLOR_TEXT
        else:
            density_str = f"固定: {self.target_density:.2%} (按 D 恢复默认)"
            color_density = COLOR_TEXT
        
        speed_str = f"{self.speed_multiplier:.2f}x"
        
        info_lines = [
            f"状态: {status}",
            f"代数: {self.generation}",
            f"耗时: {seconds:.2f} 秒",
            (f"初始密度: {density_str}", color_density), # 元组支持特定颜色
            f"速度: {speed_str}",
            f"控制: 空格=暂停 | R=重置 | Enter=输入密度 | ↑↓=±1% | ←→=±0.1%",
            f"      D=默认密度(立即生效) | H=隐藏UI | 1-9=倍速 | +/-=调速 | 滚轮=缩放"
        ]
        
        y = 10
        for item in info_lines:
            if isinstance(item, tuple):
                line, line_color = item
            else:
                line, line_color = item, color
            
            try:
                text = self.font.render(line, True, line_color)
            except:
                # Fallback font if SimHei fails
                text = pygame.font.SysFont("Arial", 20).render(line, True, line_color)
            self.screen.blit(text, (10, y))
            y += 30

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            current_ticks = pygame.time.get_ticks()
            dt = clock.tick(60) # 保持 UI 响应 60 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    key_char = event.unicode.lower() if event.unicode else ""
                    key_name = pygame.key.name(event.key).lower()
                    if event.key == pygame.K_h or key_char == "h" or key_name == "h":
                        self.ui_visible = not self.ui_visible
                        self.input_mode = False
                        self.input_buffer = ""
                        continue
                    elif event.key == pygame.K_r or key_char == "r" or key_name == "r":
                        self.input_mode = False
                        self.input_buffer = ""
                        self.randomize_grid()
                        continue
                    elif event.key == pygame.K_d or key_char == "d" or key_name == "d":
                        self.input_mode = False
                        self.input_buffer = ""
                        self.target_density = None
                        self.randomize_grid()
                        continue
                    
                    # 输入模式处理
                    if self.input_mode:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            # 确认输入
                            try:
                                val = float(self.input_buffer)
                                if 0 <= val <= 100:
                                    self.target_density = val / 100.0
                                    self.randomize_grid()
                                else:
                                    print("请输入 0-100 之间的数值")
                            except ValueError:
                                pass # 无效输入忽略
                            self.input_mode = False
                            self.input_buffer = ""
                        elif event.key == pygame.K_ESCAPE:
                            # 取消输入
                            self.input_mode = False
                            self.input_buffer = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_buffer = self.input_buffer[:-1]
                        else:
                            # 仅允许数字和小数点
                            if event.unicode.isdigit() or event.unicode == '.':
                                self.input_buffer += event.unicode
                        continue # 输入模式下拦截其他按键

                    # 常规模式按键处理
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        # 如果从暂停恢复，更新 last_update_ticks 防止瞬间跳变
                        if not self.paused:
                            self.last_update_ticks = current_ticks
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER or event.key == pygame.K_i:
                        # 进入输入模式
                        self.input_mode = True
                        self.input_buffer = ""
                        self.paused = True # 输入时自动暂停
                    elif event.key == pygame.K_UP:
                        # 增加初始密度 (+1%)
                        if self.target_density is None:
                            self.target_density = 0.35
                        self.target_density = min(0.99, self.target_density + 0.01)
                        self.randomize_grid()
                    elif event.key == pygame.K_DOWN:
                        # 减少初始密度 (-1%)
                        if self.target_density is None:
                            self.target_density = 0.35
                        self.target_density = max(0.01, self.target_density - 0.01)
                        self.randomize_grid()
                    elif event.key == pygame.K_RIGHT:
                        # 微调增加 (+0.1%)
                        if self.target_density is None:
                            self.target_density = 0.35
                        self.target_density = min(0.999, self.target_density + 0.001)
                        self.randomize_grid()
                    elif event.key == pygame.K_LEFT:
                        # 微调减少 (-0.1%)
                        if self.target_density is None:
                            self.target_density = 0.35
                        self.target_density = max(0.001, self.target_density - 0.001)
                        self.randomize_grid()
                    elif event.key == pygame.K_1:
                        self.speed_multiplier = 1.0
                    elif event.key == pygame.K_2:
                        self.speed_multiplier = 2.0
                    elif event.key == pygame.K_3:
                        self.speed_multiplier = 3.0
                    elif event.key == pygame.K_4:
                        self.speed_multiplier = 4.0
                    elif event.key == pygame.K_5:
                        self.speed_multiplier = 5.0
                    elif event.key == pygame.K_6:
                        self.speed_multiplier = 6.0
                    elif event.key == pygame.K_7:
                        self.speed_multiplier = 7.0
                    elif event.key == pygame.K_8:
                        self.speed_multiplier = 8.0
                    elif event.key == pygame.K_9:
                        self.speed_multiplier = 9.0
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.speed_multiplier = max(0.1, self.speed_multiplier / 1.25)
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.speed_multiplier = min(8.0, self.speed_multiplier * 1.25)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click to drag
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                        # 抓手光标
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                
                if event.type == pygame.MOUSEWHEEL:
                    # 滚轮缩放逻辑
                    old_cell_size = self.cell_size
                    
                    # 使用比例缩放代替固定步长，实现更平滑的缩放
                    zoom_factor = 1.1 # 缩放比例 10%
                    
                    if event.y > 0: # 向上滚动，放大
                        self.cell_size = min(self.max_cell_size, self.cell_size * zoom_factor)
                    elif event.y < 0: # 向下滚动，缩小
                        self.cell_size = max(self.min_cell_size, self.cell_size / zoom_factor)
                    
                    # 调整 offset 以保持鼠标指向的位置不变
                    if self.cell_size != old_cell_size:
                        mx, my = pygame.mouse.get_pos()
                        
                        # 计算鼠标在逻辑网格中的相对位置 (基于旧的缩放)
                        grid_x = (mx - self.offset_x) / old_cell_size
                        grid_y = (my - self.offset_y) / old_cell_size
                        
                        # 基于新的缩放重新计算 offset
                        self.offset_x = mx - grid_x * self.cell_size
                        self.offset_y = my - grid_y * self.cell_size

                if event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.offset_x += dx
                        self.offset_y += dy
                        self.last_mouse_pos = event.pos

            # 演化逻辑更新 (基于时间间隔)
            if not self.paused and not self.is_stable:
                interval_ms = max(1, int(UPDATE_INTERVAL / self.speed_multiplier))
                if current_ticks - self.last_update_ticks > interval_ms:
                    self.update_grid()
                    # 累加实际演化时间
                    self.total_elapsed_ticks += (current_ticks - self.last_update_ticks)
                    self.last_update_ticks = current_ticks
            
            self.draw()

if __name__ == "__main__":
    demo = EvolutionDemo()
    demo.run()
