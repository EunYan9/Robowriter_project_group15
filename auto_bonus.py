###
# Final project with MuJoCo
# SI100B Robotics Programming
# RoboWriter Auto Planar: Universal Text + Null-Space Posture + MinJerk
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# ================= 配置区域 =================
xml_path = '../../models/universal_robots_ur5e/scene.xml' 
simend = 180 

# 定义高度常量
Z_PAPER = 0.10  # 纸面 (写字高度)
Z_AIR   = 0.15  # 空中 (抬笔高度)

# 结束位姿
HOME_Q = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0]) 

# 【关键】定义“舒适书写姿态” (Rest Pose)
# Base=-90, Shoulder=-90, Elbow=90, Wrist1=-90 (笔尖垂直向下)
Q_REST = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# ================= 1. 字库定义 (保持原版 auto.py 逻辑) =================

CHAR_WIDTH = 0.012   # 标准字符宽度
CHAR_HEIGHT = 0.025  # 大写/数字高度
CHAR_SPACING = 0.005 # 字符间距

def get_grid(base_pos, width, height):
    cx, cy, _ = base_pos
    y_base = cy
    y_top  = cy + height
    y_mid  = cy + height * 0.55
    y_desc = cy - height * 0.3
    x_l = cx - width/2
    x_c = cx
    x_r = cx + width/2
    z = Z_PAPER
    
    return {
        'TL': np.array([x_l, y_top, z]), 'TC': np.array([x_c, y_top, z]), 'TR': np.array([x_r, y_top, z]),
        'ML': np.array([x_l, y_mid, z]), 'MC': np.array([x_c, y_mid, z]), 'MR': np.array([x_r, y_mid, z]),
        'BL': np.array([x_l, y_base, z]), 'BC': np.array([x_c, y_base, z]), 'BR': np.array([x_r, y_base, z]),
        'DL': np.array([x_l, y_desc, z]), 'DC': np.array([x_c, y_desc, z]), 'DR': np.array([x_r, y_desc, z]),
    }

def get_digit_strokes(digit, base_pos, width=CHAR_WIDTH, height=CHAR_HEIGHT):
    g = get_grid(base_pos, width, height)
    strokes = []
    if digit == '0':
        strokes.extend([[g['TL'], g['TR']], [g['TR'], g['BR']], [g['BR'], g['BL']], [g['BL'], g['TL']], [g['TR'], g['BL']]])
    elif digit == '1':
        strokes.append([g['TC'], g['BC']])
    elif digit == '2':
        strokes.extend([[g['TL'], g['TR']], [g['TR'], g['MR']], [g['MR'], g['BL']], [g['BL'], g['BR']]])
    elif digit == '3':
        strokes.extend([[g['TL'], g['TR']], [g['TR'], g['MC']], [g['MC'], g['BR']], [g['BR'], g['BL']]])
    elif digit == '4':
        strokes.extend([[g['TL'], g['ML']], [g['ML'], g['MR']], [g['TC'], g['BC']]])
    elif digit == '5':
        strokes.extend([[g['TR'], g['TL']], [g['TL'], g['ML']], [g['ML'], g['MR']], [g['MR'], g['BR']], [g['BR'], g['BL']]])
    elif digit == '6':
        strokes.extend([[g['TR'], g['TL']], [g['TL'], g['BL']], [g['BL'], g['BR']], [g['BR'], g['MR']], [g['MR'], g['ML']]])
    elif digit == '7':
        strokes.extend([[g['TL'], g['TR']], [g['TR'], g['BL']]])
    elif digit == '8':
        strokes.extend([[g['TL'], g['TR']], [g['TR'], g['BR']], [g['BR'], g['BL']], [g['BL'], g['TL']], [g['ML'], g['MR']]])
    elif digit == '9':
        strokes.extend([[g['BL'], g['BR']], [g['BR'], g['TR']], [g['TR'], g['TL']], [g['TL'], g['ML']], [g['ML'], g['MR']]])
    return strokes

def get_letter_strokes(letter, base_pos, width=CHAR_WIDTH, height=CHAR_HEIGHT):
    g = get_grid(base_pos, width, height)
    strokes = []
    
    if 'A' <= letter <= 'Z':
        if letter == 'A':
            strokes.extend([[g['BL'], g['TC']], [g['TC'], g['BR']], [g['ML'], g['MR']]])
        elif letter == 'B':
            strokes.append([g['TL'], g['BL']])
            strokes.append([g['TL'], g['TR'], g['MC']]) 
            strokes.append([g['MC'], g['BR'], g['BL']]) 
        elif letter == 'C':
            strokes.append([g['TR'], g['TL'], g['ML']])
            strokes.append([g['ML'], g['BL'], g['BR']])
        elif letter == 'D':
            strokes.append([g['TL'], g['BL']])
            strokes.append([g['TL'], g['TR'], g['MR']])
            strokes.append([g['MR'], g['BR'], g['BL']])
        elif letter == 'E':
            strokes.extend([[g['TL'], g['BL']], [g['TL'], g['TR']], [g['ML'], g['MR']], [g['BL'], g['BR']]])
        elif letter == 'F':
            strokes.extend([[g['TL'], g['BL']], [g['TL'], g['TR']], [g['ML'], g['MR']]])
        elif letter == 'G':
            strokes.extend([[g['TR'], g['TC'], g['TL']], [g['TL'], g['BL'], g['BR']], [g['BR'], g['MR']], [g['MR'], g['MC']]])
        elif letter == 'H':
            strokes.extend([[g['TL'], g['BL']], [g['TR'], g['BR']], [g['ML'], g['MR']]])
        elif letter == 'I':
            strokes.extend([[g['TL'], g['TR']], [g['TC'], g['BC']], [g['BL'], g['BR']]])
        elif letter == 'J':
            strokes.append([g['TL'], g['TR']])
            strokes.append([g['TC'], g['BC']])
            hook_ctrl = g['BC'] + np.array([0, -height*0.2, 0])
            strokes.append([g['BC'], hook_ctrl, g['BL']])
        elif letter == 'K':
            strokes.extend([[g['TL'], g['BL']], [g['ML'], g['TR']], [g['ML'], g['BR']]])
        elif letter == 'L':
            strokes.extend([[g['TL'], g['BL']], [g['BL'], g['BR']]])
        elif letter == 'M':
            strokes.extend([[g['BL'], g['TL']], [g['TL'], g['BC']], [g['BC'], g['TR']], [g['TR'], g['BR']]])
        elif letter == 'N':
            strokes.extend([[g['BL'], g['TL']], [g['TL'], g['BR']], [g['BR'], g['TR']]])
        elif letter == 'O':
            strokes.append([g['TC'], g['ML'], g['BC']]) 
            strokes.append([g['BC'], g['MR'], g['TC']]) 
        elif letter == 'P':
            strokes.append([g['BL'], g['TL']])
            strokes.append([g['TL'], g['TR'], g['MC']])
            strokes.append([g['MC'], g['ML']])
        elif letter == 'Q':
            strokes.append([g['TC'], g['ML'], g['BC']])
            strokes.append([g['BC'], g['MR'], g['TC']])
            strokes.append([g['MC'], g['BR']])
        elif letter == 'R':
            strokes.append([g['BL'], g['TL']])
            strokes.append([g['TL'], g['TR'], g['MC']])
            strokes.append([g['MC'], g['ML']])
            strokes.append([g['MC'], g['BR']])
        elif letter == 'S':
            strokes.append([g['TR'], g['TL'], g['MC']])
            strokes.append([g['MC'], g['BR'], g['BL']])
        elif letter == 'T':
            strokes.extend([[g['TL'], g['TR']], [g['TC'], g['BC']]])
        elif letter == 'U':
            strokes.append([g['TL'], g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR'], g['TR']])
        elif letter == 'V':
            strokes.extend([[g['TL'], g['BC']], [g['BC'], g['TR']]])
        elif letter == 'W':
            strokes.extend([[g['TL'], g['BL']], [g['BL'], g['MC']], [g['MC'], g['BR']], [g['BR'], g['TR']]])
        elif letter == 'X':
            strokes.extend([[g['TL'], g['BR']], [g['TR'], g['BL']]])
        elif letter == 'Y':
            strokes.extend([[g['TL'], g['MC']], [g['TR'], g['MC']], [g['MC'], g['BC']]])
        elif letter == 'Z':
            strokes.extend([[g['TL'], g['TR']], [g['TR'], g['BL']], [g['BL'], g['BR']]])

    elif 'a' <= letter <= 'z':
        y_top_x = g['ML'][1]; y_bottom = g['BL'][1]
        y_center_box = (y_top_x + y_bottom) / 2
        center_s = np.array([g['MC'][0], y_center_box, Z_PAPER])
        L_mid = np.array([g['ML'][0], y_center_box, Z_PAPER])
        R_mid = np.array([g['MR'][0], y_center_box, Z_PAPER])
        y_cross = (y_top_x + y_bottom) / 2
        cross_l = np.array([g['ML'][0], y_cross, Z_PAPER])
        cross_r = np.array([g['MR'][0], y_cross, Z_PAPER])
        
        if letter == 'a':
            strokes.append([g['MR'], g['BR']])
            strokes.append([g['MR'], g['ML'], L_mid])
            strokes.append([L_mid, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR']])
        elif letter == 'b':
            strokes.append([g['TL'], g['BL']])
            strokes.append([g['ML'], g['MR'], R_mid])
            strokes.append([R_mid, g['BR'], g['BC']])
            strokes.append([g['BC'], g['BL']])
        elif letter == 'c':
            strokes.extend([[g['MR'], g['ML'], g['BL']], [g['BL'], g['BR']]])
        elif letter == 'd':
            strokes.append([g['TR'], g['BR']])
            strokes.append([g['MR'], g['ML'], L_mid])
            strokes.append([L_mid, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR']])
        elif letter == 'e':
            strokes.append([cross_l, cross_r])
            strokes.append([cross_r, g['MR'], g['MC']])
            strokes.append([g['MC'], g['ML'], cross_l])
            strokes.append([cross_l, g['BL'], g['BR']])
        elif letter == 'f':
            strokes.append([g['TC'], g['TL'], g['ML']]) 
            strokes.append([g['ML'], g['BL']])          
            strokes.append([g['ML'], g['MC']])          
        elif letter == 'g':
            strokes.append([g['MR'], g['ML'], L_mid])
            strokes.append([L_mid, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR']])
            strokes.append([g['BR'], g['MR']])
            strokes.append([g['MR'], g['DR'], g['DC']])
        elif letter == 'h':
            strokes.append([g['TL'], g['BL']])          
            strokes.append([g['ML'], g['MC'], g['MR']]) 
            strokes.append([g['MR'], g['BR']])          
        elif letter == 'i':
            strokes.append([g['ML'], g['BL']])
            dot_center = g['ML'] + np.array([0, height * 0.4, 0])
            dot_start = dot_center + np.array([0, height * 0.02, 0])
            dot_end   = dot_center - np.array([0, height * 0.02, 0])
            strokes.append([dot_start, dot_end])
        elif letter == 'j':
            strokes.append([g['MC'], g['DC'], g['DL']]) 
            dot_center = g['MC'] + np.array([0, height * 0.4, 0])
            dot_start = dot_center + np.array([0, height * 0.02, 0])
            dot_end   = dot_center - np.array([0, height * 0.02, 0])
            strokes.append([dot_start, dot_end])
        elif letter == 'k':
            strokes.append([g['TL'], g['BL']]) 
            y_knot = (g['ML'][1] + g['BL'][1]) / 2
            k_knot = np.array([g['ML'][0], y_knot, Z_PAPER])
            strokes.append([k_knot, g['MR']]) 
            strokes.append([k_knot, g['BR']]) 
        elif letter == 'l':
            strokes.append([g['TL'], g['BL']])
        elif letter == 'm':
            strokes.append([g['ML'], g['BL']])
            strokes.append([g['ML'], g['MC'], g['BC']]) 
            strokes.append([g['MC'], g['MR'], g['BR']]) 
        elif letter == 'n':
            strokes.append([g['ML'], g['BL']])
            strokes.append([g['ML'], g['MR'], g['BR']]) 
        elif letter == 'o':
            strokes.append([g['MC'], g['ML'], L_mid])
            strokes.append([L_mid, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR'], R_mid])
            strokes.append([R_mid, g['MR'], g['MC']])
        elif letter == 'p':
            strokes.append([g['ML'], g['DL']])
            strokes.append([g['ML'], g['MR'], R_mid])
            strokes.append([R_mid, g['BR'], g['BC']])
            strokes.append([g['BC'], g['BL']])
        elif letter == 'q':
            strokes.append([g['MR'], g['DR']])
            strokes.append([g['MR'], g['ML'], L_mid])
            strokes.append([L_mid, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR']])
        elif letter == 'r':
            r_start = g['ML'] + np.array([0, -height*0.25, 0])
            strokes.append([g['ML'], g['BL']])
            strokes.append([r_start, g['ML'], g['MR']])
        elif letter == 's':
            strokes.append([g['MR'], g['ML'], center_s])
            strokes.append([center_s, g['BR'], g['BL']])
        elif letter == 't':
            t_top = (g['TL'] + g['TC']) / 2
            t_center_x = t_top[0]
            t_bot_turn = np.array([t_center_x, y_bottom + height*0.15, Z_PAPER])
            strokes.append([t_top, t_bot_turn])
            strokes.append([t_bot_turn, np.array([t_center_x, y_bottom, Z_PAPER]), g['BC']])
            bar_left = np.array([t_center_x - width*0.4, y_top_x, Z_PAPER])
            bar_right = np.array([t_center_x + width*0.4, y_top_x, Z_PAPER])
            strokes.append([bar_left, bar_right])
        elif letter == 'u':
            u_turn_l = g['BL'] + np.array([0, height*0.15, 0])
            u_turn_r = g['BR'] + np.array([0, height*0.15, 0])
            strokes.append([g['ML'], u_turn_l])
            strokes.append([u_turn_l, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR'], u_turn_r])
            strokes.append([u_turn_r, g['MR']])
            strokes.append([g['MR'], g['BR']])
        elif letter == 'v':
            strokes.append([g['ML'], g['BC']])
            strokes.append([g['BC'], g['MR']])
        elif letter == 'w':
            w_left = g['ML'] + np.array([-width*0.2, 0, 0])
            w_right = g['MR'] + np.array([width*0.2, 0, 0])
            w_b_left = g['BL'] + np.array([-width*0.1, 0, 0])
            w_b_right = g['BR'] + np.array([width*0.1, 0, 0])
            strokes.append([w_left, w_b_left])
            strokes.append([w_b_left, g['MC']])
            strokes.append([g['MC'], w_b_right])
            strokes.append([w_b_right, w_right])
        elif letter == 'x':
            strokes.append([g['ML'], g['BR']])
            strokes.append([g['MR'], g['BL']])
        elif letter == 'y':
            strokes.append([g['ML'], g['BC']])          
            strokes.append([g['MR'], g['DL']])          
        elif letter == 'z':
            strokes.append([g['ML'], g['MR']])
            strokes.append([g['MR'], g['BL']])
            strokes.append([g['BL'], g['BR']])
    return strokes

def get_punctuation_strokes(punct, base_pos, width=CHAR_WIDTH, height=CHAR_HEIGHT):
    base = np.array(base_pos)
    cx, cy, z = base
    dot_center = np.array([cx, cy + height*0.1, z]) 
    dot_top    = dot_center + np.array([0, 0.001, 0])
    dot_bot    = dot_center + np.array([0, -0.001, 0])
    strokes = []
    if punct == '.' or punct == '。': 
        strokes.append([dot_top, dot_bot])
    elif punct == ',' or punct == '，': 
        strokes.append([dot_top, dot_bot])
        tail_end = dot_bot + np.array([-0.002, -0.005, 0])
        strokes.append([dot_bot, tail_end])
    return strokes

def generate_string_strokes(text, start_x, start_y, char_width=CHAR_WIDTH, 
                           char_height=CHAR_HEIGHT, spacing=CHAR_SPACING):
    all_strokes = []
    non_space_chars = [c for c in text if c != ' ']
    total_width = len(non_space_chars) * (char_width + spacing) - spacing
    start_x_pos = start_x - total_width / 2
    char_index = 0
    
    for i, char in enumerate(text):
        if char == ' ':
            char_index += 1
            continue
        
        char_x = start_x_pos + char_index * (char_width + spacing) + char_width/2
        char_y_base = start_y 
        base_pos = np.array([char_x, char_y_base, Z_PAPER])
        
        if char in '0123456789':
            char_strokes = get_digit_strokes(char, base_pos, char_width, char_height)
        elif char in ',.，。':
            char_strokes = get_punctuation_strokes(char, base_pos, char_width, char_height)
        elif char.isalpha():
            char_strokes = get_letter_strokes(char, base_pos, char_width, char_height)
        else:
            print(f"跳过不支持字符: {char}")
            char_index += 1
            continue
        
        all_strokes.extend(char_strokes)
        char_index += 1
    
    return all_strokes

# ================= 测试输入 =================
print("请输入要写入的文本 (支持数字/大小写英文/标点):")
test_string = input()

# 调整起始位置
start_x = 0.0  
start_y = 0.30 

ALL_STROKES_RAW = []
string_strokes = generate_string_strokes(test_string, start_x, start_y)
ALL_STROKES_RAW.extend(string_strokes)

# ================= 镜像处理 =================
ALL_STROKES = []
for stroke in ALL_STROKES_RAW:
    mirrored_stroke = []
    for point in stroke:
        new_point = point.copy()
        new_point[0] = -new_point[0] 
        mirrored_stroke.append(new_point)
    ALL_STROKES.append(mirrored_stroke)

# ================= 2. 优化：MinJerk 插值核心 =================

class TrajectorySegment:
    def __init__(self, target_pos, duration, method='minjerk', ink=False, control_point=None):
        self.target_pos = np.array(target_pos)
        self.duration = duration
        self.method = method 
        self.ink = ink
        self.control_point = control_point

# 最小加加速度时间缩放函数
def get_minjerk_u(t, t_total):
    if t_total <= 0: return 1.0
    u_raw = np.clip(t / t_total, 0, 1)
    return 10 * (u_raw**3) - 15 * (u_raw**4) + 6 * (u_raw**5)

def Interpolate(start_pos, end_pos, t, t_total, method='minjerk', control_point=None):
    u = get_minjerk_u(t, t_total)
    
    if method == 'linear' or method == 'minjerk':
        return start_pos + (end_pos - start_pos) * u
    elif method == 'bezier':
        p0, p1, p2 = start_pos, control_point, end_pos
        return (1-u)**2 * p0 + 2*u*(1-u) * p1 + u**2 * p2
    return start_pos

def add_smart_stroke_logic(segments, stroke_data):
    start_pos_paper = stroke_data[0]
    
    if len(stroke_data) == 3:
        control_pos_paper = stroke_data[1]
        end_pos_paper = stroke_data[2]
        is_curve = True
    else:
        end_pos_paper = stroke_data[-1] 
        is_curve = False
        
    dist = np.linalg.norm(start_pos_paper - end_pos_paper)
    
    write_duration = 0.15 + (dist * 3.0)
    if dist < 0.01: 
        write_duration += 0.05

    start_pos_air = start_pos_paper.copy(); start_pos_air[2] = Z_AIR
    end_pos_air = end_pos_paper.copy(); end_pos_air[2] = Z_AIR

    # 【优化】全流程 MinJerk，动作更丝滑
    segments.append(TrajectorySegment(start_pos_air, duration=0.3, method='minjerk', ink=False))
    segments.append(TrajectorySegment(start_pos_paper, duration=0.2, method='minjerk', ink=False))
    segments.append(TrajectorySegment(start_pos_paper, duration=0.05, method='minjerk', ink=False)) # Dwell

    if is_curve:
        segments.append(TrajectorySegment(end_pos_paper, duration=write_duration, method='bezier', ink=True, control_point=control_pos_paper))
    else:
        segments.append(TrajectorySegment(end_pos_paper, duration=write_duration, method='minjerk', ink=True))

    segments.append(TrajectorySegment(end_pos_paper, duration=0.05, method='minjerk', ink=False)) # Dwell
    segments.append(TrajectorySegment(end_pos_air, duration=0.2, method='minjerk', ink=False))

# --- 3. 优化：零空间姿态 IK 控制器 ---
def IK_controller_robust(model, data, X_ref, q_pos):
    position_Q = data.site_xpos[0]
    
    # 1. 计算雅可比矩阵
    jacp = np.zeros((3, 6))
    mj.mj_jac(model, data, jacp, None, position_Q, 7)
    J = jacp.copy()
    
    # 2. 主要任务：到达目标点 (使用阻尼最小二乘法)
    X = position_Q.copy()
    dX = X_ref - X
    lambda_val = 0.1 
    J_T = J.T
    J_pinv = J_T @ np.linalg.inv(J @ J_T + np.eye(3) * lambda_val**2)
    
    dq_primary = J_pinv @ (dX * 15.0) 
    
    # 3. 次要任务：零空间姿态优化 (保持 "端手" 姿势)
    # 投影矩阵 P = I - J_pinv * J
    I = np.eye(6)
    null_space_projector = I - (J_pinv @ J)
    q_error = Q_REST - q_pos
    
    # 在零空间施加姿态恢复力矩
    dq_secondary = null_space_projector @ (q_error * 2.0)
    
    return q_pos + dq_primary + dq_secondary

def controller(model, data):
    pass

# --- 交互 ---
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    if (not button_left) and (not button_middle) and (not button_right): return
    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# --- 初始化 ---
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)

model = mj.MjModel.from_xml_path(abspath)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1280, 720, "RoboWriter Auto Planar Final", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 90
cam.elevation = -45
cam.distance = 1.8
cam.lookat = np.array([0.0, 0.35, 0.0])

mj.set_mjcb_control(controller)

init_qpos = np.array([-1.57, -2.0, 2.0, -1.57, -1.57, 0.0])
data.qpos[:6] = init_qpos
mj.mj_forward(model, data)

traj_points = []
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0]) 

# --- 4. 生成轨迹 ---
segments = []
for stroke in ALL_STROKES:
    add_smart_stroke_logic(segments, stroke)

print(f"Total segments: {len(segments)}")

# --- 主循环 ---
current_seg_idx = 0
seg_start_time = 0.0
current_start_pos = data.site_xpos[0].copy()
is_homing = False
homing_start_time = 0.0
homing_start_q = np.zeros(6)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        
        if current_seg_idx >= len(segments):
            if not is_homing:
                print("Writing finished! Homing...")
                is_homing = True
                homing_start_time = data.time
                homing_start_q = data.qpos.copy()
            
            t_homing = data.time - homing_start_time
            if t_homing < 5.0:
                alpha = t_homing / 5.0
                target_q = (1 - alpha) * homing_start_q + alpha * HOME_Q
                # 使用位置控制归位 (最稳)
                data.ctrl[:] = -20.0 * (data.qpos - target_q) - 1.0 * data.qvel
            else:
                data.ctrl[:] = -30.0 * (data.qpos - HOME_Q) - 2.0 * data.qvel
            mj.mj_step(model, data)
            continue

        seg = segments[current_seg_idx]
        t_local = data.time - seg_start_time
        
        if t_local >= seg.duration:
            current_start_pos = seg.target_pos.copy()
            seg_start_time = data.time
            current_seg_idx += 1
            continue
            
        # 1. 计算 MinJerk 插值目标点
        X_ref = Interpolate(
            current_start_pos, 
            seg.target_pos, 
            t_local, 
            seg.duration, 
            method=seg.method, 
            control_point=seg.control_point
        )
            
        cur_q_pos = data.qpos.copy()
        
        # 2. 使用增强版 IK (带姿态控制)
        cur_ctrl = IK_controller_robust(model, data, X_ref, cur_q_pos)
        data.ctrl[:] = cur_ctrl
        
        mj.mj_step(model, data)
        
        if seg.ink:  
            mj_end_eff_pos = data.site_xpos[0]
            traj_points.append(mj_end_eff_pos.copy())
            
        if len(traj_points) > 50000:
            traj_points.pop(0)

    if (data.time >= simend):
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    for j in range(1, len(traj_points)):
        if scene.ngeom >= scene.maxgeom: break
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.002, 0.002, 0.002])
        geom.pos[:] = traj_points[j]
        geom.mat[:] = np.eye(3)
        geom.objtype = 0; geom.objid = 0; geom.dataid = -1; geom.segid = -1
        
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()