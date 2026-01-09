###
# Final project with MuJoCo
# SI100B Robotics Programming
# Sphere Writing Task: Universal Character Support + Final Homing
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

# ================= 配置区域 =================
xml_path = '../../models/universal_robots_ur5e/scene.xml' 
simend = 180 

# 球体参数
SPHERE_CENTER = np.array([0.0, 0.35, 1.3])
SPHERE_RADIUS = 1.3

# 题目要求的结束位姿
HOME_Q = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])

# 书写中心 (投影中心)
WRITE_CENTER_XY = np.array([0.0, 0.35])

# 笔画参数
CHAR_SCALE = 1.5       
CHAR_SPACING = 0.02 * CHAR_SCALE
Z_SAFETY_OFFSET = 0.02 # 笔尖离地高度
Z_AIR = 0.25           # 抬笔高度

# 原始字库尺寸
BASE_WIDTH = 0.012
BASE_HEIGHT = 0.025

# ================= 1. 字库核心 (保持不变) =================

def get_grid(base_pos, width, height):
    """ 生成字符网格 (2D版本) """
    cx, cy = base_pos 
    y_base, y_top = cy, cy + height
    y_mid = cy + height * 0.55
    y_desc = cy - height * 0.3
    x_l, x_c, x_r = cx - width/2, cx, cx + width/2
    
    return {
        'TL': np.array([x_l, y_top]), 'TC': np.array([x_c, y_top]), 'TR': np.array([x_r, y_top]),
        'ML': np.array([x_l, y_mid]), 'MC': np.array([x_c, y_mid]), 'MR': np.array([x_r, y_mid]),
        'BL': np.array([x_l, y_base]), 'BC': np.array([x_c, y_base]), 'BR': np.array([x_r, y_base]),
        'DL': np.array([x_l, y_desc]), 'DC': np.array([x_c, y_desc]), 'DR': np.array([x_r, y_desc]),
    }

def get_digit_strokes(digit, base_pos, width, height):
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

def get_letter_strokes(letter, base_pos, width, height):
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
            hook_ctrl = g['BC'] + np.array([0, -height*0.2])
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
        center_s = np.array([g['MC'][0], y_center_box])
        L_mid = np.array([g['ML'][0], y_center_box])
        R_mid = np.array([g['MR'][0], y_center_box])
        y_cross = (y_top_x + y_bottom) / 2
        cross_l = np.array([g['ML'][0], y_cross])
        cross_r = np.array([g['MR'][0], y_cross])
        
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
            dot_center = g['ML'] + np.array([0, height * 0.4])
            dot_start = dot_center + np.array([0, height * 0.02])
            dot_end   = dot_center - np.array([0, height * 0.02])
            strokes.append([dot_start, dot_end])
        elif letter == 'j':
            strokes.append([g['MC'], g['DC'], g['DL']]) 
            dot_center = g['MC'] + np.array([0, height * 0.4])
            dot_start = dot_center + np.array([0, height * 0.02])
            dot_end   = dot_center - np.array([0, height * 0.02])
            strokes.append([dot_start, dot_end])
        elif letter == 'k':
            strokes.append([g['TL'], g['BL']]) 
            y_knot = (g['ML'][1] + g['BL'][1]) / 2
            k_knot = np.array([g['ML'][0], y_knot])
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
            r_start = g['ML'] + np.array([0, -height*0.25])
            strokes.append([g['ML'], g['BL']])
            strokes.append([r_start, g['ML'], g['MR']])
        elif letter == 's':
            strokes.append([g['MR'], g['ML'], center_s])
            strokes.append([center_s, g['BR'], g['BL']])
        elif letter == 't':
            t_top = (g['TL'] + g['TC']) / 2
            t_center_x = t_top[0]
            t_bot_turn = np.array([t_center_x, y_bottom + height*0.15])
            strokes.append([t_top, t_bot_turn])
            strokes.append([t_bot_turn, np.array([t_center_x, y_bottom]), g['BC']])
            bar_left = np.array([t_center_x - width*0.4, y_top_x])
            bar_right = np.array([t_center_x + width*0.4, y_top_x])
            strokes.append([bar_left, bar_right])
        elif letter == 'u':
            u_turn_l = g['BL'] + np.array([0, height*0.15])
            u_turn_r = g['BR'] + np.array([0, height*0.15])
            strokes.append([g['ML'], u_turn_l])
            strokes.append([u_turn_l, g['BL'], g['BC']])
            strokes.append([g['BC'], g['BR'], u_turn_r])
            strokes.append([u_turn_r, g['MR']])
            strokes.append([g['MR'], g['BR']])
        elif letter == 'v':
            strokes.append([g['ML'], g['BC']])
            strokes.append([g['BC'], g['MR']])
        elif letter == 'w':
            w_left = g['ML'] + np.array([-width*0.2, 0])
            w_right = g['MR'] + np.array([width*0.2, 0])
            w_b_left = g['BL'] + np.array([-width*0.1, 0])
            w_b_right = g['BR'] + np.array([width*0.1, 0])
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

def get_punctuation_strokes(punct, base_pos, width, height):
    base = np.array(base_pos)
    cx, cy = base
    dot_center = np.array([cx, cy + height*0.1]) 
    dot_top    = dot_center + np.array([0, 0.001])
    dot_bot    = dot_center + np.array([0, -0.001])
    strokes = []
    if punct == '.' or punct == '。': 
        strokes.append([dot_top, dot_bot])
    elif punct == ',' or punct == '，': 
        strokes.append([dot_top, dot_bot])
        tail_end = dot_bot + np.array([-0.002, -0.005])
        strokes.append([dot_bot, tail_end])
    return strokes

# ================= 2. 字符串排版生成器 =================

def generate_string_strokes_flat(text):
    all_strokes = []
    width_char = BASE_WIDTH * CHAR_SCALE
    height_char = BASE_HEIGHT * CHAR_SCALE
    spacing = CHAR_SPACING
    
    content_len = len(text)
    total_width = content_len * width_char + (content_len - 1) * spacing
    
    start_x_edge = WRITE_CENTER_XY[0] - total_width / 2
    current_x = start_x_edge + width_char / 2
    base_y = WRITE_CENTER_XY[1]
    
    for char in text:
        if char == ' ':
            current_x += (width_char + spacing)
            continue
            
        base_pos = np.array([current_x, base_y])
        strokes = []
        if char.isdigit():
            strokes = get_digit_strokes(char, base_pos, width_char, height_char)
        elif 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            strokes = get_letter_strokes(char, base_pos, width_char, height_char)
        elif char in '.,':
            strokes = get_punctuation_strokes(char, base_pos, width_char, height_char)
            
        all_strokes.extend(strokes)
        current_x += (width_char + spacing)
        
    return all_strokes

# ================= 3. 核心：球面投影 =================

def project_to_sphere(x, y):
    cx, cy, cz = SPHERE_CENTER
    R = SPHERE_RADIUS
    r_sq = (x - cx)**2 + (y - cy)**2
    if r_sq > R**2: r_sq = R**2 
    z = cz - np.sqrt(R**2 - r_sq) + Z_SAFETY_OFFSET
    return np.array([x, y, z])

def generate_trajectory(text_to_write):
    flat_strokes = generate_string_strokes_flat(text_to_write)
    trajectory = []
    if not flat_strokes: return []
    
    p0_flat = flat_strokes[0][0]
    p0_surf = project_to_sphere(p0_flat[0], p0_flat[1])
    trajectory.append((np.array([p0_surf[0], p0_surf[1], Z_AIR]), False))
    
    for stroke in flat_strokes:
        # Move
        start_2d = stroke[0]
        start_surf = project_to_sphere(start_2d[0], start_2d[1])
        start_air = start_surf.copy(); start_air[2] = Z_AIR
        trajectory.append((start_air, False))
        
        # Drop
        for i in range(5):
            t = (i+1)/5
            pos = (1-t)*start_air + t*start_surf
            trajectory.append((pos, False))
            
        # Write
        for i in range(len(stroke)-1):
            p1 = stroke[i]
            p2 = stroke[i+1]
            dist = np.linalg.norm(np.array(p2)-np.array(p1))
            steps = max(5, int(dist/0.0005)) 
            for k in range(steps):
                t = (k+1)/steps
                curr_2d = (1-t)*np.array(p1) + t*np.array(p2)
                curr_surf = project_to_sphere(curr_2d[0], curr_2d[1])
                trajectory.append((curr_surf, True)) # Ink ON
                
        # Lift
        end_2d = stroke[-1]
        end_surf = project_to_sphere(end_2d[0], end_2d[1])
        end_air = end_surf.copy(); end_air[2] = Z_AIR
        for i in range(5):
            t = (i+1)/5
            pos = (1-t)*end_surf + t*end_air
            trajectory.append((pos, False))
            
    return trajectory

# ================= MuJoCo 样板代码 =================

def IK_controller(model, data, X_ref, q_pos):
    position_Q = data.site_xpos[0]
    jacp = np.zeros((3, 6)); mj.mj_jac(model, data, jacp, None, position_Q, 7)
    J = jacp.copy()
    lam = 0.1
    J_inv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
    dX = X_ref - position_Q
    if np.linalg.norm(dX) > 0.05: dX = dX / np.linalg.norm(dX) * 0.05
    return q_pos + J_inv @ (dX * 6.0)

def controller(model, data): pass
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data); mj.mj_forward(model, data)

button_left = False; button_middle = False; button_right = False; lastx = 0; lasty = 0
def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)
def mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx = xpos - lastx; dy = ypos - lasty; lastx = xpos; lasty = ypos
    if not (button_left or button_middle or button_right): return
    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
    if button_right: action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left: action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else: action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)
def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

# Init
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
model = mj.MjModel.from_xml_path(abspath)
data = mj.MjData(model)
cam = mj.MjvCamera(); opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1280, 720, "Sphere Writer: Generic Text + Homing", None, None)
glfw.make_context_current(window); glfw.swap_interval(1)
mj.mjv_defaultCamera(cam); mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 135; cam.elevation = -30; cam.distance = 2.0
cam.lookat = np.array([0.0, 0.35, 0.3])
mj.set_mjcb_control(controller)

init_qpos = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
data.qpos[:] = init_qpos; mj.mj_forward(model, data)

# ================= 用户输入 =================
print("请输入要在球面上书写的文本 (支持 A-Z, a-z, 0-9, . ,):")
input_text = input() 
if not input_text: input_text = "Hello Sphere"

# 生成轨迹
full_traj_data = generate_trajectory(input_text)
target_points = [x[0] for x in full_traj_data]
ink_flags = [x[1] for x in full_traj_data]

traj_vis = []
# Homing 相关变量
is_homing = False
homing_start_time = 0.0
homing_start_q = np.zeros(6)

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        # 0.015s per step
        idx = int(data.time / 0.015)
        
        if idx < len(target_points):
            # === 1. 写字阶段 ===
            target = target_points[idx]
            is_ink = ink_flags[idx]
            if is_ink:
                if len(traj_vis)==0 or np.linalg.norm(target-traj_vis[-1])>0.005:
                    traj_vis.append(target.copy())
                    if len(traj_vis)>10000: traj_vis.pop(0)
            data.ctrl[:] = IK_controller(model, data, target, data.qpos.copy())
            
        else:
            # === 2. 归位阶段 (Homing) ===
            if not is_homing:
                print("Writing finished! Homing to final pose...")
                is_homing = True
                homing_start_time = data.time
                homing_start_q = data.qpos.copy()
            
            # 在 5 秒内平滑归位
            t_homing = data.time - homing_start_time
            if t_homing < 5.0:
                alpha = t_homing / 5.0
                target_q = (1 - alpha) * homing_start_q + alpha * HOME_Q
            else:
                target_q = HOME_Q
                
            data.ctrl[:] = target_q

        mj.mj_step(model, data)
        if data.time >= simend: break

    viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # 绘制墨水
    for p in traj_vis:
        if scene.ngeom >= scene.maxgeom: break
        g = scene.geoms[scene.ngeom]; scene.ngeom += 1
        g.type = mj.mjtGeom.mjGEOM_SPHERE
        g.rgba[:] = [1, 0, 0, 1]; g.size[:] = [0.003]*3; g.pos[:] = p; g.mat[:] = np.eye(3)
        g.objtype = 0; g.objid = 0; g.dataid = -1; g.segid = -1
    
    # 绘制球体参考面
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]; scene.ngeom += 1
        g.type = mj.mjtGeom.mjGEOM_SPHERE
        g.rgba[:] = [0, 0, 1, 0.15]; g.size[:] = [SPHERE_RADIUS]*3; g.pos[:] = SPHERE_CENTER
        g.mat[:] = np.eye(3); g.objtype = 0; g.objid = 0; g.dataid = -1; g.segid = -1

    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window); glfw.poll_events()

glfw.terminate()