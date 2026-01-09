###
# Final project with MuJoCo
# SI100B Robotics Programming
# Sphere Writing Task: "李昂 Li Ang 2025531120" (Clean Version + Final Homing)
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

# 【新增】题目要求的结束位姿
HOME_Q = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])

# 书写中心
WRITE_CENTER_XY = np.array([0.0, 0.35])

# 安全参数
Z_SAFETY_OFFSET = 0.02 
Z_AIR = 0.25

# ================= 1. 笔画生成 (2D) =================

def get_li_strokes_flat(scale=1.0, center_offset=np.array([0,0])):
    """ 李 """
    strokes = []
    def add(points):
        strokes.append([np.array(p) * scale + center_offset for p in points])

    # 木
    add([[-0.10, 0.10], [0.10, 0.10]])
    add([[0.00, 0.15], [0.00, 0.00]])
    add([[0.00, 0.10], [-0.08, 0.02]])
    add([[0.00, 0.10], [0.08, 0.02]])
    # 子
    add([[-0.06, -0.05], [0.06, -0.05], [0.00, -0.10]])
    curve = []
    for t in np.linspace(0, 1, 10):
        p = (1-t)**2 * np.array([0.00, -0.10]) + 2*(1-t)*t * np.array([0.02, -0.18]) + t**2 * np.array([-0.02, -0.15])
        curve.append(p)
    add(curve)
    add([[-0.10, -0.10], [0.10, -0.10]])
    return strokes

def get_ang_strokes_flat(scale=1.0, center_offset=np.array([0,0])):
    """ 昂 """
    strokes = []
    def add(points):
        strokes.append([np.array(p) * scale + center_offset for p in points])

    # 日
    add([[-0.06, 0.14], [-0.06, 0.06]])
    add([[-0.06, 0.14], [0.06, 0.14]])
    add([[0.06, 0.14], [0.06, 0.06]])
    add([[-0.06, 0.10], [0.06, 0.10]])
    add([[-0.06, 0.06], [0.06, 0.06]])
    # 卬
    add([[-0.03, 0.04], [-0.06, 0.01], [-0.09, -0.01]])
    add([[-0.06, 0.01], [-0.06, -0.10], [-0.02, -0.06]])
    add([[0.00, 0.02], [0.06, 0.02]])
    add([[0.06, 0.02], [0.06, -0.08], [0.03, -0.06]])
    add([[0.00, 0.04], [0.00, -0.15]])
    return strokes

def get_eng_strokes_flat(text, scale=1.0, start_offset=np.array([0,0])):
    """ Li Ang """
    strokes = []
    cursor = start_offset.copy()
    spacing = 0.02 * scale 
    
    def add_char(char_strokes, width):
        nonlocal cursor
        for s in char_strokes:
            strokes.append([np.array(p) * scale + cursor for p in s])
        cursor[0] += (width * scale + spacing)

    for char in text:
        if char == 'L':
            add_char([[[0, 0.1], [0, 0], [0.06, 0]]], 0.06)
        elif char == 'i':
            add_char([[[0, 0.07], [0, 0]], [[0, 0.09], [0, 0.1]]], 0.02)
        elif char == 'A':
            add_char([[[-0.04, 0], [0, 0.1]], [[0, 0.1], [0.04, 0]], [[-0.02, 0.04], [0.02, 0.04]]], 0.08)
        elif char == 'n':
            c = []
            for t in np.linspace(0, 1, 8):
                c.append((1-t)**2 * np.array([0, 0.05]) + 2*(1-t)*t * np.array([0.05, 0.08]) + t**2 * np.array([0.05, 0]))
            add_char([[[0, 0.07], [0, 0]], c, [[0.05, 0.05], [0.05, 0]]], 0.06)
        elif char == 'g':
            c_top = [np.array([0.025 + 0.025*np.cos(t), 0.025 + 0.025*np.sin(t)]) for t in np.linspace(0, 2*np.pi, 16)]
            add_char([c_top, [[0.05, 0.05], [0.05, -0.05], [0.0, -0.05]]], 0.06)
        elif char == ' ':
            cursor[0] += (0.04 * scale)

    return strokes

def get_num_strokes_flat(text, scale=1.0, start_offset=np.array([0,0])):
    """ 2025531120 """
    strokes = []
    cursor = start_offset.copy()
    w = 0.025; h = 0.05; spacing = 0.015 * scale

    def add(points):
        strokes.append([np.array(p) * scale + cursor for p in points])

    for char in text:
        if char == '0':
            add([[-w, h], [w, h], [w, -h], [-w, -h], [-w, h]])
        elif char == '1':
            add([[0, h], [0, -h]])
        elif char == '2':
            add([[-w, h], [w, h], [w, 0], [-w, 0], [-w, -h], [w, -h]])
        elif char == '3':
            add([[-w, h], [w, h], [w, 0], [-w, 0]])
            add([[w, 0], [w, -h], [-w, -h]])
        elif char == '5':
            add([[w, h], [-w, h], [-w, 0], [w, 0], [w, -h], [-w, -h]])
        cursor[0] += (2 * w * scale + spacing)
    
    return strokes

# ================= 2. 核心：球面投影 =================

def project_to_sphere(x, y):
    cx, cy, cz = SPHERE_CENTER
    R = SPHERE_RADIUS
    r_sq = (x - cx)**2 + (y - cy)**2
    if r_sq > R**2: r_sq = R**2 
    z = cz - np.sqrt(R**2 - r_sq) + Z_SAFETY_OFFSET
    return np.array([x, y, z])

# ================= 3. 轨迹生成 (带墨水标记) =================

def generate_full_trajectory():
    # 返回列表结构: [(position, is_ink), ...]
    trajectory = []
    
    # 布局
    pos_li = WRITE_CENTER_XY + np.array([-0.10, 0.12])
    pos_ang = WRITE_CENTER_XY + np.array([0.10, 0.12])
    pos_eng_start = WRITE_CENTER_XY + np.array([-0.09, -0.02]) 
    pos_num_start = WRITE_CENTER_XY + np.array([-0.11, -0.15])

    # 生成所有笔画
    s1 = get_li_strokes_flat(scale=0.5, center_offset=pos_li)
    s2 = get_ang_strokes_flat(scale=0.5, center_offset=pos_ang)
    s3 = get_eng_strokes_flat("Li Ang", scale=0.5, start_offset=pos_eng_start)
    s4 = get_num_strokes_flat("2025531120", scale=0.4, start_offset=pos_num_start)
    all_strokes = s1 + s2 + s3 + s4
    
    # 初始抬笔
    p0 = project_to_sphere(*all_strokes[0][0])
    # False 表示此时不应该画画 (在空中)
    trajectory.append((np.array([p0[0], p0[1], Z_AIR]), False))
    
    for stroke in all_strokes:
        # 1. 移动到起点上方 (Air Move) - 关墨水
        start_surf = project_to_sphere(*stroke[0])
        start_air = start_surf.copy(); start_air[2] = Z_AIR
        trajectory.append((start_air, False))
        
        # 2. 下笔 (Drop) - 关墨水 (直到接触纸面)
        for i in range(8): 
            pos = (1-(i+1)/8)*start_air + ((i+1)/8)*start_surf
            trajectory.append((pos, False))
            
        # 3. 书写 (Write) - 开墨水!
        for i in range(len(stroke)-1):
            p1, p2 = stroke[i], stroke[i+1]
            dist = np.linalg.norm(np.array(p2)-np.array(p1))
            steps = max(3, int(dist/0.002))
            for k in range(steps):
                t = (k+1)/steps
                curr_2d = (1-t)*np.array(p1) + t*np.array(p2)
                # 只有这里标记为 True
                trajectory.append((project_to_sphere(*curr_2d), True))
                
        # 4. 抬笔 (Lift) - 关墨水
        end_surf = project_to_sphere(*stroke[-1])
        end_air = end_surf.copy(); end_air[2] = Z_AIR
        for i in range(8): 
            pos = (1-(i+1)/8)*end_surf + ((i+1)/8)*end_air
            trajectory.append((pos, False))
            
    return trajectory

# ================= MuJoCo Boilerplate =================

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
window = glfw.create_window(1280, 720, "Sphere: Li Ang 2025531120 + Homing", None, None)
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

# 生成轨迹 (包含墨水信息)
full_traj_data = generate_full_trajectory()
# 分离出 目标位置 和 墨水标记
full_traj_points = [item[0] for item in full_traj_data]
full_traj_ink    = [item[1] for item in full_traj_data]

traj_vis = []

# 【新增】归位相关变量
is_homing = False
homing_start_time = 0.0
homing_start_q = np.zeros(6)

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        idx = int(data.time / 0.01)
        
        # 1. 轨迹跟随阶段
        if idx < len(full_traj_points):
            target = full_traj_points[idx]
            is_ink = full_traj_ink[idx]
            
            # 记录轨迹
            if is_ink:
                if len(traj_vis)==0 or np.linalg.norm(target-traj_vis[-1])>0.005:
                    traj_vis.append(target.copy())
                    if len(traj_vis)>10000: traj_vis.pop(0)
            
            data.ctrl[:] = IK_controller(model, data, target, data.qpos.copy())
            
        # 2. 轨迹结束后 -> 归位阶段
        else:
            if not is_homing:
                print("Writing finished! Homing to final pose...")
                is_homing = True
                homing_start_time = data.time
                homing_start_q = data.qpos.copy()
            
            # 5秒平滑插值回到 HOME_Q
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
    
    # 绘制红色字迹 (Ink)
    for p in traj_vis:
        if scene.ngeom >= scene.maxgeom: break
        g = scene.geoms[scene.ngeom]; scene.ngeom += 1
        g.type = mj.mjtGeom.mjGEOM_SPHERE
        g.rgba[:] = [1, 0, 0, 1]; g.size[:] = [0.003]*3; g.pos[:] = p; g.mat[:] = np.eye(3)
        g.objtype = 0; g.objid = 0; g.dataid = -1; g.segid = -1
        
    # 绘制透明球体 (Ghost Sphere)
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]; scene.ngeom += 1
        g.type = mj.mjtGeom.mjGEOM_SPHERE
        g.rgba[:] = [0, 0, 1, 0.15]; g.size[:] = [1.3]*3; g.pos[:] = SPHERE_CENTER
        g.mat[:] = np.eye(3); g.objtype = 0; g.objid = 0; g.dataid = -1; g.segid = -1

    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window); glfw.poll_events()

glfw.terminate()