###
# Final project with MuJoCo
# SI100B Robotics Programming
# RoboWriter Final: MinJerk Speed + Null-Space Posture Control + Pretty Digits
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# ================= 配置区域 =================
xml_path = '../../models/universal_robots_ur5e/scene.xml' 
simend = 180 

# 定义高度常量
Z_PAPER = 0.10  # 纸面
Z_AIR   = 0.15  # 抬笔

# 结束位姿
HOME_Q = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0]) 

# 【关键】定义“舒适书写姿态” (Rest Pose)
# 这会让机械臂保持“端着手”的姿势：
# Base=-90, Shoulder=-90(抬平), Elbow=90(弯曲), Wrist1=-90(垂直向下), Wrist2=-90, Wrist3=0
Q_REST = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# ================= 1. 优化的字库定义 =================

# --- 1.1 中文 & 英文名字 (保持不变) ---
C_LI = np.array([-0.12, 0.50, 0.0]) 
STROKES_LI_3D = [
    [C_LI + np.array([-0.05, 0.05, Z_PAPER]), C_LI + np.array([0.05, 0.05, Z_PAPER])], 
    [C_LI + np.array([0.00, 0.07, Z_PAPER]), C_LI + np.array([0.00, 0.00, Z_PAPER])],  
    [C_LI + np.array([0.00, 0.05, Z_PAPER]), C_LI + np.array([-0.02, 0.03, Z_PAPER]), C_LI + np.array([-0.06, 0.01, Z_PAPER])], 
    [C_LI + np.array([0.00, 0.05, Z_PAPER]), C_LI + np.array([0.02, 0.03, Z_PAPER]), C_LI + np.array([0.06, 0.01, Z_PAPER])],   
    [C_LI + np.array([-0.03, -0.01, Z_PAPER]), C_LI + np.array([0.04, -0.01, Z_PAPER]), C_LI + np.array([-0.01, -0.04, Z_PAPER])], 
    [C_LI + np.array([-0.01, -0.04, Z_PAPER]), C_LI + np.array([0.01, -0.06, Z_PAPER]), C_LI + np.array([-0.01, -0.08, Z_PAPER])], 
    [C_LI + np.array([-0.05, -0.05, Z_PAPER]), C_LI + np.array([0.05, -0.05, Z_PAPER])],
]

C_ANG = np.array([0.12, 0.50, 0.0])
STROKES_ANG_3D = [
    [C_ANG + np.array([-0.04, 0.08, Z_PAPER]), C_ANG + np.array([-0.04, 0.04, Z_PAPER])],
    [C_ANG + np.array([-0.04, 0.08, Z_PAPER]), C_ANG + np.array([0.04, 0.08, Z_PAPER])],
    [C_ANG + np.array([0.04, 0.08, Z_PAPER]), C_ANG + np.array([0.04, 0.04, Z_PAPER])],
    [C_ANG + np.array([-0.04, 0.06, Z_PAPER]), C_ANG + np.array([0.04, 0.06, Z_PAPER])],
    [C_ANG + np.array([-0.04, 0.04, Z_PAPER]), C_ANG + np.array([0.04, 0.04, Z_PAPER])],
    [C_ANG + np.array([-0.02, 0.03, Z_PAPER]), C_ANG + np.array([-0.04, 0.01, Z_PAPER]), C_ANG + np.array([-0.06, 0.00, Z_PAPER])],
    [C_ANG + np.array([-0.04, 0.01, Z_PAPER]), C_ANG + np.array([-0.04, -0.06, Z_PAPER]), C_ANG + np.array([-0.01, -0.03, Z_PAPER])],
    [C_ANG + np.array([0.00, 0.02, Z_PAPER]), C_ANG + np.array([0.04, 0.02, Z_PAPER])],
    [C_ANG + np.array([0.04, 0.02, Z_PAPER]), C_ANG + np.array([0.04, -0.05, Z_PAPER]), C_ANG + np.array([0.02, -0.04, Z_PAPER])],
    [C_ANG + np.array([0.00, 0.03, Z_PAPER]), C_ANG + np.array([0.00, -0.10, Z_PAPER])],
]

C_ENG = np.array([0.0, 0.35, 0.0])
STROKES_LiA = [
    [C_ENG + np.array([-0.09, 0.05, Z_PAPER]), C_ENG + np.array([-0.09, 0.00, Z_PAPER])],
    [C_ENG + np.array([-0.09, 0.00, Z_PAPER]), C_ENG + np.array([-0.07, 0.00, Z_PAPER])],
    [C_ENG + np.array([-0.05, 0.03, Z_PAPER]), C_ENG + np.array([-0.05, 0.00, Z_PAPER])],
    [C_ENG + np.array([-0.05, 0.05, Z_PAPER]), C_ENG + np.array([-0.05, 0.045, Z_PAPER])],
    [C_ENG + np.array([-0.01, 0.05, Z_PAPER]), C_ENG + np.array([-0.03, 0.00, Z_PAPER])],
    [C_ENG + np.array([-0.01, 0.05, Z_PAPER]), C_ENG + np.array([0.01, 0.00, Z_PAPER])],
    [C_ENG + np.array([-0.02, 0.02, Z_PAPER]), C_ENG + np.array([0.00, 0.02, Z_PAPER])],
]
USER_RAW_NG_UPDATED = [
    np.array([[0.2318, 0.3660], [0.2309, 0.3404]]),
    np.array([[0.2327, 0.3633], [0.2426, 0.3705], [0.2543, 0.3647]]),
    np.array([[0.2543, 0.3642], [0.2548, 0.3408]]),
    np.array([[0.2836, 0.3629], [0.2732, 0.3763], [0.2651, 0.3642]]),
    np.array([[0.2656, 0.3633], [0.2588, 0.3566], [0.2651, 0.3503]]),
    np.array([[0.2651, 0.3503], [0.2741, 0.3444], [0.2804, 0.3485]]),
    np.array([[0.2804, 0.3494], [0.2849, 0.3561], [0.2845, 0.3651]]),
    np.array([[0.2840, 0.3651], [0.2836, 0.3368]]),
    np.array([[0.2836, 0.3368], [0.2732, 0.3282], [0.2647, 0.3359]]),
]
STROKES_NG_PROCESSED = []
OFFSET_X = 0.04 - 0.2318 
OFFSET_Y = 0.00 - 0.3345 
SHIFT_VECTOR = np.array([OFFSET_X, OFFSET_Y])
for stroke in USER_RAW_NG_UPDATED:
    processed_stroke = []
    for p in stroke:
        p_relative = p + SHIFT_VECTOR 
        p_final = p_relative + C_ENG[:2] 
        p_3d = np.array([p_final[0], p_final[1], Z_PAPER])
        processed_stroke.append(p_3d)
    STROKES_NG_PROCESSED.append(processed_stroke)

# --- 1.2 学号美化 (Beautiful Digits) ---
C_NUM = np.array([0.0, 0.22, 0.0]) 
W_N = 0.012  
H_N = 0.025  
S_N = 0.005  

def get_beautiful_digit(d, offset_x):
    """
    使用贝塞尔曲线(3点)定义更美观的数字
    返回格式: [p_start, p_control, p_end] (曲线) 或 [p_start, p_end] (直线)
    """
    base = C_NUM + np.array([offset_x, 0, 0])
    cx, cy, z = base[0], base[1], base[2]
    
    # 定义关键点 (相对坐标)
    # TL: Top Left, etc.
    p = lambda x, y: np.array([cx + x*W_N, cy + y*H_N, Z_PAPER])
    
    strokes = []
    
    if d == '0':
        # 椭圆形 (4段贝塞尔)
        # 右上弧
        strokes.append([p(0, 0.5), p(0.5, 0.5), p(0.5, 0)]) 
        # 右下弧
        strokes.append([p(0.5, 0), p(0.5, -0.5), p(0, -0.5)])
        # 左下弧
        strokes.append([p(0, -0.5), p(-0.5, -0.5), p(-0.5, 0)])
        # 左上弧
        strokes.append([p(-0.5, 0), p(-0.5, 0.5), p(0, 0.5)])
        
    elif d == '1':
        # 简单竖线
        strokes.append([p(0, 0.5), p(0, -0.5)])
        
    elif d == '2':
        # 天鹅颈 (曲线)
        strokes.append([p(-0.4, 0.3), p(-0.4, 0.5), p(0, 0.5)]) # 起笔上扬
        strokes.append([p(0, 0.5), p(0.5, 0.5), p(0.5, 0.2)])   # 右上圆弧
        strokes.append([p(0.5, 0.2), p(0.2, 0), p(-0.5, -0.5)]) # 斜向左下
        # 底部横线
        strokes.append([p(-0.5, -0.5), p(0.5, -0.5)])
        
    elif d == '3':
        # 上半圆
        strokes.append([p(-0.4, 0.5), p(0.5, 0.5), p(0.2, 0)])
        # 下半圆
        strokes.append([p(0.2, 0), p(0.5, -0.5), p(-0.4, -0.5)])
        
    elif d == '5':
        # 顶部横线
        strokes.append([p(0.4, 0.5), p(-0.4, 0.5)])
        # 竖线
        strokes.append([p(-0.4, 0.5), p(-0.4, 0)])
        # 肚子 (大圆弧)
        strokes.append([p(-0.4, 0), p(0.6, 0.1), p(0.5, -0.4)])
        strokes.append([p(0.5, -0.4), p(0.3, -0.5), p(-0.4, -0.5)])
        
    return strokes

STUDENT_ID = "2025531120"
STROKES_ID = []
total_width = len(STUDENT_ID) * (W_N + S_N) - S_N
start_x = -total_width / 2

for i, char in enumerate(STUDENT_ID):
    curr_x = start_x + i * (W_N + S_N)
    strokes = get_beautiful_digit(char, curr_x)
    STROKES_ID.extend(strokes)

# 合并所有笔画
ALL_STROKES_RAW = STROKES_LI_3D + STROKES_ANG_3D + STROKES_LiA + STROKES_NG_PROCESSED + STROKES_ID

# 镜像处理
ALL_STROKES = []
for stroke in ALL_STROKES_RAW:
    mirrored_stroke = []
    for point in stroke:
        new_point = point.copy()
        new_point[0] = -new_point[0] 
        mirrored_stroke.append(new_point)
    ALL_STROKES.append(mirrored_stroke)

# ================= 2. MinJerk 轨迹插值 =================

class TrajectorySegment:
    def __init__(self, target_pos, duration, method='minjerk', ink=False, control_point=None):
        self.target_pos = np.array(target_pos)
        self.duration = duration
        self.method = method 
        self.ink = ink
        self.control_point = control_point

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
        end_pos_paper = stroke_data[1]
        is_curve = False
        
    dist = np.linalg.norm(start_pos_paper - end_pos_paper)
    write_duration = 0.15 + (dist * 3.0)
    if start_pos_paper[1] < 0.25: 
        write_duration = 0.1 + (dist * 1.5)
    
    start_pos_air = start_pos_paper.copy(); start_pos_air[2] = Z_AIR
    end_pos_air = end_pos_paper.copy(); end_pos_air[2] = Z_AIR

    # 动作序列 (全部使用 MinJerk 以保证丝滑)
    segments.append(TrajectorySegment(start_pos_air, duration=0.3, method='minjerk', ink=False)) 
    segments.append(TrajectorySegment(start_pos_paper, duration=0.2, method='minjerk', ink=False)) 
    segments.append(TrajectorySegment(start_pos_paper, duration=0.05, method='minjerk', ink=False)) 

    if is_curve:
        segments.append(TrajectorySegment(end_pos_paper, duration=write_duration, method='bezier', ink=True, control_point=control_pos_paper)) 
    else:
        segments.append(TrajectorySegment(end_pos_paper, duration=write_duration, method='minjerk', ink=True)) 

    segments.append(TrajectorySegment(end_pos_paper, duration=0.05, method='minjerk', ink=False)) 
    segments.append(TrajectorySegment(end_pos_air, duration=0.2, method='minjerk', ink=False)) 

# --- 3. 增强版 IK 控制器 (Null-Space Control) ---
# 这是解决“手腕触地/乱扭”的关键
def IK_controller_robust(model, data, X_ref, q_pos):
    position_Q = data.site_xpos[0]
    
    # 1. 计算雅可比矩阵
    jacp = np.zeros((3, 6))
    mj.mj_jac(model, data, jacp, None, position_Q, 7)
    J = jacp.copy() # 3x6 Matrix
    
    # 2. 计算主要任务 (Primary Task: Reach X_ref)
    X = position_Q.copy()
    dX = X_ref - X
    
    # 使用 Damped Least Squares 防止奇异
    lambda_val = 0.1 
    J_T = J.T
    # J_pinv = J.T @ inv(J*J.T + lambda*I)
    J_pinv = J_T @ np.linalg.inv(J @ J_T + np.eye(3) * lambda_val**2)
    
    dq_primary = J_pinv @ (dX * 20.0)
    
    # 3. 计算次要任务 (Secondary Task: Maintain Posture)
    # Null Space Projection: (I - J_pinv * J)
    I = np.eye(6)
    null_space_projector = I - (J_pinv @ J)
    
    # 我们希望 q 接近 Q_REST (舒适姿态)
    # 姿态误差 = Q_REST - q_pos
    q_error = Q_REST - q_pos
    
    # 在零空间内优化姿态 (增益设为 1.0)
    dq_secondary = null_space_projector @ (q_error * 2.0)
    
    # 总控制量
    dq_total = dq_primary + dq_secondary
    
    return q_pos + dq_total

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
window = glfw.create_window(1280, 720, "RoboWriter Final (Posture + Font)", None, None)
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
            
        # 插值计算
        X_ref = Interpolate(
            current_start_pos, 
            seg.target_pos, 
            t_local, 
            seg.duration, 
            method=seg.method, 
            control_point=seg.control_point
        )
            
        cur_q_pos = data.qpos.copy()
        
        # 【关键】使用带姿态优化的 IK
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