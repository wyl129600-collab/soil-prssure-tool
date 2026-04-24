import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="基坑土压力计算工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 工程参数
# =========================
GAMMA_W = 10.0  # 水重度 kN/m³

# =========================
# 核心计算函数
# =========================
def calc_Ka(phi):
    phi_rad = np.radians(phi)
    return np.tan(np.pi/4 - phi_rad/2) ** 2

def calc_vertical_stress(layers, z):
    stress = 0
    current_depth = 0
    for _, row in layers.iterrows():
        h_layer = row['h']
        if z <= current_depth:
            break
        dz = min(h_layer, z - current_depth)
        stress += row['gamma'] * dz
        current_depth += dz
    return stress

def calc_active_pressure(layers, z, water_level, mode):
    current_depth = 0
    sigma_a = 0.0
    for _, row in layers.iterrows():
        h = row['h']
        gamma = row['gamma']
        phi = row['phi']
        c = row['c']
        Ka = calc_Ka(phi)
        if current_depth < z <= current_depth + h:
            sigma_v = calc_vertical_stress(layers, z)
            sigma_a = Ka * sigma_v - 2 * c * np.sqrt(Ka)
            sigma_a = max(sigma_a, 0.0)
            break
        current_depth += h
    return sigma_a

def calc_water_pressure(z, water_level):
    if z <= water_level:
        return 0.0
    return GAMMA_W * (z - water_level)

def integrate_force(depths, pressures):
    force = np.trapz(pressures, depths)
    moment = np.trapz(pressures * depths, depths)
    if force < 1e-6:
        return 0.0, 0.0
    return force, moment / force

# =========================
# UI
# =========================
st.title("朗肯主动土压力计算工具")
st.sidebar.header("输入参数")

num_layers = st.sidebar.number_input("土层数量", min_value=1, max_value=10, value=3)
layers_data = []

for i in range(num_layers):
    st.sidebar.subheader(f"第{i+1}层土")
    h = st.sidebar.number_input(f"厚度 h{i+1} (m)", value=2.0, key=f"h{i}")
    gamma = st.sidebar.number_input(f"天然重度 γ{i+1} (kN/m³)", value=18.0, key=f"g{i}")
    phi = st.sidebar.number_input(f"内摩擦角 φ{i+1} (°)", value=20.0, key=f"p{i}")
    c = st.sidebar.number_input(f"粘聚力 c{i+1} (kPa)", value=10.0, key=f"c{i}")
    layers_data.append([h, gamma, phi, c])

layers = pd.DataFrame(layers_data, columns=['h', 'gamma', 'phi', 'c'])
water_level = st.sidebar.number_input("地下水位深度 (m)", value=3.0)
mode = st.sidebar.selectbox("计算模式", ["水土分算", "水土合算"])

if st.button("开始计算"):
    total_depth = layers['h'].sum()
    depths = np.linspace(0, total_depth, 200)

    sigma_a_list = []
    u_list = []
    total_p_list = []

    for z in depths:
        sigma_a = calc_active_pressure(layers, z, water_level, mode)
        u = calc_water_pressure(z, water_level)

        if mode == "水土分算":
            total_p = sigma_a + u
        else:
            total_p = sigma_a

        sigma_a_list.append(sigma_a)
        u_list.append(u)
        total_p_list.append(total_p)

    df = pd.DataFrame({
        "深度(m)": depths,
        "主动土压力(kPa)": sigma_a_list,
        "水压力(kPa)": u_list,
        "总侧压力(kPa)": total_p_list
    })

    st.subheader("📊 计算结果")
    st.dataframe(df.round(2), use_container_width=True)

    Ea, z_bar = integrate_force(depths, np.array(total_p_list))

    st.subheader("🔍 关键输出")
    col1, col2 = st.columns(2)
    col1.metric("总主动侧压力 Eₐ", f"{Ea:.2f} kN/m")
    col2.metric("合力作用点深度", f"{z_bar:.2f} m")

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(total_p_list, depths, 'b-', linewidth=2.5, label="总侧压力")
    ax.plot(sigma_a_list, depths, 'r--', linewidth=2, label="主动土压力")
    ax.plot(u_list, depths, 'g-.', linewidth=1.5, label="水压力")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("压力 (kPa)")
    ax.set_ylabel("深度 (m)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)