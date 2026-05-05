import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 页面配置
st.set_page_config(page_title="基坑土压力计算工具", layout="wide", initial_sidebar_state="expanded")

# 常量定义
GAMMA_W = 10.0  # 水的重度 kN/m³


# =========================
# 核心计算函数
# =========================

# 主动土压力系数
def calc_Ka(phi):
    phi_rad = np.radians(phi)
    return np.tan(np.pi / 4 - phi_rad / 2) ** 2


# 被动土压力系数
def calc_Kp(phi):
    phi_rad = np.radians(phi)
    return np.tan(np.pi / 4 + phi_rad / 2) ** 2


# 计算竖向自重应力
def calc_vertical_stress(layers, z):
    stress = 0.0
    depth = 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            stress += row['gamma'] * dz
            depth += row['h']
        else:
            break
    return stress


# 计算主动土压力（修正版）
def calc_active_pressure(layers, z):
    current_depth = 0.0
    # 遍历土层，找到当前深度z所在的土层
    for _, row in layers.iterrows():
        layer_top = current_depth
        layer_bottom = current_depth + row['h']

        if layer_top <= z < layer_bottom:
            # 计算竖向应力
            sigma_v = calc_vertical_stress(layers, z)
            Ka = calc_Ka(row['phi'])
            # 库伦主动土压力公式
            sigma_a = Ka * sigma_v - 2 * row['c'] * np.sqrt(Ka)
            return max(sigma_a, 0.0)

        current_depth = layer_bottom
    # 深度超过总土层厚度，取最底层计算
    sigma_v = calc_vertical_stress(layers, z)
    last_row = layers.iloc[-1]
    Ka = calc_Ka(last_row['phi'])
    sigma_a = Ka * sigma_v - 2 * last_row['c'] * np.sqrt(Ka)
    return max(sigma_a, 0.0)


# 计算被动土压力
def calc_passive_pressure(layers, z, reduction=1.0):
    current_depth = 0.0
    # 遍历土层，找到当前深度z所在的土层
    for _, row in layers.iterrows():
        layer_top = current_depth
        layer_bottom = current_depth + row['h']

        if layer_top <= z < layer_bottom:
            sigma_v = calc_vertical_stress(layers, z)
            Kp = calc_Kp(row['phi'])
            sigma_p = Kp * sigma_v + 2 * row['c'] * np.sqrt(Kp)
            return max(sigma_p, 0.0) * reduction

        current_depth = layer_bottom
    # 深度超过总土层厚度，取最底层计算
    sigma_v = calc_vertical_stress(layers, z)
    last_row = layers.iloc[-1]
    Kp = calc_Kp(last_row['phi'])
    sigma_p = Kp * sigma_v + 2 * last_row['c'] * np.sqrt(Kp)
    return max(sigma_p, 0.0) * reduction


# 计算水压力
def calc_water_pressure(z, water_level):
    if z <= water_level:
        return 0.0
    return GAMMA_W * (z - water_level)


# 积分计算合力和作用点
def integrate_force(depths, pressures):
    depths = np.asarray(depths)
    pressures = np.asarray(pressures)

    force = np.trapezoid(pressures, depths)
    moment = np.trapezoid(pressures * depths, depths)

    if force == 0:
        return 0.0, 0.0

    return force, moment / force


# =========================
# 界面UI
# =========================
st.title("土压力工程计算工具（主动+被动）")

st.sidebar.header("📥 输入参数")

# 土层数量设置
num_layers = st.sidebar.number_input("土层数量", min_value=1, max_value=10, value=3)

layers_data = []
for i in range(num_layers):
    st.sidebar.subheader(f"第{i + 1}层土参数")
    h = st.sidebar.number_input(f"厚度 h{i + 1}(m)", value=2.0, key=f"h{i}")
    gamma = st.sidebar.number_input(f"重度 γ{i + 1}(kN/m³)", value=18.0, key=f"g{i}")
    phi = st.sidebar.number_input(f"内摩擦角 φ{i + 1}(°)", value=20.0, key=f"p{i}")
    c = st.sidebar.number_input(f"粘聚力 c{i + 1}(kPa)", value=10.0, key=f"c{i}")
    layers_data.append([h, gamma, phi, c])

# 转换为DataFrame
layers = pd.DataFrame(layers_data, columns=['h', 'gamma', 'phi', 'c'])

# 其他参数
water_level = st.sidebar.number_input("地下水位深度 (m)", value=3.0)
mode = st.sidebar.selectbox("计算类型", ["主动土压力", "被动土压力"])
reduction = st.sidebar.slider("被动土压力折减系数", 0.1, 1.0, 0.7)

# 计算按钮
if st.button("🚀 开始计算"):
    total_depth = layers['h'].sum()
    depths = np.linspace(0, total_depth, 200)

    soil_p = []
    water_p = []
    total_p = []

    # 逐深度计算压力
    for z in depths:
        if mode == "主动土压力":
            sigma = calc_active_pressure(layers, z)
        else:
            sigma = calc_passive_pressure(layers, z, reduction)

        u = calc_water_pressure(z, water_level)
        total = sigma + u

        soil_p.append(sigma)
        water_p.append(u)
        total_p.append(total)

    # 结果表格
    df = pd.DataFrame({
        "深度(m)": depths,
        "土压力(kPa)": soil_p,
        "水压力(kPa)": water_p,
        "总压力(kPa)": total_p
    })

    # 展示结果
    st.subheader("📊 计算结果表")
    st.dataframe(df.round(2), use_container_width=True)

    # 计算合力与作用点
    E, z_bar = integrate_force(depths, total_p)

    st.subheader("🎯 关键计算结果")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("计算类型", mode)
        st.metric("总侧压力合力", f"{E:.2f} kN/m")
    with col2:
        st.metric("被动土压力折减系数", f"{reduction}" if mode == "被动土压力" else "不适用")
        st.metric("合力作用点深度", f"{z_bar:.2f} m")

    # 绘制压力分布图
    st.subheader("📈 土压力分布图")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(total_p, depths, label="总压力", color="#FF4B4B", linewidth=2)
    ax.plot(soil_p, depths, linestyle='--', label="土压力", color="#2E86AB", linewidth=2)
    ax.plot(water_p, depths, linestyle=':', label="水压力", color="#A23B72", linewidth=2)

    ax.invert_yaxis()  # 深度向下为正
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("压力 (kPa)", fontsize=12)
    ax.set_ylabel("深度 (m)", fontsize=12)
    ax.set_title(f"{mode}分布曲线", fontsize=14)
    ax.legend(loc="best")

    st.pyplot(fig)

# 底部说明
st.markdown("---")
st.markdown("""
### 工具说明
1. 基于**库伦土压力理论**计算，适用于基坑支护、挡土墙等工程场景
2. 支持多层土、地下水位、被动土压力折减
3. 自动输出：压力分布表、合力大小、合力作用点深度、压力分布曲线
""")