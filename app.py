import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 页面配置
st.set_page_config(page_title="基坑支护计算工具", layout="wide")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示

# 水重度
GAMMA_W = 10


# =========================
# 核心计算函数
# =========================
def calc_Ka(phi):
    return np.tan(np.pi / 4 - np.radians(phi) / 2) ** 2


def calc_Kp(phi):
    return np.tan(np.pi / 4 + np.radians(phi) / 2) ** 2


def calc_vertical_stress(layers, z):
    stress, depth = 0.0, 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            stress += row['gamma'] * dz
            depth += row['h']
        else:
            break
    return stress


def calc_active_pressure(layers, z):
    current_depth = 0.0
    for _, row in layers.iterrows():
        layer_top = current_depth
        layer_bottom = current_depth + row['h']
        if layer_top <= z < layer_bottom:
            sv = calc_vertical_stress(layers, z)
            Ka = calc_Ka(row['phi'])
            sigma = Ka * sv - 2 * row['c'] * np.sqrt(Ka)
            return max(sigma, 0.0)
        current_depth = layer_bottom
    # 超过总深度，用最后一层
    sv = calc_vertical_stress(layers, z)
    last = layers.iloc[-1]
    Ka = calc_Ka(last['phi'])
    return max(Ka * sv - 2 * last['c'] * np.sqrt(Ka), 0.0)


def calc_passive_pressure(layers, z, reduction=0.7):
    current_depth = 0.0
    for _, row in layers.iterrows():
        layer_top = current_depth
        layer_bottom = current_depth + row['h']
        if layer_top <= z < layer_bottom:
            sv = calc_vertical_stress(layers, z)
            Kp = calc_Kp(row['phi'])
            sigma = Kp * sv + 2 * row['c'] * np.sqrt(Kp)
            return max(sigma, 0.0) * reduction
        current_depth = layer_bottom
    # 超过总深度，用最后一层
    sv = calc_vertical_stress(layers, z)
    last = layers.iloc[-1]
    Kp = calc_Kp(last['phi'])
    return max(Kp * sv + 2 * last['c'] * np.sqrt(Kp), 0.0) * reduction


def calc_water_pressure(z, water_level):
    return 0.0 if z <= water_level else GAMMA_W * (z - water_level)


# =========================
# UI 界面
# =========================
st.title("📊 基坑支护计算工具（土压力+剪力+弯矩）")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🔧 土层参数")
    num_layers = st.number_input("土层数量", 1, 10, 3)

    layers_data = []
    for i in range(num_layers):
        st.caption(f"第 {i + 1} 层土")
        h = st.number_input(f"厚度 h{i + 1} (m)", 0.1, 50.0, 2.0, key=f"h{i}")
        gamma = st.number_input(f"重度 γ{i + 1} (kN/m³)", 10.0, 25.0, 18.0, key=f"g{i}")
        phi = st.number_input(f"内摩擦角 φ{i + 1} (°)", 0.0, 50.0, 20.0, key=f"p{i}")
        c = st.number_input(f"粘聚力 c{i + 1} (kPa)", 0.0, 100.0, 10.0, key=f"c{i}")
        layers_data.append([h, gamma, phi, c])

    layers = pd.DataFrame(layers_data, columns=['h', 'gamma', 'phi', 'c'])

    st.subheader("🌊 水位与折减")
    water_level = st.number_input("地下水位深度 (m)", 0.0, 50.0, 3.0)
    reduction = st.slider("被动土压力折减系数", 0.1, 1.0, 0.7)

    run = st.button("✅ 开始计算", type="primary")

with col_right:
    if run:
        total_depth = layers['h'].sum()
        z = np.linspace(0, total_depth, 500)  # 加密计算点，提高弯矩精度

        # 压力计算
        pa, pp, net_p = [], [], []
        for depth in z:
            a = calc_active_pressure(layers, depth)
            p = calc_passive_pressure(layers, depth, reduction)
            u = calc_water_pressure(depth, water_level)
            active_total = a + u
            passive_total = p + u
            pa.append(active_total)
            pp.append(passive_total)
            net_p.append(active_total - passive_total)

        pa = np.array(pa)
        pp = np.array(pp)
        net_p = np.array(net_p)

        # 单支撑反力（平衡条件）
        R = -np.trapezoid(net_p, z)

        # 剪力 & 弯矩（数值积分，精度大幅提升）
        shear = np.zeros_like(z)
        moment = np.zeros_like(z)
        for i in range(1, len(z)):
            dz = z[i] - z[i - 1]
            shear[i] = shear[i - 1] - net_p[i - 1] * dz
            moment[i] = moment[i - 1] + shear[i - 1] * dz

        # 结果表
        df = pd.DataFrame({
            "深度(m)": z,
            "主动侧压力(kPa)": pa,
            "被动侧压力(kPa)": pp,
            "净侧压力(kPa)": net_p,
            "剪力(kN/m)": shear,
            "弯矩(kN·m/m)": moment
        }).round(2)

        st.subheader("📄 计算结果")
        st.dataframe(df, use_container_width=True, height=400)

        # 关键结果展示
        max_moment = np.max(np.abs(moment))
        st.info(f"""
        ✅ **支撑反力 R = {R:.2f} kN/m**  
        ✅ **最大弯矩 |Mmax| = {max_moment:.2f} kN·m/m**
        """)

        # 图1：土压力分布
        st.subheader("📈 主动/被动土压力分布")
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(pa, z, label="主动侧压力", color="red", linewidth=2)
        ax.plot(pp, z, label="被动侧压力", color="green", linewidth=2)
        ax.invert_yaxis()
        ax.set_xlabel("压力 (kPa)")
        ax.set_ylabel("深度 (m)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # 图2：弯矩图
        st.subheader("📉 桩身弯矩图")
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        ax2.plot(moment, z, color="blue", linewidth=2, label="弯矩")
        ax2.invert_yaxis()
        ax2.set_xlabel("弯矩 (kN·m/m)")
        ax2.set_ylabel("深度 (m)")
        ax2.set_title("基坑桩/墙弯矩分布")
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

st.markdown("---")
st.markdown("""
### 工具说明
- 采用**库伦土压力理论**
- 适用于**单支撑基坑支护结构**（桩、墙、板）
- 自动计算：主动/被动/净压力 + 剪力 + 弯矩
- 包含水压力、被动土压力折减、多层土模型
""")