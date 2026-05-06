iimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="基坑土压力计算工具", layout="wide")

GAMMA_W = 10

# 计算函数
def calc_Ka(phi):
    return np.tan(np.pi/4 - np.radians(phi)/2) ** 2

def calc_Kp(phi):
    return np.tan(np.pi/4 + np.radians(phi)/2) ** 2

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

def calc_effective_stress(layers, z, water_level):
    sigma = calc_vertical_stress(layers, z)
    u = 0 if z <= water_level else GAMMA_W * (z - water_level)
    return sigma - u

def calc_active_pressure(layers, z, water_level, mode):
    depth, sigma_a = 0.0, 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            Ka = calc_Ka(row['phi'])
            if mode == "水土分算":
                sv = calc_vertical_stress(layers, depth + dz)
            else:
                sv = calc_effective_stress(layers, depth + dz, water_level)
            sigma = Ka * sv - 2 * row['c'] * np.sqrt(Ka)
            sigma_a = max(sigma, 0.0)
            depth += row['h']
        else:
            break
    return sigma_a

def calc_passive_pressure(layers, z, water_level, mode, reduction=0.7):
    depth, sigma_p = 0.0, 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            Kp = calc_Kp(row['phi'])
            if mode == "水土分算":
                sv = calc_vertical_stress(layers, depth + dz)
            else:
                sv = calc_effective_stress(layers, depth + dz, water_level)
            sigma = Kp * sv + 2 * row['c'] * np.sqrt(Kp)
            sigma_p = max(sigma, 0.0) * reduction
            depth += row['h']
        else:
            break
    return sigma_p

def calc_water_pressure(z, water_level):
    return 0.0 if z <= water_level else GAMMA_W * (z - water_level)

# UI界面
st.title("📊 基坑支护计算工具（最大弯矩+最大剪力）")

col1, col2 = st.columns([1,2])

with col1:
    num_layers = st.number_input("土层数量", 1, 10, 3)
    layers_data = []
    for i in range(num_layers):
        h = st.number_input(f"土层{i+1}厚度h(m)", 2.0, key=f"h{i}")
        gamma = st.number_input(f"土层{i+1}重度γ(kN/m³)", 18.0, key=f"g{i}")
        phi = st.number_input(f"土层{i+1}内摩擦角φ(°)", 20.0, key=f"p{i}")
        c = st.number_input(f"土层{i+1}黏聚力c(kPa)", 10.0, key=f"c{i}")
        layers_data.append([h, gamma, phi, c])

    layers = pd.DataFrame(layers_data, columns=['h','gamma','phi','c'])

    use_water = st.checkbox("考虑地下水", value=True)
    water_out = st.number_input("坑外水位(m)", 3.0, disabled=not use_water)
    water_in = st.number_input("坑内水位(m)", 0.0, disabled=not use_water)

    mode = st.selectbox("计算模式", ["水土分算","水土合算"])
    reduction = st.slider("被动土压力折减系数", 0.1, 1.0, 0.7)
    run = st.button("开始计算")

with col2:
    if run:
        total_depth = layers['h'].sum()
        z = np.linspace(0, total_depth, 200)

        pa, pp, net = [], [], []
        for depth in z:
            a = calc_active_pressure(layers, depth, water_out, mode)
            p = calc_passive_pressure(layers, depth, water_in, mode, reduction)

            u_out = calc_water_pressure(depth, water_out)
            u_in = calc_water_pressure(depth, water_in)

            if mode == "水土分算":
                pa_val = a + u_out
                pp_val = p + u_in
            else:
                pa_val = a
                pp_val = p

            pa.append(pa_val)
            pp.append(pp_val)
            net.append(pa_val - pp_val)

        pa = np.array(pa)
        pp = np.array(pp)
        net = np.array(net)

        # 总侧向合力
        R = np.trapezoid(net, z)

        # 计算剪力、弯矩
        shear, moment = [], []
        for i in range(len(z)):
            V = R - np.trapezoid(net[:i+1], z[:i+1])
            M = np.trapezoid([R - np.trapezoid(net[:j+1], z[:j+1]) for j in range(i+1)], z[:i+1])
            shear.append(V)
            moment.append(M)

        shear = np.array(shear)
        moment = np.array(moment)

        # 极值求解
        Mmax = np.max(np.abs(moment))
        idx_m = np.argmax(np.abs(moment))
        z_m = z[idx_m]

        Vmax = np.max(np.abs(shear))
        idx_v = np.argmax(np.abs(shear))
        z_v = z[idx_v]

        # 结果表格
        df = pd.DataFrame({
            "深度(m)": z,
            "主动土压力(kPa)": pa,
            "被动土压力(kPa)": pp,
            "净压力(kPa)": net,
            "剪力(kN)": shear,
            "弯矩(kN·m)": moment
        })
        st.dataframe(df, use_container_width=True)

        st.markdown(f"### 最大弯矩：Mmax = {Mmax:.2f} kN·m  深度：{z_m:.2f} m")
        st.markdown(f"### 最大剪力：Vmax = {Vmax:.2f} kN    深度：{z_v:.2f} m")

        # 图1 土压力分布图
        fig1, ax1 = plt.subplots(figsize=(6,8))
        ax1.plot(pa, z, label="主动土压力", linewidth=2)
        ax1.plot(pp, z, label="被动土压力", linewidth=2)
        ax1.plot(net, z, label="净压力", linewidth=2, linestyle='--')
        ax1.invert_yaxis()
        ax1.set_xlabel("压力(kPa)")
        ax1.set_ylabel("深度(m)")
        ax1.grid(alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)

        # 图2 剪力图
        fig2, ax2 = plt.subplots(figsize=(6,8))
        ax2.plot(shear, z, label="剪力", color='darkred', linewidth=2)
        ax2.scatter(shear[idx_v], z_v, color='red', s=50)
        ax2.text(shear[idx_v], z_v, f"Vmax={Vmax:.1f}", fontsize=9)
        ax2.invert_yaxis()
        ax2.set_title("剪力分布图")
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

        # 图3 弯矩图
        fig3, ax3 = plt.subplots(figsize=(6,8))
        ax3.plot(moment, z, label="弯矩", color='darkgreen', linewidth=2)
        ax3.scatter(moment[idx_m], z_m, color='green', s=50)
        ax3.text(moment[idx_m], z_m, f"Mmax={Mmax:.1f}", fontsize=9)
        ax3.invert_yaxis()
        ax3.set_title("弯矩分布图")
        ax3.grid(alpha=0.3)
        ax3.legend()
        st.pyplot(fig3)

st.markdown("#### 说明：已修复语法、缩进、重复代码，支持多层土、水土分算/合算、被动土压力折减，自动计算最大剪力&最大弯矩并绘图。")