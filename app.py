import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="基坑土压力计算工具", layout="wide")

GAMMA_W = 10

# 计算函数（保持原函数不变）
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

# 主动土压力（加入地表超载q）
def calc_active_pressure(layers, z, water_level, mode, q=0.0):
    depth, sigma_a = 0.0, 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            Ka = calc_Ka(row['phi'])
            sv = calc_vertical_stress(layers, depth + dz) if mode=="水土分算" else calc_effective_stress(layers, depth + dz, water_level)
            # 地表均布荷载产生附加应力 q*Ka
            sigma = Ka * (sv + q) - 2 * row['c'] * np.sqrt(Ka)
            sigma_a = max(sigma, 0.0)
            depth += row['h']
        else:
            break
    return sigma_a

# 被动土压力（加入地表超载q）
def calc_passive_pressure(layers, z, water_level, mode, reduction=0.7, q=0.0):
    depth, sigma_p = 0.0, 0.0
    for _, row in layers.iterrows():
        if z > depth:
            dz = min(row['h'], z - depth)
            Kp = calc_Kp(row['phi'])
            sv = calc_vertical_stress(layers, depth + dz) if mode=="水土分算" else calc_effective_stress(layers, depth + dz, water_level)
            # 地表均布荷载产生附加应力 q*Kp
            sigma = Kp * (sv + q) + 2 * row['c'] * np.sqrt(Kp)
            sigma_p = max(sigma, 0.0) * reduction
            depth += row['h']
        else:
            break
    return sigma_p

def calc_water_pressure(z, water_level):
    return 0.0 if z <= water_level else GAMMA_W * (z - water_level)

# UI 优化
st.markdown("## 🏗️ 多层土压力计算工具（含地表均布荷载）")
st.markdown("### 👉 支持：主动/被动 + 水土分算/合算 + 地表超载 + 支撑 + 弯矩剪力")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📥 基本参数")

    H = st.number_input("挡土墙高度 H (m)", value=6.0)
    # 新增：地表均布荷载
    q_surface = st.number_input("地表均布荷载 q (kPa)", value=10.0, min_value=0.0)

    num_layers = st.number_input("土层数量",1,10,2)

    st.markdown("---")
    st.subheader("🌍 土层参数")

    layers_data = []
    for i in range(num_layers):
        st.markdown(f"第{i+1}层")
        h = st.number_input(f"厚度 h{i+1}",0,key=f"h{i}")
        gamma = st.number_input(f"重度 γ{i+1}",0,key=f"g{i}")
        phi = st.number_input(f"内摩擦角 φ{i+1}",0,key=f"p{i}")
        c = st.number_input(f"粘聚力 c{i+1}",0,key=f"c{i}")
        layers_data.append([h,gamma,phi,c])

    layers = pd.DataFrame(layers_data,columns=['h','gamma','phi','c'])

    st.markdown("---")
    st.subheader("💧 水位参数")

    use_water = st.checkbox("考虑地下水", True)
    water_out = st.number_input("坑外水位",3.0, disabled=not use_water)
    water_in = st.number_input("坑内水位",0.0, disabled=not use_water)

    mode = st.selectbox("计算模式", ["水土分算","水土合算"])

    reduction = st.slider("被动折减系数",0.1,1.0,0.7)

    run = st.button("🚀 开始计算")

with col2:
    if run:
        # 挡土墙高度校核
        total_layer_depth = layers['h'].sum()

        if H > total_layer_depth:
            st.warning(f"⚠️ 挡土墙高度 H={H:.2f} m 大于土层总厚度 {total_layer_depth:.2f} m")
            last_layer = layers.iloc[-1]
            extra_h = H - total_layer_depth

            layers = pd.concat([
                layers,
                pd.DataFrame([[extra_h, last_layer['gamma'], last_layer['phi'], last_layer['c']]],
                             columns=['h','gamma','phi','c'])
            ], ignore_index=True)

            st.info(f"已自动补充 {extra_h:.2f} m 土层（按最后一层参数延续）")
        z = np.linspace(0,H,200)

        pa, pp, net = [], [], []

        for depth in z:
            # 传入地表荷载 q_surface
            a = calc_active_pressure(layers, depth, water_out, mode, q=q_surface)
            p = calc_passive_pressure(layers, depth, water_in, mode, reduction, q=q_surface)

            u_out = calc_water_pressure(depth, water_out) if use_water else 0
            u_in = calc_water_pressure(depth, water_in) if use_water else 0

            pa_val = a + u_out if mode=="水土分算" else a
            pp_val = p + u_in if mode=="水土分算" else p

            pa.append(pa_val)
            pp.append(pp_val)
            net.append(pa_val - pp_val)

        pa, pp, net = np.array(pa), np.array(pp), np.array(net)

        # 支撑反力
        R = np.trapezoid(net, z)

        shear, moment = [], []
        for i in range(len(z)):
            V = R - np.trapezoid(net[:i+1], z[:i+1])
            M = np.trapezoid([R - np.trapezoid(net[:j+1], z[:j+1]) for j in range(i+1)], z[:i+1])
            shear.append(V)
            moment.append(M)

        shear = np.array(shear)
        moment = np.array(moment)

        # 极值
        Mmax = np.max(np.abs(moment))
        z_m = z[np.argmax(np.abs(moment))]

        Vmax = np.max(np.abs(shear))
        z_v = z[np.argmax(np.abs(shear))]

        # 顶部结果卡片
        c1,c2,c3,c4,c5 = st.columns(5)

        c1.metric("最大弯矩", f"{Mmax:.2f}", f"位置 {z_m:.2f}m")
        c2.metric("最大剪力", f"{Vmax:.2f}", f"位置 {z_v:.2f}m")
        c3.metric("支撑反力", f"{R:.2f}")
        c4.metric("挡土墙高度", f"{H:.2f} m")
        c5.metric("地表超载", f"{q_surface:.2f} kPa")

        st.markdown("---")

        # 计算结果表
        df = pd.DataFrame({
            "深度(m)": z,
            "主动土压力(kPa)": pa,
            "被动土压力(kPa)": pp,
            "净土压力(kPa)": net,
            "剪力(kN)": shear,
            "弯矩(kN·m)": moment
        })

        st.subheader("📋 计算结果表（含地表超载）")

        st.dataframe(
            df.style.format({
                "深度(m)": "{:.2f}",
                "主动土压力(kPa)": "{:.2f}",
                "被动土压力(kPa)": "{:.2f}",
                "净土压力(kPa)": "{:.2f}",
                "剪力(kN)": "{:.2f}",
                "弯矩(kN·m)": "{:.2f}"
            }),
            use_container_width=True
        )

        st.markdown("---")

        # 图形
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
        fig, ax = plt.subplots()
        ax.plot(pa, z, label="主动土压力")
        ax.plot(pp, z, label="被动土压力")
        ax.plot(net, z, label="净压力")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("土压力分布（含地表均布荷载）")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(shear, z, label="剪力")
        ax2.scatter(Vmax, z_v, color='red')
        ax2.text(Vmax, z_v, f"Vmax={Vmax:.1f}")
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.set_title("剪力图")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(moment, z, label="弯矩")
        ax3.scatter(Mmax, z_m, color='red')
        ax3.text(Mmax, z_m, f"Mmax={Mmax:.1f}")
        ax3.invert_yaxis()
        ax3.grid(alpha=0.3)
        ax3.set_title("弯矩图")
        st.pyplot(fig3)

st.markdown("---")
st.markdown("说明:无。")
