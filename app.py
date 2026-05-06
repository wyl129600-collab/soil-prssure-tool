import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="基坑土压力计算工具", layout="wide")

GAMMA_W = 9.81

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
st.markdown("## 🏗️ 多层土压力计算工具（含地表超载+土压力/水压力合力）")
st.markdown("### 👉 支持：主动/被动 + 水土分算/合算 + 地表超载 + 合力 + 支撑反力 + 弯矩剪力")

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
        h = st.number_input(f"厚度 h{i+1}",2.0,key=f"h{i}")
        gamma = st.number_input(f"重度 γ{i+1}",18.0,key=f"g{i}")
        phi = st.number_input(f"内摩擦角 φ{i+1}",20.0,key=f"p{i}")
        c = st.number_input(f"粘聚力 c{i+1}",10.0,key=f"c{i}")
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
        wa_out, wa_in = [], []

        for depth in z:
            a = calc_active_pressure(layers, depth, water_out, mode, q=q_surface)
            p = calc_passive_pressure(layers, depth, water_in, mode, reduction, q=q_surface)

            u_out = calc_water_pressure(depth, water_out) if use_water else 0
            u_in = calc_water_pressure(depth, water_in) if use_water else 0

            pa_val = a + u_out if mode=="水土分算" else a
            pp_val = p + u_in if mode=="水土分算" else p

            pa.append(pa_val)
            pp.append(pp_val)
            net.append(pa_val - pp_val)
            wa_out.append(u_out)
            wa_in.append(u_in)

        pa, pp, net = np.array(pa), np.array(pp), np.array(net)
        wa_out, wa_in = np.array(wa_out), np.array(wa_in)

        # ========== 新增：各类合力积分计算 ==========
        Ea_total = np.trapezoid(pa, z)        # 主动总土压力合力(含水压力)
        Ep_total = np.trapezoid(pp, z)        # 被动总土压力合力(含水压力)
        Ew_out = np.trapezoid(wa_out, z)      # 坑外水压力合力
        Ew_in = np.trapezoid(wa_in, z)        # 坑内水压力合力
        Ea_soil_only = Ea_total - Ew_out      # 主动土压力纯土部分合力
        Ep_soil_only = Ep_total - Ew_in       # 被动土压力纯土部分合力

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

        # 顶部结果卡片扩展为多列展示合力
        st.subheader("📊 合力与内力汇总")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("主动总合力", f"{Ea_total:.2f} kN/m")
        c2.metric("被动总合力", f"{Ep_total:.2f} kN/m")
        c3.metric("坑外水压力合力", f"{Ew_out:.2f} kN/m")
        c4.metric("坑内水压力合力", f"{Ew_in:.2f} kN/m")

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("主动纯土合力", f"{Ea_soil_only:.2f} kN/m")
        c6.metric("被动纯土合力", f"{Ep_soil_only:.2f} kN/m")
        c7.metric("支撑反力", f"{R:.2f} kN/m")
        c8.metric("地表超载", f"{q_surface:.2f} kPa")

        st.markdown("---")
        c9,c10 = st.columns(2)
        c9.metric("最大弯矩", f"{Mmax:.2f} kN·m/m", f"位置 {z_m:.2f}m")
        c10.metric("最大剪力", f"{Vmax:.2f} kN/m", f"位置 {z_v:.2f}m")

        st.markdown("---")

        # 计算结果表
        df = pd.DataFrame({
            "深度(m)": z,
            "主动土压力(kPa)": pa,
            "被动土压力(kPa)": pp,
            "净土压力(kPa)": net,
            "坑外水压力(kPa)": wa_out,
            "坑内水压力(kPa)": wa_in,
            "剪力(kN/m)": shear,
            "弯矩(kN·m/m)": moment
        })

        st.subheader("📋 计算结果表（含水土压力分项）")
        st.dataframe(
            df.style.format({
                "深度(m)": "{:.2f}",
                "主动土压力(kPa)": "{:.2f}",
                "被动土压力(kPa)": "{:.2f}",
                "净土压力(kPa)": "{:.2f}",
                "坑外水压力(kPa)": "{:.2f}",
                "坑内水压力(kPa)": "{:.2f}",
                "剪力(kN/m)": "{:.2f}",
                "弯矩(kN·m/m)": "{:.2f}"
            }),
            use_container_width=True
        )

        st.markdown("---")

        # 图形
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
        fig, ax = plt.subplots(figsize=(6,8))
        ax.plot(pa, z, label="主动土压力")
        ax.plot(pp, z, label="被动土压力")
        ax.plot(net, z, label="净土压力")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("土压力分布（含地表超载）")
        ax.set_xlabel("压力(kPa)")
        ax.set_ylabel("深度(m)")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6,8))
        ax2.plot(shear, z, label="剪力", color="#2E86AB")
        ax2.scatter(Vmax, z_v, color='red')
        ax2.text(Vmax, z_v, f"Vmax={Vmax:.1f}")
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.set_title("剪力图")
        ax2.set_xlabel("剪力(kN/m)")
        ax2.set_ylabel("深度(m)")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6,8))
        ax3.plot(moment, z, label="弯矩", color="#A23B72")
        ax3.scatter(Mmax, z_m, color='red')
        ax3.text(Mmax, z_m, f"Mmax={Mmax:.1f}")
        ax3.invert_yaxis()
        ax3.grid(alpha=0.3)
        ax3.set_title("弯矩图")
        ax3.set_xlabel("弯矩(kN·m/m)")
        ax3.set_ylabel("深度(m)")
        st.pyplot(fig3)

st.markdown("---")
st.markdown("说明：保留原有全部计算逻辑，新增：主动/被动总合力、纯土合力、坑内外水压力合力，结果表格新增水压力分项，单位均按每延米(kN/m)工程惯例。")
