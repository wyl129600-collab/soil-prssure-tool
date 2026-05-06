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
            sigma = Kp * (sv + q) + 2 * row['c'] * np.sqrt(Kp)
            sigma_p = max(sigma, 0.0) * reduction
            depth += row['h']
        else:
            break
    return sigma_p

def calc_water_pressure(z, water_level):
    return 0.0 if z <= water_level else GAMMA_W * (z - water_level)

# UI 优化
st.markdown("## 🏗️ 多层土压力计算工具（含合力计算）")
st.markdown("### 👉 支持：主动/被动/水压力合力 + 水土分算/合算 + 地表超载 + 弯矩剪力")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📥 基本参数")

    H = st.number_input("挡土墙高度 H (m)", value=6.0)
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
    water_out = st.number_input("坑外水位(m)",3.0, disabled=not use_water)
    water_in = st.number_input("坑内水位(m)",0.0, disabled=not use_water)

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

        # 初始化所有压力数组
        pa, pp, net = [], [], []
        u_out_list, u_in_list = [], []  # 新增：存储每个深度的水压力

        for depth in z:
            a = calc_active_pressure(layers, depth, water_out, mode, q=q_surface)
            p = calc_passive_pressure(layers, depth, water_in, mode, reduction, q=q_surface)

            u_out = calc_water_pressure(depth, water_out) if use_water else 0
            u_in = calc_water_pressure(depth, water_in) if use_water else 0

            # 保存水压力值
            u_out_list.append(u_out)
            u_in_list.append(u_in)

            pa_val = a + u_out if mode=="水土分算" else a
            pp_val = p + u_in if mode=="水土分算" else p

            pa.append(pa_val)
            pp.append(pp_val)
            net.append(pa_val - pp_val)

        # 转成numpy数组
        pa = np.array(pa)
        pp = np.array(pp)
        net = np.array(net)
        u_out = np.array(u_out_list)
        u_in = np.array(u_in_list)

        # =========================
        # 新增：所有合力计算（沿深度积分）
        # =========================
        pa_total = np.trapezoid(pa, z)       # 总主动土压力合力（含分算水压力）
        pp_total = np.trapezoid(pp, z)       # 总被动土压力合力（含分算水压力）
        u_out_total = np.trapezoid(u_out, z) # 坑外水压力合力
        u_in_total = np.trapezoid(u_in, z)   # 坑内水压力合力
        u_total = u_out_total - u_in_total   # 净水压力合力

        # 支撑反力（原逻辑不变）
        R = np.trapezoid(net, z)

        # 剪力弯矩计算（原逻辑不变）
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

        # =========================
        # 顶部结果卡片（分两行显示，更清晰）
        # =========================
        st.subheader("📊 核心计算结果")
        
        # 第一行：内力极值
        c1,c2,c3 = st.columns(3)
        c1.metric("最大弯矩", f"{Mmax:.2f} kN·m", f"位置 {z_m:.2f}m")
        c2.metric("最大剪力", f"{Vmax:.2f} kN", f"位置 {z_v:.2f}m")
        c3.metric("支撑反力", f"{R:.2f} kN/m")

        # 第二行：各力合力
        c4,c5,c6 = st.columns(3)
        c4.metric("总主动土压力合力", f"{pa_total:.2f} kN/m")
        c5.metric("总被动土压力合力", f"{pp_total:.2f} kN/m")
        c6.metric("净水压力合力", f"{u_total:.2f} kN/m", f"坑外{u_out_total:.1f}/坑内{u_in_total:.1f}")

        st.markdown("---")

        # =========================
        # 结果表格（新增水压力列）
        # =========================
        df = pd.DataFrame({
            "深度(m)": z,
            "主动土压力(kPa)": pa,
            "被动土压力(kPa)": pp,
            "坑外水压力(kPa)": u_out,
            "坑内水压力(kPa)": u_in,
            "净土压力(kPa)": net,
            "剪力(kN)": shear,
            "弯矩(kN·m)": moment
        })

        st.subheader("📋 详细计算结果表")
        st.dataframe(
            df.style.format({
                "深度(m)": "{:.2f}",
                "主动土压力(kPa)": "{:.2f}",
                "被动土压力(kPa)": "{:.2f}",
                "坑外水压力(kPa)": "{:.2f}",
                "坑内水压力(kPa)": "{:.2f}",
                "净土压力(kPa)": "{:.2f}",
                "剪力(kN)": "{:.2f}",
                "弯矩(kN·m)": "{:.2f}"
            }),
            use_container_width=True,
            height=400
        )

        st.markdown("---")

        # =========================
        # 图形（原逻辑不变，更新标题）
        # =========================
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
        
        fig, ax = plt.subplots(figsize=(6,8))
        ax.plot(pa, z, label="主动土压力", linewidth=2)
        ax.plot(pp, z, label="被动土压力", linewidth=2)
        ax.plot(net, z, label="净土压力", linewidth=2)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("土压力沿深度分布")
        ax.set_xlabel("压力(kPa)")
        ax.set_ylabel("深度(m)")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6,8))
        ax2.plot(shear, z, label="剪力", linewidth=2, color="#2E86AB")
        ax2.scatter(Vmax, z_v, color="red", s=40)
        ax2.text(Vmax, z_v, f"Vmax={Vmax:.1f}", fontsize=9)
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.set_title("剪力分布图")
        ax2.set_xlabel("剪力(kN)")
        ax2.set_ylabel("深度(m)")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6,8))
        ax3.plot(moment, z, label="弯矩", linewidth=2, color="#A23B72")
        ax3.scatter(Mmax, z_m, color="red", s=40)
        ax3.text(Mmax, z_m, f"Mmax={Mmax:.1f}", fontsize=9)
        ax3.invert_yaxis()
        ax3.grid(alpha=0.3)
        ax3.set_title("弯矩分布图")
        ax3.set_xlabel("弯矩(kN·m)")
        ax3.set_ylabel("深度(m)")
        st.pyplot(fig3)

st.markdown("---")
st.markdown("""
说明：
1. 所有合力单位均为 **kN/m**（单位长度挡土墙）
2. 总主动/被动土压力合力已包含对应计算模式下的水压力
3. 净水压力合力 = 坑外水压力合力 - 坑内水压力合力
4. 保留原所有功能：分层土、地表超载、水土分算/合算、自动补层
""")
