import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("ワンウェイ式カーシェア・シミュレーション")

# --- ユーザー入力 ---
st.sidebar.header("シミュレーション設定")

num_ports = st.sidebar.slider("ポート数", 2, 10, 3)
simulation_hours = st.sidebar.slider("シミュレーション時間（h）", 6, 72, 24)
num_vehicles = st.sidebar.number_input("車両台数", 1, 100, 10)

daily_fixed_cost = st.sidebar.number_input("1日あたりの固定費（円）", 0, 1000000, 5000)
vehicle_unit_cost = st.sidebar.number_input("車両1台あたりの運用コスト（円）", 0, 10000, 200)
ride_price = st.sidebar.number_input("1回あたりの料金（円）", 100, 10000, 1000)

rebalancing = st.sidebar.checkbox("再配置あり", value=True)

# --- 初期条件 ---
max_capacity = [5] * num_ports
initial_distribution = [num_vehicles // num_ports] * num_ports
stock = np.array(initial_distribution)
usage_log = np.zeros(num_vehicles)

# --- 簡易OD行列（確率でポート間移動） ---
od_matrix = np.random.dirichlet(np.ones(num_ports), size=num_ports)

# --- ログ変数 ---
total_demand = 0
unsatisfied = 0
revenue = 0
vehicle_id = 0
port_vehicles = [[] for _ in range(num_ports)]
vehicle_location = []

# 車両をポートに割り当て
for port in range(num_ports):
    for _ in range(initial_distribution[port]):
        port_vehicles[port].append(vehicle_id)
        vehicle_location.append(port)
        vehicle_id += 1

# --- シミュレーション本体 ---
stock_over_time = []

for t in range(simulation_hours):
    stock_snapshot = []
    for port in range(num_ports):
        demand = np.random.poisson(2)  # 平均2回の出発需要
        total_demand += demand
        for _ in range(demand):
            if port_vehicles[port]:
                vid = port_vehicles[port].pop()
                dest = np.random.choice(range(num_ports), p=od_matrix[port])
                if len(port_vehicles[dest]) < max_capacity[dest]:
                    port_vehicles[dest].append(vid)
                    usage_log[vid] += 1
                    revenue += ride_price
                else:
                    unsatisfied += 1
            else:
                unsatisfied += 1
        stock_snapshot.append(len(port_vehicles[port]))
    stock_over_time.append(stock_snapshot)

# --- 結果計算 ---
utilization_rate = usage_log.sum() / (num_vehicles * simulation_hours)
loss_rate = unsatisfied / total_demand
total_cost = daily_fixed_cost + vehicle_unit_cost * num_vehicles
profit = revenue - total_cost

# --- 結果表示 ---
# --- 結果表示（横並びにする） ---
st.subheader("シミュレーション結果")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("稼働率", f"{utilization_rate:.2%}")
with col2:
    st.metric("機会損失率", f"{loss_rate:.2%}")
with col3:
    st.metric("総収益", f"¥{revenue:,}")
with col4:
    st.metric("総コスト", f"¥{total_cost:,}")
with col5:
    st.metric("利益", f"¥{profit:,}")

# --- グラフ描画 ---
df_stock = pd.DataFrame(stock_over_time, columns=[f"Port {i+1}" for i in range(num_ports)])
st.line_chart(df_stock)