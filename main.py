import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

WFRadialDivisions = 30  # ウェーハの半径方向分割数
WFAngularDivisions = 72  # ウェーハの角度方向分割数
PadRadialDivisions = 250  # padの半径方向分割数
PadAngularDivisions = 144  # padの角度方向分割数
WFRadius = 150  # ウェーハ半径
PadRadiusMin = 377  # pad半径最小
PadRadiusMAx = 1063  # pad半径最大
COLUMNS = ['nr', 'ns', 'r', 's', 'xg', 'yg']

# ウェーハの格子点生成
wf_lattice_points = [[0, 0, 0., 0., 0., 0.]]
wf_ds = 2 * np.pi / WFAngularDivisions
wf_r0 = WFRadius / np.sqrt(WFRadialDivisions)
for nr_i in tqdm(range(WFRadialDivisions), desc='ウェーハ格子点'):
    r_i = wf_r0 * np.sqrt(nr_i + 1)
    for ns_i in range(WFAngularDivisions):
        s_i = wf_ds * ns_i
        x_i = r_i * np.cos(s_i)
        y_i = r_i * np.sin(s_i)
        wf_lattice_points.append([nr_i + 1, ns_i, r_i, s_i, x_i, y_i])
wf_lattice_points = pd.DataFrame(wf_lattice_points, columns=COLUMNS)

# ウェーハの要素生成
wf_elements = []
for nr_i in tqdm(range(WFRadialDivisions), desc='ウェーハ要素'):
    # 要素の特徴量を記しておく
    r_im1 = wf_r0 * np.sqrt(nr_i)  # 要素の内側の円の半径
    r_i = wf_r0 * np.sqrt(nr_i + 1)  # 要素の外側の円の半径
    a = (r_i - r_im1) * np.cos(wf_ds / 2) / 2
    b = r_im1 * np.sin(wf_ds / 2)
    c = r_i * np.sin(wf_ds / 2)
    el_r = (r_i + r_im1) * np.cos(wf_ds / 2) / 2
    for ns_i in range(WFAngularDivisions):
        # 再内周の三角形要素のみ特殊な数え方をする
        if nr_i == 0:
            nr_list = [0, nr_i+1, nr_i+1, 0]
            ns_list = [0, ns_i, ns_i + 1, 0]
        else:
            nr_list = [nr_i, nr_i + 1, nr_i + 1, nr_i]
            ns_list = [ns_i, ns_i, ns_i + 1, ns_i + 1]
        # 要素を反時計回りに対応させていく
        nodes_i = []
        for nr_j, ns_j in zip(nr_list, ns_list):
            chk_nr_j = wf_lattice_points['nr'] == nr_j
            chk_ns_j = wf_lattice_points['ns'] == ns_j % WFAngularDivisions
            node_j = wf_lattice_points[chk_nr_j & chk_ns_j].index.values[0]
            nodes_i.append(node_j)
        # データを格納する
        el_s = wf_ds / 2 + wf_ds * ns_i
        for v_tmp in [a, b, c, el_r, el_s]:
            nodes_i.append(v_tmp)
        wf_elements.append(nodes_i)

wf_elements = pd.DataFrame(wf_elements, columns=['node0', 'node1', 'node2', 'node3', 'a', 'b', 'c', 'r', 's'])

# padの格子点生成
pad_lattice_points = []
pad_ds = 2 * np.pi / 5 / PadAngularDivisions
for nr_i in tqdm(range(PadRadialDivisions + 1), desc='パッド格子点'):
    r_i = np.sqrt(PadRadiusMin ** 2 + nr_i / PadRadialDivisions * (PadRadiusMAx ** 2 - PadRadiusMin ** 2))
    for ns_i in range(PadAngularDivisions + 1):
        s_i = -np.pi / 5 + pad_ds * ns_i
        x_i = r_i * np.cos(s_i)
        y_i = r_i * np.sin(s_i)
        pad_lattice_points.append([nr_i, ns_i, r_i, s_i, x_i, y_i])
pad_lattice_points = pd.DataFrame(pad_lattice_points, columns=COLUMNS)

# padの要素生成
pad_elements = []
for nr_i in tqdm(range(PadRadialDivisions), desc='パッド要素'):
    # 要素の特徴量を記しておく
    r_i = np.sqrt(PadRadiusMin ** 2 + nr_i / PadRadialDivisions * (PadRadiusMAx ** 2 - PadRadiusMin ** 2))  # 要素の内側の円の半径
    r_ip1 = np.sqrt(PadRadiusMin ** 2 + (nr_i + 1) / PadRadialDivisions * (PadRadiusMAx ** 2 - PadRadiusMin ** 2))  # 要素の外側の円の半径
    a = (r_ip1 - r_i) * np.cos(pad_ds / 2) / 2
    b = r_i * np.sin(pad_ds / 2)
    c = r_ip1 * np.sin(pad_ds / 2)
    el_r = (r_i + r_ip1) * np.cos(pad_ds / 2) / 2
    for ns_i in range(PadAngularDivisions):
        # 台形要素のみなので特に場合分け必要なし
        nr_list = [nr_i, nr_i + 1, nr_i + 1, nr_i]
        ns_list = [ns_i, ns_i, ns_i + 1, ns_i + 1]
        # 要素を反時計回りに対応させていく
        nodes_i = []
        for nr_j, ns_j in zip(nr_list, ns_list):
            chk_nr_j = pad_lattice_points['nr'] == nr_j
            chk_ns_j = pad_lattice_points['ns'] == ns_j
            node_j = pad_lattice_points[chk_nr_j & chk_ns_j].index.values[0]
            nodes_i.append(node_j)
        # データを格納する
        el_s = pad_ds / 2 + pad_ds * ns_i - np.pi / 5
        for v_tmp in [a, b, c, el_r, el_s]:
            nodes_i.append(v_tmp)
        pad_elements.append(nodes_i)
        # 要素の特徴量を記しておく
pad_elements = pd.DataFrame(pad_elements, columns=['node0', 'node1', 'node2', 'node3', 'a', 'b', 'c', 'r', 's'])

# ファイル出力
wf_lattice_points.to_csv('wf_lp.csv', index=False)
pad_lattice_points.to_csv('pad_lp.csv', index=False)
wf_elements.to_csv('wf_el.csv', index=False)
pad_elements.to_csv('pad_el.csv', index=False)

# グラフで確認
fig, ax = plt.subplots(2, 2, tight_layout=True)
ax[0, 0].set_aspect('equal')
ax[0, 1].set_aspect('equal')
ax[0, 0].set_xlabel('x(mm)')
ax[0, 0].set_ylabel('y(mm)')
ax[0, 1].set_xlabel('x(mm)')
ax[0, 1].set_ylabel('y(mm)')

#ax[0].scatter(wf_lattice_points['x'], wf_lattice_points['y'], s=10, color='black')
#ax[1].scatter(pad_lattice_points['x'], pad_lattice_points['y'], s=10, color='black')

#for _, nodes_i in tqdm(wf_elements[['node0', 'node1', 'node2', 'node3']].iterrows(),
#                       desc='ウェーハ要素プロット', total=len(wf_elements)):
#    ax[0, 0].plot(wf_lattice_points.iloc[nodes_i]['xg'], wf_lattice_points.iloc[nodes_i]['yg'], lw=1)
#for _, nodes_i in tqdm(pad_elements[['node0', 'node1', 'node2', 'node3']].iterrows(),
#                       desc='ウェーハ格子点', total=len(wf_elements)):
#    ax[0, 1].plot(pad_lattice_points.iloc[nodes_i]['xg'], pad_lattice_points.iloc[nodes_i]['yg'], lw=1)

#for _, (el_r_i, el_s_i) in tqdm(wf_elements[['r', 's']].iterrows(),
#                                desc='ウェーハ要素プロット', total=len(wf_elements)):
#    ax[0, 0].scatter(el_r_i * np.cos(el_s_i), el_r_i * np.sin(el_s_i), color='black', s=3)
#for _, (el_r_i, el_s_i) in tqdm(pad_elements[['r', 's']].iterrows(),
#                                desc='パッド要素プロット', total=len(pad_elements)):
#    ax[0, 1].scatter(el_r_i * np.cos(el_s_i), el_r_i * np.sin(el_s_i), color='black', s=3)

ax[0, 0].scatter(wf_elements['r'] * np.cos(wf_elements['s']),
                 wf_elements['r'] * np.sin(wf_elements['s']),
                 color='black', s=3)
ax[0, 1].scatter(pad_elements['r'] * np.cos(pad_elements['s']),
                 pad_elements['r'] * np.sin(pad_elements['s']),
                 color='black', s=3)

r_unique_wf = wf_elements['r'].unique()
ax[1, 0].plot(list(range(len(r_unique_wf)-1)), r_unique_wf[1:]-r_unique_wf[:-1], lw=3)
r_unique_pad = pad_elements['r'].unique()
ax[1, 1].plot(list(range(len(r_unique_pad)-1)), r_unique_pad[1:]-r_unique_pad[:-1], lw=3)

plt.show()
