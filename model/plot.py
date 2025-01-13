import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go

from model import calc
from model.hubbard_model import _spin

k_points = {}
k_points["Γ"]       = [0.0, 0.0]
k_points["X"]        = [np.pi, 0.0]
k_points["Y"]        = [0.0, np.pi]
k_points["M"]        = [np.pi, np.pi]
k_points["Σ"]       = [np.pi/2, np.pi/2]
k_points["M'"]       = [-np.pi, np.pi]
k_points["Σ'"]      = [-np.pi/2, np.pi/2]

path       = [("Γ","Y"),("Y","M'"),("M'","Σ'"),("Σ'","Γ"),
                           ("Γ","Σ"),("Σ","M"),("M","X"),("X","Γ")]

def nsite(model, folder_path="./output/temp/", is_plt_show = True):
    if(model.Ef_scf.size < 2):
        print("SCF calculation wasn't done yet.")
        return

    plt.figure(figsize=[12.8,4.8])
    plt.subplot(121)
    for i in range(model.n_orbit):
        plt.plot(model.N_site_scf[:,i], label = "site {:d} = {:.3f}".format(i, model.N_site_scf[-1, i]))
    plt.legend()
    plt.subplot(122)
    for i in range(model.n_orbit,model.n_orbit*2):
        plt.plot(model.N_site_scf[:,i], label = "site {:d} = {:.3f}".format(i, model.N_site_scf[-1, i]))
    plt.legend()

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_path = folder_path +"nsite"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if is_plt_show:
        plt.show()
    else:
        plt.close()

    return


def scf(model, folder_path="./output/temp/", is_plt_show = True):
    if(model.Ef_scf.size < 2):
        print("SCF calculation wasn't done yet.")
        return

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("scf loop")

    ax1.set_ylabel("Delta")
    ax1.plot(model.Delta_scf, label="Delta = {:.5f}".format(model.Delta_scf[-1]), color = "tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Ef (eV)")
    ax2.plot(model.Ef_scf, label="Ef = {:.5f}".format(model.Ef_scf[-1]), color = "tab:orange")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_path = folder_path +"scf"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if is_plt_show:
        plt.show()
    else:
        plt.close()

    return


def __gen_kpath(path, npoints = 50):
    """バンド図を書くときの対称点に沿った波数ベクトルの列を作る

    Args:
        path (list[tuple[str, str]]): プロットする対称点と終点の tuple 列
        npoints (int, optional): 対称点の間の点の数 Defaults to 50.

    Returns:
        _type_: 経路の座標、プロット用のラベル、プロット用のラベルの位置、プロット用のx軸の値
    """
    k_path = []
    labels = []
    labels_loc = []
    distances = []
    total_distance = 0.0
    for (spoint, epoint) in path :
        k_start = k_points[spoint]
        k_end   = k_points[epoint]
        # 線形補完でnpoints個のk点の生成
        segment = np.linspace(k_start, k_end, npoints)
        k_path.extend(segment)

        labels.append(spoint)
        labels_loc.append(total_distance)

        distance = np.linalg.norm(np.array(k_end)-np.array(k_start))
        segment_dist = np.linspace(total_distance, total_distance+distance, npoints)
        distances.extend(segment_dist)
        total_distance += distance
    # del spoint, epoint

    labels.append(path[-1][1])
    labels_loc.append(total_distance)

    return k_path, labels, labels_loc, distances


def band(model, folder_path="./output/temp/", is_plt_show = True):
    if(model.Ef_scf.size < 2):
        print("SCF calculation wasn't done yet.")
        return

    k_path, label, label_loc, distances = __gen_kpath(path)

    bands = []
    spins = []

    for kxy in k_path:
        enes, eigenstate = model.Hamiltonian(kxy[0], kxy[1], model.delta)
        bands.append(enes)
        spin = _spin(enes, eigenstate)
        spins.append(spin)
    del kxy

    bands = np.array(bands)
    spins = np.array(spins)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.xlabel("k points")
    plt.ylabel("Energy (eV)")


    Ymin = np.min(bands)*1.05
    Ymax = np.max(bands)*1.05
    plt.xticks(label_loc, label)
    plt.xlim(label_loc[0], label_loc[-1])
    plt.ylim(Ymin, Ymax)

    colors = ["tab:blue", "tab:green","tab:orange"]
    cmap_name = LinearSegmentedColormap.from_list("custom",colors, 10)

    for i in range(model.n_orbit*2):
        plt.scatter(distances, bands[:,i], c=spins[:,i]/2, cmap=cmap_name, vmin=-0.5, vmax=0.5, s=1)
    del i

    plt.vlines(label_loc[1:-1], Ymin,Ymax, "grey", "dashed")
    plt.hlines(model.ef, distances[0], distances[-1], "grey")
    plt.title("$E_f$ = {:.5f}".format(model.ef))
    plt.colorbar()

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_path = folder_path +"band"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if is_plt_show:
        plt.show()
    else:
        plt.close()

    return


def band3d(model):
# 参考 https://qiita.com/okumakito/items/3b2ccc9966c43a5e84d0

    if(model.Ef_scf.size < 2):
        print("SCF calculation wasn't done yet.")
        return

    kx, ky = model._gen_kmesh()

    fig = go.Figure()

    contours = dict(
        x=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
        y=dict(highlight=False, show=True, color='grey', start=-3.5, end=3.5, size=0.5),
        z=dict(highlight=False, show=False, start=-8, end = 8, size=0.5)
    )

    fig.add_trace(go.Surface(
            z=model.enes[:,:,0]-model.ef,
            x=kx,
            y=ky,
            surfacecolor=model.spins[:,:,0],
            colorscale = "viridis",
            cmin=-1.5,
            cmax=1.5,
            showscale = False,
            hoverinfo="skip",
            # opacity=0.8,
            # hidesurface=True,
        )
    )
    for i in range(1, model.n_orbit*2):
        fig.add_trace(go.Surface(
                z=model.enes[:,:,i]-model.ef,
                x=kx,
                y=ky,
                surfacecolor=model.spins[:,:,i],
                colorscale = "viridis",
                cmin=-1.5,
                cmax=1.5,
                showscale = False,
                hoverinfo="skip",
                contours=contours,
                # opacity=0.8,
                # hidesurface=True,
            )
        )
    del i

    axis = dict(visible=True)
    fig.update_scenes(
        xaxis=axis,
        yaxis=axis,
        zaxis=axis,
        aspectratio=dict(x=1,y=1,z=1.5)
    )
    fig.update_layout(
        width=800,   # グラフの幅
        height=800   # グラフの高さ
    )
    fig.show()
    return


def dos(model, folder_path="./output/temp/", is_plt_show =True):
    if(model.dos.size < 2):
        calc.dos(model)

    E = np.linspace(np.min(model.enes)-0.1, np.max(model.enes)+0.1, model.dos.size)

    ysacale = np.max(model.dos)
    plt.ylim(-0.04*ysacale, 1.04*ysacale)

    plt.plot(E, model.dos)

    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.vlines(model.ef, -0.04*ysacale, 1.04*ysacale, color="gray", linestyles="dashed")
    plt.title("Ef={:.2f} eV".format(model.ef))

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_path = folder_path +"dos"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if is_plt_show:
        plt.show()
    else:
        plt.close()

    return


def fermi_surface(model, folder_path="./output/temp/", is_plt_show = True):
    if(model.kF_index.size == 3):
        calc.kF_index(model)

    kx, ky = model._gen_kmesh()

    colors = np.full(model.kF_index.shape[0], "tab:green")
    colors[model.spins[model.kF_index[:, 0], model.kF_index[:, 1], model.kF_index[:, 2]] < -0.1] = "tab:blue"
    colors[model.spins[model.kF_index[:, 0], model.kF_index[:, 1], model.kF_index[:, 2]] > 0.1] = "#ff7f0e"
    # colors[model.spins[model.kF_index[:, 0], model.kF_index[:, 1], model.kF_index[:, 2]] > 0.1] = "tab:orange" # これだとエラーが出る

    rotate_kx = (kx[model.kF_index[:, 0], model.kF_index[:, 1]] + ky[model.kF_index[:, 0], model.kF_index[:, 1]]) / 2
    rotate_ky = (-kx[model.kF_index[:, 0], model.kF_index[:, 1]] + ky[model.kF_index[:, 0], model.kF_index[:, 1]]) / 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.scatter(rotate_kx, rotate_ky, c=colors, s=1)
    plt.scatter(rotate_kx+np.pi, rotate_ky+np.pi, c=colors, s=1)
    plt.scatter(rotate_kx+np.pi, rotate_ky-np.pi, c=colors, s=1)
    plt.scatter(rotate_kx-np.pi, rotate_ky+np.pi, c=colors, s=1)
    plt.scatter(rotate_kx-np.pi, rotate_ky-np.pi, c=colors, s=1)

    plt.plot([np.pi, 0, -np.pi, 0, np.pi], [0, np.pi, 0, -np.pi, 0], linestyle = "dashed", c = "grey")
    plt.arrow(-2.1,2.1, 4.2, -4.2, width=0.01,head_width=0.05,head_length=0.2,length_includes_head=True, color ="grey")
    plt.arrow(-2.1, -2.1, 4.2, 4.2, width=0.01,head_width=0.05,head_length=0.2,length_includes_head=True, color = "grey")
    plt.text(2.3, -2.3, "$k_x$")
    plt.text(2.3, 2.3, "$k_y$")

    plt.xlabel("$k_x'$")
    plt.ylabel("$k_y'$")

    plt.axis("square")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    image_path = folder_path +"fermi"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if is_plt_show:
        plt.show()
    else:
        plt.close()
    return
