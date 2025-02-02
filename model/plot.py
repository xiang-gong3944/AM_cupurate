import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go

from model import calc
from model.hubbard_model import _spin

from util import post

# デフォルトのプロットオプション
defaults = {
        "folder_path": "./output/temp/",
        "is_plt_show": True,
        "is_post": False,
    }

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


def band(model, **kwargs):
    option = {**defaults, **kwargs}

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

    colors = ["tab:blue", "tab:purple","tab:red"]
    cmap_name = LinearSegmentedColormap.from_list("custom",colors)

    for i in range(model.n_orbit*2):
        plt.scatter(distances, bands[:,i], c=spins[:,i]/2, cmap=cmap_name, vmin=-0.5, vmax=0.5, s=1)
    del i

    plt.vlines(label_loc[1:-1], Ymin,Ymax, "grey", "dashed")
    plt.hlines(model.ef, distances[0], distances[-1], "grey")
    plt.title("$E_f$ = {:.5f}".format(model.ef))
    plt.colorbar()

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] +"band"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
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


def dos(model, **kwargs):
    option = {**defaults, **kwargs}

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

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] +"dos"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()

    return


def fermi_surface(model, beta=500, **kwargs):
    option = {**defaults, **kwargs}

    # プロットエリアの整備
    fig, ax = plt.subplots()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # スピン分裂の表示
    kx, ky = model._gen_kmesh()
    spin = np.sum(model.spins * calc.fermi_dist(model.enes, model.ef, beta), axis=2).clip(-0.5,0.5)
    mappable = ax.pcolormesh(kx, ky, spin, cmap="seismic", vmax=0.5, vmin = -0.5)
    plt.colorbar(mappable, ax=ax)

    # フェルミ面の表示 スピン分裂がないところだけ散布図でプロットすることであたかも重なって紫になってるように見える。
    abs_spin_spit = 1 - np.abs(spin)*2
    fermi_surf = np.sum(-calc.fermi_dist_diff(model.enes, model.ef, beta),  axis=2)
    fermi_surf = fermi_surf / np.max(fermi_surf)
    fermi_surf = fermi_surf * abs_spin_spit
    ax.scatter(kx, ky, c="tab:purple", alpha=fermi_surf, s=0.1)

    plt.title("$N$ = {:1.1f}".format(model.n_carrier))
    ax.set_box_aspect(1)

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] +"fermi"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()
        print("generated fermi surface\n")

    return


def spin_current(model, mu, beta=500, **kwargs):
    option = {**defaults, **kwargs}

    fig, ax = plt.subplots()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")

    velocity = np.zeros((model.k_mesh, model.k_mesh), np.complex128)
    J_matrices = calc.spin_current_matrix(model, mu)

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):
            velocity[i,j] = np.sum(np.diag(J_matrices[i,j]) * calc.fermi_dist(model.enes[i,j], model.ef, beta))

    real_v = velocity.real
    velocity_max = np.max(np.abs(real_v))
    velocity_min = -velocity_max

    kx, ky = model._gen_kmesh()
    mappable = ax.pcolormesh(kx, ky, real_v, cmap="seismic", vmax=velocity_max, vmin=velocity_min)
    plt.colorbar(mappable, ax=ax)

    plt.title("$J_{{ {:s} }}^s,\,N =$ {:1.1f} ".format(mu, model.n_carrier))
    ax.axis("equal")

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] + "spin_J" + mu + model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()
        print("generated spin current on fermi surface\n")

    return


def electrical_current(model, mu, beta=500, **kwargs):
    option = {**defaults, **kwargs}

    fig, ax = plt.subplots()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")

    velocity = np.zeros((model.k_mesh, model.k_mesh), np.complex128)
    J_matrices = calc.electrical_current_matrix(model, mu)

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):
            velocity[i,j] = np.sum(np.diag(J_matrices[i,j]) * calc.fermi_dist(model.enes[i,j], model.ef, beta))

    real_v = velocity.real
    velocity_max = np.max(np.abs(real_v))
    velocity_min = -velocity_max

    kx, ky = model._gen_kmesh()
    mappable = ax.pcolormesh(kx, ky, real_v, cmap="seismic", vmax=velocity_max, vmin=velocity_min)
    plt.colorbar(mappable, ax=ax)

    plt.title("$J_{{ {:s} }}^e,\,N =$ {:1.1f} ".format(mu, model.n_carrier))
    ax.axis("equal")

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] + "electrical_J" + mu + model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()
        print("generated electrical current on fermi surface\n")

    return


def spin_conductivity(model, mu: str, nu: str, **kwargs):
    option = {**defaults, **kwargs}

    munu = mu + nu
    if(munu == "xy"):
        chi = model.chi_xy
    elif(munu == "yx"):
        chi = model.chi_yx
    elif(munu == "xx"):
        chi = model.chi_xx
    elif(munu == "yy"):
        chi = model.chi_yy
    else:
        print("invalid arguments given")
        return

    if(chi is None):
        print("spin_conducticity has not calculated")
        return

    chi = chi.real

    kx, ky = model._gen_kmesh()
    fig, ax = plt.subplots()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    chi_max = np.max(np.abs(chi))
    chi_min = -chi_max

    mappable = ax.pcolormesh(kx, ky, chi, cmap="seismic", vmax = chi_max, vmin=chi_min)
    plt.colorbar(mappable, ax=ax)

    plt.title("$\chi_{{ {:s} }} =$ {:1.2f}, $N =$ {:1.1f} ".format(munu, np.sum(chi), model.n_carrier))

    ax.set_box_aspect(1)


    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path = option["folder_path"] +f"spin_conductivity_{munu}"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()
    return


def electrical_conductivity(model, mu: str, nu: str, **kwargs):
    option = {**defaults, **kwargs}

    munu = mu + nu
    if(munu == "xy"):
        sigma = model.sigma_xy
    elif(munu == "yx"):
        sigma = model.sigma_yx
    elif(munu == "xx"):
        sigma = model.sigma_xx
    elif(munu == "yy"):
        sigma = model.sigma_yy
    else:
        print("invalid arguments given")
        return

    if(sigma is None):
        print("electrical conducticity has not calculated")
        return

    sigma = sigma.real

    kx, ky = model._gen_kmesh()
    fig, ax = plt.subplots()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])

    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")

    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    sigma_max = np.max(np.abs(sigma))
    sigma_min = -sigma_max

    mappable = ax.pcolormesh(kx, ky, sigma, cmap="seismic", vmax = sigma_max, vmin=sigma_min)
    plt.colorbar(mappable, ax=ax)

    plt.title("$\sigma_{{ {:s} }} =$ {:1.2f}, $N =$ {:1.1f} ".format(munu, np.sum(sigma), model.n_carrier))

    ax.set_box_aspect(1)

    if not os.path.isdir(option["folder_path"]):
        os.makedirs(option["folder_path"])

    image_path =option["folder_path"] +f"electric_conductivity_{munu}"+ model.file_index
    plt.savefig(image_path, bbox_inches='tight')

    if(option["is_post"]):
        post.image(image_path, image_path)

    if option["is_plt_show"]:
        plt.show()
    else:
        plt.close()

    return
