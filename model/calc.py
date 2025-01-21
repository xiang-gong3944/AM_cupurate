import numpy as np


def dos(model,E_fineness=1000,sigma2 = 0.0001):

    model.E = np.linspace(np.min(model.enes)-0.1,np.max(model.enes)+0.1,E_fineness)
    model.dos = np.array([])

    for e in model.E:
        model.dos = np.append(
            model.dos,np.sum(np.exp(-(e-model.enes)**2 / 2 / sigma2 ) / np.sqrt(2 * np.pi * sigma2)))
    del e

    model.dos /= np.sum(model.dos)*(model.E[1]-model.E[0])

    return


def kF_index(model):
    """
    フェルミ面のインデックスを計算する
    """
    if(model.kF_index.size != 3):
        return

    print("kF index calculation start")

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):
            for m in range(model.n_orbit*2):
                candidate_kF_index  = np.array([i,j,m])
                ene_ij = model.enes[i,j,m]
                # 八方で確かめる
                if(i < model.k_mesh-1): # 南方向
                    ene_delta = model.enes[i+1,j,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(j < model.k_mesh-1): # 東方向
                    ene_delta = model.enes[i,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i > 0): # 北方向
                    ene_delta = model.enes[i-1,j,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(j > 0): # 西方向
                    ene_delta = model.enes[i,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i < model.k_mesh-1 and j < model.k_mesh-1): # 南東方向
                    ene_delta = model.enes[i+1,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index ))
                        continue

                if(i > 0 and j < model.k_mesh-1): # 北東方向
                    ene_delta = model.enes[i-1,j+1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(i > 0 and j > 0): # 北西方向
                    ene_delta = model.enes[i-1,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue

                if(i < model.k_mesh-1 and j > 0): # 南西方向
                    ene_delta = model.enes[i+1,j-1,m]
                    if((ene_ij-model.ef)*(ene_delta-model.ef) < 0
                        and np.abs(ene_ij-model.ef) < np.abs(ene_delta-model.ef)):
                        model.kF_index = np.vstack((model.kF_index,candidate_kF_index))
                        continue
    del i,j,m

    model.kF_index = np.delete(model.kF_index,0,0)
    print("kF index calculation finished\n")
    return


def spin_current_matrix(model,mu: str):
    """固有状態基底でのスピン流演算子を返す

    Args:
        model : 強束縛模型
        mu (str): 流れの方向

    Returns:
        spin_current (ndarray): (k_mesh, k_mesh, n_orbit, n_orbit) のスピン流演算子
    """

    kx, ky = model._gen_kmesh()
    k_mesh = model.k_mesh
    n_orbit = model.n_orbit*2

    # もとの基底のスピン行列
    spin_current = np.array([
        model.SpinCurrent(kx[i, j], ky[i, j], mu)
        for i in range(k_mesh) for j in range(k_mesh)
    ]).reshape(k_mesh, k_mesh, n_orbit, n_orbit)

    # 固有状態の基底のスピン行列
    spin_current = np.einsum(
        '...ji,...jk,...kl->...il',
        np.conjugate(model.eigenStates),  # 転置共役
        spin_current,                     # スピン流演算子
        model.eigenStates                 # 固有状態
    )

    return spin_current


def electrical_current_matrix(model,mu: str):
    """固有状態基底での電流演算子を返す

    Args:
        model : 強束縛模型
        mu (str): 流れの方向

    Returns:
        electrical_current (ndarray): (k_mesh, k_mesh, n_orbit, n_orbit) の電流演算子
    """

    kx, ky = model._gen_kmesh()
    k_mesh = model.k_mesh
    n_orbit = model.n_orbit*2

    # もとの基底のスピン行列
    electrical_current = np.array([
        model.Current(kx[i, j], ky[i, j], mu)
        for i in range(k_mesh) for j in range(k_mesh)
    ]).reshape(k_mesh, k_mesh, n_orbit, n_orbit)

    # 固有状態の基底のスピン行列
    electrical_current = np.einsum(
        '...ji,...jk,...kl->...il',
        np.conjugate(model.eigenStates),  # 転置共役
        electrical_current,               # スピン流演算子
        model.eigenStates                 # 固有状態
    )
    return electrical_current


def spin_conductivity(model,mu,nu,omega=0,gamma=0.0001,beta=500):
    """スピン伝導度の計算

    Args:
        mu (str): スピンの流れる方向.
        nu (str): 電場を加える方向.
        omega (float, optional): 電場の振動数. Defaults to 0
        gamma (float, optional): ダンピングファクター. Defaults to 0.0001.

    Returns:
        complex: 複素伝導度が帰ってくる
    """

    print("spin conductivity calculation start.")

    # スピン伝導度 複素数として初期化
    chis = np.zeros((model.k_mesh, model.k_mesh), np.complex128)

    # ブリュアンゾーンのメッシュの生成
    kx,ky = model._gen_kmesh()

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):

            chi_ij = 0.0 + 0.0*1j
            Jmu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.SpinCurrent(kx[i,j],ky[i,j],mu) @ model.eigenStates[i,j]
            Jnu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.Current(kx[i,j],ky[i,j],nu) @ model.eigenStates[i,j]

            for m in range(model.n_orbit*2):
                for n in range(model.n_orbit*2):

                    Jmu = Jmu_matrix[m,n]
                    Jnu = Jnu_matrix[n,m]

                    # バンド間遷移 (van Vleck 項)
                    if(np.abs(model.enes[i,j][m]-model.enes[i,j][n])>1e-6):

                        efm = fermi_dist(model.enes[i,j][m],model.ef, beta)
                        efn = fermi_dist(model.enes[i,j][n],model.ef, beta)

                        add_chi = Jmu * Jnu * (efm - efn) / (
                            (model.enes[i,j][m]-model.enes[i,j][n])*(model.enes[i,j][m]-model.enes[i,j][n] + omega + 1j*gamma))
                        chi_ij += add_chi

                    # バンド内遷移
                    else:
                        # フェルミ分布の微分
                        f_diff = (fermi_dist_diff(model.enes[i,j][m],model.ef)
                                  +fermi_dist_diff(model.enes[i,j][n],model.ef))/2

                        add_chi = Jmu * Jnu * f_diff / (omega + 1j*gamma)
                        chi_ij += add_chi

            chis[i,j] = chi_ij

    chis /= (model.k_mesh*model.k_mesh*1j)
    chi = np.sum(chis)

    munu = mu + nu
    if (munu == "xx"):
        model.chi_xx = chis
    elif (munu == "yy"):
        model.chi_yy = chis
    elif (munu == "xy"):
        model.chi_xy = chis
    elif (munu == "yx"):
        model.chi_yx = chis

    print("spin conductivity calculation finished")
    print("ReChi = {:1.2e}, ImChi = {:1.2e}\n".format(np.real(chi),np.imag(chi)))

    return chi


def spin_cond_omega(model, mu: str, nu: str, omegas, beta: float = 500):
    """スピン伝導度の周波数特性

    Args:
        model (hubbardmodel.HubbardModel): ハバード模型のインスタンス
        mu (str): スピン流の流れる方向
        nu (str): 電場をかける方向
        omegas (ndarray): 調べる振動数の配列

    Returns:
        chis (ndarray): 複素スピン伝導度のリスト
    """

    chis = np.array([])
    for omega in omegas:
        print(f"omega = {omega}")
        chi = spin_conductivity(model, "x", "y", omega)
        chis = np.append(chis, chi)

    return chis


def electrical_conductivity(model,mu:str,nu:str,omega:float=0,gamma:float=0.0001,beta:float=500):
    """電気伝導度の計算

    Args:
        mu (str): キャリアの流れる方向.
        nu (str): 電場を加える方向.
        omega (float, optional): 電場の振動数. Defaults to 0
        gamma (float, optional): ダンピングファクター. Defaults to 0.0001.

    Returns:
        complex: 複素伝導度が帰ってくる
    """

    print("electrical conductivity calculation start.")

    # 電気伝導度 複素数として初期化
    sigmas = np.zeros((model.k_mesh, model.k_mesh), np.complex128)

    # ブリュアンゾーンのメッシュの生成
    kx,ky = model._gen_kmesh()

    for i in range(model.k_mesh):
        for j in range(model.k_mesh):

            sigma_ij = 0.0 + 0.0*1j
            Jmu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.Current(kx[i,j],ky[i,j],mu) @ model.eigenStates[i,j]
            Jnu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.Current(kx[i,j],ky[i,j],nu) @ model.eigenStates[i,j]

            for m in range(model.n_orbit*2):
                for n in range(model.n_orbit*2):

                    Jmu = Jmu_matrix[m,n]
                    Jnu = Jnu_matrix[n,m]

                    # バンド間遷移 (van Vleck 項)
                    if(np.abs(model.enes[i,j][m]-model.enes[i,j][n])>1e-6):

                        efm = fermi_dist(model.enes[i,j][m],model.ef, beta)
                        efn = fermi_dist(model.enes[i,j][n],model.ef, beta)

                        add_sigma = Jmu * Jnu * (efm - efn) / (
                            (model.enes[i,j][m]-model.enes[i,j][n])*(model.enes[i,j][m]-model.enes[i,j][n] + omega + 1j*gamma))
                        sigma_ij += add_sigma

                    # バンド内遷移
                    else:
                        # フェルミ分布の微分
                        f_diff = (fermi_dist_diff(model.enes[i,j][m],model.ef)
                                  +fermi_dist_diff(model.enes[i,j][n],model.ef))/2

                        add_sigma = Jmu * Jnu * f_diff / (omega + 1j*gamma)
                        sigma_ij += add_sigma

            sigmas[i,j] = sigma_ij

    sigmas /= (model.k_mesh*model.k_mesh*1j)
    sigma = np.sum(sigmas)

    munu = mu + nu
    if (munu == "xx"):
        model.sigma_xx = sigmas
    elif (munu == "yy"):
        model.sigma_yy = sigmas
    elif (munu == "xy"):
        model.sigma_xy = sigmas
    elif (munu == "yx"):
        model.sigma_yx = sigmas

    print("electrical conductivity calculation finished")
    print("ReSigma = {:1.2e}, ImSigma = {:1.2e}\n".format(np.real(sigma),np.imag(sigma)))

    return sigma


def electrical_cond_omega(model, mu: str, nu: str, omegas, beta = 500):
    """電気伝導度の周波数特性

    Args:
        model (hubbardmodel.HubbardModel): ハバード模型のインスタンス
        mu (str): キャリアの流れる方向
        nu (str): 電場をかける方向
        omegas (ndarray): 調べる振動数の配列

    Returns:
        chis (ndarray): 複素スピン伝導度のリスト
    """

    sigmas = np.array([])
    for omega in omegas:
        print(f"omega = {omega}")
        sigma = electrical_conductivity(model, "x", "y", omega, beta)
        sigmas = np.append(sigmas, sigma)

    return sigmas


def fermi_dist(ene,ef: float,beta: float=1000):
    """フェルミ分布関数

    Args:
        ene (float): エネルギー
        ef (float): フェルミエネルギー
        beta (float, optional): 逆温度の値. Defaults to 1000.

    Returns:
        float: f(E,ef) の値
    """

    # ene が数字でも配列でも numpy 配列に変換
    a = np.array(beta*(ene-ef))
    # オーバーフローを防ぐ
    a = np.clip(a, -700, 700)

    return 1/(np.exp(a)+1)


def fermi_dist_diff(ene,ef: float,beta: float=1000):
    """フェルミ分布関数の微分

    Args:
        ene (float): エネルギー
        ef (float): フェルミエネルギー
        beta (float, optional): 逆温度の値. Defaults to 1000.

    Returns:
        float: df/dE の値
    """

    # ene が数字でも配列でも numpy 配列に変換
    a = np.array(beta*(ene-ef))
    # オーバーフローを防ぐ
    a = np.clip(a, -700, 700)

    return -beta/(2*np.cosh(a/2))**2

