import scipy.linalg
import numpy as np


# Slater-Koster パラメータ
Vpps = 1.0
Vppp = -0.6
Vpds = 1.85

# ホッピングパラメータ
Tpp0 = (Vpps-Vppp) / 2
Tpd0 = np.sqrt(3) * Vpds / 2

# クーロン相互作用の強さ
Ud = 8.0
Up = 4.0
Upd = 1.0

# 軌道エネルギー差
Ep = 3.0

# 軌道の数
n_orbit = 6

def Hamiltonian(kx, ky, delta, a0, Ud):
    """ある波数のでの平均場ハミルトニアン
    Args:
        (float) kx: 波数のx成分
        (float) ky: 波数のy成分
        (float) delta: 反強磁性分子場の強さ
        (float) a0: Cu-Cu間距離に対するCu-O間距離の比率
        (float) U: オンサイト相互作用の強さ

    Returns:
        ハミルトニアンの固有値[0]と固有ベクトルの行列[1]
    """

    H = np.zeros((n_orbit*2, n_orbit*2), dtype=np.complex128)
    Tpp = Tpd0 * 2* a0 * (1-a0) / (a0*a0 + (1-a0)*(1-a0))
    Tpd1 = Tpd0 * 2 * (1-a0)    # 短いホッピング
    Tpd2 = Tpd0 * 2 * a0        # 長いホッピング

    # ホッピング項
    H[0,2] = Tpd1 * np.exp(1j*(-kx+ky)*a0/2)
    H[0,3] = Tpd2 * np.exp(1j*(kx+ky)*(1-a0)/2)
    H[0,4] = Tpd1 * np.exp(1j*(kx-ky)*a0/2)
    H[0,5] = Tpd2 * np.exp(-1j*(kx+ky)*(1-a0)/2)

    H[1,2] = Tpd2 * np.exp(1j*(kx-ky)*(1-a0)/2)
    H[1,3] = Tpd1 * np.exp(-1j*(kx+ky)*a0/2)
    H[1,4] = Tpd2 * np.exp(1j*(-kx+ky)*(1-a0)/2)
    H[1,5] = Tpd1 * np.exp(1j*(kx+ky)*a0/2)

    H[2,3] = -2*Tpp * np.cos(kx/2) * np.exp(1j*ky*(1-2*a0)/2)
    H[2,5] = -2*Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

    H[3,4] = -2*Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

    H[4,5] = -2*Tpp * np.cos(kx/2) * np.exp(-1j*ky*(1-2*a0)/2)

    #エルミート化
    for i in range(1,n_orbit):
        for j in range(0, i):
            H[i][j] = H[j][i].conjugate()
    del i, j

    # 軌道準位差
    H[2,2] = Ep
    H[3,3] = Ep
    H[4,4] = Ep
    H[5,5] = Ep

    # 反対向きスピンの分
    for i in range(n_orbit):
        for j in range(n_orbit):
            H[i+n_orbit,j+n_orbit] = H[i,j]
    del i, j

    # Cu イオンでのクーロン相互作用(ハートリー項のみ)
    H[0,0] = -Ud * delta /2
    H[1,1] = Ud * delta /2
    H[0+n_orbit,0+n_orbit] = Ud * delta /2
    H[1+n_orbit,1+n_orbit] = -Ud * delta /2

    # 無限大や非数が含まれていないか確認
    if not np.isfinite(H).all():
        raise ValueError("Hamiltonian matrix contains inf or NaN values")

    return scipy.linalg.eigh(H)

def Current(kx, ky, mu, a0):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける
        a0 (float): Cu-Cu間距離に対するCu-O間距離の比率

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    J = np.zeros((n_orbit*2, n_orbit*2), dtype=np.complex128)
    Tpp = Tpd0 * 2* a0 * (1-a0) / (a0*a0 + (1-a0)*(1-a0))
    Tpd1 = Tpd0 * 2 * (1-a0)    # 短いホッピング
    Tpd2 = Tpd0 * 2 * a0        # 長いホッピング

    if (mu == "x"):
        J[0,2] = -1j*a0/2 * Tpd1 * np.exp(1j*(-kx+ky)*a0/2)
        J[0,3] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx+ky)*(1-a0)/2)
        J[0,4] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx-ky)*a0/2)
        J[0,5] = -1j*(1-a0)/2 * Tpd2 * np.exp(-1j*(kx+ky)*(1-a0)/2)

        J[1,2] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx-ky)*(1-a0)/2)
        J[1,3] = -1j*a0/2 * Tpd1 * np.exp(-1j*(kx+ky)*a0/2)
        J[1,4] = -1j*(1-a0)/2 * Tpd2 * np.exp(1j*(-kx+ky)*(1-a0)/2)
        J[1,5] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx+ky)*a0/2)

        J[2,3] = Tpp * np.sin(kx/2) * np.exp(1j*ky*(1-2*a0)/2)
        J[2,5] = 1j*(1-2*a0) * Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[3,4] = 1j*(1-2*a0) * Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[4,5] = Tpp * np.sin(kx/2) * np.exp(-1j*ky*(1-2*a0)/2)

    elif (mu == "y"):
        J[0,2] = 1j*a0/2 * Tpd1 * np.exp(1j*(-kx+ky)*a0/2)
        J[0,3] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx+ky)*(1-a0)/2)
        J[0,4] = -1j*a0/2 * Tpd1 * np.exp(1j*(kx-ky)*a0/2)
        J[0,5] = -1j*(1-a0)/2 * Tpd2 * np.exp(-1j*(kx+ky)*(1-a0)/2)

        J[1,2] = -1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx-ky)*(1-a0)/2)
        J[1,3] = -1j*a0/2 * Tpd1 * np.exp(-1j*(kx+ky)*a0/2)
        J[1,4] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(-kx+ky)*(1-a0)/2)
        J[1,5] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx+ky)*a0/2)

        J[2,3] = -1j*(1-2*a0) * Tpp * np.cos(kx/2) * np.exp(1j*ky*(1-2*a0)/2)
        J[2,5] = Tpp * np.sin(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[3,4] = Tpp * np.sin(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[4,5] = 1j * (1-2*a0) * Tpp * np.cos(kx/2) * np.exp(-1j*ky*(1-2*a0)/2)


    elif (mu == "z"):
        J = J

    else :
        print("The current direction is incorrect.")
        return

    # 反対向きスピンの分
    for i in range(n_orbit):
        for j in range(n_orbit):
            J[i+n_orbit,j+n_orbit] = J[i,j]
    del i, j

    #エルミート化
    for i in range(1,n_orbit*2):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()
    del i, j

    return J

def SpinCurrent(kx, ky, mu, a0):
    """ある波数での電流演算子行列

    Args:
        kx (float): 波数のx成分
        ky (float): 波数のy成分
        mu (sring): 電流の方向. "x", "y", "z" のみ受け付ける
        a0 (float): Cu-Cu間距離に対するCu-O間距離の比率

    Return:
        J (ndarray): 8x8の電流演算子行列
    """

    J = np.zeros((n_orbit*2, n_orbit*2), dtype=np.complex128)
    Tpp = Tpd0 * 2* a0 * (1-a0) / (a0*a0 + (1-a0)*(1-a0))
    Tpd1 = Tpd0 * 2 * (1-a0)    # 短いホッピング
    Tpd2 = Tpd0 * 2 * a0        # 長いホッピング

    if (mu == "x"):
        J[0,2] = -1j*a0/2 * Tpd1 * np.exp(1j*(-kx+ky)*a0/2)
        J[0,3] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx+ky)*(1-a0)/2)
        J[0,4] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx-ky)*a0/2)
        J[0,5] = -1j*(1-a0)/2 * Tpd2 * np.exp(-1j*(kx+ky)*(1-a0)/2)

        J[1,2] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx-ky)*(1-a0)/2)
        J[1,3] = -1j*a0/2 * Tpd1 * np.exp(-1j*(kx+ky)*a0/2)
        J[1,4] = -1j*(1-a0)/2 * Tpd2 * np.exp(1j*(-kx+ky)*(1-a0)/2)
        J[1,5] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx+ky)*a0/2)

        J[2,3] = Tpp * np.sin(kx/2) * np.exp(1j*ky*(1-2*a0)/2)
        J[2,5] = 1j*(1-2*a0) * Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[3,4] = 1j*(1-2*a0) * Tpp * np.cos(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[4,5] = Tpp * np.sin(kx/2) * np.exp(-1j*ky*(1-2*a0)/2)

    elif (mu == "y"):
        J[0,2] = 1j*a0/2 * Tpd1 * np.exp(1j*(-kx+ky)*a0/2)
        J[0,3] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx+ky)*(1-a0)/2)
        J[0,4] = -1j*a0/2 * Tpd1 * np.exp(1j*(kx-ky)*a0/2)
        J[0,5] = -1j*(1-a0)/2 * Tpd2 * np.exp(-1j*(kx+ky)*(1-a0)/2)

        J[1,2] = -1j*(1-a0)/2 * Tpd2 * np.exp(1j*(kx-ky)*(1-a0)/2)
        J[1,3] = -1j*a0/2 * Tpd1 * np.exp(-1j*(kx+ky)*a0/2)
        J[1,4] = 1j*(1-a0)/2 * Tpd2 * np.exp(1j*(-kx+ky)*(1-a0)/2)
        J[1,5] = 1j*a0/2 * Tpd1 * np.exp(1j*(kx+ky)*a0/2)

        J[2,3] = -1j*(1-2*a0) * Tpp * np.cos(kx/2) * np.exp(1j*ky*(1-2*a0)/2)
        J[2,5] = Tpp * np.sin(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[3,4] = Tpp * np.sin(ky/2) * np.exp(-1j*kx*(1-2*a0)/2)

        J[4,5] = 1j * (1-2*a0) * Tpp * np.cos(kx/2) * np.exp(-1j*ky*(1-2*a0)/2)


    elif (mu == "z"):
        J = J

    else :
        print("The current direction is incorrect.")
        return

    # 反対向きスピンの分
    for i in range(n_orbit):
        for j in range(n_orbit):
            J[i+n_orbit,j+n_orbit] = -J[i,j]
    del i, j

    #エルミート化
    for i in range(1,n_orbit*2):
        for j in range(0, i):
            J[i][j] = J[j][i].conjugate()
    del i, j

    return J/2
