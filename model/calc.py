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


def spin_conductivity(model,mu,nu,gamma=0.0001):
    """直流スピン伝導度の計算

    Args:
        mu (str, optional): スピンの流れる方向. Defaults to "x".
        nu (str, optional): 電場を加える方向. Defaults to "y".
        gamma (float, optional): ダンピングファクター. Defaults to 0.0001.

    Returns:
        complex: 複素伝導度が帰ってくる
    """
    if(model.enes[0,0,0] == 0):
        print("NSCF calculation wasn't done yet.")
        return

    # フェルミ面の計算をしていなかったらする
    if(model.kF_index.size == 3):
        kF_index(model)

    print("SpinConductivity calculation start.")

    # スピン伝導度 複素数として初期化
    chi = 0.0 + 0.0*1j

    # ブリュアンゾーンのメッシュの生成
    kx,ky = model._gen_kmesh()

    # バンド間遷移
    for i in range(model.k_mesh):
        for j in range(model.k_mesh):

            Jmu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.SpinCurrent(kx[i,j],ky[i,j],mu) @ model.eigenStates[i,j]
            Jnu_matrix = np.conjugate(model.eigenStates[i,j].T) @     model.Current(kx[i,j],ky[i,j],nu) @ model.eigenStates[i,j]

            for m in range(model.n_orbit*2):
                for n in range(model.n_orbit*2):

                    Jmu = Jmu_matrix[m,n]
                    Jnu  = Jnu_matrix[n,m]

                    if(np.abs(model.enes[i,j,m]-model.enes[i,j,n]) > 1e-6):
                        # フェルミ分布
                        efm = 1 if (model.enes[i,j][m]<model.ef) else 0
                        efn = 1 if (model.enes[i,j][n]<model.ef) else 0

                        add_chi = Jmu * Jnu * (efm - efn) / ((model.enes[i,j][m]-model.enes[i,j][n])*(model.enes[i,j][m]-model.enes[i,j][n]+1j*gamma))
                        chi += add_chi
    del i,j,m,n

    # バンド内遷移
    for i,j,m in model.kF_index:

            Jmu_matrix = np.conjugate(model.eigenStates[i,j].T) @ model.SpinCurrent(kx[i,j],ky[i,j],mu) @ model.eigenStates[i,j]
            Jnu_matrix = np.conjugate(model.eigenStates[i,j].T) @     model.Current(kx[i,j],ky[i,j],nu) @ model.eigenStates[i,j]

            Jmu = Jmu_matrix[m,m]
            Jnu = Jnu_matrix[m,m]

            chi += 1j * Jmu * Jnu / gamma

    # del i,j,m

    chi /= (model.k_mesh*model.k_mesh*1j)

    print("Spin Conductivity calculation finished")
    print("ReChi = {:1.2e}, ImChi = {:1.2e}\n".format(np.real(chi),np.imag(chi)))

    return chi