import numpy as np
from functools import partial

from model import operators as op

# ちょくちょくつかう関数
def _delta(N_site):
    """反強磁性磁化の大きさ

    Args:
        N_site (list[float]): 各サイトにある電子数

    Returns:
       delta (float): 反強磁性磁化の大きさ
    """
    delta = np.abs(N_site[0] - N_site[1] - N_site[6] + N_site[7]) / 2

    return delta

def _Steffensen(array):
    """Steffensen の反復法で収束を早めるための処理

    Args:
        array : SCF 計算で出てきた2つの値

    Returns:
        val : 収束が早い数列の値
    """

    if array.size <= 3:
        raise ValueError("Array must contain at least three elements for Steffensen's method.")

    try:
        p0, p1, p2 = array[-3], array[-2], array[-1]
        denominator = (p0 - 2 * p1 + p2)
        if denominator == 0:
            raise ZeroDivisionError("Denominator in Steffensen's method is zero, causing division by zero.")
        res = p0 - (p0 - p1) ** 2 / denominator
        return res
    except ZeroDivisionError as e:
        # print(f"Error: {e}")
        return array[-1]  # 収束しない場合は最新の値を返す
    except Exception as e:
        # print(f"Unexpected error: {e}")
        return array[-1]  # 収束しない場合は最新の値を返す

def _spin(enes, eigenstate):
    """各サイトのスピンの大きさ

    Args:
        enes: ある波数の固有エネルギー
        eigenstate :  ある波数の固有ベクトル

    Returns:
        spin : 各サイトのスピンの大きさ

    Remarks:
        ハミルトニアンの作り方に依存するため注意
    """
    spin = []
    n_orbit = 6
    for l in range(n_orbit*2):
        sp = (np.abs(eigenstate[0,l])**2
                +np.abs(eigenstate[1,l])**2
                +np.abs(eigenstate[2,l])**2
                +np.abs(eigenstate[3,l])**2
                +np.abs(eigenstate[4,l])**2
                +np.abs(eigenstate[5,l])**2
                -np.abs(eigenstate[6,l])**2
                -np.abs(eigenstate[7,l])**2
                -np.abs(eigenstate[8,l])**2
                -np.abs(eigenstate[9,l])**2
                -np.abs(eigenstate[10,l])**2
                -np.abs(eigenstate[11,l])**2
                )
        spin.append(sp)
    del l

    for l in range(n_orbit):
        if(np.abs(enes[2*l]-enes[2*l + 1])<1e-12):
            spin[2*l] = 0
            spin[2*l+1] = 0
    del l

    return np.array(spin)

class HubbardModel:
    def __init__(self,Np=2.0,a=0.5,k_mesh=31,U=8,iteration=400,err=1e-6,fineness=5):
        """モデルのパラメータの設定

        Args:
            U (float): オンサイト相互作用の大きさ Defaults to 8.0
            Np (float,optional): 単位胞内でのホールの数 Defaults to 2.0
            a: 格子のひずみ具合 Defaults to a=0.5 でひずみ無し
            k_mesh (int,optional): k点の細かさ Defaults to 31.
        """
        self.Np = Np
        self.a = a
        self.k_mesh = k_mesh
        self.k_mesh_fine = k_mesh
        self.U = U

        self.Hamiltonian = partial(op.Hamiltonian,a0 = self.a,Ud = self.U)
        self.SpinCurrent = partial(op.SpinCurrent,a0 = self.a)
        self.Current = partial(op.SpinCurrent,a0 = self.a)
        self.n_orbit = op.n_orbit

        ne1 = Np/4.0 + 0.1
        ne2 = Np/4.0 - 0.1
        self.N_site_scf = np.array([
                        [ne1,ne2,0,0,0,0,ne2,ne1,0,0,0,0]])
        self.Ef_scf  = np.array([0])
        self.Delta_scf = np.array([-0.2])
        self.Etot_scf = np.array([0.8])

        self.delta = 0
        self.ef  = 0

        self.enes  = 0
        self.eigenStates = 0
        self.spins = 0

        self.E          = 0
        self.dos        = np.array([])

        self.kF_index = np.array([[-1,-1,-1]])

        self._calc_scf(iteration ,err)
        self._calc_nscf(fineness)


    def _calc_scf(self,iteration = 400,err = 1e-6):
        """自己無頓着計算を行う。delta と ef を決定する。

        Args:
            iteration (int,optional): 繰り返す回数の上限. Defaults to 100.
            err (float,optional): 収束条件. Defaults to 1e-6.
        """

        # 一度やったらもうやらない。
        if(self.Ef_scf.size > 1):
            print("SCF calculation was already done.")
            return

        print("SCF calculation start. Np = {:1.2f}, Ud = {:1.2f}, err < {:1.1e}".format(self.Np,self.U,err))

        kx,ky = self._gen_kmesh()

        # ここから自己無頓着方程式のループになる
        for scf_iteration in range(1, iteration+1):
            # Steffensen の反復法
            for m in range(3):
                # フェルミエネルギーを求める
                enes = []
                eigenEnes = np.zeros((self.k_mesh,self.k_mesh,self.n_orbit*2))
                eigenStates = np.zeros((self.k_mesh,self.k_mesh,self.n_orbit*2,self.n_orbit*2),dtype=np.complex128)
                Delta   = _delta(self.N_site_scf[-1])

                # ブリュアンゾーン内の全探査
                for i in range(self.k_mesh):
                    for j in range(self.k_mesh):
                        eigenEnergy,eigenState = self.Hamiltonian(kx[i][j],ky[i][j],Delta)
                        enes = np.append(enes,eigenEnergy)
                        eigenEnes[i,j] = eigenEnergy
                        eigenStates[i,j] = eigenState
                del i,j

                # 求めたエネルギー固有値をソートして下から何番目というのを探してやればよい
                # 絶縁体のときのことを考えると平均をとる必要がある
                sorted_enes = np.sort(enes)
                ef = (sorted_enes[int(self.k_mesh * self.k_mesh * self.Np) - 1]
                      + sorted_enes[int(self.k_mesh * self.k_mesh * self.Np)])/2
                self.Ef_scf = np.append(self.Ef_scf,ef)

                # scf で求める値の初期化
                nsite  = np.zeros((self.n_orbit*2))
                etot   = 0

                nsite += sum(np.abs(eigenStates[i,j][:,l])**2
                    for i in range(self.k_mesh)
                    for j in range(self.k_mesh)
                    for l in range(self.n_orbit*2)
                    if eigenEnes[i,j,l] <= ef)

                etot += sum(eigenEnes[i,j,l]
                        for i in range(self.k_mesh)
                        for j in range(self.k_mesh)
                        for l in range(self.n_orbit*2)
                        if eigenEnes[i,j,l] <= ef)

                # 規格化して足す
                nsite /= self.k_mesh * self.k_mesh
                self.N_site_scf = np.vstack((self.N_site_scf,nsite))

                self.Delta_scf = np.append(self.Delta_scf,_delta(nsite))

                etot /= self.k_mesh * self.k_mesh
                self.Etot_scf = np.append(self.Etot_scf,etot)


            del m

            # Steffensen の反復法
            ef = _Steffensen(self.Ef_scf)
            self.Ef_scf = np.append(self.Ef_scf,ef)

            delta = _Steffensen(self.Delta_scf)
            self.Delta_scf = np.append(self.Delta_scf,delta)

            etot = _Steffensen(self.Etot_scf)
            self.Etot_scf = np.append(self.Etot_scf,etot)

            nsite = _Steffensen(self.N_site_scf)
            self.N_site_scf = np.vstack((self.N_site_scf,nsite))


            # 与えられた誤差の範囲に収まったら終了する
            if(np.abs(self.Delta_scf[-1]-self.Delta_scf[-2]) < err) :

                self.delta = self.Delta_scf[-1]
                self.ef    = self.Ef_scf[-1]

                print("SCF loop converged.  Np = {:1.2f}, err < {:1.1e}, loop = {:2d}, delta = {:1.2e}\n"
                      .format(self.Np,err,scf_iteration*3,self.delta))

                return

        del scf_iteration

        # 収束しなかったときの処理
        self.delta = self.Delta_scf[-1]
        self.ef    = self.Ef_scf[-1]
        print('\033[41m'+"Calculation didn't converge. err > {:1.1e},Np = {:1.2f} loop = {:2d},delta = {:1.2e}"
              .format(err,self. Np,iteration*3,self.delta)+'\033[0m')
        print(f"latter deltas are {self.Delta_scf[-4:-1]}\n")

        return


    def _calc_nscf(self,fineness=5):
        """
        delta と ef が与えられたときの各k点の固有状態のエネルギー、状態ベクトル、スピンの大きさの計算をする

        Args:
            fineness (int,optional): k_mesh の倍率. Defaults to 5.
        """

        print("NSCF calculation start.")

        if(self.enes == 0):
            self.k_mesh *= fineness
        self.enes = np.zeros((self.k_mesh,self.k_mesh,self.n_orbit*2))
        self.eigenStates = np.zeros((self.k_mesh,self.k_mesh,self.n_orbit*2,self.n_orbit*2),dtype=np.complex128)
        self.spins = np.zeros((self.k_mesh,self.k_mesh,self.n_orbit*2))

        # ブリュアンゾーンのメッシュの生成
        kx,ky = self._gen_kmesh()

        # メッシュの各点でのエネルギー固有値の計算
        for i in range(self.k_mesh):
            for j in range(self.k_mesh):
                enes,eigenstate = self.Hamiltonian(kx[i][j],ky[i][j],self.delta)
                spin = _spin(enes,eigenstate)
                self.enes[i,j]         = enes
                self.eigenStates[i,j]  = eigenstate
                self.spins[i,j]        = np.array(spin)
        del i,j

        print("NSCF calculation finished.\n")
        return

    def _gen_kmesh(self):
        kx = np.linspace(- np.pi,np.pi,self.k_mesh)
        ky = np.linspace(- np.pi,np.pi,self.k_mesh)
        return(np.meshgrid(kx, ky))
