
# 概要

交代磁性についての Smejkal-Sinova の Phys.Rev.X 12040501 にある Ru2 の模式図を見ると、
銅酸化物の2次元正方格子を歪ませたように見える。
なので Toy model として銅酸化物をひずませた模型を作って、
交代磁性が出るかを確認してみるというアイデアが出てもおかしくはない。
しかし、遠山先生曰くなぜかやった結果というのが出回っていないらしい。

なのでこのモデルを調べて、相図やスピン感受率の計算をするという目標である。

# 構造

フォルダの構造は以下のとおりである。

``` sh
AM_cupurate
│  CuO2.ipynb           # 簡単なノート
│  readme.md            # これ
│  run.ipynb            # 実行テンプレート
│
├─image         # PowerPoint 等で策表する図表
│
├─model
│  │  calc.py                   # 物理量の計算
│  │  hubbard_model.py          # 平均場ハバード模型の自己無撞着計算と変数
│  │  operators.py              # 平均場ハミルトニアンと流れ演算子
│  │  plot.py                   # 計算結果のグラフの出力
│  │  __init__.py
│
├─output            # 計算結果の図の保存先
│
├─util
│  │  post.py           # Discord への通知
│  │  __init__.py
│
└─資料          # 関連しそうな文献
    ├─交代磁性
    │
    ├─動的スピン感受率
    │
    └─銅酸化物
```

# 使い方

こんな感じで計算できる

``` python
CuO2 = HubbardModel(Np=2.1, a = 0.45, k_mesh=31, U=8, iteration=400, err=1e-6, fineness=5)
plot.fermi_surface(CuO2)
plot.dos(CuO2)
plot.nsite(CuO2)
plot.scf(CuO2)
plot.band(CuO2)
```
