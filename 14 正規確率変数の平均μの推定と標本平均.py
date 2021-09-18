#---------------------------------------------------------------------------------------------------
# 14章
# 赤鉛筆をn回転がして、標本平均を得る
# このような標本平均をn_sinbunsya個作って、ヒストグラムを描く。
# これによって、標本平均の分布を求めることができる
#
# 対応する正規分布のヒストグラムも描く
#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


def hyouhon_heikin_uniform(n):

    # 標本数を変えて実験
    # n=1,2, 10, 100
    #n=10

    # 平均を何個作るか
    n_sinbunsya=100000

    # 赤鉛筆を(n*n_sinbunsya)回、ころがして、n_sinbunsya行、n列の行列に記録する
    aka=pd.DataFrame(np.random.rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

    # 平均をn_sinbunsyaだけ作成する
    heikin=pd.DataFrame(aka.mean(1),columns=['平均'])

    # 平均0.5、分散 (1/12)/nの正規乱数をn_sinbunsyaだけ作る
    # seikiransu=pd.DataFrame([normal(0.5,np.sqrt((1/12)/n)) for i in range(n_sinbunsya)],columns=['対応する正規乱数'])

    myu=0.5
    sigma=np.sqrt((1/12)/n)

    seikiransu=pd.DataFrame(normal(0.5,np.sqrt((1/12)/n),n_sinbunsya),columns=['対応する正規乱数'])

    # 標本数nの一様乱数の平均と平均0.5、分散 (1/12)/nの正規乱数とのヒストグラム作成用データ
    hikaku=heikin.join(seikiransu)

    # 以下、n個の一様乱数から得られた標本平均とそれに対応する正規分布の描写
    # to do:x軸を固定
    fig=plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    kaikyu=[0+0.01*i for i in range(100)]

    ax1.hist(hikaku['平均'],bins=kaikyu)
    ax2.hist(hikaku['対応する正規乱数'],bins=kaikyu,color='blue')

    title1='%s個のデータの標本平均の分布\n元の確率変数は\n0-1区間の一様確率変数' % n
    title2='平均=%s, 分散=1/(12*%s)の正規分布' % (myu, n)

    ax1.set_title(title1,fontsize=15)
    ax2.set_title(title2,fontsize=15)

    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=20)

    plt.show()

hyouhon_heikin_uniform(1)   # 一様乱数の確認。対応する正規分布はｘ軸を切り取っている。

hyouhon_heikin_uniform(2)   # 三角関数。対応する正規分布は、結構0，1区間に入っている

hyouhon_heikin_uniform(10)  # 正規近似がかなり正確になる。「中心極限定理」の紹介。

hyouhon_heikin_uniform(100) # ほぼ完璧な近似。

