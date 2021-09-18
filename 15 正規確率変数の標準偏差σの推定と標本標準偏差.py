#------------------------------------------------------------------------------
# 15章
# 正規乱数から標本平均、標本分散、χ2乗変数を求める
# これをnoexp回行って、標本平均、標本分散、χ2乗変数のヒストグラムを作成する
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def hyouhon_bunsan_normal(n):

    # 正規乱数のμとσ
    myu=170
    sigma=6

    # 標本数を変えて実験
    # n=2, 3, 4, 10, 100
    #n=2

    # 平均、標本分散、χ2乗を何個作るか
    noexp=10000

    # 正規乱数を(n*noexp)回、ころがして、noexp行、n列の行列に記録する
    seiki=pd.DataFrame(normal(myu,sigma,n*noexp).reshape(noexp,n) )

    # 標本平均をnoexpだけ作成する
    heikin=pd.DataFrame(seiki.mean(1),columns=['標本平均'])

    # 標本分散をnoexpだけ作成する
    bunsan=pd.DataFrame(seiki.var(1),columns=['標本分散'])

    # χ2乗変数をnoexpだけ作成する
    henkan=lambda x: x*(n-1)/(sigma**2)
    chai=bunsan.applymap(henkan).rename(columns={'標本分散':'χ2乗変数'})

    # 標本平均、標本分散、χ2乗変数を一つのDataFrameにまとめる
    matomeru=heikin.join(bunsan).join(chai)

    # 標本平均、標本分散、χ2乗変数のヒストグラム
    fig=plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    kaikyu1=[140+0.6*i for i in range(100)]
    kaikyu2=[0  +4*i   for i in range(100)]

    ax1.hist(matomeru['標本平均'],bins=kaikyu1)
    ax2.hist(matomeru['標本分散'],bins=kaikyu2)

    title1=(
'%s個のデータの標本平均のヒストグラム\n元の確率変数は\n平均=%s,分散=%sの正規確率変数' 
% (n,myu,sigma**2))

    title2='%s個のデータの標本分散のヒストグラム' % n

    ax1.set_title(title1,fontsize=15)
    ax2.set_title(title2,fontsize=15)

    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=20)

    plt.show()

hyouhon_bunsan_normal(2) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(3) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(4) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(10) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(100) # n=2, 3, 4, 10, 100
