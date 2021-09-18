#---------------------------------------------------------------------------------------------------
# 第5章
# このプログラムはn=2,10,101,3000のおのおのについて以下のことを行う(p=0.7)
#   ・点推定を1万個作成し、そのヒストグラムを描く
#   ・点推定値が区間(p-1.96rmsq(p(1-p)/n), p+1.96rmsq(p(1-p)/n)) の間に入る確率が95%であることの確認
#   ・z=(N/n-p)/sqrt(p(1-p)/n)が標準正規分布になっていることの確認
#
#   点推定量の不偏性と一致性とを確認することが眼目
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


# nを変えて実験
# n=2, 10, 101, 3000
n=101

# 新聞社の数
n_sinbunsya=10000

# 首相の真の支持率
p=0.7

# 定理5のμとσ
myu=p
sigma=np.sqrt(p*(1-p)/n)

# 赤鉛筆の結果から、首相支持、不支持の変数をつくるラムダ関数の定義
NHK = lambda x: 1 if x<p else 0

# 赤鉛筆を(n*n_sinbunsya)回、ころがして、n_sinbunsya行、n列のDataFrameに記録する
aka=pd.DataFrame(np.random.rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

# 赤鉛筆の結果から、首相支持、不支持の調査結果を得る。
# さらにそれを、n_sinbunsya行、n列の行列に記録する。
aka2=aka.applymap(NHK)

# 新聞社ごとに点推定を行う
tensuitei=aka2.mean(1) 

# 点推定が区間（μ-1.96σ,μ+1.96σ)に入っていたら'入っている'、
# そうでなければ'入っていない'となるラムダ関数の定義
in_or_out = lambda x: '入っている' if (x>=myu-1.96*sigma) & (x<=myu+1.96*sigma) else '入っていない'

# 新聞社ごとの点推定が（μ-1.96σ,μ+1.96σ)に入っていたら'入っている' 、そうでなければ'入っていない'とする。
# その結果をtensuitei2に記録する
tensuitei2=tensuitei.map(in_or_out)

# 点推定から、ヒストグラム作成
title='1万個の点推定値から作られたヒストグラム(n=%s)' % n
tensuitei.plot(kind='hist', fontsize=20,title=title,range=(0,1),bins=200)
plt.show()

# z=(N/n-p)/sqrt(p(1-p)/n)が標準正規分布になっていることの確認
z_henkan=lambda x: (x-p)/np.sqrt(p*(1-p)/n)
z=tensuitei.apply(z_henkan)

title='標準化(n=%s)' % n
z.plot(kind='hist', fontsize=20,title=title,range=(-3,3),bins=200)
plt.show()

# 点推定が区間（μ-1.96σ,μ+1.96σ)に入っているか否かのクロス表
pd.crosstab(tensuitei2,columns='割合(%)',normalize=True)*100
