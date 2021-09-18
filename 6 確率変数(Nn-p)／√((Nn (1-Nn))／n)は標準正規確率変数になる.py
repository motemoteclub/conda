#---------------------------------------------------------------------------------------------------
# 第6章
# 確率変数(N/n-p)/sqrt((N/n)(1-N/n)/n)は標準正規確率変数になる
#
#   以下の二つの事項をnを変化させながら、確認する
#
#   ・二つの標準化された変数が標準正規確率変数になる様子をグラフ化する
#       (N/n-p)/sqrt(p(1-p)/n)と(N/n-p)/sqrt((N/n)(1-N/n)/n)との比較
#
#   ・95％信頼区間の原形と95％信頼区間がpを含む確率を比較する
#       95％信頼区間の原形: (N/n-1.96sqrt(p(1-p)/n),N/n+1.96sqrt(p(1-p)/n))
#       95％信頼区間      : (N/n-1.96sqrt(N/n(1-N/n)/n),N/n+1.96sqrt(N/n(1-N/n)/n))
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


# nを変えて実験
# n=2, 10, 101, 3000
n=10

# 新聞社の数
n_sinbunsya=10000

# 首相の真の支持率
p=0.7

# 定理5のμとσ(これを正規確率変数の平均と標準偏差にする)
myu=p
sigma=np.sqrt(p*(1-p)/n)

# 赤鉛筆の結果から、首相支持、不支持の変数をつくるラムダ関数の定義
NHK = lambda x: 1 if x<p else 0

# 赤鉛筆を(n*n_sinbunsya)回、ころがして、n_sinbunsya行、n列のDataFrameに記録する
aka=pd.DataFrame(rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

# 赤鉛筆の結果から、首相支持、不支持の調査結果を得る。
# さらにそれを、n_sinbunsya行、n列の行列に記録する。
aka2=aka.applymap(NHK)

# 新聞社ごとに点推定を行う
tensuitei=pd.DataFrame(aka2.mean(axis=1),columns=['点推定'])

#------------------------------------------------------------------------------
# 標準化を行う(ここでは、(N/n-p)/sqrt((N/n)(1-N/n)/n)を計算する)
#------------------------------------------------------------------------------
# 標準化　5章と6章の標準化を行うラムダ関数の定義
z_henkan_chapter5=lambda x: (x-myu)/(np.sqrt(p*(1-p)/n))
z_henkan_chapter6=lambda x: np.nan if x==0 else np.nan if x==1 else (x-myu)/(np.sqrt(x*(1-x)/n))

# 標準化を行う
tensuitei['第5章の標準化']=tensuitei['点推定'].map(z_henkan_chapter5)
tensuitei['第6章の標準化']=tensuitei['点推定'].map(z_henkan_chapter6)

#------------------------------------------------------------------------------
# μが95％信頼区間、95％信頼区間の原形(5章)に入っているか否かのチェック
#------------------------------------------------------------------------------
# μが95％信頼区間、95％信頼区間の原形に入っているか否かをチェックする関数定義
myu_check_chapter5=(lambda x: '入っている' if (myu>=x-1.96*sigma) & (myu<=x+1.96*sigma)
                                else '入っていない！！！' )

myu_check_chapter6=(lambda x: '入っている' if (myu>=x-1.96*np.sqrt(x*(1-x)/n)) & 
                                (myu<=x+1.96*np.sqrt(x*(1-x)/n)) else '入っていない！！！' )

# 新聞社ごとにμが95％信頼区間、あるいは、95％信頼区間の原形に入っているか否かのチェックを実行
tensuitei['μは95%信頼区間の原形に入っているか']=tensuitei['点推定'].map(myu_check_chapter5)
tensuitei['μは95%信頼区間に入っているか']=tensuitei['点推定'].map(myu_check_chapter6)


#------------------------------------------------------------------------------
# グラフ作成・クロス表作成
#------------------------------------------------------------------------------
# 点推定から、ヒストグラム作成（第5章と第6章の標準化の比較）
title='第5章と第6章の標準化の比較(n=%s)' % n
tensuitei[['第5章の標準化','第6章の標準化']].plot(kind='hist',bins=300,range=(-3,3),grid=False,title=title)
plt.show()

# 点推定が区間（μ-1.96σ,μ+1.96σ)に入っているか否かのクロス表
print(pd.crosstab(tensuitei['μは95%信頼区間の原形に入っているか'],
columns='割合(%)',normalize=True)*100)
print(pd.crosstab(tensuitei['μは95%信頼区間に入っているか'],
columns='割合(%)',normalize=True)*100)
