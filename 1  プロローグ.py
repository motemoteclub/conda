#---------------------------------------------------------------------------------------------------
# 第1章
# 赤鉛筆をn回転がして、頂点から0線まで測定した結果のデータを用いて、ヒストグラムを描く。
# また、普通のサイコロをn回転がして、出た目のデータを用いて、ヒストグラムを描く。
#---------------------------------------------------------------------------------------------------

from numpy.random import *
import pandas as pd
import matplotlib.pyplot as plt

# 赤鉛筆を1回転がす
n=1

R=rand(n)      # 赤鉛筆を転がす

# 赤鉛筆を10000回、転がす
n=10000 

akaenpitu=pd.Series(rand(n))

# ヒストグラムを描いた
akaenpitu.plot(kind='hist',fontsize=20,title='赤鉛筆Rのヒストグラム') 
plt.show()

# サイコロを転がすための準備
saikoro=pd.Series([1,2,3,4,5,6]) # Seriesを作る

# サイコロをn回転がす
dice=saikoro.sample(n,replace=True)

dice.plot(kind='hist', fontsize=20,title='サイコロのヒストグラム',bins=[1,2,3,4,5,6,7])  
plt.show()
