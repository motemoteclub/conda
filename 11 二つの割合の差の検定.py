#---------------------------------------------------------------------------------------------------
# 11章
# 二つの割合の差の検定で使われる定理を確認するプログラム
#
# 定理：
# ( N1/n1 - N2/n2 )は平均 p1-p2、分散 p1(1-p1)/n1 + p2(1-p2)/n2 の正規分布に従う。
#
# パラメータ
# n1, p1, n2, p2
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats

# パラメータ設定
n1=300
p1=0.7

n2=200
p2=0.5

# 平均の差を作る回数
n_sinbunsya=10000 # 2つの平均の差を作る回数

# 対応する正規分布のパラメータ
mean=p1-p2
std=np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

# 赤鉛筆の結果から、首相支持、不支持の変数をつくるラムダ関数の定義
NHK1 = lambda x: 1 if x<p1 else 0
NHK2 = lambda x: 1 if x<p2 else 0

# 赤鉛筆を(n*n_sinbunsya)回、ころがして、n_sinbunsya行、n列の行列に記録する
aka1=pd.DataFrame(np.random.rand(n1*n_sinbunsya).reshape(n_sinbunsya,n1) )
aka2=pd.DataFrame(np.random.rand(n2*n_sinbunsya).reshape(n_sinbunsya,n2) )

# 赤鉛筆の結果から、首相支持、不支持の調査結果を得る。
# さらにそれを、n_sinbunsya行、n列の行列に記録する。
tyousa_kekka1=aka1.applymap(NHK1)
tyousa_kekka2=aka2.applymap(NHK2)

# 新聞社ごとに点推定を行う
tensuitei1=pd.DataFrame(tyousa_kekka1.mean(1),columns=['点推定'])
tensuitei2=pd.DataFrame(tyousa_kekka2.mean(1),columns=['点推定'])

# 二つの割合の差を計算する
sa=tensuitei1-tensuitei2
sa=sa.rename(columns={'点推定':'二つの割合の差'})
#sa.drop('点推定',axis=1,inplace=True)

#対応する正規乱数を作る
seikiransu=pd.DataFrame(normal(mean,std,n_sinbunsya),columns=['正規乱数'])

# 合体させる
result=sa.join(seikiransu) # 強制合体。共通の変数がない場合はこれが便利

# 点推定から、ヒストグラム作成(割合の差の分布と、それに対応する正規分布がほぼ等しいこと)
fig=plt.figure(figsize=(8,5),tight_layout=True)
ax1=fig.add_subplot(1,2,1) # fig.add_subplot(m,n,l) m×nのキャンバスのl番目
ax2=fig.add_subplot(1,2,2) 

result['二つの割合の差'].plot(kind='hist',bins=100,range=(0,1),ax=ax1)
result['正規乱数'].plot(kind='hist',bins=100,range=(0,1),ax=ax2)

ax1.set_title('二つの割合の差(N1/n1 - N2/n2)のヒストグラム\n（n1=%s, p1=%s,n2=%s,p2=%s）' 
% (n1,p1,n2,p2))
ax2.set_title(
'正規乱数のヒストグラム\n（平均=(p1-p2), 標準偏差=sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)')

plt.show()

#=============================================================
# 検定統計量の計算とヒストグラム
# 標準誤差の計算
p=(tyousa_kekka1.sum(1)+tyousa_kekka2.sum(1))/(n1+n2)
std=np.sqrt(p*(1-p)*(1/n1+1/n2))

# 検定統計量の計算
kentei=(sa['二つの割合の差']-(p1-p2))/std

# 描画
fig2=plt.figure(figsize=(8,5),tight_layout=True)
ax3=fig2.add_subplot(1,1,1)
ax3.set_title('検定統計量のヒストグラム')
ax3.grid(False) # グリッド線消去
ax3.hist(kentei.values, bins=69)
plt.show()
