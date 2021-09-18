#---------------------------------------------------------------------------------------------------
# 第13章 
"""
  分割表における独立性の検定で使われる定理を確認するプログラム(以下の設定はAB型を無視している)

確率の記号(()内はカテゴリ番号)
　--------------------------------------
 |       性格→|　積極的  |　消極的    |合計
 |血液型↓     |          |            |
 --------------------------------------
 | A型         | p11(1)   | p12(2)     |p1_
 | O,B型       | p21(3)   | p22(4)     |p2_
 --------------------------------------
  合計           p_1        p_2          1

観測値の記号
　--------------------------------------
 |       性格→|　積極的  |　消極的    |合計
 |血液型↓     |          |            |
 --------------------------------------
 | A型         | sumy11   | sumy12     |
 | O,B型       | sumy21   | sumy22     |
 --------------------------------------
  合計                                   n|

"""
from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *

import matplotlib as mpl
import matplotlib.pyplot as plt

n=1000
n_chi=10000
p1_ = 0.4  # A  型の真の確率 
p2_ = 0.6  # O,B型の真の確率 
p_1 = 0.2  #　積極的な人の真の確率 
p_2 = 0.8  #　消極的な人の真の確率 
# 以下、血液型と性格が独立な場合の各セルの確率 
p11 = p1_ * p_1  # A  型で積極的な確率 
p12 = p1_ * p_2  # A  型で消極的な確率 
p21 = p2_ * p_1  # O,B型で積極的な確率 
p22 = p2_ * p_2  # O,B型で消極的な確率 

aka=pd.DataFrame(np.random.rand(n*n_chi).reshape(n_chi,n) )
# クロス表のどこに実現するかを決定
to_cross=lambda x: 1 if x<p11 else 2 if x<(p11+p12) else 3 if x<(p11+p12+p21) else 4 
cross=aka.applymap(to_cross)
cross_summary=lambda x:x.value_counts() # それぞれの目が何回出たかカウントし出力
obs=cross.apply(cross_summary,axis=1) # n回の出生ごとに、目の度数を計算する
# それぞれの目が何回出たか記録するための準備
obs_head=pd.DataFrame(columns=[1,2,3,4])
# 任意の目が出ていない場合の対策
obs=pd.concat([obs_head,obs])
# 欠損値対策
obs.fillna(0,inplace=True)
# Qab_1の計算
chi_square=(lambda x: (x[1]-(n*p11))**2/(n*p11)+(x[2]-(n*p12))**2/(n*p12)+(x[3]-(n*p21))**2/(n*p21)+
            (x[4]-(n*p22))**2/(n*p22))
Qab_1=obs.apply(chi_square,axis=1)

# Qab_1_a_1_b_1の計算
chi_square2=(lambda x: (x[1]-(n*((x[1]+x[2])/n)*((x[1]+x[3])/n)))**2/(n*((x[1]+x[2])/n)*((x[1]+x[3])/n))+
                       (x[2]-(n*((x[1]+x[2])/n)*((x[2]+x[4])/n)))**2/(n*((x[1]+x[2])/n)*((x[2]+x[4])/n))+
                       (x[3]-(n*((x[3]+x[4])/n)*((x[1]+x[3])/n)))**2/(n*((x[3]+x[4])/n)*((x[1]+x[3])/n))+
                       (x[4]-(n*((x[3]+x[4])/n)*((x[2]+x[4])/n)))**2/(n*((x[3]+x[4])/n)*((x[2]+x[4])/n)) )
Qab_1_a_1_b_1=obs.apply(chi_square2,axis=1)

# 二つの確率変数　Qab_1　と　Qab_1_a_1_b_1　の理論分布と実現値の分布とを比較する
print(stats.chi2.ppf(0.95, 4-1)) # χ2乗変数の右裾5％点を返す
print(Qab_1.describe(percentiles=[0.90,0.95,0.99])) # パーセンタイルを返す

print(stats.chi2.ppf(0.95, (2-1)*(2-1))) # χ2乗変数の右裾5％点を返す
print(Qab_1_a_1_b_1.describe(percentiles=[0.90,0.95,0.99])) # パーセンタイルを返す

# ヒストグラム描画
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

ax1.set_title('χ2乗（期待度数は真の確率使用）\n n=%s' % n,fontsize=20)
ax2.set_title('χ2乗（期待度数は推定された確率使用）\n n=%s' % n,fontsize=20)

Qab_1.plot(kind='hist',ax=ax1,bins=35)
Qab_1_a_1_b_1.plot(kind='hist',ax=ax2,bins=35)
plt.show()
