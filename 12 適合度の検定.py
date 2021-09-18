#---------------------------------------------------------------------------------------------------
# 第12章
# サイコロの目を題材にして、χ2乗変数の分布を示す
#---------------------------------------------------------------------------------------------------

from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *

import matplotlib as mpl
import matplotlib.pyplot as plt

n=100
n_chi=10000

# サイコロを転がすための準備
saikoro=pd.Series([1,2,3,4,5,6]) # Seriesを作る

# サイコロをn回転がすということをn_chi回繰り返す
dice=pd.DataFrame(saikoro.sample(n*n_chi,replace=True).values.reshape(n_chi,n))

# それぞれの目が何回出たか記録するための準備
obs_head=pd.DataFrame(columns=[1,2,3,4,5,6])

dice_summary=lambda x:x.value_counts() # それぞれの目が何回出たか、行ごとにカウントし出力

obs=dice.apply(dice_summary,axis=1) # n回サイコロを振るごとに、目の度数を計算する

# 任意の目が出ていない場合の対策
obs=pd.concat([obs_head,obs])

# 欠損値対策
obs.fillna(0,inplace=True)

# chi 2乗の計算
ex=n*(1/6) # 期待度数の計算
chi_square=lambda x:(   (x[1]-ex)**2/ex+(x[2]-ex)**2/ex+(x[3]-ex)**2/ex+
                        (x[4]-ex)**2/ex+(x[5]-ex)**2/ex+(x[6]-ex)**2/ex
                    )

chi2=obs.apply(chi_square,axis=1)

# chi2の実現値のデータのパーセンタイルを返す
print(chi2.describe(percentiles=[0.90,0.95,0.99])) 

print(stats.chi2.ppf(0.95, 6-1)) # χ2乗変数の右裾5％点を返す

# 図示
chi2.plot(kind='hist',bins=25,title='サイコロの適合度の検定に使われるカイ2乗変数のヒストグラム(n=%s)' % n)
plt.show()
