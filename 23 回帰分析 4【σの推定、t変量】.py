#---------------------------------------------------------------------------------------------------
# 第23章
# 身長が155、160、165、170、175センチメートルの人の仮想データ(5個)を作成し、
# 標本相関係数と標本分散を計算する。そして、回帰方程式を求め表示・図示する。
# さらに決定係数を計算し、表示する。
# さらに、σの推定値、αにかんするt変量、βにかんするt変量を計算、表示する。
# 
# 仮想データの作り方
#   -80+0.83身長＋誤差項（誤差項は平均0，標準偏差6.83の正規確率変数）
#---------------------------------------------------------------------------------------------------
from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
# 身長が155、160、165、170、175センチメートルの人の仮想データを作成する。
height_weight=pd.DataFrame( {'身長':[155, 160, 165, 170, 175],
                             '体重':[-80+0.83*i+6.83*normal(0,1) for i in [155, 160, 165, 170, 175] ],
                             '体重・サイコロ振らず':[-80+0.83*i  for i in [155, 160, 165, 170, 175] ]})
print(height_weight.corr())
print(height_weight.cov())
b=height_weight.cov().loc['身長','体重']/(height_weight.var().loc['身長']) # 身長・体重の共分散/身長の分散
a=height_weight.mean().loc['体重']-b*height_weight.mean().loc['身長'] # 体重の平均 - b・身長の平均
height_weight['定数項'] = 1 # sm.add_constant(height_weight)でも可。"const"ができる。
x=height_weight[['定数項','身長']]
y=height_weight[['体重']]
model = sm.OLS(y, x)
results = model.fit() # dir(results)で属性が分かる
print( results.summary() ) # results.scaleはσ2乗の推定値
y_hat=results.fittedvalues
y_souhendo=y.var()*(len(y)-1)          # 総変動(Series)
y_hat_hendo=y_hat.var()*(len(y_hat)-1) # 回帰変動(Series)
zansa_hendo=y_souhendo-y_hat_hendo     # 残差変動
kettei_keisu=y_hat_hendo/y_souhendo
print(kettei_keisu)
#------------------------------------------------------------------------------
# 神様のサイコロの標準偏差の推定値（σハット）
# αに関するｔ変量
# βに関するｔ変量
#------------------------------------------------------------------------------
# σの推定値
sigma_hat=np.sqrt( (results.resid ** 2).sum() / (len(y)-2) )
x_souhendo=(x[['身長']].var()*(len(x)-1))[0] # xの総変動
x_bar=x[['身長']].mean()[0]                  # xの平均
alpha_ni_kansuru_T_henryo=(a-(-80))/np.sqrt((sigma_hat**2)*(1/len(y)+x_bar**2/x_souhendo))
beta_ni_kansuru_T_henryo=(b-0.83)/np.sqrt((sigma_hat**2)/x_souhendo)
print('σハット=',sigma_hat)
print('αに関するｔ変量=',alpha_ni_kansuru_T_henryo)
print('βに関するｔ変量=',beta_ni_kansuru_T_henryo)
#------------------------------------------------------------------------------------
# 以下、グラフの描画設定
#------------------------------------------------------------------------------------
# グラフ全体のフォント指定
fsz=20                 # 図全体のフォントサイズ
fti=np.floor(fsz*1.2)  # 図タイトルのフォントサイズ
flg=np.floor(fsz*0.5)  # 凡例のフォントサイズ
flgti=flg              # 凡例のタイトルのフォントサイズ

plt.rcParams["font.size"] = fsz # 図全体のフォントサイズ指定
plt.rcParams['font.family'] ='IPAexGothic' # 図全体のフォント
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.set_xlabel('身長',fontsize=20)
ax1.set_ylabel('体重',fontsize=20)
ax1.set_ylim(0,120)
ax1.set_title('神様がサイコロを振って作ったデータの描画\n 体重=-80+0.83×身長+サイコロ',fontsize=20)
ax1.scatter(x=height_weight['身長'].values, y=height_weight['体重'].values,s=500)
ax1.plot(height_weight['身長'].values, results.fittedvalues,c='blue',linestyle='solid',
    label='回帰方程式：y = {0:.1f} + {1:.2f}x'.format(a,b))
(ax1.plot(height_weight['身長'].values, height_weight['体重・サイコロ振らず'].values,c='red',
    linestyle='solid',label='y=-80+0.83x'))
ax1.legend(loc='upper right',fontsize=10, title_fontsize=10)
ax2.axis('off') # 図を囲む実線を消去
ax2.set_title('回帰分析の結果',fontsize=30)
ax2.text(0.2, 0.8, '標本相関係数=  {0:.2f}'.format(height_weight.corr().loc['身長','体重']), size = 20)
ax2.text(0.2, 0.7, '標本共分散=  {0:.1f}'.format(height_weight.cov().loc['身長','体重']), size = 20)
ax2.text(0.1, 0.6, '回帰方程式：y = {0:.1f} + {1:.2f}x'.format(a,b), size = 20)
ax2.text(0.1, 0.5, '決定係数={0:.2f}'.format(kettei_keisu[0]), size = 20)
ax2.text(0.1, 0.4, 'σの推定値={0:.2f}'.format(sigma_hat), size = 20, color = "blue")
ax2.text(0.1, 0.3, 'αに関するｔ変量={0:.2f}'.format(alpha_ni_kansuru_T_henryo), size = 20, color = "blue")
ax2.text(0.1, 0.2, 'βに関するｔ変量={0:.2f}'.format(beta_ni_kansuru_T_henryo), size = 20, color = "blue")
plt.show()
