#------------------------------------------------------------------------------
# 第24章
# 身長と体重の5個のデータから、
# 
#   αに関するt変量
#   βに関するt変量
#   αハット
#   βハット
#   χ2乗変数
#   回帰方程式の標準誤差の2乗
#
# を計算する。これを、1万回繰り返すことにより得られる、上記の変数のデータから
# 6種のヒストグラムを描く
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

# 本プログラムでは、statsmodelではなく、sklearnを使ってみる
# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
clf = linear_model.LinearRegression()
X=pd.DataFrame([150, 160, 170, 180],columns=['説明変数']).values # 教科書にあわせた
alpha=-80
beta=0.8
sigma=8
hozon=pd.DataFrame([])
for i in range(10000):
    Y=pd.DataFrame([ alpha+beta*i + normal(0,sigma) for i in list(X)],columns=['目的変数']).values
    clf.fit(X, Y)
    a=clf.intercept_
    b=clf.coef_.flatten() # 2次元配列を1次元配列にしておく
    Y_hat=clf.predict(X)
    e=Y-Y_hat
    sigma_sq_hat=np.dot(e.T,e)/(len(Y)-2) # sigma_sq_hatの期待値はsigma**2
    t_beta=(b-beta)/np.sqrt(sigma_sq_hat/(np.dot((X-np.mean(X)).T,(X-np.mean(X)))))
    t_alpha=(a-alpha)/np.sqrt(sigma_sq_hat*(1/len(Y)+
            np.mean(X)**2/(np.dot((X-np.mean(X)).T,(X-np.mean(X))))))
    chai=sigma_sq_hat*(len(Y)-2)/sigma**2 # chiの期待値は 5-2=3
    save=pd.DataFrame([[a[0],b[0],
                        sigma_sq_hat[0,0],chai[0,0],
                        t_beta[0,0],t_alpha[0,0]]],
                        columns=['定数項','傾き',
                                '回帰方程式の標準誤差の2乗','χ2乗',
                                'βに関するt変量','αに関するt変量'])

    hozon=hozon.append(save)

#------------------------------------------------------------------------------------
# 以下、グラフの描画設定と描画
#------------------------------------------------------------------------------------

# グラフ全体のフォント指定
fsz=10                 # 図全体のフォントサイズ
fti=np.floor(fsz*1.2)  # 図タイトルのフォントサイズ
flg=np.floor(fsz*0.5)  # 凡例のフォントサイズ
flgti=flg              # 凡例のタイトルのフォントサイズ

plt.rcParams["font.size"] = fsz # 図全体のフォントサイズ指定
#plt.rcParams['font.family'] ='sans-serif' # 図全体のフォント
plt.rcParams['font.family'] ='IPAexGothic' # 図全体のフォント

# ヒストグラム描画
figure=plt.figure(figsize=(8,5),tight_layout=True)

ax1=figure.add_subplot(3,2,1,title='αに関するt変量(自由度:n-2=2)')
ax2=figure.add_subplot(3,2,2,title='βに関するt変量(自由度:n-2=2)')
ax3=figure.add_subplot(3,2,3,title='αハット')
ax4=figure.add_subplot(3,2,4,title='βハット')
ax5=figure.add_subplot(3,2,5,title='χ2乗(自由度:n-2=2)')
ax6=figure.add_subplot(3,2,6,title='回帰方程式の標準誤差の2乗')

ax1.set_xlim(-20,20)
ax2.set_xlim(-20,20)
ax3.axvline(x=-80,c='blue')
ax4.axvline(x=0.8,c='blue')
ax6.axvline(x=64,c='blue')

hozon['αに関するt変量'].hist(bins=200,grid=False,ax=ax1)
hozon['βに関するt変量'].hist(bins=200,grid=False,ax=ax2)
hozon['定数項'].hist(bins=200,grid=False,ax=ax3)
hozon['傾き'].hist(bins=200,grid=False,ax=ax4)
hozon['χ2乗'].hist(bins=200,grid=False,ax=ax5)
hozon['回帰方程式の標準誤差の2乗'].hist(bins=200,grid=False,ax=ax6)

plt.show()
