#------------------------------------------------------------------------------
# 第19章
# 小標本法によるμの検定のシミュレーション
# 　元の確率変数が、平均0.5、分散1/12の正規分布の場合で行う
# 　検定回数:10万回
# 　検定統計量は以下のとおり。
#        z=(標本平均-myu)/(標本平均の標準誤差)
#   棄却域は、-t0より小さいかt0より大きい。
#
#   daihyouhon_heikin_kentei(2,'正規分布')   # サンプルサイズが小さくてもいける
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kentei(n,bunpu):
    noexp=100000
    myu=1/2
    sigma2=1/12

    if bunpu=='一様分布':
        # 一様確率変数を(n*noexp)回、ころがして、noexp行、n列の行列に記録する
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='正規分布':
        # 平均myu、分散 1/12 の正規乱数をn*noexpだけ作る
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('分布間違ってますよ')
        stop

    # 帰無仮説におけるμの値(以下の設定では、帰無仮説が真の世界と一致している)
    myu0=myu
    heikin=pd.DataFrame(itiyou.mean(1),columns=['標本平均'])
    bunsan=pd.DataFrame(itiyou.var(1),columns=['標本分散'])
    std_error=np.sqrt(bunsan.rename(columns={'標本分散':'標本平均の標準誤差'})/n)
    hidari=-stats.t.ppf(q=0.975, df=n-1)
    migi=stats.t.ppf(q=0.975, df=n-1)
    all=pd.concat([heikin,bunsan,std_error],axis=1)
    all['元の確率変数']=itiyou.loc[:,0]
    z_henkan = lambda x: (x['標本平均']-myu0)/x['標本平均の標準誤差']
    all['z']=all.apply(z_henkan,axis=1)
    kentei=lambda x: '帰無仮説棄却' if (x['z'] < hidari or x['z'] > migi) else '帰無仮説受容'
    #all=all.assign( test=all.apply(kentei,axis=1)).rename(columns={'test':'検定結果'})
    all['検定結果']=all.apply(kentei,axis=1)
    result=pd.crosstab(all['検定結果'],columns='実数')

    # グラフ全体のフォント指定
    fsz=20                 # 図全体のフォントサイズ
    fti=np.floor(fsz*1.2)  # 図タイトルのフォントサイズ

    figure=plt.figure(figsize=(8,5),tight_layout=True)
    axes_1 = figure.add_subplot(2,2,1)
    axes_2 = figure.add_subplot(2,2,2)
    axes_3 = figure.add_subplot(2,2,3)
    axes_4 = figure.add_subplot(2,2,4)
    axes_1.hist(all['元の確率変数'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('元の確率変数' ,loc='center',fontsize=fti)
    axes_1.legend()
    axes_2.hist(all['標本平均'].values, bins=100, alpha=0.3, histtype='stepfilled', color='b',label='X_bar')
    axes_2.set_title('標本平均\n(n=%i)' %n,loc='center',fontsize=fti) #図タイトルの位置とサイズ
    axes_2.legend()
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_3.hist(all['z'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='g',label='z')
    axes_3.set_title('z\n 実は自由度(n-1)のt分布と判明(n=%i)' %n,loc='center',fontsize=15) #図タイトルの位置とサイズ
    axes_3.legend()
    str1='帰無仮説受容：  '+str(result.iloc[0,0])+'回'
    str2='帰無仮説棄却：　'+str(result.iloc[1,0])+'回'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    axes_4.set_title('検定結果（小標本法）\n(%d回検定を行った,n=%d)' %(noexp,n),loc='center',fontsize=20)
    #axes_4.set_xlabel('XXX')
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()
    plt.show()

daihyouhon_heikin_kentei(2,'正規分布')   # サンプルサイズが小さくてもいける
daihyouhon_heikin_kentei(3,'正規分布')   # サンプルサイズが小さくてもいける
daihyouhon_heikin_kentei(100,'正規分布') # 当然、これもいける
