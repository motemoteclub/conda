#------------------------------------------------------------------------------
# 第18章
# 小標本法によるμの区間推定のシミュレーション
# 　元の確率変数は②平均0.5、分散1/12の正規分布の場合で行う
#
#   daihyouhon_heikin_kukansuitei(2,'正規分布') # サンプルサイズが小さくともいける
#
#   daihyouhon_heikin_kukansuitei(3,'正規分布') # サンプルサイズが小さくともいける
#
#   daihyouhon_heikin_kukansuitei(100,'正規分布') # サンプルサイズが大きくなるとt分布による検定と
#                                                 # 大標本法による検定はほぼ同じ
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kukansuitei(n, bunpu):
    myu=1/2
    sigma2=1/12
    noexp=100000
    if bunpu=='一様分布':
        # 一様確率変数を(n*noexp)回、ころがして、noexp行、n列の行列に記録する
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='正規分布':
        # 平均myu、分散 1/12 の正規乱数をn*noexpだけ作る
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('分布間違ってますよ')
        stop
    heikin=pd.DataFrame(itiyou.mean(1),columns=['標本平均'])
    bunsan=pd.DataFrame(itiyou.var(1),columns=['標本分散'])
    std_error=np.sqrt(bunsan/n)
    std_error=std_error.rename(columns={'標本分散':'標本平均の標準誤差'})
    t0=stats.t.ppf(q=0.975, df=n-1)
    hidari=heikin['標本平均']-t0*std_error['標本平均の標準誤差']
    migi=heikin['標本平均']+t0*std_error['標本平均の標準誤差']
    c_interval=pd.DataFrame({'下側信頼限界':hidari,'上側信頼限界':migi})
    all=pd.concat([heikin,bunsan,std_error,c_interval],axis=1)
    all['元の確率変数']=itiyou.loc[:,0]
    all['標本平均の標準化']=(all['標本平均']-myu)/all['標本平均の標準誤差']
    estimate=(lambda x: '成功：myuを含んでいる' if (myu>=x['下側信頼限界'])&(myu<=x['上側信頼限界']) 
                                                else '失敗：myuを含んでいない')
    all=all.assign( y=all.apply(estimate,axis=1)).rename(columns={'y':'信頼区間が myu を含むか否か'})
    result=pd.crosstab(all['信頼区間が myu を含むか否か'],columns='実数')

    # グラフ全体のフォント指定
    fsz=20                 # 図全体のフォントサイズ
    fti=np.floor(fsz*1.2)  # 図タイトルのフォントサイズ
    flg=np.floor(fsz*0.5)  # 凡例のフォントサイズ
    flgti=flg              # 凡例のタイトルのフォントサイズ

    plt.rcParams["font.size"] = fsz # 図全体のフォントサイズ指定
    plt.rcParams['font.family'] ='IPAexGothic' # 図全体のフォント

    # グラフの配置設定
    figure = plt.figure()
    gs_master = GridSpec(nrows=2, ncols=2, height_ratios=[1, 1],hspace=0.5) #hspaceでスペース作成

    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1,subplot_spec=gs_master[0:1, 0]) #gs_masterで描画位置設定
    axes_1 = figure.add_subplot(gs_1[:, :])

    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1:2, 0])
    axes_2 = figure.add_subplot(gs_2[:, :])

    gs_4 = GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_master[0:2, 1])
    axes_4 = figure.add_subplot(gs_4[:, :])

    axes_1.hist(all['元の確率変数'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('元の確率変数',loc='center',fontsize=fti)
    axes_1.legend()
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_2.hist(all['標本平均の標準化'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='b',label='男性')
    axes_2.set_title('標本平均の標準化されたもの\n(実は自由度(n-1)のt分布と判明　n=%i)' %n,loc='center',fontsize=20) 
    str1='失敗：μを含んでいない:  '+str(result.iloc[0,0])+'回'
    str2='成功：μを含んでいる　:　'+str(result.iloc[1,0])+'回'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    (axes_4.set_title('小標本法による信頼区間が\nμを含んでいるか否か\n(%d回の区間推定,n=%d)' 
         %(noexp,n),loc='center',fontsize=15))
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()
    plt.show()

daihyouhon_heikin_kukansuitei(2,'正規分布') 
daihyouhon_heikin_kukansuitei(3,'正規分布') 
daihyouhon_heikin_kukansuitei(100,'正規分布')
