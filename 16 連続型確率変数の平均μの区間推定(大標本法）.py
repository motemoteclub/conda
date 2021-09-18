#------------------------------------------------------------------------------
# 16章
# 大標本法によるμの区間推定のシミュレーション
# 　元の確率変数が①一様分布の場合と、②平均0.5、分散1/12の正規分布の場合で行う
#
#   daihyouhon_heikin_kukansuitei(2,'一様分布') # データ数が小さいとボロボロ
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kukansuitei(n, bunpu):

    # (0,1)区間の一様確率変数の平均と分散
    myu=1/2
    sigma2=1/12    
    noexp=100000 # 平均、標本分散、χ2乗を何個作るか
    if bunpu=='一様分布':
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='正規分布':
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('分布間違ってますよ')
        stop    
    heikin=pd.DataFrame(itiyou.mean(1),columns=['標本平均']) # 標本平均をnoexpだけ作成する    
    bunsan=pd.DataFrame(itiyou.var(1),columns=['標本分散']) # 標本分散をnoexpだけ作成する
    std_error=np.sqrt(bunsan/n) # 標本平均の標準誤差を計算する
    std_error=std_error.rename(columns={'標本分散':'標本平均の標準誤差'})
    hidari=heikin['標本平均']-1.96*std_error['標本平均の標準誤差']
    migi=heikin['標本平均']+1.96*std_error['標本平均の標準誤差']
    c_interval=pd.DataFrame({'下側信頼限界':hidari,'上側信頼限界':migi})
    all=pd.concat([heikin,bunsan,std_error,c_interval],axis=1)
    all['元の確率変数']=itiyou.loc[:,0]
    all['標本平均の標準化']=(all['標本平均']-myu)/all['標本平均の標準誤差']

    # 信頼区間が myu を含むか否か
    hantei=(lambda x: '成功：myuを含んでいる' if (myu>=x['下側信頼限界']) & 
              (myu<=x['上側信頼限界']) else '失敗：myuを含んでいない')
    all[ '信頼区間が myu を含むか否か']=all.apply(hantei,axis=1)
    result=pd.crosstab(all['信頼区間が myu を含むか否か'],columns='実数') # 信頼区間が myu を含むか否かのクロス表

    figure=plt.figure(figsize=(8,5),tight_layout=True)
    axes_1 = plt.subplot2grid((2,2),(0,0))
    axes_2 = plt.subplot2grid((2,2),(1,0))
    axes_4 = plt.subplot2grid((2,2),(0,1),rowspan=2)

    # 元の確率変数のヒストグラム
    axes_1.hist(all['元の確率変数'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('元の確率変数',loc='center',fontsize=24)
    axes_1.legend()

    # 標本平均の標準化されたもののヒストグラム
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_2.hist(all['標本平均の標準化'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='b',label='男性')
    axes_2.set_title('標本平均の標準化されたもの\n(n=%i)' %n,loc='center',fontsize=24) #図タイトルの位置とサイズ
    str1='失敗：μを含んでいない:  '+str(result.iloc[0,0])+'回'
    str2='成功：μを含んでいる　:　'+str(result.iloc[1,0])+'回'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    (axes_4.set_title('大標本法による信頼区間が\nμを含んでいるか否か\n(%d回の区間推定,n=%d)' %(noexp,n),
     loc='center',fontsize=15))
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()    
    plt.show()

daihyouhon_heikin_kukansuitei(2,'一様分布') # データ数が小さいとボロボロ
daihyouhon_heikin_kukansuitei(2,'正規分布') # データ数が小さいとボロボロ

daihyouhon_heikin_kukansuitei(30,'一様分布') # データ数が30になるともうほぼ理論どおり
daihyouhon_heikin_kukansuitei(30,'正規分布') # データ数が30になるともうほぼ理論どおり

daihyouhon_heikin_kukansuitei(100,'一様分布') # 完璧
daihyouhon_heikin_kukansuitei(100,'正規分布') # 完璧
