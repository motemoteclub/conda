#------------------------------------------------------------------------------
# 第３章
# 点推定と区間推定を多数回行い、点推定のヒストグラムと
# 区間推定の成功の割合を求める
#
# 条件は以下のとおり
# 新聞社の数＝1万
#
#   nの設定↓       pの設定
#   2               0.7
#   30              0.7
#   101             0.7
#   2000            0.7
#   2               0.5
#   30              0.5
#   101             0.5
#   2000            0.5
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
#from statistics import mean, median,variance,stdev
import matplotlib.pyplot as plt

def tensuitei_kukansuitei_imi(n,p):

    # 調査を行う新聞社の数
    n_newspapers=10000

    # 首相を支持するか否かを決定するラムダ関数
    # xは赤鉛筆
    support = lambda x: 1 if x<=p else 0

    # 信頼区間を計算するラムダ関数の定義
    # xはN/nである
    hidari = lambda x: x-1.96*np.sqrt(x*(1-x)/n)
    migi   = lambda x: x+1.96*np.sqrt(x*(1-x)/n)

    # 赤鉛筆を(n_newspapers*n)回、転がして、その結果をn_newspapers行,n列の行列に保管する
    aka=pd.DataFrame(rand(n_newspapers*n).reshape(n_newspapers,n))

    # newspapersの各要素には首相支持なら1，不支持なら0が入る
    newspapers=aka.applymap(support) 

    # 新聞社ごとの点推定
    newspapers=newspapers.assign( tensuitei=newspapers.mean(axis=1) )

    # 95%信頼区間の計算
    newspapers=newspapers.assign( 
                                    hidari=hidari(newspapers.tensuitei),
                                    migi=migi(newspapers.tensuitei)
                                )

    # 信頼区間にpが入っているか否かを判定するラムダ関数の定義
    # x.hidariは左信頼限界、x.migiは右信頼限界
    hantei = lambda x: '成功・区間推定' if (p>=x.hidari) & (p<=x.migi) else '失敗・区間推定'

    # 95%信頼区間にpが含まれているか否かの判定
    newspapers=newspapers.assign( 
                                    judge=newspapers.apply( hantei,axis=1)
                                )

    # 変数名の変更
    newspapers.rename(columns={'tensuitei':'点推定値','hidari':'左信頼限界','migi':'右信頼限界'},inplace=True)

    # 信頼区間の成功、失敗の数
    print(newspapers.judge.value_counts())

    # 点推定値のヒストグラム
    newspapers['点推定値'].plot(kind='hist',bins=[i*0.01 for i in range(101)],title='点推定値のヒストグラム')
    plt.show()

# 実験開始
tensuitei_kukansuitei_imi(2,0.7) # n=2 一つの新聞社の標本数, p=0.7 首相を支持する真の確率

tensuitei_kukansuitei_imi(30,0.7) 
tensuitei_kukansuitei_imi(101,0.7)
tensuitei_kukansuitei_imi(2000,0.7) 
tensuitei_kukansuitei_imi(2,0.5) 
tensuitei_kukansuitei_imi(30,0.5)
tensuitei_kukansuitei_imi(101,0.5) 
tensuitei_kukansuitei_imi(2000,0.5) 
