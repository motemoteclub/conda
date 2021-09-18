#------------------------------------------------------------------------------
# 第9章
# 有意水準5％で、検定を行う。
#   ①まともなサイコロで1000回検定
#   ②いかまさサイコロで1000回検定
#   最後に、①と②とをまとめた表を作成する
#
# どのような条件で実験するかは以下のとおり
#
# test(0.5,0.60,100)   n=100で、p0とp1とが遠い値の場合
# test(0.5,0.55,100)   n=100で、p0とp1とが近い値の場合
# test(0.5,0.55,1000)  n=1000で、p0とp1とが近い値の場合
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def test( p0, p1, n ):

    test005=(lambda x:'まとも' if (x>0.5-1.96*np.sqrt(0.5*(1-0.5)/n)) and (x<0.5+1.96*np.sqrt(0.5*(1-0.5)/n))
                               else 'いかさま')
    test010=(lambda x:'まとも' if (x>0.5-1.64*np.sqrt(0.5*(1-0.5)/n)) and (x<0.5+1.64*np.sqrt(0.5*(1-0.5)/n))
                               else 'いかさま')
    aka_matomo =lambda x: 1 if x<p0 else 0
    aka_ikasama=lambda x: 1 if x<p1 else 0

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # まともなサイコロをn回転がして、5％有意水準で検定する。これを1000回行う
    test_result_in_matomo=aka.applymap(aka_matomo).mean(axis=1).map(test005)

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # いかさまサイコロをn回転がして、5％有意水準で検定する。これを1000回行う
    test_result_in_ikasama=aka.applymap(aka_ikasama).mean(axis=1).map(test005)

    # クロス集計
    test_result_in_matomo_crosstab=pd.crosstab(test_result_in_matomo,columns='回数',normalize=False)
    test_result_in_ikasama_crosstab=pd.crosstab(test_result_in_ikasama,columns='回数',normalize=False)

    # 変数名などの整理
    test_result_in_matomo_crosstab['コインの真の性質']='まとも'
    test_result_in_matomo_crosstab['有意水準']='5％'
    test_result_in_matomo_crosstab.index.name='検定結果'
#    test_result_in_matomo_crosstab.reset_index().set_index(['コインの真の性質','有意水準','検定結果'])

    # 変数名などの整理
    test_result_in_ikasama_crosstab['コインの真の性質']='いかさま'
    test_result_in_ikasama_crosstab['有意水準']='5％'
    test_result_in_ikasama_crosstab.index.name='検定結果'
#    test_result_in_ikasama_crosstab.reset_index().set_index(['コインの真の性質','有意水準','検定結果'])

    # ①まともなサイコロで1000回検定し、さらに②いかまさサイコロで1000回検定したうえで
    # 最後に、①と②とをまとめたDataFrameを作成する
    test_result=pd.concat([test_result_in_matomo_crosstab,test_result_in_ikasama_crosstab])

    temp=test_result

    # test_resultのインデックス等を修正する。修正した結果は以下のとおり
    #
    #　　　　　　　　　　　　　　　　　'回数'
    # 'コインの真の性質','検定結果'
    # まとも　　　　　　　まとも        値1
    # まとも　　　　　　　いかさま      値2
    # いかさま　　　　　　まとも        値3
    # いかさま　　　　　　いかさま      値4
    #
    #                                               ↓マルチインデックス設定
    test_result=test_result.reset_index().set_index(['コインの真の性質','検定結果']).loc[
        [('まとも','まとも'),('まとも','いかさま'),('いかさま','まとも'),('いかさま','いかさま')],:]

    # タイトル設定
    titlename='p0='+str(p0)+'  p1='+str(p1)+'  n='+str(n)

    # 1000回の検定結果の描画
    # .pivot_table()の使い方注意（stacked=Trueで累積棒グラフになる）
    test_result.pivot_table(index=['コインの真の性質'],columns='検定結果',values='回数').plot.barh(
        stacked=True,fontsize=20,title=titlename)
    plt.show()
    return test_result

test(0.5,0.60,100)
test(0.5,0.55,100)
test(0.5,0.55,1000)
