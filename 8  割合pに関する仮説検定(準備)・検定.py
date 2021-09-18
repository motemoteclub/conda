#---------------------------------------------------------------------------------------------------
# 第8章
# ゲーマーAとゲーマーBとが、それぞれ、「まともなコイン」で1000ゲーム、「いかさまコイン」で1000ゲームを
# 行う。「いかさまコイン」で表が出る確率pは関数game()の引数p1で与えられる。「まともなコイン」で
# 表が出る確率は関数game()の引数p0で与えられる。コインを投げ上げる回数はgame()の引数nで与えられる。
# 実験は以下の３つの場合について行う。
#
# p0=0.5,p1=0.55,n=100 
# p0=0.5,p1=0.60,n=100 
# p0=0.5,p1=0.55,n=1000
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def game( p0, p1, n ):
    handan_A=(lambda x:'まとも' if (x>=0.5-1.96*np.sqrt(0.5*(1-0.5)/n)) and 
                                (x<=0.5+1.96*np.sqrt(0.5*(1-0.5)/n)) else 'いかさま')
    handan_B=(lambda x:'まとも' if (x>=0.5-1.64*np.sqrt(0.5*(1-0.5)/n)) and 
                                (x<=0.5+1.64*np.sqrt(0.5*(1-0.5)/n)) else 'いかさま')
    aka_sizi_husizi0=lambda x: 1 if x<p0 else 0
    aka_sizi_husizi1=lambda x: 1 if x<p1 else 0

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer Aの判断（p=p0の下での実験）
    gamer_a_p0=aka.applymap(aka_sizi_husizi0).mean(axis=1).map(handan_A)

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer Aの判断（p=p1の下での実験）
    gamer_a_p1=aka.applymap(aka_sizi_husizi1).mean(axis=1).map(handan_A)

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer Bの判断（p=p0の下での実験）
    gamer_b_p0=aka.applymap(aka_sizi_husizi0).mean(axis=1).map(handan_B)

    # 赤鉛筆をn回転がすという実験を 1000 回繰り返す
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer Bの判断（p=p1の下での実験）
    gamer_b_p1=aka.applymap(aka_sizi_husizi1).mean(axis=1).map(handan_B)

    # クロス集計
    gamer_a_p0_crosstab=pd.crosstab(gamer_a_p0,columns='回数',normalize=False)
    gamer_a_p1_crosstab=pd.crosstab(gamer_a_p1,columns='回数',normalize=False)
    gamer_b_p0_crosstab=pd.crosstab(gamer_b_p0,columns='回数',normalize=False)
    gamer_b_p1_crosstab=pd.crosstab(gamer_b_p1,columns='回数',normalize=False)

    # 変数名などの整理
    gamer_a_p0_crosstab['真のサイコロの性質']='まとも'
    gamer_a_p0_crosstab['プレーヤー']='ゲーマーA'
    gamer_a_p0_crosstab.index.name='判断'
    gamer_a_p0_crosstab.reset_index().set_index(['真のサイコロの性質','プレーヤー','判断'])

    gamer_a_p1_crosstab['真のサイコロの性質']='いかさま'
    gamer_a_p1_crosstab['プレーヤー']='ゲーマーA'
    gamer_a_p1_crosstab.index.name='判断'
    gamer_a_p1_crosstab.reset_index().set_index(['真のサイコロの性質','プレーヤー','判断'])

    gamer_b_p0_crosstab['真のサイコロの性質']='まとも'
    gamer_b_p0_crosstab['プレーヤー']='ゲーマーB'
    gamer_b_p0_crosstab.index.name='判断'
    gamer_b_p0_crosstab.reset_index().set_index(['真のサイコロの性質','プレーヤー','判断'])

    gamer_b_p1_crosstab['真のサイコロの性質']='いかさま'
    gamer_b_p1_crosstab['プレーヤー']='ゲーマーB'
    gamer_b_p1_crosstab.index.name='判断'
    gamer_b_p1_crosstab.reset_index().set_index(['真のサイコロの性質','プレーヤー','判断'])

    # まとめのクロス表
    game_result=pd.concat([gamer_a_p0_crosstab,gamer_a_p1_crosstab,gamer_b_p0_crosstab,
                            gamer_b_p1_crosstab])

    titlename='p0='+str(p0)+'  p1='+str(p1)+'  n='+str(n)

    (game_result.reset_index().pivot_table(index=['プレーヤー','真のサイコロの性質'],
        values='回数',columns='判断').plot.barh(stacked=True,fontsize=20,title=titlename))

    return game_result

game(0.5,0.55,100).reset_index()
game(0.5,0.60,100).reset_index()
game(0.5,0.55,1000).reset_index()
