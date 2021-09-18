#---------------------------------------------------------------------------------------------------
# 第2章
# 真の支持率は以下の、p=で指示する。その上で、
# n人の有権者に首相を支持するか否かをきいて、支持するのであれば1、支持しないのであれば0となる
# サンプルを得る（asahi_sinbun）。このサンプルを使って、点推定と95％信頼区間を作成する（suitei）。
#---------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from numpy.random import *
import matplotlib.pyplot as plt

n=10 # ランダムサンプルサイズ

akaenpitu=pd.Series(rand(n)) # 赤鉛筆をn回、転がす

# 赤鉛筆の結果から首相支持、不支持をの値を返す関数、prime_minister
# 首相を支持する＝1，しない＝0
# 首相を支持する真の確率=p
def prime_minister(R,p):
    if R <= p:
        X = 1
    else :
        X = 0
    return X 

# 朝日新聞の世論調査n人分の結果が得られる
asahi_sinbun=akaenpitu.apply(prime_minister,p=0.6) # applyがmapだと不可

# 首相を支持する人の真の割合の点推定を１個得る
asahi_sinbun_tensuitei=asahi_sinbun.mean()

# 信頼区間を計算するラムダ関数の定義
# xは点推定でなければならない
hidari = lambda x: x-1.96*np.sqrt(x*(1-x)/n)
migi   = lambda x: x+1.96*np.sqrt(x*(1-x)/n)

# 点推定、区間推定を含むDataFrameの作成：１行からなる
suitei =pd.DataFrame([[asahi_sinbun_tensuitei,hidari(asahi_sinbun_tensuitei),
                       migi(asahi_sinbun_tensuitei)]], 
                       columns=['点推定','左信頼限界','右信頼限界'])

suitei
