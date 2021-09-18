#---------------------------------------------------------------------------------------------------
# 第10章
# 帰無仮説下のN/nの分布と対立仮説下のN/nの分布を同時に描き
# 第1種の過誤の確率と第2種の過誤の確率とを図示する
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats

def N_over_n( p0,p1,n ,number):

    #===========================
    # N/nの分布（帰無仮説の下）

    x = [i/n for i in range(n+1)]

    # 帰無仮説下の2項分布
    binom=stats.binom.pmf([i for i in range(n+1)], n, p0) #pdfではなく、pmf

    # 帰無仮説下の分布をbinom0に保管
    binom0=binom

    # 棄却限界値
    hidari=p0-1.96*np.sqrt(p0*(1-p0)/n)
    migi  =p0+1.96*np.sqrt(p0*(1-p0)/n)

    fig=plt.figure(number,figsize=(8,5),tight_layout=True)
    ax1=fig.add_subplot(2,1,1) # fig.add_subplot(m,n,l) m×nのキャンバスのl番目

    ax1.plot(x, binom,color='r',label='帰無仮説:p=%s' % p0) # 2項分布描画（ｘは変換済み）
    ax1.fill_between(x,0,binom,where=x<=hidari,color="r",alpha=0.5) # 左裾着色
    ax1.fill_between(x,0,binom,where=x>=migi,color="r",alpha=0.5)   # 右裾着色

    #====================================
    # N/nの分布（対立仮説の下）

    # 対立仮説下の2項分布
    binom=stats.binom.pmf([i for i in range(n+1)], n, p1) #pdfではなく、pmf

    # 対立仮説下の分布をbinom1に保管
    binom1=binom

    ax1.plot(x, binom,color='blue',label='対立仮説:p=%s' % p1) # 2項分布描画（ｘは変換済み）
    ax1.fill_between(x,0,binom,where=((x>hidari)&(x<migi)),color="blue",alpha=0.5) # (hidari,migi)区間の着色

    ax1.set_title('第1種の過誤の確率と第2種の過誤の確率(n=%s)' % n)

    ax1.legend(loc='upper right',fontsize=10)

    #plt.show()

    all=pd.DataFrame({'x':x,'帰無仮説下の二項分布':binom0,'対立仮説下の二項分布':binom1})

    #まとめの表
    c00=all[(all.x>=hidari)&(all.x<=migi)]['帰無仮説下の二項分布'].sum()
    c01=all[(all.x<hidari)|(all.x>migi)]['帰無仮説下の二項分布'].sum()
    
    c10=all[(all.x>=hidari)&(all.x<=migi)]['対立仮説下の二項分布'].sum()
    c11=all[(all.x<hidari)|(all.x>migi)]['対立仮説下の二項分布'].sum()

    #print(c00,c01,c10,c11)

    ax2=fig.add_subplot(2,1,2)
    data = {"A":3.2,"B":2.1,"C":1.2,"D":0.5,"E":0.2,"F":0.1}

    #表を描写
    ax2.table(
            cellText=[['帰無仮説',c00,c01],['対立仮説',c10,c11]],
            cellColours=[['white','white','lightcoral'],['white','cornflowerblue','white']],
            colLabels=['真の世界↓','帰無仮説が正しいと判断','対立仮説が正しいと判断'],
            loc='center')   

    ax2.axis('off')

    plt.show()

N_over_n( 0.5, 0.6, 100 ,1)
N_over_n( 0.5, 0.6, 500 ,2)
N_over_n( 0.5, 0.6, 1000,3)
N_over_n( 0.5, 0.4, 100 ,4)
N_over_n( 0.5, 0.7, 100 ,5)
N_over_n( 0.5, 0.9, 100 ,6)
N_over_n( 0.6, 0.5, 100 ,7)
N_over_n( 0.6, 0.7, 100 ,8)
N_over_n( 0.6, 0.9, 100 ,9)
