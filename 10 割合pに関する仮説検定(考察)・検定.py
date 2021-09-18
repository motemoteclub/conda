#---------------------------------------------------------------------------------------------------
# ��10��
# �A����������N/n�̕��z�ƑΗ���������N/n�̕��z�𓯎��ɕ`��
# ��1��̉ߌ�̊m���Ƒ�2��̉ߌ�̊m���Ƃ�}������
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats

def N_over_n( p0,p1,n ,number):

    #===========================
    # N/n�̕��z�i�A�������̉��j

    x = [i/n for i in range(n+1)]

    # �A����������2�����z
    binom=stats.binom.pmf([i for i in range(n+1)], n, p0) #pdf�ł͂Ȃ��Apmf

    # �A���������̕��z��binom0�ɕۊ�
    binom0=binom

    # ���p���E�l
    hidari=p0-1.96*np.sqrt(p0*(1-p0)/n)
    migi  =p0+1.96*np.sqrt(p0*(1-p0)/n)

    fig=plt.figure(number,figsize=(8,5),tight_layout=True)
    ax1=fig.add_subplot(2,1,1) # fig.add_subplot(m,n,l) m�~n�̃L�����o�X��l�Ԗ�

    ax1.plot(x, binom,color='r',label='�A������:p=%s' % p0) # 2�����z�`��i���͕ϊ��ς݁j
    ax1.fill_between(x,0,binom,where=x<=hidari,color="r",alpha=0.5) # �������F
    ax1.fill_between(x,0,binom,where=x>=migi,color="r",alpha=0.5)   # �E�����F

    #====================================
    # N/n�̕��z�i�Η������̉��j

    # �Η���������2�����z
    binom=stats.binom.pmf([i for i in range(n+1)], n, p1) #pdf�ł͂Ȃ��Apmf

    # �Η��������̕��z��binom1�ɕۊ�
    binom1=binom

    ax1.plot(x, binom,color='blue',label='�Η�����:p=%s' % p1) # 2�����z�`��i���͕ϊ��ς݁j
    ax1.fill_between(x,0,binom,where=((x>hidari)&(x<migi)),color="blue",alpha=0.5) # (hidari,migi)��Ԃ̒��F

    ax1.set_title('��1��̉ߌ�̊m���Ƒ�2��̉ߌ�̊m��(n=%s)' % n)

    ax1.legend(loc='upper right',fontsize=10)

    #plt.show()

    all=pd.DataFrame({'x':x,'�A���������̓񍀕��z':binom0,'�Η��������̓񍀕��z':binom1})

    #�܂Ƃ߂̕\
    c00=all[(all.x>=hidari)&(all.x<=migi)]['�A���������̓񍀕��z'].sum()
    c01=all[(all.x<hidari)|(all.x>migi)]['�A���������̓񍀕��z'].sum()
    
    c10=all[(all.x>=hidari)&(all.x<=migi)]['�Η��������̓񍀕��z'].sum()
    c11=all[(all.x<hidari)|(all.x>migi)]['�Η��������̓񍀕��z'].sum()

    #print(c00,c01,c10,c11)

    ax2=fig.add_subplot(2,1,2)
    data = {"A":3.2,"B":2.1,"C":1.2,"D":0.5,"E":0.2,"F":0.1}

    #�\��`��
    ax2.table(
            cellText=[['�A������',c00,c01],['�Η�����',c10,c11]],
            cellColours=[['white','white','lightcoral'],['white','cornflowerblue','white']],
            colLabels=['�^�̐��E��','�A���������������Ɣ��f','�Η��������������Ɣ��f'],
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
