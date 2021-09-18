#------------------------------------------------------------------------------
# 15��
# ���K��������W�{���ρA�W�{���U�A��2��ϐ������߂�
# �����noexp��s���āA�W�{���ρA�W�{���U�A��2��ϐ��̃q�X�g�O�������쐬����
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def hyouhon_bunsan_normal(n):

    # ���K�����̃ʂƃ�
    myu=170
    sigma=6

    # �W�{����ς��Ď���
    # n=2, 3, 4, 10, 100
    #n=2

    # ���ρA�W�{���U�A��2�������邩
    noexp=10000

    # ���K������(n*noexp)��A���낪���āAnoexp�s�An��̍s��ɋL�^����
    seiki=pd.DataFrame(normal(myu,sigma,n*noexp).reshape(noexp,n) )

    # �W�{���ς�noexp�����쐬����
    heikin=pd.DataFrame(seiki.mean(1),columns=['�W�{����'])

    # �W�{���U��noexp�����쐬����
    bunsan=pd.DataFrame(seiki.var(1),columns=['�W�{���U'])

    # ��2��ϐ���noexp�����쐬����
    henkan=lambda x: x*(n-1)/(sigma**2)
    chai=bunsan.applymap(henkan).rename(columns={'�W�{���U':'��2��ϐ�'})

    # �W�{���ρA�W�{���U�A��2��ϐ������DataFrame�ɂ܂Ƃ߂�
    matomeru=heikin.join(bunsan).join(chai)

    # �W�{���ρA�W�{���U�A��2��ϐ��̃q�X�g�O����
    fig=plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    kaikyu1=[140+0.6*i for i in range(100)]
    kaikyu2=[0  +4*i   for i in range(100)]

    ax1.hist(matomeru['�W�{����'],bins=kaikyu1)
    ax2.hist(matomeru['�W�{���U'],bins=kaikyu2)

    title1=(
'%s�̃f�[�^�̕W�{���ς̃q�X�g�O����\n���̊m���ϐ���\n����=%s,���U=%s�̐��K�m���ϐ�' 
% (n,myu,sigma**2))

    title2='%s�̃f�[�^�̕W�{���U�̃q�X�g�O����' % n

    ax1.set_title(title1,fontsize=15)
    ax2.set_title(title2,fontsize=15)

    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=20)

    plt.show()

hyouhon_bunsan_normal(2) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(3) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(4) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(10) # n=2, 3, 4, 10, 100

hyouhon_bunsan_normal(100) # n=2, 3, 4, 10, 100
