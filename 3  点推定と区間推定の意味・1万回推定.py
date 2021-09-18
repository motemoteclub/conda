#------------------------------------------------------------------------------
# ��R��
# �_����Ƌ�Ԑ���𑽐���s���A�_����̃q�X�g�O������
# ��Ԑ���̐����̊��������߂�
#
# �����͈ȉ��̂Ƃ���
# �V���Ђ̐���1��
#
#   n�̐ݒ聫       p�̐ݒ�
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

    # �������s���V���Ђ̐�
    n_newspapers=10000

    # �񑊂��x�����邩�ۂ������肷�郉���_�֐�
    # x�͐ԉ��M
    support = lambda x: 1 if x<=p else 0

    # �M����Ԃ��v�Z���郉���_�֐��̒�`
    # x��N/n�ł���
    hidari = lambda x: x-1.96*np.sqrt(x*(1-x)/n)
    migi   = lambda x: x+1.96*np.sqrt(x*(1-x)/n)

    # �ԉ��M��(n_newspapers*n)��A�]�����āA���̌��ʂ�n_newspapers�s,n��̍s��ɕۊǂ���
    aka=pd.DataFrame(rand(n_newspapers*n).reshape(n_newspapers,n))

    # newspapers�̊e�v�f�ɂ͎񑊎x���Ȃ�1�C�s�x���Ȃ�0������
    newspapers=aka.applymap(support) 

    # �V���Ђ��Ƃ̓_����
    newspapers=newspapers.assign( tensuitei=newspapers.mean(axis=1) )

    # 95%�M����Ԃ̌v�Z
    newspapers=newspapers.assign( 
                                    hidari=hidari(newspapers.tensuitei),
                                    migi=migi(newspapers.tensuitei)
                                )

    # �M����Ԃ�p�������Ă��邩�ۂ��𔻒肷�郉���_�֐��̒�`
    # x.hidari�͍��M�����E�Ax.migi�͉E�M�����E
    hantei = lambda x: '�����E��Ԑ���' if (p>=x.hidari) & (p<=x.migi) else '���s�E��Ԑ���'

    # 95%�M����Ԃ�p���܂܂�Ă��邩�ۂ��̔���
    newspapers=newspapers.assign( 
                                    judge=newspapers.apply( hantei,axis=1)
                                )

    # �ϐ����̕ύX
    newspapers.rename(columns={'tensuitei':'�_����l','hidari':'���M�����E','migi':'�E�M�����E'},inplace=True)

    # �M����Ԃ̐����A���s�̐�
    print(newspapers.judge.value_counts())

    # �_����l�̃q�X�g�O����
    newspapers['�_����l'].plot(kind='hist',bins=[i*0.01 for i in range(101)],title='�_����l�̃q�X�g�O����')
    plt.show()

# �����J�n
tensuitei_kukansuitei_imi(2,0.7) # n=2 ��̐V���Ђ̕W�{��, p=0.7 �񑊂��x������^�̊m��

tensuitei_kukansuitei_imi(30,0.7) 
tensuitei_kukansuitei_imi(101,0.7)
tensuitei_kukansuitei_imi(2000,0.7) 
tensuitei_kukansuitei_imi(2,0.5) 
tensuitei_kukansuitei_imi(30,0.5)
tensuitei_kukansuitei_imi(101,0.5) 
tensuitei_kukansuitei_imi(2000,0.5) 
