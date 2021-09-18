#---------------------------------------------------------------------------------------------------
# ��2��
# �^�̎x�����͈ȉ��́Ap=�Ŏw������B���̏�ŁA
# n�l�̗L���҂Ɏ񑊂��x�����邩�ۂ��������āA�x������̂ł����1�A�x�����Ȃ��̂ł����0�ƂȂ�
# �T���v���𓾂�iasahi_sinbun�j�B���̃T���v�����g���āA�_�����95���M����Ԃ��쐬����isuitei�j�B
#---------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from numpy.random import *
import matplotlib.pyplot as plt

n=10 # �����_���T���v���T�C�Y

akaenpitu=pd.Series(rand(n)) # �ԉ��M��n��A�]����

# �ԉ��M�̌��ʂ���񑊎x���A�s�x�����̒l��Ԃ��֐��Aprime_minister
# �񑊂��x�����遁1�C���Ȃ���0
# �񑊂��x������^�̊m��=p
def prime_minister(R,p):
    if R <= p:
        X = 1
    else :
        X = 0
    return X 

# �����V���̐��_����n�l���̌��ʂ�������
asahi_sinbun=akaenpitu.apply(prime_minister,p=0.6) # apply��map���ƕs��

# �񑊂��x������l�̐^�̊����̓_������P����
asahi_sinbun_tensuitei=asahi_sinbun.mean()

# �M����Ԃ��v�Z���郉���_�֐��̒�`
# x�͓_����łȂ���΂Ȃ�Ȃ�
hidari = lambda x: x-1.96*np.sqrt(x*(1-x)/n)
migi   = lambda x: x+1.96*np.sqrt(x*(1-x)/n)

# �_����A��Ԑ�����܂�DataFrame�̍쐬�F�P�s����Ȃ�
suitei =pd.DataFrame([[asahi_sinbun_tensuitei,hidari(asahi_sinbun_tensuitei),
                       migi(asahi_sinbun_tensuitei)]], 
                       columns=['�_����','���M�����E','�E�M�����E'])

suitei
