#---------------------------------------------------------------------------------------------------
# ��1��
# �ԉ��M��n��]�����āA���_����0���܂ő��肵�����ʂ̃f�[�^��p���āA�q�X�g�O������`���B
# �܂��A���ʂ̃T�C�R����n��]�����āA�o���ڂ̃f�[�^��p���āA�q�X�g�O������`���B
#---------------------------------------------------------------------------------------------------

from numpy.random import *
import pandas as pd
import matplotlib.pyplot as plt

# �ԉ��M��1��]����
n=1

R=rand(n)      # �ԉ��M��]����

# �ԉ��M��10000��A�]����
n=10000 

akaenpitu=pd.Series(rand(n))

# �q�X�g�O������`����
akaenpitu.plot(kind='hist',fontsize=20,title='�ԉ��MR�̃q�X�g�O����') 
plt.show()

# �T�C�R����]�������߂̏���
saikoro=pd.Series([1,2,3,4,5,6]) # Series�����

# �T�C�R����n��]����
dice=saikoro.sample(n,replace=True)

dice.plot(kind='hist', fontsize=20,title='�T�C�R���̃q�X�g�O����',bins=[1,2,3,4,5,6,7])  
plt.show()
