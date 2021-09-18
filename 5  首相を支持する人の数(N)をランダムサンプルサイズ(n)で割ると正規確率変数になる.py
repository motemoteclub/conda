#---------------------------------------------------------------------------------------------------
# ��5��
# ���̃v���O������n=2,10,101,3000�̂��̂��̂ɂ��Ĉȉ��̂��Ƃ��s��(p=0.7)
#   �E�_�����1���쐬���A���̃q�X�g�O������`��
#   �E�_����l�����(p-1.96rmsq(p(1-p)/n), p+1.96rmsq(p(1-p)/n)) �̊Ԃɓ���m����95%�ł��邱�Ƃ̊m�F
#   �Ez=(N/n-p)/sqrt(p(1-p)/n)���W�����K���z�ɂȂ��Ă��邱�Ƃ̊m�F
#
#   �_����ʂ̕s�ΐ��ƈ�v���Ƃ��m�F���邱�Ƃ����
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


# n��ς��Ď���
# n=2, 10, 101, 3000
n=101

# �V���Ђ̐�
n_sinbunsya=10000

# �񑊂̐^�̎x����
p=0.7

# �藝5�̃ʂƃ�
myu=p
sigma=np.sqrt(p*(1-p)/n)

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̕ϐ������郉���_�֐��̒�`
NHK = lambda x: 1 if x<p else 0

# �ԉ��M��(n*n_sinbunsya)��A���낪���āAn_sinbunsya�s�An���DataFrame�ɋL�^����
aka=pd.DataFrame(np.random.rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̒������ʂ𓾂�B
# ����ɂ�����An_sinbunsya�s�An��̍s��ɋL�^����B
aka2=aka.applymap(NHK)

# �V���Ђ��Ƃɓ_������s��
tensuitei=aka2.mean(1) 

# �_���肪��ԁi��-1.96��,��+1.96��)�ɓ����Ă�����'�����Ă���'�A
# �����łȂ����'�����Ă��Ȃ�'�ƂȂ郉���_�֐��̒�`
in_or_out = lambda x: '�����Ă���' if (x>=myu-1.96*sigma) & (x<=myu+1.96*sigma) else '�����Ă��Ȃ�'

# �V���Ђ��Ƃ̓_���肪�i��-1.96��,��+1.96��)�ɓ����Ă�����'�����Ă���' �A�����łȂ����'�����Ă��Ȃ�'�Ƃ���B
# ���̌��ʂ�tensuitei2�ɋL�^����
tensuitei2=tensuitei.map(in_or_out)

# �_���肩��A�q�X�g�O�����쐬
title='1���̓_����l������ꂽ�q�X�g�O����(n=%s)' % n
tensuitei.plot(kind='hist', fontsize=20,title=title,range=(0,1),bins=200)
plt.show()

# z=(N/n-p)/sqrt(p(1-p)/n)���W�����K���z�ɂȂ��Ă��邱�Ƃ̊m�F
z_henkan=lambda x: (x-p)/np.sqrt(p*(1-p)/n)
z=tensuitei.apply(z_henkan)

title='�W����(n=%s)' % n
z.plot(kind='hist', fontsize=20,title=title,range=(-3,3),bins=200)
plt.show()

# �_���肪��ԁi��-1.96��,��+1.96��)�ɓ����Ă��邩�ۂ��̃N���X�\
pd.crosstab(tensuitei2,columns='����(%)',normalize=True)*100
