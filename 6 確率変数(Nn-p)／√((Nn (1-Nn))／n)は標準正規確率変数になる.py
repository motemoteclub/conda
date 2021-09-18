#---------------------------------------------------------------------------------------------------
# ��6��
# �m���ϐ�(N/n-p)/sqrt((N/n)(1-N/n)/n)�͕W�����K�m���ϐ��ɂȂ�
#
#   �ȉ��̓�̎�����n��ω������Ȃ���A�m�F����
#
#   �E��̕W�������ꂽ�ϐ����W�����K�m���ϐ��ɂȂ�l�q���O���t������
#       (N/n-p)/sqrt(p(1-p)/n)��(N/n-p)/sqrt((N/n)(1-N/n)/n)�Ƃ̔�r
#
#   �E95���M����Ԃ̌��`��95���M����Ԃ�p���܂ފm�����r����
#       95���M����Ԃ̌��`: (N/n-1.96sqrt(p(1-p)/n),N/n+1.96sqrt(p(1-p)/n))
#       95���M�����      : (N/n-1.96sqrt(N/n(1-N/n)/n),N/n+1.96sqrt(N/n(1-N/n)/n))
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


# n��ς��Ď���
# n=2, 10, 101, 3000
n=10

# �V���Ђ̐�
n_sinbunsya=10000

# �񑊂̐^�̎x����
p=0.7

# �藝5�̃ʂƃ�(����𐳋K�m���ϐ��̕��ςƕW���΍��ɂ���)
myu=p
sigma=np.sqrt(p*(1-p)/n)

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̕ϐ������郉���_�֐��̒�`
NHK = lambda x: 1 if x<p else 0

# �ԉ��M��(n*n_sinbunsya)��A���낪���āAn_sinbunsya�s�An���DataFrame�ɋL�^����
aka=pd.DataFrame(rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̒������ʂ𓾂�B
# ����ɂ�����An_sinbunsya�s�An��̍s��ɋL�^����B
aka2=aka.applymap(NHK)

# �V���Ђ��Ƃɓ_������s��
tensuitei=pd.DataFrame(aka2.mean(axis=1),columns=['�_����'])

#------------------------------------------------------------------------------
# �W�������s��(�����ł́A(N/n-p)/sqrt((N/n)(1-N/n)/n)���v�Z����)
#------------------------------------------------------------------------------
# �W�����@5�͂�6�͂̕W�������s�������_�֐��̒�`
z_henkan_chapter5=lambda x: (x-myu)/(np.sqrt(p*(1-p)/n))
z_henkan_chapter6=lambda x: np.nan if x==0 else np.nan if x==1 else (x-myu)/(np.sqrt(x*(1-x)/n))

# �W�������s��
tensuitei['��5�͂̕W����']=tensuitei['�_����'].map(z_henkan_chapter5)
tensuitei['��6�͂̕W����']=tensuitei['�_����'].map(z_henkan_chapter6)

#------------------------------------------------------------------------------
# �ʂ�95���M����ԁA95���M����Ԃ̌��`(5��)�ɓ����Ă��邩�ۂ��̃`�F�b�N
#------------------------------------------------------------------------------
# �ʂ�95���M����ԁA95���M����Ԃ̌��`�ɓ����Ă��邩�ۂ����`�F�b�N����֐���`
myu_check_chapter5=(lambda x: '�����Ă���' if (myu>=x-1.96*sigma) & (myu<=x+1.96*sigma)
                                else '�����Ă��Ȃ��I�I�I' )

myu_check_chapter6=(lambda x: '�����Ă���' if (myu>=x-1.96*np.sqrt(x*(1-x)/n)) & 
                                (myu<=x+1.96*np.sqrt(x*(1-x)/n)) else '�����Ă��Ȃ��I�I�I' )

# �V���Ђ��ƂɃʂ�95���M����ԁA���邢�́A95���M����Ԃ̌��`�ɓ����Ă��邩�ۂ��̃`�F�b�N�����s
tensuitei['�ʂ�95%�M����Ԃ̌��`�ɓ����Ă��邩']=tensuitei['�_����'].map(myu_check_chapter5)
tensuitei['�ʂ�95%�M����Ԃɓ����Ă��邩']=tensuitei['�_����'].map(myu_check_chapter6)


#------------------------------------------------------------------------------
# �O���t�쐬�E�N���X�\�쐬
#------------------------------------------------------------------------------
# �_���肩��A�q�X�g�O�����쐬�i��5�͂Ƒ�6�͂̕W�����̔�r�j
title='��5�͂Ƒ�6�͂̕W�����̔�r(n=%s)' % n
tensuitei[['��5�͂̕W����','��6�͂̕W����']].plot(kind='hist',bins=300,range=(-3,3),grid=False,title=title)
plt.show()

# �_���肪��ԁi��-1.96��,��+1.96��)�ɓ����Ă��邩�ۂ��̃N���X�\
print(pd.crosstab(tensuitei['�ʂ�95%�M����Ԃ̌��`�ɓ����Ă��邩'],
columns='����(%)',normalize=True)*100)
print(pd.crosstab(tensuitei['�ʂ�95%�M����Ԃɓ����Ă��邩'],
columns='����(%)',normalize=True)*100)
