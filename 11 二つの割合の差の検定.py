#---------------------------------------------------------------------------------------------------
# 11��
# ��̊����̍��̌���Ŏg����藝���m�F����v���O����
#
# �藝�F
# ( N1/n1 - N2/n2 )�͕��� p1-p2�A���U p1(1-p1)/n1 + p2(1-p2)/n2 �̐��K���z�ɏ]���B
#
# �p�����[�^
# n1, p1, n2, p2
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from scipy import stats

# �p�����[�^�ݒ�
n1=300
p1=0.7

n2=200
p2=0.5

# ���ς̍�������
n_sinbunsya=10000 # 2�̕��ς̍�������

# �Ή����鐳�K���z�̃p�����[�^
mean=p1-p2
std=np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̕ϐ������郉���_�֐��̒�`
NHK1 = lambda x: 1 if x<p1 else 0
NHK2 = lambda x: 1 if x<p2 else 0

# �ԉ��M��(n*n_sinbunsya)��A���낪���āAn_sinbunsya�s�An��̍s��ɋL�^����
aka1=pd.DataFrame(np.random.rand(n1*n_sinbunsya).reshape(n_sinbunsya,n1) )
aka2=pd.DataFrame(np.random.rand(n2*n_sinbunsya).reshape(n_sinbunsya,n2) )

# �ԉ��M�̌��ʂ���A�񑊎x���A�s�x���̒������ʂ𓾂�B
# ����ɂ�����An_sinbunsya�s�An��̍s��ɋL�^����B
tyousa_kekka1=aka1.applymap(NHK1)
tyousa_kekka2=aka2.applymap(NHK2)

# �V���Ђ��Ƃɓ_������s��
tensuitei1=pd.DataFrame(tyousa_kekka1.mean(1),columns=['�_����'])
tensuitei2=pd.DataFrame(tyousa_kekka2.mean(1),columns=['�_����'])

# ��̊����̍����v�Z����
sa=tensuitei1-tensuitei2
sa=sa.rename(columns={'�_����':'��̊����̍�'})
#sa.drop('�_����',axis=1,inplace=True)

#�Ή����鐳�K���������
seikiransu=pd.DataFrame(normal(mean,std,n_sinbunsya),columns=['���K����'])

# ���̂�����
result=sa.join(seikiransu) # �������́B���ʂ̕ϐ����Ȃ��ꍇ�͂��ꂪ�֗�

# �_���肩��A�q�X�g�O�����쐬(�����̍��̕��z�ƁA����ɑΉ����鐳�K���z���قړ���������)
fig=plt.figure(figsize=(8,5),tight_layout=True)
ax1=fig.add_subplot(1,2,1) # fig.add_subplot(m,n,l) m�~n�̃L�����o�X��l�Ԗ�
ax2=fig.add_subplot(1,2,2) 

result['��̊����̍�'].plot(kind='hist',bins=100,range=(0,1),ax=ax1)
result['���K����'].plot(kind='hist',bins=100,range=(0,1),ax=ax2)

ax1.set_title('��̊����̍�(N1/n1 - N2/n2)�̃q�X�g�O����\n�in1=%s, p1=%s,n2=%s,p2=%s�j' 
% (n1,p1,n2,p2))
ax2.set_title(
'���K�����̃q�X�g�O����\n�i����=(p1-p2), �W���΍�=sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)')

plt.show()

#=============================================================
# ���蓝�v�ʂ̌v�Z�ƃq�X�g�O����
# �W���덷�̌v�Z
p=(tyousa_kekka1.sum(1)+tyousa_kekka2.sum(1))/(n1+n2)
std=np.sqrt(p*(1-p)*(1/n1+1/n2))

# ���蓝�v�ʂ̌v�Z
kentei=(sa['��̊����̍�']-(p1-p2))/std

# �`��
fig2=plt.figure(figsize=(8,5),tight_layout=True)
ax3=fig2.add_subplot(1,1,1)
ax3.set_title('���蓝�v�ʂ̃q�X�g�O����')
ax3.grid(False) # �O���b�h������
ax3.hist(kentei.values, bins=69)
plt.show()
