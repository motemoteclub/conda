#---------------------------------------------------------------------------------------------------
# ��13�� 
"""
  �����\�ɂ�����Ɨ����̌���Ŏg����藝���m�F����v���O����(�ȉ��̐ݒ��AB�^�𖳎����Ă���)

�m���̋L��(()���̓J�e�S���ԍ�)
�@--------------------------------------
 |       ���i��|�@�ϋɓI  |�@���ɓI    |���v
 |���t�^��     |          |            |
 --------------------------------------
 | A�^         | p11(1)   | p12(2)     |p1_
 | O,B�^       | p21(3)   | p22(4)     |p2_
 --------------------------------------
  ���v           p_1        p_2          1

�ϑ��l�̋L��
�@--------------------------------------
 |       ���i��|�@�ϋɓI  |�@���ɓI    |���v
 |���t�^��     |          |            |
 --------------------------------------
 | A�^         | sumy11   | sumy12     |
 | O,B�^       | sumy21   | sumy22     |
 --------------------------------------
  ���v                                   n|

"""
from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *

import matplotlib as mpl
import matplotlib.pyplot as plt

n=1000
n_chi=10000
p1_ = 0.4  # A  �^�̐^�̊m�� 
p2_ = 0.6  # O,B�^�̐^�̊m�� 
p_1 = 0.2  #�@�ϋɓI�Ȑl�̐^�̊m�� 
p_2 = 0.8  #�@���ɓI�Ȑl�̐^�̊m�� 
# �ȉ��A���t�^�Ɛ��i���Ɨ��ȏꍇ�̊e�Z���̊m�� 
p11 = p1_ * p_1  # A  �^�ŐϋɓI�Ȋm�� 
p12 = p1_ * p_2  # A  �^�ŏ��ɓI�Ȋm�� 
p21 = p2_ * p_1  # O,B�^�ŐϋɓI�Ȋm�� 
p22 = p2_ * p_2  # O,B�^�ŏ��ɓI�Ȋm�� 

aka=pd.DataFrame(np.random.rand(n*n_chi).reshape(n_chi,n) )
# �N���X�\�̂ǂ��Ɏ������邩������
to_cross=lambda x: 1 if x<p11 else 2 if x<(p11+p12) else 3 if x<(p11+p12+p21) else 4 
cross=aka.applymap(to_cross)
cross_summary=lambda x:x.value_counts() # ���ꂼ��̖ڂ�����o�����J�E���g���o��
obs=cross.apply(cross_summary,axis=1) # n��̏o�����ƂɁA�ڂ̓x�����v�Z����
# ���ꂼ��̖ڂ�����o�����L�^���邽�߂̏���
obs_head=pd.DataFrame(columns=[1,2,3,4])
# �C�ӂ̖ڂ��o�Ă��Ȃ��ꍇ�̑΍�
obs=pd.concat([obs_head,obs])
# �����l�΍�
obs.fillna(0,inplace=True)
# Qab_1�̌v�Z
chi_square=(lambda x: (x[1]-(n*p11))**2/(n*p11)+(x[2]-(n*p12))**2/(n*p12)+(x[3]-(n*p21))**2/(n*p21)+
            (x[4]-(n*p22))**2/(n*p22))
Qab_1=obs.apply(chi_square,axis=1)

# Qab_1_a_1_b_1�̌v�Z
chi_square2=(lambda x: (x[1]-(n*((x[1]+x[2])/n)*((x[1]+x[3])/n)))**2/(n*((x[1]+x[2])/n)*((x[1]+x[3])/n))+
                       (x[2]-(n*((x[1]+x[2])/n)*((x[2]+x[4])/n)))**2/(n*((x[1]+x[2])/n)*((x[2]+x[4])/n))+
                       (x[3]-(n*((x[3]+x[4])/n)*((x[1]+x[3])/n)))**2/(n*((x[3]+x[4])/n)*((x[1]+x[3])/n))+
                       (x[4]-(n*((x[3]+x[4])/n)*((x[2]+x[4])/n)))**2/(n*((x[3]+x[4])/n)*((x[2]+x[4])/n)) )
Qab_1_a_1_b_1=obs.apply(chi_square2,axis=1)

# ��̊m���ϐ��@Qab_1�@�Ɓ@Qab_1_a_1_b_1�@�̗��_���z�Ǝ����l�̕��z�Ƃ��r����
print(stats.chi2.ppf(0.95, 4-1)) # ��2��ϐ��̉E��5���_��Ԃ�
print(Qab_1.describe(percentiles=[0.90,0.95,0.99])) # �p�[�Z���^�C����Ԃ�

print(stats.chi2.ppf(0.95, (2-1)*(2-1))) # ��2��ϐ��̉E��5���_��Ԃ�
print(Qab_1_a_1_b_1.describe(percentiles=[0.90,0.95,0.99])) # �p�[�Z���^�C����Ԃ�

# �q�X�g�O�����`��
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

ax1.set_title('��2��i���ғx���͐^�̊m���g�p�j\n n=%s' % n,fontsize=20)
ax2.set_title('��2��i���ғx���͐��肳�ꂽ�m���g�p�j\n n=%s' % n,fontsize=20)

Qab_1.plot(kind='hist',ax=ax1,bins=35)
Qab_1_a_1_b_1.plot(kind='hist',ax=ax2,bins=35)
plt.show()
