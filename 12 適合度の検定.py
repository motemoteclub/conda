#---------------------------------------------------------------------------------------------------
# ��12��
# �T�C�R���̖ڂ��ނɂ��āA��2��ϐ��̕��z������
#---------------------------------------------------------------------------------------------------

from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *

import matplotlib as mpl
import matplotlib.pyplot as plt

n=100
n_chi=10000

# �T�C�R����]�������߂̏���
saikoro=pd.Series([1,2,3,4,5,6]) # Series�����

# �T�C�R����n��]�����Ƃ������Ƃ�n_chi��J��Ԃ�
dice=pd.DataFrame(saikoro.sample(n*n_chi,replace=True).values.reshape(n_chi,n))

# ���ꂼ��̖ڂ�����o�����L�^���邽�߂̏���
obs_head=pd.DataFrame(columns=[1,2,3,4,5,6])

dice_summary=lambda x:x.value_counts() # ���ꂼ��̖ڂ�����o�����A�s���ƂɃJ�E���g���o��

obs=dice.apply(dice_summary,axis=1) # n��T�C�R����U�邲�ƂɁA�ڂ̓x�����v�Z����

# �C�ӂ̖ڂ��o�Ă��Ȃ��ꍇ�̑΍�
obs=pd.concat([obs_head,obs])

# �����l�΍�
obs.fillna(0,inplace=True)

# chi 2��̌v�Z
ex=n*(1/6) # ���ғx���̌v�Z
chi_square=lambda x:(   (x[1]-ex)**2/ex+(x[2]-ex)**2/ex+(x[3]-ex)**2/ex+
                        (x[4]-ex)**2/ex+(x[5]-ex)**2/ex+(x[6]-ex)**2/ex
                    )

chi2=obs.apply(chi_square,axis=1)

# chi2�̎����l�̃f�[�^�̃p�[�Z���^�C����Ԃ�
print(chi2.describe(percentiles=[0.90,0.95,0.99])) 

print(stats.chi2.ppf(0.95, 6-1)) # ��2��ϐ��̉E��5���_��Ԃ�

# �}��
chi2.plot(kind='hist',bins=25,title='�T�C�R���̓K���x�̌���Ɏg����J�C2��ϐ��̃q�X�g�O����(n=%s)' % n)
plt.show()
