#---------------------------------------------------------------------------------------------------
# ��21��
# �g����155�A160�A165�A170�A175�Z���`���[�g���̐l�̉��z�f�[�^(5��)���쐬���A
# �W�{���֌W���ƕW�{���U���v�Z����B�����āA��A�����������ߕ\���E�}������B
# 
# ���z�f�[�^�̍���
#   -80+0.83�g���{�덷���i�덷���͕���0�C�W���΍�6.83�̐��K�m���ϐ��j
#---------------------------------------------------------------------------------------------------
from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

# �g����155�A160�A165�A170�A175�Z���`���[�g���̐l�̉��z�f�[�^���쐬����B
height_weight=pd.DataFrame( {'�g��':[155, 160, 165, 170, 175],
                             '�̏d':[-80+0.83*i+6.83*normal(0,1) for i in [155, 160, 165, 170, 175] ],
                             '�̏d�E�T�C�R���U�炸':[-80+0.83*i  for i in [155, 160, 165, 170, 175] ]})
#------------------------------------------------------------------------------
# �W�{���U�ƕW�{���֌W��
#------------------------------------------------------------------------------
# �g���Ƒ̏d�̕W�{���֌W��
print(height_weight.corr())
# �g���Ƒ̏d�̕W�{�����U
print(height_weight.cov())
#------------------------------------------------------------------------------
# �ŏ�2��@
#------------------------------------------------------------------------------
# ��A�p�����[�^(a,b)
b=height_weight.cov().loc['�g��','�̏d']/(height_weight.var().loc['�g��']) # �g���E�̏d�̋����U/�g���̕��U
a=height_weight.mean().loc['�̏d']-b*height_weight.mean().loc['�g��'] # �̏d�̕��� - b�E�g���̕���
#------------------------------------------------------------------------------
# statsmodel �ɂ���A����
#------------------------------------------------------------------------------
# statsmodel�ɂ���A����
height_weight['�萔��'] = 1 # sm.add_constant(height_weight)�ł��B"const"���ł���B
x=height_weight[['�萔��','�g��']]
y=height_weight[['�̏d']]
model = sm.OLS(y, x)
results = model.fit() # dir(results)�ő�����������
# ���ʂ̊T�v��\��
print( results.summary() )
#------------------------------------------------------------------------------------
# �ȉ��A�O���t�̕`��ݒ�
#------------------------------------------------------------------------------------
# �O���t�S�̂̃t�H���g�w��
fsz=20                 # �}�S�̂̃t�H���g�T�C�Y
fti=np.floor(fsz*1.2)  # �}�^�C�g���̃t�H���g�T�C�Y
flg=np.floor(fsz*0.5)  # �}��̃t�H���g�T�C�Y
flgti=flg              # �}��̃^�C�g���̃t�H���g�T�C�Y

plt.rcParams["font.size"] = fsz # �}�S�̂̃t�H���g�T�C�Y�w��
plt.rcParams['font.family'] ='IPAexGothic' # �}�S�̂̃t�H���g

#------------------------------------------------------------------------------
# ax�ɕ`��
#------------------------------------------------------------------------------
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
#------------------------------------------------------------------------------
# ax1�ɕ`��
#------------------------------------------------------------------------------
ax1.set_xlabel('�g��',fontsize=20)
ax1.set_ylabel('�̏d',fontsize=20)
ax1.set_ylim(0,120)
ax1.set_title('�_�l���T�C�R����U���č�����f�[�^�̕`��\n �̏d=-80+0.83�~�g��+�T�C�R��',fontsize=20)
ax1.scatter(x=height_weight['�g��'].values, y=height_weight['�̏d'].values,s=500)

# ��A�����`��
ax1.plot(height_weight['�g��'].values, results.fittedvalues,c='blue',linestyle='solid',
    label='��A�������Fy = {0:.1f} + {1:.2f}x'.format(a,b))
# �T�C�R���U�炸
ax1.plot(height_weight['�g��'].values, height_weight['�̏d�E�T�C�R���U�炸'].values,
    c='red',linestyle='solid',label='y=-80+0.83x')
ax1.legend(loc='upper right',fontsize=10, title_fontsize=10)
#------------------------------------------------------------------------------
# ax2�ɐ��l���ʂ�`��
#------------------------------------------------------------------------------
ax2.axis('off') # �}���͂ގ���������
ax2.set_title('��A���͂̌���',fontsize=30)
# ax2�Ƀe�L�X�g��ǉ�
ax2.text(0.2, 0.8, '�W�{���֌W��=  {0:.2f}'.format(height_weight.corr().loc['�g��','�̏d']), size = 20)
ax2.text(0.2, 0.7, '�W�{�����U=  {0:.1f}'.format(height_weight.cov().loc['�g��','�̏d']), size = 20)
ax2.text(0.1, 0.6, '��A�������Fy = {0:.1f} + {1:.2f}x'.format(a,b), size = 20, color = "blue")
plt.show()
