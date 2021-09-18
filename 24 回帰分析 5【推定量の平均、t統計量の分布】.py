#------------------------------------------------------------------------------
# ��24��
# �g���Ƒ̏d��5�̃f�[�^����A
# 
#   ���Ɋւ���t�ϗ�
#   ���Ɋւ���t�ϗ�
#   ���n�b�g
#   ���n�b�g
#   ��2��ϐ�
#   ��A�������̕W���덷��2��
#
# ���v�Z����B������A1����J��Ԃ����Ƃɂ�蓾����A��L�̕ϐ��̃f�[�^����
# 6��̃q�X�g�O������`��
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

# �{�v���O�����ł́Astatsmodel�ł͂Ȃ��Asklearn���g���Ă݂�
# sklearn.linear_model.LinearRegression �N���X��ǂݍ���
from sklearn import linear_model
clf = linear_model.LinearRegression()
X=pd.DataFrame([150, 160, 170, 180],columns=['�����ϐ�']).values # ���ȏ��ɂ��킹��
alpha=-80
beta=0.8
sigma=8
hozon=pd.DataFrame([])
for i in range(10000):
    Y=pd.DataFrame([ alpha+beta*i + normal(0,sigma) for i in list(X)],columns=['�ړI�ϐ�']).values
    clf.fit(X, Y)
    a=clf.intercept_
    b=clf.coef_.flatten() # 2�����z���1�����z��ɂ��Ă���
    Y_hat=clf.predict(X)
    e=Y-Y_hat
    sigma_sq_hat=np.dot(e.T,e)/(len(Y)-2) # sigma_sq_hat�̊��Ғl��sigma**2
    t_beta=(b-beta)/np.sqrt(sigma_sq_hat/(np.dot((X-np.mean(X)).T,(X-np.mean(X)))))
    t_alpha=(a-alpha)/np.sqrt(sigma_sq_hat*(1/len(Y)+
            np.mean(X)**2/(np.dot((X-np.mean(X)).T,(X-np.mean(X))))))
    chai=sigma_sq_hat*(len(Y)-2)/sigma**2 # chi�̊��Ғl�� 5-2=3
    save=pd.DataFrame([[a[0],b[0],
                        sigma_sq_hat[0,0],chai[0,0],
                        t_beta[0,0],t_alpha[0,0]]],
                        columns=['�萔��','�X��',
                                '��A�������̕W���덷��2��','��2��',
                                '���Ɋւ���t�ϗ�','���Ɋւ���t�ϗ�'])

    hozon=hozon.append(save)

#------------------------------------------------------------------------------------
# �ȉ��A�O���t�̕`��ݒ�ƕ`��
#------------------------------------------------------------------------------------

# �O���t�S�̂̃t�H���g�w��
fsz=10                 # �}�S�̂̃t�H���g�T�C�Y
fti=np.floor(fsz*1.2)  # �}�^�C�g���̃t�H���g�T�C�Y
flg=np.floor(fsz*0.5)  # �}��̃t�H���g�T�C�Y
flgti=flg              # �}��̃^�C�g���̃t�H���g�T�C�Y

plt.rcParams["font.size"] = fsz # �}�S�̂̃t�H���g�T�C�Y�w��
#plt.rcParams['font.family'] ='sans-serif' # �}�S�̂̃t�H���g
plt.rcParams['font.family'] ='IPAexGothic' # �}�S�̂̃t�H���g

# �q�X�g�O�����`��
figure=plt.figure(figsize=(8,5),tight_layout=True)

ax1=figure.add_subplot(3,2,1,title='���Ɋւ���t�ϗ�(���R�x:n-2=2)')
ax2=figure.add_subplot(3,2,2,title='���Ɋւ���t�ϗ�(���R�x:n-2=2)')
ax3=figure.add_subplot(3,2,3,title='���n�b�g')
ax4=figure.add_subplot(3,2,4,title='���n�b�g')
ax5=figure.add_subplot(3,2,5,title='��2��(���R�x:n-2=2)')
ax6=figure.add_subplot(3,2,6,title='��A�������̕W���덷��2��')

ax1.set_xlim(-20,20)
ax2.set_xlim(-20,20)
ax3.axvline(x=-80,c='blue')
ax4.axvline(x=0.8,c='blue')
ax6.axvline(x=64,c='blue')

hozon['���Ɋւ���t�ϗ�'].hist(bins=200,grid=False,ax=ax1)
hozon['���Ɋւ���t�ϗ�'].hist(bins=200,grid=False,ax=ax2)
hozon['�萔��'].hist(bins=200,grid=False,ax=ax3)
hozon['�X��'].hist(bins=200,grid=False,ax=ax4)
hozon['��2��'].hist(bins=200,grid=False,ax=ax5)
hozon['��A�������̕W���덷��2��'].hist(bins=200,grid=False,ax=ax6)

plt.show()
