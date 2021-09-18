#---------------------------------------------------------------------------------------------------
# 14��
# �ԉ��M��n��]�����āA�W�{���ς𓾂�
# ���̂悤�ȕW�{���ς�n_sinbunsya����āA�q�X�g�O������`���B
# ����ɂ���āA�W�{���ς̕��z�����߂邱�Ƃ��ł���
#
# �Ή����鐳�K���z�̃q�X�g�O�������`��
#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


def hyouhon_heikin_uniform(n):

    # �W�{����ς��Ď���
    # n=1,2, 10, 100
    #n=10

    # ���ς�����邩
    n_sinbunsya=100000

    # �ԉ��M��(n*n_sinbunsya)��A���낪���āAn_sinbunsya�s�An��̍s��ɋL�^����
    aka=pd.DataFrame(np.random.rand(n*n_sinbunsya).reshape(n_sinbunsya,n) )

    # ���ς�n_sinbunsya�����쐬����
    heikin=pd.DataFrame(aka.mean(1),columns=['����'])

    # ����0.5�A���U (1/12)/n�̐��K������n_sinbunsya�������
    # seikiransu=pd.DataFrame([normal(0.5,np.sqrt((1/12)/n)) for i in range(n_sinbunsya)],columns=['�Ή����鐳�K����'])

    myu=0.5
    sigma=np.sqrt((1/12)/n)

    seikiransu=pd.DataFrame(normal(0.5,np.sqrt((1/12)/n),n_sinbunsya),columns=['�Ή����鐳�K����'])

    # �W�{��n�̈�l�����̕��ςƕ���0.5�A���U (1/12)/n�̐��K�����Ƃ̃q�X�g�O�����쐬�p�f�[�^
    hikaku=heikin.join(seikiransu)

    # �ȉ��An�̈�l�������瓾��ꂽ�W�{���ςƂ���ɑΉ����鐳�K���z�̕`��
    # to do:x�����Œ�
    fig=plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    kaikyu=[0+0.01*i for i in range(100)]

    ax1.hist(hikaku['����'],bins=kaikyu)
    ax2.hist(hikaku['�Ή����鐳�K����'],bins=kaikyu,color='blue')

    title1='%s�̃f�[�^�̕W�{���ς̕��z\n���̊m���ϐ���\n0-1��Ԃ̈�l�m���ϐ�' % n
    title2='����=%s, ���U=1/(12*%s)�̐��K���z' % (myu, n)

    ax1.set_title(title1,fontsize=15)
    ax2.set_title(title2,fontsize=15)

    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=20)

    plt.show()

hyouhon_heikin_uniform(1)   # ��l�����̊m�F�B�Ή����鐳�K���z�͂�����؂����Ă���B

hyouhon_heikin_uniform(2)   # �O�p�֐��B�Ή����鐳�K���z�́A���\0�C1��Ԃɓ����Ă���

hyouhon_heikin_uniform(10)  # ���K�ߎ������Ȃ萳�m�ɂȂ�B�u���S�Ɍ��藝�v�̏Љ�B

hyouhon_heikin_uniform(100) # �قڊ����ȋߎ��B

