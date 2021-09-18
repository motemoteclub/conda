#---------------------------------------------------------------------------------------------------
# ��8��
# �Q�[�}�[A�ƃQ�[�}�[B�Ƃ��A���ꂼ��A�u�܂Ƃ��ȃR�C���v��1000�Q�[���A�u�������܃R�C���v��1000�Q�[����
# �s���B�u�������܃R�C���v�ŕ\���o��m��p�͊֐�game()�̈���p1�ŗ^������B�u�܂Ƃ��ȃR�C���v��
# �\���o��m���͊֐�game()�̈���p0�ŗ^������B�R�C���𓊂��グ��񐔂�game()�̈���n�ŗ^������B
# �����͈ȉ��̂R�̏ꍇ�ɂ��čs���B
#
# p0=0.5,p1=0.55,n=100 
# p0=0.5,p1=0.60,n=100 
# p0=0.5,p1=0.55,n=1000
#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def game( p0, p1, n ):
    handan_A=(lambda x:'�܂Ƃ�' if (x>=0.5-1.96*np.sqrt(0.5*(1-0.5)/n)) and 
                                (x<=0.5+1.96*np.sqrt(0.5*(1-0.5)/n)) else '��������')
    handan_B=(lambda x:'�܂Ƃ�' if (x>=0.5-1.64*np.sqrt(0.5*(1-0.5)/n)) and 
                                (x<=0.5+1.64*np.sqrt(0.5*(1-0.5)/n)) else '��������')
    aka_sizi_husizi0=lambda x: 1 if x<p0 else 0
    aka_sizi_husizi1=lambda x: 1 if x<p1 else 0

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer A�̔��f�ip=p0�̉��ł̎����j
    gamer_a_p0=aka.applymap(aka_sizi_husizi0).mean(axis=1).map(handan_A)

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer A�̔��f�ip=p1�̉��ł̎����j
    gamer_a_p1=aka.applymap(aka_sizi_husizi1).mean(axis=1).map(handan_A)

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer B�̔��f�ip=p0�̉��ł̎����j
    gamer_b_p0=aka.applymap(aka_sizi_husizi0).mean(axis=1).map(handan_B)

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # gamer B�̔��f�ip=p1�̉��ł̎����j
    gamer_b_p1=aka.applymap(aka_sizi_husizi1).mean(axis=1).map(handan_B)

    # �N���X�W�v
    gamer_a_p0_crosstab=pd.crosstab(gamer_a_p0,columns='��',normalize=False)
    gamer_a_p1_crosstab=pd.crosstab(gamer_a_p1,columns='��',normalize=False)
    gamer_b_p0_crosstab=pd.crosstab(gamer_b_p0,columns='��',normalize=False)
    gamer_b_p1_crosstab=pd.crosstab(gamer_b_p1,columns='��',normalize=False)

    # �ϐ����Ȃǂ̐���
    gamer_a_p0_crosstab['�^�̃T�C�R���̐���']='�܂Ƃ�'
    gamer_a_p0_crosstab['�v���[���[']='�Q�[�}�[A'
    gamer_a_p0_crosstab.index.name='���f'
    gamer_a_p0_crosstab.reset_index().set_index(['�^�̃T�C�R���̐���','�v���[���[','���f'])

    gamer_a_p1_crosstab['�^�̃T�C�R���̐���']='��������'
    gamer_a_p1_crosstab['�v���[���[']='�Q�[�}�[A'
    gamer_a_p1_crosstab.index.name='���f'
    gamer_a_p1_crosstab.reset_index().set_index(['�^�̃T�C�R���̐���','�v���[���[','���f'])

    gamer_b_p0_crosstab['�^�̃T�C�R���̐���']='�܂Ƃ�'
    gamer_b_p0_crosstab['�v���[���[']='�Q�[�}�[B'
    gamer_b_p0_crosstab.index.name='���f'
    gamer_b_p0_crosstab.reset_index().set_index(['�^�̃T�C�R���̐���','�v���[���[','���f'])

    gamer_b_p1_crosstab['�^�̃T�C�R���̐���']='��������'
    gamer_b_p1_crosstab['�v���[���[']='�Q�[�}�[B'
    gamer_b_p1_crosstab.index.name='���f'
    gamer_b_p1_crosstab.reset_index().set_index(['�^�̃T�C�R���̐���','�v���[���[','���f'])

    # �܂Ƃ߂̃N���X�\
    game_result=pd.concat([gamer_a_p0_crosstab,gamer_a_p1_crosstab,gamer_b_p0_crosstab,
                            gamer_b_p1_crosstab])

    titlename='p0='+str(p0)+'  p1='+str(p1)+'  n='+str(n)

    (game_result.reset_index().pivot_table(index=['�v���[���[','�^�̃T�C�R���̐���'],
        values='��',columns='���f').plot.barh(stacked=True,fontsize=20,title=titlename))

    return game_result

game(0.5,0.55,100).reset_index()
game(0.5,0.60,100).reset_index()
game(0.5,0.55,1000).reset_index()
