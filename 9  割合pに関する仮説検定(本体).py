#------------------------------------------------------------------------------
# ��9��
# �L�Ӑ���5���ŁA������s���B
#   �@�܂Ƃ��ȃT�C�R����1000�񌟒�
#   �A�����܂��T�C�R����1000�񌟒�
#   �Ō�ɁA�@�ƇA�Ƃ��܂Ƃ߂��\���쐬����
#
# �ǂ̂悤�ȏ����Ŏ������邩�͈ȉ��̂Ƃ���
#
# test(0.5,0.60,100)   n=100�ŁAp0��p1�Ƃ������l�̏ꍇ
# test(0.5,0.55,100)   n=100�ŁAp0��p1�Ƃ��߂��l�̏ꍇ
# test(0.5,0.55,1000)  n=1000�ŁAp0��p1�Ƃ��߂��l�̏ꍇ
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

def test( p0, p1, n ):

    test005=(lambda x:'�܂Ƃ�' if (x>0.5-1.96*np.sqrt(0.5*(1-0.5)/n)) and (x<0.5+1.96*np.sqrt(0.5*(1-0.5)/n))
                               else '��������')
    test010=(lambda x:'�܂Ƃ�' if (x>0.5-1.64*np.sqrt(0.5*(1-0.5)/n)) and (x<0.5+1.64*np.sqrt(0.5*(1-0.5)/n))
                               else '��������')
    aka_matomo =lambda x: 1 if x<p0 else 0
    aka_ikasama=lambda x: 1 if x<p1 else 0

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # �܂Ƃ��ȃT�C�R����n��]�����āA5���L�Ӑ����Ō��肷��B�����1000��s��
    test_result_in_matomo=aka.applymap(aka_matomo).mean(axis=1).map(test005)

    # �ԉ��M��n��]�����Ƃ��������� 1000 ��J��Ԃ�
    aka=pd.DataFrame([ [rand() for j in range(n) ] for i in range(1000)]) 

    # �������܃T�C�R����n��]�����āA5���L�Ӑ����Ō��肷��B�����1000��s��
    test_result_in_ikasama=aka.applymap(aka_ikasama).mean(axis=1).map(test005)

    # �N���X�W�v
    test_result_in_matomo_crosstab=pd.crosstab(test_result_in_matomo,columns='��',normalize=False)
    test_result_in_ikasama_crosstab=pd.crosstab(test_result_in_ikasama,columns='��',normalize=False)

    # �ϐ����Ȃǂ̐���
    test_result_in_matomo_crosstab['�R�C���̐^�̐���']='�܂Ƃ�'
    test_result_in_matomo_crosstab['�L�Ӑ���']='5��'
    test_result_in_matomo_crosstab.index.name='���茋��'
#    test_result_in_matomo_crosstab.reset_index().set_index(['�R�C���̐^�̐���','�L�Ӑ���','���茋��'])

    # �ϐ����Ȃǂ̐���
    test_result_in_ikasama_crosstab['�R�C���̐^�̐���']='��������'
    test_result_in_ikasama_crosstab['�L�Ӑ���']='5��'
    test_result_in_ikasama_crosstab.index.name='���茋��'
#    test_result_in_ikasama_crosstab.reset_index().set_index(['�R�C���̐^�̐���','�L�Ӑ���','���茋��'])

    # �@�܂Ƃ��ȃT�C�R����1000�񌟒肵�A����ɇA�����܂��T�C�R����1000�񌟒肵��������
    # �Ō�ɁA�@�ƇA�Ƃ��܂Ƃ߂�DataFrame���쐬����
    test_result=pd.concat([test_result_in_matomo_crosstab,test_result_in_ikasama_crosstab])

    temp=test_result

    # test_result�̃C���f�b�N�X�����C������B�C���������ʂ͈ȉ��̂Ƃ���
    #
    #�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@'��'
    # '�R�C���̐^�̐���','���茋��'
    # �܂Ƃ��@�@�@�@�@�@�@�܂Ƃ�        �l1
    # �܂Ƃ��@�@�@�@�@�@�@��������      �l2
    # �������܁@�@�@�@�@�@�܂Ƃ�        �l3
    # �������܁@�@�@�@�@�@��������      �l4
    #
    #                                               ���}���`�C���f�b�N�X�ݒ�
    test_result=test_result.reset_index().set_index(['�R�C���̐^�̐���','���茋��']).loc[
        [('�܂Ƃ�','�܂Ƃ�'),('�܂Ƃ�','��������'),('��������','�܂Ƃ�'),('��������','��������')],:]

    # �^�C�g���ݒ�
    titlename='p0='+str(p0)+'  p1='+str(p1)+'  n='+str(n)

    # 1000��̌��茋�ʂ̕`��
    # .pivot_table()�̎g�������Ӂistacked=True�ŗݐϖ_�O���t�ɂȂ�j
    test_result.pivot_table(index=['�R�C���̐^�̐���'],columns='���茋��',values='��').plot.barh(
        stacked=True,fontsize=20,title=titlename)
    plt.show()
    return test_result

test(0.5,0.60,100)
test(0.5,0.55,100)
test(0.5,0.55,1000)
