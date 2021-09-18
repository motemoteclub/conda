#------------------------------------------------------------------------------
# 16��
# ��W�{�@�ɂ��ʂ̋�Ԑ���̃V�~�����[�V����
# �@���̊m���ϐ����@��l���z�̏ꍇ�ƁA�A����0.5�A���U1/12�̐��K���z�̏ꍇ�ōs��
#
#   daihyouhon_heikin_kukansuitei(2,'��l���z') # �f�[�^�����������ƃ{���{��
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kukansuitei(n, bunpu):

    # (0,1)��Ԃ̈�l�m���ϐ��̕��ςƕ��U
    myu=1/2
    sigma2=1/12    
    noexp=100000 # ���ρA�W�{���U�A��2�������邩
    if bunpu=='��l���z':
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='���K���z':
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('���z�Ԉ���Ă܂���')
        stop    
    heikin=pd.DataFrame(itiyou.mean(1),columns=['�W�{����']) # �W�{���ς�noexp�����쐬����    
    bunsan=pd.DataFrame(itiyou.var(1),columns=['�W�{���U']) # �W�{���U��noexp�����쐬����
    std_error=np.sqrt(bunsan/n) # �W�{���ς̕W���덷���v�Z����
    std_error=std_error.rename(columns={'�W�{���U':'�W�{���ς̕W���덷'})
    hidari=heikin['�W�{����']-1.96*std_error['�W�{���ς̕W���덷']
    migi=heikin['�W�{����']+1.96*std_error['�W�{���ς̕W���덷']
    c_interval=pd.DataFrame({'�����M�����E':hidari,'�㑤�M�����E':migi})
    all=pd.concat([heikin,bunsan,std_error,c_interval],axis=1)
    all['���̊m���ϐ�']=itiyou.loc[:,0]
    all['�W�{���ς̕W����']=(all['�W�{����']-myu)/all['�W�{���ς̕W���덷']

    # �M����Ԃ� myu ���܂ނ��ۂ�
    hantei=(lambda x: '�����Fmyu���܂�ł���' if (myu>=x['�����M�����E']) & 
              (myu<=x['�㑤�M�����E']) else '���s�Fmyu���܂�ł��Ȃ�')
    all[ '�M����Ԃ� myu ���܂ނ��ۂ�']=all.apply(hantei,axis=1)
    result=pd.crosstab(all['�M����Ԃ� myu ���܂ނ��ۂ�'],columns='����') # �M����Ԃ� myu ���܂ނ��ۂ��̃N���X�\

    figure=plt.figure(figsize=(8,5),tight_layout=True)
    axes_1 = plt.subplot2grid((2,2),(0,0))
    axes_2 = plt.subplot2grid((2,2),(1,0))
    axes_4 = plt.subplot2grid((2,2),(0,1),rowspan=2)

    # ���̊m���ϐ��̃q�X�g�O����
    axes_1.hist(all['���̊m���ϐ�'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('���̊m���ϐ�',loc='center',fontsize=24)
    axes_1.legend()

    # �W�{���ς̕W�������ꂽ���̂̃q�X�g�O����
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_2.hist(all['�W�{���ς̕W����'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='b',label='�j��')
    axes_2.set_title('�W�{���ς̕W�������ꂽ����\n(n=%i)' %n,loc='center',fontsize=24) #�}�^�C�g���̈ʒu�ƃT�C�Y
    str1='���s�F�ʂ��܂�ł��Ȃ�:  '+str(result.iloc[0,0])+'��'
    str2='�����F�ʂ��܂�ł���@:�@'+str(result.iloc[1,0])+'��'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    (axes_4.set_title('��W�{�@�ɂ��M����Ԃ�\n�ʂ��܂�ł��邩�ۂ�\n(%d��̋�Ԑ���,n=%d)' %(noexp,n),
     loc='center',fontsize=15))
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()    
    plt.show()

daihyouhon_heikin_kukansuitei(2,'��l���z') # �f�[�^�����������ƃ{���{��
daihyouhon_heikin_kukansuitei(2,'���K���z') # �f�[�^�����������ƃ{���{��

daihyouhon_heikin_kukansuitei(30,'��l���z') # �f�[�^����30�ɂȂ�Ƃ����قڗ��_�ǂ���
daihyouhon_heikin_kukansuitei(30,'���K���z') # �f�[�^����30�ɂȂ�Ƃ����قڗ��_�ǂ���

daihyouhon_heikin_kukansuitei(100,'��l���z') # ����
daihyouhon_heikin_kukansuitei(100,'���K���z') # ����
