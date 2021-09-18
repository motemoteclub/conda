#------------------------------------------------------------------------------
# ��18��
# ���W�{�@�ɂ��ʂ̋�Ԑ���̃V�~�����[�V����
# �@���̊m���ϐ��͇A����0.5�A���U1/12�̐��K���z�̏ꍇ�ōs��
#
#   daihyouhon_heikin_kukansuitei(2,'���K���z') # �T���v���T�C�Y���������Ƃ�������
#
#   daihyouhon_heikin_kukansuitei(3,'���K���z') # �T���v���T�C�Y���������Ƃ�������
#
#   daihyouhon_heikin_kukansuitei(100,'���K���z') # �T���v���T�C�Y���傫���Ȃ��t���z�ɂ�錟���
#                                                 # ��W�{�@�ɂ�錟��͂قړ���
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kukansuitei(n, bunpu):
    myu=1/2
    sigma2=1/12
    noexp=100000
    if bunpu=='��l���z':
        # ��l�m���ϐ���(n*noexp)��A���낪���āAnoexp�s�An��̍s��ɋL�^����
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='���K���z':
        # ����myu�A���U 1/12 �̐��K������n*noexp�������
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('���z�Ԉ���Ă܂���')
        stop
    heikin=pd.DataFrame(itiyou.mean(1),columns=['�W�{����'])
    bunsan=pd.DataFrame(itiyou.var(1),columns=['�W�{���U'])
    std_error=np.sqrt(bunsan/n)
    std_error=std_error.rename(columns={'�W�{���U':'�W�{���ς̕W���덷'})
    t0=stats.t.ppf(q=0.975, df=n-1)
    hidari=heikin['�W�{����']-t0*std_error['�W�{���ς̕W���덷']
    migi=heikin['�W�{����']+t0*std_error['�W�{���ς̕W���덷']
    c_interval=pd.DataFrame({'�����M�����E':hidari,'�㑤�M�����E':migi})
    all=pd.concat([heikin,bunsan,std_error,c_interval],axis=1)
    all['���̊m���ϐ�']=itiyou.loc[:,0]
    all['�W�{���ς̕W����']=(all['�W�{����']-myu)/all['�W�{���ς̕W���덷']
    estimate=(lambda x: '�����Fmyu���܂�ł���' if (myu>=x['�����M�����E'])&(myu<=x['�㑤�M�����E']) 
                                                else '���s�Fmyu���܂�ł��Ȃ�')
    all=all.assign( y=all.apply(estimate,axis=1)).rename(columns={'y':'�M����Ԃ� myu ���܂ނ��ۂ�'})
    result=pd.crosstab(all['�M����Ԃ� myu ���܂ނ��ۂ�'],columns='����')

    # �O���t�S�̂̃t�H���g�w��
    fsz=20                 # �}�S�̂̃t�H���g�T�C�Y
    fti=np.floor(fsz*1.2)  # �}�^�C�g���̃t�H���g�T�C�Y
    flg=np.floor(fsz*0.5)  # �}��̃t�H���g�T�C�Y
    flgti=flg              # �}��̃^�C�g���̃t�H���g�T�C�Y

    plt.rcParams["font.size"] = fsz # �}�S�̂̃t�H���g�T�C�Y�w��
    plt.rcParams['font.family'] ='IPAexGothic' # �}�S�̂̃t�H���g

    # �O���t�̔z�u�ݒ�
    figure = plt.figure()
    gs_master = GridSpec(nrows=2, ncols=2, height_ratios=[1, 1],hspace=0.5) #hspace�ŃX�y�[�X�쐬

    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1,subplot_spec=gs_master[0:1, 0]) #gs_master�ŕ`��ʒu�ݒ�
    axes_1 = figure.add_subplot(gs_1[:, :])

    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1:2, 0])
    axes_2 = figure.add_subplot(gs_2[:, :])

    gs_4 = GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_master[0:2, 1])
    axes_4 = figure.add_subplot(gs_4[:, :])

    axes_1.hist(all['���̊m���ϐ�'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('���̊m���ϐ�',loc='center',fontsize=fti)
    axes_1.legend()
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_2.hist(all['�W�{���ς̕W����'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='b',label='�j��')
    axes_2.set_title('�W�{���ς̕W�������ꂽ����\n(���͎��R�x(n-1)��t���z�Ɣ����@n=%i)' %n,loc='center',fontsize=20) 
    str1='���s�F�ʂ��܂�ł��Ȃ�:  '+str(result.iloc[0,0])+'��'
    str2='�����F�ʂ��܂�ł���@:�@'+str(result.iloc[1,0])+'��'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    (axes_4.set_title('���W�{�@�ɂ��M����Ԃ�\n�ʂ��܂�ł��邩�ۂ�\n(%d��̋�Ԑ���,n=%d)' 
         %(noexp,n),loc='center',fontsize=15))
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()
    plt.show()

daihyouhon_heikin_kukansuitei(2,'���K���z') 
daihyouhon_heikin_kukansuitei(3,'���K���z') 
daihyouhon_heikin_kukansuitei(100,'���K���z')
