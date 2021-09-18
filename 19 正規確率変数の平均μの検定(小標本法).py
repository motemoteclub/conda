#------------------------------------------------------------------------------
# ��19��
# ���W�{�@�ɂ��ʂ̌���̃V�~�����[�V����
# �@���̊m���ϐ����A����0.5�A���U1/12�̐��K���z�̏ꍇ�ōs��
# �@�����:10����
# �@���蓝�v�ʂ͈ȉ��̂Ƃ���B
#        z=(�W�{����-myu)/(�W�{���ς̕W���덷)
#   ���p��́A-t0��菬������t0���傫���B
#
#   daihyouhon_heikin_kentei(2,'���K���z')   # �T���v���T�C�Y���������Ă�������
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def daihyouhon_heikin_kentei(n,bunpu):
    noexp=100000
    myu=1/2
    sigma2=1/12

    if bunpu=='��l���z':
        # ��l�m���ϐ���(n*noexp)��A���낪���āAnoexp�s�An��̍s��ɋL�^����
        itiyou=pd.DataFrame(np.random.rand(n*noexp).reshape(noexp,n) )
    elif bunpu=='���K���z':
        # ����myu�A���U 1/12 �̐��K������n*noexp�������
        itiyou=pd.DataFrame(normal(myu,np.sqrt(sigma2),n*noexp).reshape(noexp,n))
    else:
        print('���z�Ԉ���Ă܂���')
        stop

    # �A�������ɂ�����ʂ̒l(�ȉ��̐ݒ�ł́A�A���������^�̐��E�ƈ�v���Ă���)
    myu0=myu
    heikin=pd.DataFrame(itiyou.mean(1),columns=['�W�{����'])
    bunsan=pd.DataFrame(itiyou.var(1),columns=['�W�{���U'])
    std_error=np.sqrt(bunsan.rename(columns={'�W�{���U':'�W�{���ς̕W���덷'})/n)
    hidari=-stats.t.ppf(q=0.975, df=n-1)
    migi=stats.t.ppf(q=0.975, df=n-1)
    all=pd.concat([heikin,bunsan,std_error],axis=1)
    all['���̊m���ϐ�']=itiyou.loc[:,0]
    z_henkan = lambda x: (x['�W�{����']-myu0)/x['�W�{���ς̕W���덷']
    all['z']=all.apply(z_henkan,axis=1)
    kentei=lambda x: '�A���������p' if (x['z'] < hidari or x['z'] > migi) else '�A��������e'
    #all=all.assign( test=all.apply(kentei,axis=1)).rename(columns={'test':'���茋��'})
    all['���茋��']=all.apply(kentei,axis=1)
    result=pd.crosstab(all['���茋��'],columns='����')

    # �O���t�S�̂̃t�H���g�w��
    fsz=20                 # �}�S�̂̃t�H���g�T�C�Y
    fti=np.floor(fsz*1.2)  # �}�^�C�g���̃t�H���g�T�C�Y

    figure=plt.figure(figsize=(8,5),tight_layout=True)
    axes_1 = figure.add_subplot(2,2,1)
    axes_2 = figure.add_subplot(2,2,2)
    axes_3 = figure.add_subplot(2,2,3)
    axes_4 = figure.add_subplot(2,2,4)
    axes_1.hist(all['���̊m���ϐ�'].values, bins=100, alpha=0.3, histtype='stepfilled', color='r',label='X')
    axes_1.set_title('���̊m���ϐ�' ,loc='center',fontsize=fti)
    axes_1.legend()
    axes_2.hist(all['�W�{����'].values, bins=100, alpha=0.3, histtype='stepfilled', color='b',label='X_bar')
    axes_2.set_title('�W�{����\n(n=%i)' %n,loc='center',fontsize=fti) #�}�^�C�g���̈ʒu�ƃT�C�Y
    axes_2.legend()
    kaikyu=[-4+(8/100)*i for i in range(100)]
    axes_3.hist(all['z'].values, bins=kaikyu, alpha=0.3, histtype='stepfilled', color='g',label='z')
    axes_3.set_title('z\n ���͎��R�x(n-1)��t���z�Ɣ���(n=%i)' %n,loc='center',fontsize=15) #�}�^�C�g���̈ʒu�ƃT�C�Y
    axes_3.legend()
    str1='�A��������e�F  '+str(result.iloc[0,0])+'��'
    str2='�A���������p�F�@'+str(result.iloc[1,0])+'��'
    axes_4.text(0.1, 0.8,str1 , size = 20, color = "red")
    axes_4.text(0.1, 0.6,str2 , size = 20, color = "blue")
    axes_4.set_title('���茋�ʁi���W�{�@�j\n(%d�񌟒���s����,n=%d)' %(noexp,n),loc='center',fontsize=20)
    #axes_4.set_xlabel('XXX')
    axes_4.grid(False)
    axes_4.set_facecolor('w')
    axes_4.set_axis_off()
    plt.show()

daihyouhon_heikin_kentei(2,'���K���z')   # �T���v���T�C�Y���������Ă�������
daihyouhon_heikin_kentei(3,'���K���z')   # �T���v���T�C�Y���������Ă�������
daihyouhon_heikin_kentei(100,'���K���z') # ���R�A�����������
