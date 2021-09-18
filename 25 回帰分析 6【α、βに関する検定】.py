#------------------------------------------------------------------------------
# ��25��
# �R�[�i�������̎��v���ƃn���Y�}���̎��v���Ƃ����ꂼ���A���͂��āA
# ���҂̉�A���͂̌��ʂ��A��r�ł���`�ŁA�`�悷��
#------------------------------------------------------------------------------
from scipy import stats
import pandas as pd
import numpy as np
from numpy.random import *
from datetime import datetime, date, time, timedelta
from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import statsmodels.api as sm
import statsmodels.formula.api as smf

#--------------------------------------------
# �f�[�^�̓ǂݍ���
#--------------------------------------------
path=r'Q:\Dropbox\�S���V�������v�w����E��\190711�E���v�w�U�ŏI\�i�z.xlsx'
# header=0�Ȃ̂ŁAnrows=51�̓G�N�Z���łQ�s�ڂ��琔���āA51�s�܂Łi�܂�G�N�Z����52�s�܂Łj
# �ǂݍ��ނƂ����Ӗ��ɂȂ�B
data_yomikomi=pd.read_excel(path,'Sheet1',header=0,usecols=[0,4,9,14])

# �����������o��
var_names=data_yomikomi.columns # �ϐ����擾
target_name=str(var_names[2])
target_name_return=target_name+'���v��'
target_name2=str(var_names[3])
target_name2_return=target_name2+'���v��'
data=data_yomikomi.dropna().copy()

# excel�̓��t�� datetime �ɕϊ�����֐�
def excel_date(num):
    from datetime import datetime, timedelta
    return(datetime(1899, 12, 30) + timedelta(days=num))

data['���t2']=data['���t'].map(excel_date) # datetime�֕ϊ�

data.set_index('���t',inplace=True) # ���t��index�ɂ���

data.sort_index(inplace=True) # ���t���Â����̂���V�������̂֕��בւ�

# ���v���̌v�Z
data['���o���ώ��v��']=(data['���o����']/data['���o����'].shift(1)-1)*100
data[target_name_return]=(data[target_name]/data[target_name].shift(1)-1)*100
data[target_name2_return]=(data[target_name2]/data[target_name2].shift(1)-1)*100

# �����l�̂���ϑ��l���폜����
data=data.dropna()

# �L�q���v
# �ʉ��@data.loc[:,['���o���ώ��v��',target_name_return,target_name2_return]]
print(data[['���o���ώ��v��',target_name_return,target_name2_return]].describe())

# ��A����1
model=target_name_return+'~ ���o���ώ��v��'
kaiki = smf.ols(model, data=data)
koreha_instance=kaiki.fit()
print(koreha_instance.summary())

# �\���l�v�Z
data['�\���l']=koreha_instance.predict()

# ��A����2
model2=target_name2_return+'~ ���o���ώ��v��'
kaiki2 = smf.ols(model2, data=data)
koreha_instance2=kaiki2.fit()
print(koreha_instance2.summary())

# �\���l�v�Z2
data['�\���l2']=koreha_instance2.predict()

#------------------------------------------------------------------------------------
# �ȉ��A�O���t�̕`��ݒ�ƕ`��
#------------------------------------------------------------------------------------
# �O���t�S�̂̃t�H���g�w��
fsz=20                 # �}�S�̂̃t�H���g�T�C�Y
fti=np.floor(fsz*1.2)  # �}�^�C�g���̃t�H���g�T�C�Y
flg=np.floor(fsz*0.5)  # �}��̃t�H���g�T�C�Y
flgti=flg              # �}��̃^�C�g���̃t�H���g�T�C�Y

plt.rcParams["font.size"] = fsz # �}�S�̂̃t�H���g�T�C�Y�w��
plt.rcParams['font.family'] =' IPAexGothic' # �}�S�̂̃t�H���g


# �O���t�̔z�u�ݒ�
#figure = plt.figure(figsize=(8,5),tight_layout=True)

axes_1 = plt.subplot2grid((3,2),(0,0),rowspan=2)
axes_3 = plt.subplot2grid((3,2),(2,0))

axes_2 = plt.subplot2grid((3,2),(0,1),rowspan=2, sharex=axes_1, sharey=axes_1)
axes_4 = plt.subplot2grid((3,2),(2,1))

# kind='line'�ŕ����̃��C����}����ŕ`�悷�邽�߂ɁA�]���ȗ��r������B
data_draw=data[['���o���ώ��v��',target_name_return,'�\���l']] 

axes_1.scatter(data['���o���ώ��v��'].values,data[target_name_return].values,label='�����l')
axes_1.plot(data['���o���ώ��v��'].values,data['�\���l'].values,color='blue',label='�\���l')

axes_1.set_title(target_name_return)
axes_1.set_xlabel('���o���ώ��v��')

# �}��
axes_1.legend(fontsize=10)

#----------------------
# ��A�������̕`��
#----------------------
ketteikeisu='%.2f' % koreha_instance.rsquared
kaikihouteisikino_hyouzyunhensa='%.2f' % np.sqrt(koreha_instance.mse_resid)

line1='����W��= ' + ketteikeisu + '     ��A�������̕W���덷= ' + kaikihouteisikino_hyouzyunhensa


teisuko='%.3f'  % koreha_instance.params[0]
katamuki='%.3f' % koreha_instance.params[1]
line2='y= ' + teisuko + ' + ' + katamuki + 'x'

p1='%.3f'  % koreha_instance.pvalues[0]
p2='%.3f'  % koreha_instance.pvalues[1]
line3='   (' + p1 + ')' + '   ' + '(' + p2 + ')' + '      ()����p�l'

text_size=15
axes_3.text(0.0, 0.4,line1, size = text_size, color = "m")
axes_3.text(0.0, 0.2,line2, size = text_size, color = "m")
axes_3.text(0.05, 0.0,line3, size = text_size-3, color = "m")

axes_3.grid(False)        # �O���b�h������
axes_3.set_facecolor('w') # �n�̐F�̓z���C�g
axes_3.set_axis_off()     # ���̃���������


#----------------------
# ��A������2�̕`��
#----------------------
data_draw2=data[['���o���ώ��v��',target_name2_return,'�\���l2']] 

axes_2.scatter(data['���o���ώ��v��'].values,data[target_name2_return].values,label='�����l')
axes_2.plot(data['���o���ώ��v��'].values,data['�\���l2'].values,color='blue',label='�\���l')

axes_2.set_title(target_name2_return)
axes_2.set_xlabel('���o���ώ��v��')

# �}��
axes_2.legend(fontsize=10)


# ��A�������̕`��
ketteikeisu2='%.2f' % koreha_instance2.rsquared
kaikihouteisikino_hyouzyunhensa2='%.2f' % np.sqrt(koreha_instance2.mse_resid)

line1_2='����W��= ' + ketteikeisu2 + '     ��A�������̕W���덷= ' + kaikihouteisikino_hyouzyunhensa2


teisuko2='%.3f'  % koreha_instance2.params[0]
katamuki2='%.3f' % koreha_instance2.params[1]
line2_2='y= ' + teisuko2 + ' + ' + katamuki2 + 'x'

p1_2='%.3f'  % koreha_instance2.pvalues[0]
p2_2='%.3f'  % koreha_instance2.pvalues[1]
line3_2='   (' + p1_2 + ')' + '   ' + '(' + p2_2 + ')' + '      ()����p�l'

text_size=15
axes_4.text(0.0, 0.4,line1_2, size = text_size, color = "m")
axes_4.text(0.0, 0.2,line2_2, size = text_size, color = "m")
axes_4.text(0.05, 0.0,line3_2, size = text_size-3, color = "m")

axes_4.grid(False)        # �O���b�h������
axes_4.set_facecolor('w') # �n�̐F�̓z���C�g
axes_4.set_axis_off()     # ���̃���������

plt.show()
