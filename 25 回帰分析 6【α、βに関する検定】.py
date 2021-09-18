#------------------------------------------------------------------------------
# 第25章
# コーナン商事の収益率とハンズマンの収益率とをそれぞれ回帰分析して、
# 両者の回帰分析の結果を、比較できる形で、描画する
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
# データの読み込み
#--------------------------------------------
path=r'Q:\Dropbox\全く新しい統計学入門・昼\190711・統計学Ⅱ最終\ナホ.xlsx'
# header=0なので、nrows=51はエクセルで２行目から数えて、51行まで（つまりエクセルの52行まで）
# 読み込むという意味になる。
data_yomikomi=pd.read_excel(path,'Sheet1',header=0,usecols=[0,4,9,14])

# 銘柄名を取り出す
var_names=data_yomikomi.columns # 変数名取得
target_name=str(var_names[2])
target_name_return=target_name+'収益率'
target_name2=str(var_names[3])
target_name2_return=target_name2+'収益率'
data=data_yomikomi.dropna().copy()

# excelの日付を datetime に変換する関数
def excel_date(num):
    from datetime import datetime, timedelta
    return(datetime(1899, 12, 30) + timedelta(days=num))

data['日付2']=data['日付'].map(excel_date) # datetimeへ変換

data.set_index('日付',inplace=True) # 日付をindexにする

data.sort_index(inplace=True) # 日付を古いものから新しいものへ並べ替え

# 収益率の計算
data['日経平均収益率']=(data['日経平均']/data['日経平均'].shift(1)-1)*100
data[target_name_return]=(data[target_name]/data[target_name].shift(1)-1)*100
data[target_name2_return]=(data[target_name2]/data[target_name2].shift(1)-1)*100

# 欠損値のある観測値を削除する
data=data.dropna()

# 記述統計
# 別解　data.loc[:,['日経平均収益率',target_name_return,target_name2_return]]
print(data[['日経平均収益率',target_name_return,target_name2_return]].describe())

# 回帰分析1
model=target_name_return+'~ 日経平均収益率'
kaiki = smf.ols(model, data=data)
koreha_instance=kaiki.fit()
print(koreha_instance.summary())

# 予測値計算
data['予測値']=koreha_instance.predict()

# 回帰分析2
model2=target_name2_return+'~ 日経平均収益率'
kaiki2 = smf.ols(model2, data=data)
koreha_instance2=kaiki2.fit()
print(koreha_instance2.summary())

# 予測値計算2
data['予測値2']=koreha_instance2.predict()

#------------------------------------------------------------------------------------
# 以下、グラフの描画設定と描画
#------------------------------------------------------------------------------------
# グラフ全体のフォント指定
fsz=20                 # 図全体のフォントサイズ
fti=np.floor(fsz*1.2)  # 図タイトルのフォントサイズ
flg=np.floor(fsz*0.5)  # 凡例のフォントサイズ
flgti=flg              # 凡例のタイトルのフォントサイズ

plt.rcParams["font.size"] = fsz # 図全体のフォントサイズ指定
plt.rcParams['font.family'] =' IPAexGothic' # 図全体のフォント


# グラフの配置設定
#figure = plt.figure(figsize=(8,5),tight_layout=True)

axes_1 = plt.subplot2grid((3,2),(0,0),rowspan=2)
axes_3 = plt.subplot2grid((3,2),(2,0))

axes_2 = plt.subplot2grid((3,2),(0,1),rowspan=2, sharex=axes_1, sharey=axes_1)
axes_4 = plt.subplot2grid((3,2),(2,1))

# kind='line'で複数のラインを凡例つきで描画するために、余分な列を排除する。
data_draw=data[['日経平均収益率',target_name_return,'予測値']] 

axes_1.scatter(data['日経平均収益率'].values,data[target_name_return].values,label='実測値')
axes_1.plot(data['日経平均収益率'].values,data['予測値'].values,color='blue',label='予測値')

axes_1.set_title(target_name_return)
axes_1.set_xlabel('日経平均収益率')

# 凡例
axes_1.legend(fontsize=10)

#----------------------
# 回帰方程式の描写
#----------------------
ketteikeisu='%.2f' % koreha_instance.rsquared
kaikihouteisikino_hyouzyunhensa='%.2f' % np.sqrt(koreha_instance.mse_resid)

line1='決定係数= ' + ketteikeisu + '     回帰方程式の標準誤差= ' + kaikihouteisikino_hyouzyunhensa


teisuko='%.3f'  % koreha_instance.params[0]
katamuki='%.3f' % koreha_instance.params[1]
line2='y= ' + teisuko + ' + ' + katamuki + 'x'

p1='%.3f'  % koreha_instance.pvalues[0]
p2='%.3f'  % koreha_instance.pvalues[1]
line3='   (' + p1 + ')' + '   ' + '(' + p2 + ')' + '      ()内はp値'

text_size=15
axes_3.text(0.0, 0.4,line1, size = text_size, color = "m")
axes_3.text(0.0, 0.2,line2, size = text_size, color = "m")
axes_3.text(0.05, 0.0,line3, size = text_size-3, color = "m")

axes_3.grid(False)        # グリッド線消去
axes_3.set_facecolor('w') # 地の色はホワイト
axes_3.set_axis_off()     # 軸のメモリ消去


#----------------------
# 回帰方程式2の描写
#----------------------
data_draw2=data[['日経平均収益率',target_name2_return,'予測値2']] 

axes_2.scatter(data['日経平均収益率'].values,data[target_name2_return].values,label='実測値')
axes_2.plot(data['日経平均収益率'].values,data['予測値2'].values,color='blue',label='予測値')

axes_2.set_title(target_name2_return)
axes_2.set_xlabel('日経平均収益率')

# 凡例
axes_2.legend(fontsize=10)


# 回帰方程式の描写
ketteikeisu2='%.2f' % koreha_instance2.rsquared
kaikihouteisikino_hyouzyunhensa2='%.2f' % np.sqrt(koreha_instance2.mse_resid)

line1_2='決定係数= ' + ketteikeisu2 + '     回帰方程式の標準誤差= ' + kaikihouteisikino_hyouzyunhensa2


teisuko2='%.3f'  % koreha_instance2.params[0]
katamuki2='%.3f' % koreha_instance2.params[1]
line2_2='y= ' + teisuko2 + ' + ' + katamuki2 + 'x'

p1_2='%.3f'  % koreha_instance2.pvalues[0]
p2_2='%.3f'  % koreha_instance2.pvalues[1]
line3_2='   (' + p1_2 + ')' + '   ' + '(' + p2_2 + ')' + '      ()内はp値'

text_size=15
axes_4.text(0.0, 0.4,line1_2, size = text_size, color = "m")
axes_4.text(0.0, 0.2,line2_2, size = text_size, color = "m")
axes_4.text(0.05, 0.0,line3_2, size = text_size-3, color = "m")

axes_4.grid(False)        # グリッド線消去
axes_4.set_facecolor('w') # 地の色はホワイト
axes_4.set_axis_off()     # 軸のメモリ消去

plt.show()
