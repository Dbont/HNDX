import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r'E:\Work\HNDX\Keras\Data\RawData')
import pandas as pd
import numpy as np
import re
from WindPy import *
w.start()
#to test 16 index and not to 标准化
def LoadMinuteData(code = "000300.SH", date0 = "2009-01-01 9:00:00", date1 = "2017-05-09 15:01:00"):

	# set the key number
	index = "open,high,low,close,volume,amt,pct_chg,BIAS,BOLL,DMI,EXPMA,KDJ,MA,MACD,RSI"
	index = re.split(',', index) 

	# download the data
	r_data = w.wsi(code, index, date0, date1,  "BIAS_N=12;BOLL_N=26;BOLL_Width=2;BOLL_IO=1;DMI_N=14;DMI_N1=6;DMI_IO=1;EXPMA_N=12;KDJ_N=9;KDJ_M1=3;KDJ_M2=3;KDJ_IO=1;MA_N=5;MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=1;RSI_N=6;PriceAdj=F")
	# make the pd_list and transform it
	df1 = pd.DataFrame(r_data.Data)
	df = df1.T

	# add the time data
	df['time'] = r_data.Times

	# rename the name of columns
	index.append('time')
	df.columns = index

	# resort the order of columns
	cols = list(df)
	cols.insert(0, cols.pop(cols.index('time')))
	df = df.ix[:, cols]

	# calculate Y
	df['Y'] = np.nan
	n = len(df)
	for i in range(n-1):
		if df.close[i+1] > df.close[i]:
			df.Y[i] = 1
		elif df.close[i+1] < df.close[i]:
			df.Y[i] = 0
		else:
			df.Y[i] = -1

	# del Y=-1 and the first row(which Y is nan)
	df = df[df.Y != -1]
	df = df.dropna()

	# set data normal
	# time data to be saved
	#t = df.time
	#df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	#df.time = t

	# save the handled data
	name = code + ".csv"
	os.chdir(r'E:\Work\HNDX\Keras\Data\RawData')
	df.to_csv(name,index=None)


