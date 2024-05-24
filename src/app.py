from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd

df = pd.read_csv('../data/raw/River_Arno.csv')
df.info()
df_st = df[['Date','Hydrometry_Nave_di_Rosano']]
df_st['Date'] = pd.to_datetime(df_st['Date'],format="%d/%m/%Y").dt.date
df_st = df_st.set_index('Date')

df_st

import matplotlib.pyplot as plt
import seaborn as sns

plt.subplots(figsize=(10,5))
sns.lineplot(data=df_st)
plt.tight_layout()
plt.show()

df_st.dropna(inplace=True)
from statsmodels.tsa.seasonal import seasonal_decompose

descomposion = seasonal_decompose(df_st,period=365)
trend = descomposion.trend

plt.subplots(figsize=(10,5))
sns.lineplot(data=df_st)
sns.lineplot(data=trend)
plt.tight_layout()
plt.show()

estacionalidad = descomposion.seasonal

plt.subplots(figsize=(10,5))
sns.lineplot(data=df_st)
sns.lineplot(data=estacionalidad)
plt.tight_layout()
plt.show()

resid = descomposion.resid

plt.subplots(figsize=(10,5))
sns.lineplot(data=df_st)
sns.lineplot(data=resid)
plt.tight_layout()
plt.show()

df_train = df_st[df_st.index<pd.to_datetime('2020-06-01').date()]
df_test = df_st[df_st.index>=pd.to_datetime('2020-06-01').date()]

from pmdarima import auto_arima

model = auto_arima(df_train,seasonal=True,trace=True,m=30)

forecast = model.predict(30)
from statsmodels.tsa.arima.model import ARIMA

model_a = ARIMA(df_train['Hydrometry_Nave_di_Rosano'], order=(3,1,2))
model_fit = model_a.fit()

forecast = model.predict(30)

from sklearn.metrics import mean_squared_error
mean_squared_error(df_test,forecast)

forecast.index = df_test.index

plt.subplots(figsize=(10,5))
sns.lineplot(data=df_test)
sns.lineplot(data=forecast)
plt.tight_layout()
plt.show()

























