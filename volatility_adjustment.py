import time
import urllib.request
import io
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, ks_1samp
import matplotlib.pyplot as plt

days_back = 365 * 9

secs_in_one_day = 86400
_now = int(time.time())
print("The period from ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_now-  secs_in_one_day * days_back)), " to ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_now)))

target_url_spx = 'https://query2.finance.yahoo.com/v7/finance/download/^GSPC?period1=' + str(_now - secs_in_one_day * days_back) + '&period2=' + str(_now) + '&interval=1d&events=history&crumb=2MbBGaxDo3l'

target_url_vix = 'https://query2.finance.yahoo.com/v7/finance/download/^VIX?period1=' + str(_now - secs_in_one_day * days_back) + '&period2=' + str(_now) + '&interval=1d&events=history&crumb=2MbBGaxDo3l'

with urllib.request.urlopen(target_url_spx) as response:
   csv1 = response.read()
   
with urllib.request.urlopen(target_url_vix) as response:
   csv2 = response.read()

spx = pd.read_csv(io.BytesIO(csv1), encoding='utf8', index_col='Date')
vix = pd.read_csv(io.BytesIO(csv2), encoding='utf8', index_col='Date')

spx_close = spx.loc[:,'Close']
spx_close = np.log(spx_close).diff().dropna()

vix_close = vix.loc[1:,'Close']

# Normalization without VIX-adjustment
spx_close_unadjusted = (spx_close - spx_close.mean()) / spx_close.std()
spx_close_unadjusted = spx_close_unadjusted.sort_values()

# Normalization with VIX-adjustment
df = spx_close / vix_close
df = (df - df.mean()) / df.std()
df_sorted = df.sort_values()

for x in (1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0):
    print("The actual estimated probability of deviation beyond %3.1f standard deviations is %10.8f vs. the theoretical probability of %10.8f" % (x, (sum(df > x) / df.shape[0] + sum(df < -x) / df.shape[0])/2.0, 1.0 - norm.cdf(x)))

print("Shapiro test: ", shapiro(df_sorted))
print("Kolmogorov-Smirnov test: ", ks_1samp(df_sorted, norm.cdf))

normal_quantiles = norm.cdf(df_sorted)
real_quantiles = np.linspace(0.0, 1.0, df_sorted.shape[0])

plt.plot(spx_close_unadjusted, real_quantiles)
plt.plot(df_sorted, normal_quantiles)
plt.plot(df_sorted, real_quantiles)
plt.show()
