import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from mpl_finance import candlestick_ohlc
plt.style.use('ggplot')

def SMA(ts, timeperiod):
    new_ts = [np.nan for _ in range(len(ts))]
    for i in range(timeperiod-1, len(ts)):
        new_ts[i] = np.mean(ts[i-timeperiod+1:i+1])
    return np.array(new_ts)

df = pd.read_csv('EURTRYUSDTRY.csv')
df['Gmt time'] = pd.to_datetime(df['Gmt time'])
data = df[df.columns][(df['EURUSD_Volume'].apply(lambda x: x>0)) & \
                        (df['EURTRY_Volume'].apply(lambda x: x>0)) & \
                        (df['USDTRY_Volume'].apply(lambda x: x>0))]
data['EURTRY_fast'] = SMA(data['EURTRY_Close'], timeperiod=14)
data['EURTRY_slow'] = SMA(data['EURTRY_Close'], timeperiod=28)
data['USDTRY_fast'] = SMA(data['USDTRY_Close'], timeperiod=14)
data['USDTRY_slow'] = SMA(data['USDTRY_Close'], timeperiod=28)
starttime = pd.to_datetime('2018-12-14 09:00')
quotes = data[data.columns][data['Gmt time'] >= starttime]
quotes.dropna(inplace=True)
quotes.reset_index(drop=True, inplace=True)

fig, (AB, AC, BC) = plt.subplots(3, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.75)
fig.subplots_adjust(bottom=0.2)

# plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(AB, zip(mdates.date2num(quotes['Gmt time']),
                        quotes['EURUSD_Open'], quotes['EURUSD_High'],
                        quotes['EURUSD_Low'], quotes['EURUSD_Close']),
                        width=0.005)
AB.xaxis_date()
AB.autoscale_view()

candlestick_ohlc(AC, zip(mdates.date2num(quotes['Gmt time']),
                        quotes['EURTRY_Open'], quotes['EURTRY_High'],
                        quotes['EURTRY_Low'], quotes['EURTRY_Close']),
                        width=0.005)
AC.xaxis_date()
AC.autoscale_view()
AC.plot(mdates.date2num(quotes['Gmt time']),quotes['EURTRY_fast'], color='purple')
AC.plot(mdates.date2num(quotes['Gmt time']),quotes['EURTRY_slow'], color='green')

candlestick_ohlc(BC, zip(mdates.date2num(quotes['Gmt time']),
                        quotes['USDTRY_Open'], quotes['USDTRY_High'],
                        quotes['USDTRY_Low'], quotes['USDTRY_Close']),
                        width=0.005)
BC.xaxis_date()
BC.autoscale_view()
BC.plot(mdates.date2num(quotes['Gmt time']),quotes['USDTRY_fast'], color='purple')
BC.plot(mdates.date2num(quotes['Gmt time']),quotes['USDTRY_slow'], color='green')

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

