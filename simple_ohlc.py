import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mpl_finance import candlestick_ohlc
plt.style.use('ggplot')

def SMA(ts, timeperiod):
    new_ts = [np.nan for _ in range(len(ts))]
    for i in range(timeperiod-1, len(ts)):
        new_ts[i] = np.mean(ts[i-timeperiod+1:i+1])
    return np.array(new_ts)

df = pd.read_csv('EURUSD.csv')
df['Gmt time'] = pd.to_datetime(df['Gmt time'])
data = df[df.columns][df['Volume'].apply(lambda x: x>0)]
data['fast'] = SMA(data['Close'], timeperiod=13)
data['slow'] = SMA(data['Close'], timeperiod=26)

starttime = pd.to_datetime('2018-12-14 8:00')
deadline = pd.to_datetime('2018-12-14 21:00')
quotes = data[data.columns][(starttime <= data['Gmt time'])&(data['Gmt time']<= deadline)]
quotes.dropna(inplace=True)
quotes.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots()
# make a little extra space between the subplots
#fig.subplots_adjust(hspace=0.75)
fig.subplots_adjust(bottom=0.2)

# plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(ax, zip(mdates.date2num(quotes['Gmt time']),
                        quotes['Open'], quotes['High'],
                        quotes['Low'], quotes['Close']),
                        width=0.005)

ax.xaxis_date()
ax.autoscale_view()

ax.plot(mdates.date2num(quotes['Gmt time']),quotes['fast'], color='purple')
ax.plot(mdates.date2num(quotes['Gmt time']),quotes['slow'], color='green')


plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

