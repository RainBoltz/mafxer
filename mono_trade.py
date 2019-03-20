import pandas as pd
import numpy as np
#from tqdm import trange
import matplotlib.pyplot as plt
plt.style.use('ggplot')

QUANTITY = 1e5
'''
def SMA(ts, timeperiod):
    new_ts = [np.nan for _ in range(len(ts))]
    for i in range(timeperiod-1, len(ts)):
        new_ts[i] = ts[i-timeperiod+1:i+1].mean()
    return np.array(new_ts)
'''
#from talib import SMA
def check_buy(fast, slow, i):
    return fast[i-1] < slow[i-1] and fast[i] > slow[i]

def check_sell(fast, slow, i):
    return fast[i-1] > slow[i-1] and fast[i] < slow[i]

def start_trade(price, quantity, i):
    this_trade = {
        'start_price': price[i],
        'start_index': i,
        'quantity': quantity,
        'end_price': None,
        'end_index': None
    }
    return this_trade

def close_trade(the_trade, price, i):
    the_trade['end_price'] = price[i]
    the_trade['end_index'] = i
    return the_trade

def GO_TRADE(fast_tp, slow_tp):
    df = pd.read_csv('EURUSD.csv')
    data = df[df.columns][df['Volume'].apply(lambda x: x>0)]
    data['fast'] = data['Close'].rolling(fast_tp).mean()
    data['slow'] = data['Close'].rolling(slow_tp).mean()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    trades = []
    for i in range(1, len(data)):
        if len(trades) == 0:
            if check_buy(data['fast'],data['slow'],i):
                trades.append(start_trade(data['Close'], QUANTITY, i))
            elif check_sell(data['fast'],data['slow'],i):
                trades.append(start_trade(data['Close'], -QUANTITY, i))
        else:
            is_holding = trades[-1]['end_price'] == None
            trade_type = 'sell' if trades[-1]['quantity'] < 0 else 'buy'
            if is_holding:
                if trade_type == 'sell' and check_buy(data['fast'],data['slow'],i):
                    trades[-1] = close_trade(trades[-1],data['Close'],i)
                    #trades.append(start_trade(data['Close'], QUANTITY, i))
                elif trade_type == 'buy' and check_sell(data['fast'],data['slow'],i):
                    trades[-1] = close_trade(trades[-1],data['Close'],i)
                    #trades.append(start_trade(data['Close'], -QUANTITY, i))
            else:
                if check_buy(data['fast'],data['slow'],i):
                    trades.append(start_trade(data['Close'], QUANTITY, i))
                elif check_sell(data['fast'],data['slow'],i):
                    trades.append(start_trade(data['Close'], -QUANTITY, i))

    return trades, data


returnN = []
tradeN = []
PFN = []


test_list = [(i,i*2) for i in range(5,30+1)]
plot_x = [d[0] for d in test_list]
output = {'fast_tp':[], 'slow_tp':[], 'n':[],'return':[],'pf':[]}
for fast_tp, slow_tp in test_list:
    trades, _ =  GO_TRADE(fast_tp,slow_tp)

    acc_ret = 0.0
    loss = 0.0
    profit = 0.0
    for trade in trades:
        if trade['end_price'] == None:
            continue
        diff = (trade['end_price']-trade['start_price'])*trade['quantity']
        acc_ret += diff
        if diff < 0:
            loss += diff
        else:
            profit += diff

    returnN.append(acc_ret)    
    tradeN.append(len(trades))
    pf = profit/-loss if loss != 0 else float('Inf')
    PFN.append(pf if pf != float('Inf') else float('NaN'))
    
    print('(%02d,%02d)x%d: return=%d, pf=%.3f'%(fast_tp,
                                                slow_tp,
                                                len(trades),
                                                int(acc_ret),
                                                pf))
    
    output['fast_tp'].append(fast_tp)
    output['slow_tp'].append(slow_tp)
    output['n'].append(len(trades))
    output['return'].append(int(acc_ret))
    output['pf'].append(pf)
    
pd.DataFrame(output).to_csv('output.csv')
    
'''
plt.plot(plot_x,returnN)
plt.title('Total Returns')
plt.xlabel('moving average-N  (5 min.)')
plt.ylabel('pipettes (EURUSD)')
plt.grid(True)
plt.show()
plt.plot(plot_x,tradeN)
plt.title('Trade Amounts')
plt.xlabel('moving average-N  (5 min.)')
plt.ylabel('times')
plt.grid(True)
plt.show()
plt.plot(plot_x,PFN)
plt.title('Profit Factors')
plt.xlabel('moving average-N  (5 min.)')
plt.ylabel('profit/loss')
plt.grid(True)
plt.show()


trades, data =  GO_TRADE(13,26)
mdd = 0.0
profit = 0.0
loss = 0.0
cost = 0.0
plot_ret = []
acc_ret = 0.0
if trades[-1]['end_index'] == None:
    trades = trades[:-1]
for trade in trades:
    for i in range(trade['start_index']+1, trade['end_index']):
        trade_type = 'sell' if trades[-1]['quantity'] < 0 else 'buy'
        if trade_type == 'sell':
            dd = (trade['start_price'] - data['Close'][i])/trade['start_price']
        else:
            dd = (data['Close'][i] - trade['start_price'])/trade['start_price']
    mdd = dd if dd < mdd else mdd
    diff = (trade['end_price']-trade['start_price'])*trade['quantity']
    acc_ret += diff
    plot_ret.append(acc_ret)
    if diff < 0:
        loss += diff
    else:
        profit += diff
    cost += trade['start_price']*abs(trade['quantity'])
print("RET: %f\nROI: %f\nPF: %f\nN: %d\nMDD: %f"%(profit+loss,(profit-loss)/cost,profit/loss,len(trades),mdd))
plt.plot(range(len(plot_ret)),plot_ret)
plt.title('Accumulated Return')
plt.xlabel('trades (#id)')
plt.ylabel('pipettes (EURUSD)')
plt.show()
'''