import pandas as pd
import matplotlib.pyplot as plt
import tda, config
import mplfinance as mplf
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
client = tda.auth.client_from_token_file('kiran.json', config.TD_CLIENT_ID)
from statistics import mean
import tickers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def fw_draw_bands_df(bars, length=3, window=60, offset=0, interval='day', exmul=1, verbose = False, result_window=0, vol_avg=10):
    match interval:
        case 'day':
            extrapolation = 1*exmul
        case 'hour':
            extrapolation = (60.0/1440)*exmul
        case '30min':
            extrapolation = (30.0/1440)*exmul
        case '15min':
            extrapolation = (15.0/1440)*exmul
        case '10min':
            extrapolation = (10.0/1440)*exmul
        case '5min':
            extrapolation = (5.0/1440)*exmul
        case 'min':
            extrapolation = (1.0/1440)*exmul
        case _:
            return 'please use a proper interval'
    candles = pd.DataFrame(bars)
    candles.datetime = candles.datetime.apply(lambda x: datetime.fromtimestamp(x/1000))
    candles['vol_avg'] = candles.rolling(vol_avg).mean()['volume'].fillna(0)
    candles = candles[['datetime','open','high','low','close','volume','vol_avg']]
    candles.index = candles.datetime
    lowlist, highlist = [],[]
    adjustedcandles = candles.iloc[-(window+offset):min(-1,-offset)]
    adjustedcandleswindow = candles.iloc[-(window+offset):min(-1,-(offset-result_window))]
    for x in range(adjustedcandles.shape[0]-length):
        if adjustedcandles.iloc[x]['low'] == min(adjustedcandles.iloc[max(x-length,0):min(x+length+1,adjustedcandles.__len__()-1)]['low']):
            lowlist.append((str(adjustedcandles.iloc[x].datetime),adjustedcandles.iloc[x].low))
        if adjustedcandles.iloc[x]['high'] == max(adjustedcandles.iloc[max(x-length,0):min(x+length+1,adjustedcandles.__len__()-1)]['high']):
            highlist.append((str(adjustedcandles.iloc[x].datetime),adjustedcandles.iloc[x].high))
    highs = [x[1] for x in highlist]
    highdates = mdates.date2num([x[0] for x in highlist])
    lows = [x[1] for x in lowlist]
    lowdates = mdates.date2num([x[0] for x in lowlist])
    hab,hr,_,_,_ = np.polyfit(highdates,highs,1,full=True)
    lab,lr,_,_,_ = np.polyfit(lowdates,lows,1,full=True)
    ha,hb = hab
    la,lb = lab
    highdates = np.append(highdates,[mdates.date2num(adjustedcandles.iloc[-1].datetime)])
    highdates = np.append([mdates.date2num(adjustedcandles.iloc[0].datetime)],highdates)
    lowdates = np.append(lowdates,[mdates.date2num(adjustedcandles.iloc[-1].datetime)])
    lowdates = np.append([mdates.date2num(adjustedcandles.iloc[0].datetime)],lowdates)
    searr = np.array([mdates.date2num(adjustedcandles.iloc[0].datetime),mdates.date2num(adjustedcandles.iloc[-1].datetime)])
    hregpoints = (ha*highdates)+hb
    lregpoints = (la*lowdates)+lb
    hreglist = [(mdates.num2date(highdates[x]),hregpoints[x]) for x in range(len(hregpoints))]
    lreglist = [(mdates.num2date(lowdates[x]),lregpoints[x]) for x in range(len(lregpoints))]
    havdif = abs(mean([(hregpoints[x]-highs[x-1])/hregpoints[x] for x in range(1,len(hregpoints)-1)]))
    lavdif = abs(mean([(lregpoints[x]-lows[x-1])/lregpoints[x] for x in range(1,len(lregpoints)-1)]))
    vavg = mplf.make_addplot(adjustedcandleswindow['vol_avg'], color='b',panel = 1)
    if verbose: 
        mplf.plot(adjustedcandleswindow,alines=dict(alines=[lreglist,hreglist],colors=['r','g']),type='candle',volume=True, style="yahoo", addplot=vavg)
    return (ha*(highdates[-1]+extrapolation)+hb,(la*(lowdates[-1]+extrapolation)+lb)),havdif,lavdif,(ha,la), adjustedcandles
