# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:13:55 2023
modified on Sun Oct 29 21:55:55 2023
@author: Guoheng Qi, qi.gh@outlook.com

Reference: http://geophydog.cool/post/signal_pws_stack/
"""

import math,os,obspy,scipy,threading,time
import numpy as np
import pandas as pd
import numpy.fft as fftpack
import scipy.signal as sgn
from scipy.fftpack import rfft,irfft,fftfreq
from scipy.optimize import curve_fit
from stockwell import st
from scipy.signal import hilbert
from multiprocessing import Process
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import font_manager,cm
from matplotlib.colors import Normalize
from matplotlib.dates import MINUTELY,AutoDateLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,FuncFormatter
from obspy import Trace,Stream,UTCDateTime,read
from obspy.signal.util import smooth

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['Times New Roman']

#============================================================================
'''
      phase stack (c)
'''
#============================================================================

def get_coh(st, v, sm = False, sl = 20):
    '''
    输入：st, a stream;  v>=0,锐度;   sm, a bool;  sl, smooth window width
    输出：t, phase stack
    '''
    m  = len(st)
    n  = st[0].stats.npts
    dt = st[0].stats.delta
    t  = np.arange(n) * dt
    ht = np.zeros((m, n), dtype=complex)
    c  = np.zeros(n)
    for i, tr in enumerate(st):
        ht[i] = hilbert(tr.data)
    pha = ht / abs(ht)
    for i in range(n):
        c[i] = abs( sum(pha[:, i]) )
    # Smooth the coherence if necessary.
    if sm:
        c = np.convolve(c/m, np.ones(sl)/sl, 'same') ** v
    else:
        c = ( c/m ) ** v
    return t, c

#============================================================================
'''
      phase cross-correlation (c)
'''
#============================================================================

def PCC(tr1,tr2,v,t,m):
    '''
    input: tr1,tr2 are traces
           v >=0, 锐度
           t is max delayed time 
           m is number of delayed time
    output:phase cross-correlation
    '''
#    print('PCC is start!' )
    n   = tr1.stats.npts
    dt  = tr1.stats.delta
    pcc = np.zeros(m)
    ph1 = hilbert(tr1.data) / abs(hilbert(tr1.data))
    ph2 = hilbert(tr2.data) / abs(hilbert(tr2.data))

    for i in np.arange(m):
        ph0 = ph1*0
        lag = i*t/m
        et  = int(lag/dt)
        ph0[:n-et] = ph2[et:]
#        print('lag time is {} is end'.format(lag))
        pcc[i] = 1/2/n * ( sum( abs(ph1+ph0)**v - abs(ph0-ph1)**v ) )
#    plt.plot(pcc)
    Pcc = Trace()
    Pcc.data = pcc
#    print('PCC is end!' )
    return Pcc
#============================================================================
'''
      phase-weight stack (PWS)
'''
#============================================================================

def PWS(st, v, sm=False, sl=15):
    
    print('PWS is start!' )
    m = len(st)
    n = st[0].stats.npts
#    dt = st[0].stats.delta
#    t = np.arange(n) * dt
    c = np.zeros(n, dtype=complex)
    for i, tr in enumerate(st):
        h = hilbert(tr.data)
        c += h/abs(h)
    c = abs(c/m)
    if sm:
        operator = np.ones(sl) / sl
        c = np.convolve(c, operator, 'same')
    stc = st.copy()
    stc.stack()
    tr = stc[0]
    tr.data = tr.data*c**v
    
    print('PWS is end!' )
    return tr


#============================================================================
'''
      time-frequency phase-weight stack (tf-pws)
'''
#============================================================================

def tf_PWS(stream,t, v, fmin, fmax):
    '''
    f1 is fmin
    f2 is fmax
    '''
#    dt = stream[0].stats.delta
#    b = stream[0].stats.sac.b
#    t = np.arange(stream[0].stats.npts) * dt + b
#    df = 1 / (stream[0].stats.sac.e-b)

    df = 1 / (t[-1]-t[0]) ## sampling step in frequency domain (Hz)
    
#    fn1 = int(f1/df); fn2 = int(f2/df)
    fmin_samples = int(fmin/df)
    fmax_samples = int(fmax/df)
    
    
    for i, tr in enumerate(stream):
        d = tr.data
        s = st.st(d, fmin_samples, fmax_samples)
        if i < 1:
            f = np.linspace(fmin, fmax, len(s[:, 0]))
            T, F = np.meshgrid(t, f)
            c = np.zeros_like(s)
        ph = s / abs(s) * np.exp(2j*np.pi*F*T)
        c += ph
    c  /= len(stream)
    stc = stream.copy()
    stc.stack()
    ds  = st.st(stc[0].data, fmin_samples, fmax_samples)
    pws = st.ist(ds*abs(c)**v, fmin_samples, fmax_samples)
    tr = Trace()
    tr.data = pws
    return tr

#============================================================================
    
# 找字符串substr在str中第time次出现的位置

#============================================================================

def findSubStrIndex(substr, str, time):
    times = str.count(substr)
    if (times == 0) or (times < time):
        pass
    else:
        i = 0
        index = -1
        while i < time:
            index = str.find(substr, index+1)
            i+=1
        return index
    
#============================================================================

#   初始化Trace.stats的特征信息

#============================================================================
   
def InitStats(stats, station, locationCH, channel, starttime, sampling_rate):
    '''
    station, locationCH, channel should be str
    sampling_rate should be integer
    starttime: ...
    '''
    stats.network       = 'HS'
    stats.station       = station
    stats.location      = locationCH
    stats.channel       = channel
    stats.starttime     = starttime
    stats.sampling_rate = sampling_rate

#============================================================================

#   读取txt文件，存入stream

#============================================================================

def txt_to_stream(txtname,path,station):
       
#    txtname = 'SemiPhase 1.0k 100d 20230301003301.947.txt'
    pos_1       = findSubStrIndex(' ',txtname,2)     # 第2次出现空格的索引 
    pos_2       = txtname.index('d')                 # d的索引
    f           = 1000/int(txtname[pos_1+1:pos_2])
    starttime   = txtname[pos_2+2:pos_2+16]
    phase       = np.loadtxt(path+'/'+txtname)
    print(txtname+' is loaded!')
#    num_chl     = np.shape(phase)[1]
    Phase       = Stream()
    channel     = ston_chl[station][0]
    ch          = ston_chl[station][1]
    ck          = ston_chl[station][2]
    #for i in range(num_chl):
    for i in ch:  
        tr      = Trace()
        tr.data = phase[:,i] - phase[:,ck]      #   参考补偿
#        tr.detrend()
        tr.detrend("spline", order = 3, dspline = 30*f)
#        tr.resample(1)
#        f = 1
        InitStats(tr.stats , station, str(i+1).zfill(2) , channel[i] , UTCDateTime(starttime) , f ) # 初始化头文件
        Phase.append(tr)
    Phase.resample(10)
    return Phase
#============================================================================

#   寻找信号的局部极大值

#============================================================================

def find_local_peaks(t,signal):
    """
    寻找信号的局部极大值
    
    参数：
    signal -- 一个列表，表示输入的信号
    
    返回值：
    一个列表，包含所有局部极大值的索引
    """
    if len(signal) < 3:
        return []  # 信号长度不足以找到局部极大值
    
    peaks = []
    peaks_index = []
    for i in range(2, len(signal)-3):
        if signal[i] >= signal[i-1] and signal[i] >= signal[i+1] and signal[i] > signal[i-2] and signal[i] > signal[i+2] :#and signal[i] > 0:
            peaks_index.append(t[i])
            peaks.append(signal[i])
    
    return np.array(peaks), np.array(peaks_index)
#============================================================================

#   对信号的局部极大值进行多项式拟合

#============================================================================

def fit_local_peaks(t,signal):
    """
    对信号的局部极大值进行多项式拟合
    
    参数：
    signal -- 一个列表，表示输入的信号
    degree -- 用于拟合的多项式的次数。默认值为 2
    
    返回值：
    一个元组，包含两个数组：
    - 第一个数组包含拟合函数在每个信号点的值
    - 第二个数组包含拟合函数的二阶导数在每个信号点的值
    """
    # 找到局部极大值
#    signal = np.convolve(signal, np.ones(2)/2, 'same')
    peaks,peaks_index = find_local_peaks(t,signal)
    
    # 提取局部极大值的 x 和 y 坐标
    x = np.array(peaks_index).T
    y = np.array(peaks)
    
    # 进行多项式拟合
#    p = np.polyfit(x, y, degree)
    
    # 计算拟合函数和它的二阶导数
#    print(x)
#    print(y)
#    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(x[0], x[-1], 400)
#    y_new = f(x_new)
    
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(x, y)
    y_2d_new = cs(x_new,2)
    y_new = cs(x_new)
#    fit_y_2d = np.polyval(np.polyder(f, 2), x_new[2:-4])
    
    return x_new, y_new, y_2d_new


#------------------------------------------------------------------------------------------------------------

XSMchl   = ['E' ,'E'  ,'N' ,'N'  ,'NW','NW','NE','NE' ,'a-E','a-E','a-N','a-N','con','con','con','Tem']
HJCchl   = ['NE','NE' ,'NW','NW' ,'E' ,'E' ,'N' ,'N'  ,'a-N','a-E','con','con','con','con','con','Tem']
BYAchl   = ['NE','NE' ,'NW','con','E' ,'E' ,'N' ,'N'  ,'a-E','a-N','a-E','con','con','con','con','Tem']
ZJWchl   = ['NE','con','NW','NW' ,'E' ,'E' ,'N' ,'con','E'  ,'N'  ,'a-E','a-E','con','con','con','Tem']
ston_chl = {'XSM': [ XSMchl, [0,3,4,7], 12],
            'HJC': [ HJCchl, [4,7,3,1], 11],
            'BYA': [ BYAchl, [5,7,2,0], 12],
            'ZJW': [ ZJWchl, [5,6,2,4],   12]}

#path = 'F:/霍山数据/1西石门/2023-03'
#txtnames = os.listdir(path)
#numtxt = len(txtnames)
lagt = 50
m = 500
#pcc = np.zeros(shape=(300,1))
t = np.linspace(0,lagt,m)
#pcc=t*0
#txtnames = ['SemiPhase 1.0k 100d 20230331103404.654.txt']#SemiPhase 1.0k 100d 20230331233404.786
#pcc_st = Stream()
#-----------------------------------------------------------------------------------------------------
'''
      get pcc 
'''
def gerpac(path, fmin, fmax, station, outfilename):
    '''
    e.g. 
    
    path = 'F:/霍山数据/1西石门/2023-03'
    station = 'XSM'
    outfilename = 'xsm04_pcc_202303.mseed'
    
    '''
    print('now the path is '+path)
    txtnames = os.listdir(path)
    
    pac1 = Stream()
    pac2 = Stream()
    pac3 = Stream()
    pac4 = Stream()
    
    for txtname in txtnames:
        phase = txt_to_stream(txtname,path,station)
        if phase[1].stats.npts == 36000 or phase[1].stats.npts ==360000:
            
            phase.filter('bandpass', freqmin = fmin, freqmax = fmax, zerophase = True) 
            pac1 += PCC(phase[0], phase[0], 2, lagt, m)
            pac2 += PCC(phase[1], phase[1], 2, lagt, m)
            pac3 += PCC(phase[2], phase[2], 2, lagt, m)
            pac4 += PCC(phase[3], phase[3], 2, lagt, m)

        del phase
#    print(find_local_peaks(t,pcc))
            
    pac1.write(outfilename+'_01.mseed', format='MSEED')
    pac2.write(outfilename+'_02.mseed', format='MSEED') 
    pac3.write(outfilename+'_03.mseed', format='MSEED')
    pac4.write(outfilename+'_04.mseed', format='MSEED')
#    
    print(outfilename+' from 1 to 4 compenonts is saved successfully!')
    
#            PCC(phase[0], phase[0], 2, lagt, m).write(station+txtname+'_01.mseed', format='MSEED')
#            PCC(phase[1], phase[1], 2, lagt, m).write(station+txtname+'_02.mseed', format='MSEED')
#            PCC(phase[2], phase[2], 2, lagt, m).write(station+txtname+'_03.mseed', format='MSEED')
#            PCC(phase[3], phase[3], 2, lagt, m).write(station+txtname+'_04.mseed', format='MSEED')
#
#        del phase

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#gerpac('F:/霍山数据/1西石门/2023-03', 0.5, 1, 'XSM', 'xsm0510_pcc_202303')
#gerpac('F:/霍山数据/2画家村/202303', 1, 2, 'HJC', 'hjc12_pcc_202303')
#gerpac('F:/霍山数据/3白云庵/202303', 1, 2, 'BYA', 'bya12_pcc_202303')
#gerpac('F:/霍山数据/4张家塆/202303', 1, 2, 'ZJW', 'zjw12_pcc_202303')
#gerpac('F:/霍山数据/1西石门/2023-02', 1, 2, 'XSM', 'xsm12_pcc_202302')
#gerpac('F:/霍山数据/2画家村/202302', 1, 2, 'HJC', 'hjc12_pcc_202302')
#gerpac('F:/霍山数据/3白云庵/202302', 1, 2, 'BYA', 'bya12_pcc_202302')
#gerpac('F:/霍山数据/4张家塆/202304', 1, 2, 'ZJW', 'zjw12_pcc_202304')
#
#gerpac('F:/霍山数据/1西石门/2023-01', 1, 2, 'XSM', 'xsm12_pcc_202301')
#gerpac('F:/霍山数据/2画家村/202301', 1, 2, 'HJC', 'hjc12_pcc_202301')
#gerpac('F:/霍山数据/3白云庵/202301', 1, 2, 'BYA', 'bya12_pcc_202301')
#gerpac('F:/霍山数据/4张家塆/202301', 1, 2, 'ZJW', 'zjw12_pcc_202301')

#gerpac('F:/霍山数据/1西石门/2022-12', 1, 2, 'XSM', 'xsm12_pcc_202212')
#gerpac('F:/霍山数据/1西石门/2022-1013-1116', 2, 4, 'XSM', 'xsm24_pcc_202303')
#gerpac('F:/霍山数据/1西石门/2023-02', 2, 4, 'XSM', 'xsm24_pcc_202302')
#gerpac('F:/霍山数据/1西石门/2023-01', 2, 4, 'XSM', 'xsm24_pcc_202301')
#gerpac('F:/霍山数据/1西石门/2023-02', 0.5, 1.0, 'XSM', 'xsm0510_pcc_202302')
#gerpac('F:/霍山数据/1西石门/2023-01', 0.5, 1.0, 'XSM', 'xsm0510_pcc_202301')


#added_thread1 = threading.Thread(target = gerpac,args=('F:/霍山数据/1西石门/2023-04', 1, 2, 'XSM', 'xsm12_pcc_202304'))
#added_thread2 = threading.Thread(target = gerpac,args=('F:/霍山数据/2画家村/202304', 1, 2, 'HJC', 'hjc12_pcc_202304'))
#added_thread3 = threading.Thread(target = gerpac,args=('F:/霍山数据/3白云庵/202304', 1, 2, 'BYA', 'bya12_pcc_202304'))
#
#added_thread1.start()
#added_thread2.start()
#added_thread3.start()

#__spec__ = None
#__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

#if __name__ == '__main__':
##    __spec__ = None
#    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#    p1 = Process(target = gerpac, args = ('F:/霍山数据/4张家塆/202303', 1, 2, 'ZJW', 'zjw12_pcc_202303',))
#    p2 = Process(target = gerpac, args = ('F:/霍山数据/1西石门/2023-02', 1, 2, 'XSM', 'xsm12_pcc_202302',))
#    p1.start()
#    p2.start()
#    p1.join()
#    p2.join()
    
##-------------------------------------------------------------------------------------------------------------------------------------------------

#pcc=t*0
##txtnames = ['SemiPhase 1.0k 100d 20230331103404.654.txt']#SemiPhase 1.0k 100d 20230331233404.786
#pcc_st = Stream()
#for txtname in txtnames:
#    phase = txt_to_stream(txtname,path,'XSM')
#    if phase[0].stats.npts == 36000:
#        phase.filter('bandpass', freqmin = 1, freqmax = 2, zerophase = True) 
#        pcc_tr = Trace()
#        pcc_tr.data = PCC(phase[2],phase[2],2,lagt,m)
#        pcc_st += pcc_tr
#        pcc += pcc_tr.data
##    del phase
#
#pcc_st.write('xsm05_pcc_202303.mseed',format='MSEED')
#
#pcc=t*0
##txtnames = ['SemiPhase 1.0k 100d 20230331103404.654.txt']#SemiPhase 1.0k 100d 20230331233404.786
#pcc_st = Stream()
#for txtname in txtnames:
#    phase = txt_to_stream(txtname,path,'XSM')
#    if phase[0].stats.npts == 36000:
#        phase.filter('bandpass', freqmin = 1, freqmax = 2, zerophase = True) 
#        pcc_tr = Trace()    
#        pcc_tr.data = PCC(phase[3],phase[3],2,lagt,m)
#        pcc_st += pcc_tr
#        pcc += pcc_tr.data
##    del phase
#
#pcc_st.write('xsm08_pcc_202303.mseed',format='MSEED')
#-----------------------------------------------------------------------------------------------------


#-------------------------------地震---------------------------------------------------------------------
#path = 'F:/霍山数据/1西石门/2023-02'
#txtname = 'SemiPhase 1.0k 100d 20230206094233.173.txt'
#txtname2 = 'SemiPhase 1.0k 100d 20230206104233.185.txt'
#
#phase1 = txt_to_stream(txtname,path,'XSM')
#phase2 = txt_to_stream(txtname2,path,'XSM')
#
#phase = phase1+phase2
#phase = phase.slice(starttime = UTCDateTime('20230206103000'), endtime =UTCDateTime('20230206114000'))
#=============================================================================================================
#=============================================================================================================
'''
        绘图
'''
##----------------- read PAC --------------------------

#pccst = read('xsm01_pcc_202303.mseed')  
pccst1 = read('bya12_pcc_202303_01.mseed')+read('bya12_pcc_202302_01.mseed')+read('bya12_pcc_202301_01.mseed')
pccst2 = read('bya12_pcc_202303_02.mseed')+read('bya12_pcc_202302_02.mseed')+read('bya12_pcc_202301_02.mseed')
pccst3 = read('bya12_pcc_202303_03.mseed')+read('bya12_pcc_202302_03.mseed')+read('bya12_pcc_202301_03.mseed')
pccst4 = read('bya12_pcc_202303_04.mseed')+read('bya12_pcc_202302_04.mseed')+read('bya12_pcc_202301_04.mseed')

pccst5 = read('hjc12_pcc_202303_01.mseed')+read('hjc12_pcc_202302_01.mseed')+read('hjc12_pcc_202301_01.mseed')
pccst6 = read('hjc12_pcc_202303_02.mseed')+read('hjc12_pcc_202302_02.mseed')+read('hjc12_pcc_202301_02.mseed')
pccst7 = read('hjc12_pcc_202303_03.mseed')+read('hjc12_pcc_202302_03.mseed')+read('hjc12_pcc_202301_03.mseed')
pccst8 = read('hjc12_pcc_202303_04.mseed')+read('hjc12_pcc_202302_04.mseed')+read('hjc12_pcc_202301_04.mseed')

pccst9 = read('xsm01_pcc_202303.mseed')+read('xsm12_pcc_202302_01.mseed')+read('xsm12_pcc_202301_01.mseed')+read('xsm12_pcc_202304_01.mseed')
pccst10 = read('xsm04_pcc_202303.mseed')+read('xsm12_pcc_202302_02.mseed')+read('xsm12_pcc_202301_02.mseed')+read('xsm12_pcc_202304_02.mseed')
pccst11 = read('xsm05_pcc_202303.mseed')+read('xsm12_pcc_202302_03.mseed')+read('xsm12_pcc_202301_03.mseed')+read('xsm12_pcc_202304_03.mseed')
pccst12 = read('xsm08_pcc_202303.mseed')+read('xsm12_pcc_202302_04.mseed')+read('xsm12_pcc_202301_04.mseed')+read('xsm12_pcc_202304_04.mseed')

pccst13 = read('zjw12_pcc_202303_01.mseed')+read('zjw12_pcc_202304_01.mseed')+read('zjw12_pcc_202301_01.mseed')
pccst14 = read('zjw12_pcc_202303_02.mseed')+read('zjw12_pcc_202304_02.mseed')+read('zjw12_pcc_202301_02.mseed')
pccst15 = read('zjw12_pcc_202303_03.mseed')+read('zjw12_pcc_202304_03.mseed')+read('zjw12_pcc_202301_03.mseed')
pccst16 = read('zjw12_pcc_202303_04.mseed')+read('zjw12_pcc_202304_04.mseed')+read('zjw12_pcc_202301_04.mseed')

pccsts = [pccst1,pccst2,pccst3,pccst4,pccst5,pccst6,pccst7,pccst8,pccst9,pccst10,pccst11,pccst12,pccst13,pccst14,pccst15,pccst16]

#======================================================    西石门四个方向 自相关函数    ===========================================================================
#pccsts = [pccst9,pccst10,pccst11,pccst12]
pccsts = [pccst12]
#
#k = 0
#
#for pccst in pccsts:    
#    
#    k = k+1
##    plt.subplot(2,2,k)
#    plt.figure(figsize=(7,8))
#    num = len(pccst)
#    pcc_day = Stream()
#    j = 0
#    pcc = 0
#    day = 0.001
#    pccsum = pccst[0].data*0
#    for i in range(num):
#        if j < 24:
#            tr = Trace()
#            tr.data = pccst[i].data
#            pcc_day += tr
#            j += 1
#        else:
##            pcc = PWS(pcc_day,2).data
#            pcc = pcc_day.stack()[0].data/24*1000
#            pcc[pcc>1] = 1
#            pcc[pcc<-1] = -1
#            pccsum += pcc
#            plt.plot(pcc+day, t, 'k', label = 'PAC')
#            plt.ylim([30,0])
#            plt.fill_betweenx(t, pcc+day, day, where = (pcc+day > day), color='k')
#            
#            day += 1#0.001
#            j = 0
#            pcc = 0
#            pcc_day = Stream()
#    
#    pccsum = pccsum/num*24
#    pccsum[pccsum>1] = 1
#    pccsum[pccsum<-1] = -1
#    plt.plot(pccsum+day+1,t,'r',)
#    plt.fill_betweenx(t, pccsum+day+1, day+1, where = (pccsum+day+1 > day+1), color='r')
#    
#    plt.xlim([-0.5,day+2])
#    plt.xlabel('DAY',fontsize=18,weight = 'bold')
#    plt.ylabel('Lag time (s)',fontsize=18,weight = 'bold')
#    plt.xticks(color='k',size = 18)
#    plt.text(day-3,31.4,'Stack',fontsize=18,color='r')
#    plt.yticks(size = 18)
#    if k == 1 :
#        plt.title('(a)  E',fontsize=18,weight = 'bold')
#    elif k == 2:
#        plt.title('(b)  N',fontsize=18,weight = 'bold')
#    elif k == 3:
#        plt.title('(c)  NW',fontsize=18,weight = 'bold')
#    elif k == 4:
#        plt.title('(d)  NE',fontsize=18,weight = 'bold')
#========================================================================================================================================
#
#def rms(x,y,x1,x2):
#    
#    yy = [y[i] for i in range(len(x)) if x[i] > x1 and x[i] < x2]
##    yy_av = sum(yy)/len(yy)
#    return math.sqrt( sum([yi**2 for yi in yy]) / len(yy))
#
#'''
#        两种叠加结果对比
#'''
#
#for pccst in pccsts: 
#    
#    pcc_lines = pccst.copy().stack()[0].data
#    pcc_lines = np.convolve(pcc_lines, np.ones(2)/2, 'same') / 0.008
#    pcc_pws2 = PWS(pccst.copy(),2).data
#    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same') / 0.000018
##    pcc_pws1 = PWS(pccst9,1)
#    
##    plt.figure(figsize=(8,8))
#    fig,axes = plt.subplots(1,2,figsize=(8,8))
#    plt.subplot(1,2,2)
#    plt.plot(pcc_pws2,t,'k')
#    plt.fill_betweenx(t,pcc_pws2, 0, where = (pcc_pws2 > 0), color='k')
#    plt.fill_between([-1,1], 18.81, 20.79, alpha = 0.3, color='b')
#    plt.fill_between([-1,1], 20.79, 22.79, alpha = 0.3, color='k')
#    plt.xlim([-1, 1])
#    plt.ylim([30,0])
#    plt.xlabel('Normalized amplitude', fontsize=18, weight = 'bold')
##    plt.ylabel('Lag time (s)', fontsize=16, weight = 'bold')
#    plt.title('(b)  Phase-weight stack', fontsize=18, weight = 'bold')
#    plt.xticks(size=16)
#    plt.yticks([],[],size=16)
#    
#    plt.subplot(1,2,1)
##    plt.plot(pcc_lines)
#    plt.plot(pcc_lines,t,'k')
#    plt.fill_betweenx(t,pcc_lines, 0, where = (pcc_lines > 0), color='k')
#    plt.fill_between([-1,1], 18.81, 20.79, alpha = 0.3, color='b')
#    plt.fill_between([-1,1], 20.79, 22.79, alpha = 0.3, color='k')
#    plt.xlim([-1, 1])
#    plt.ylim([30,0])
#    plt.xlabel('Normalized amplitude', fontsize=18, weight = 'bold')
#    plt.ylabel('Lag time (s)', fontsize=18, weight = 'bold')
#    plt.title('(a)  Linear stack', fontsize=18, weight = 'bold')
#    plt.xticks([-1,-0.5,0,0.5], size=16)
#    plt.yticks(size=16)
#    
#    fig.subplots_adjust(wspace=0)

#========================================================================================================================================
##====================================================单方向 自相关 求导====================================================================
#z = [0.21, -0.089,-0.09, -9.88, -9.89, -22.49, -22.5, -35.10, -35.11, -45]
#vs= [1.07, 1.07,  3.6,   3.6,   3.7,   3.7,    3.9,   3.9,    4.4,    4.4]
#
#k = 0
#for pccst in pccsts: 
#    k+=1
##    pcc_lines = pccst.copy().stack()[0].data
##    pcc_lines = np.convolve(pcc_lines, np.ones(2)/2, 'same')
#    pcc_pws2 = PWS(pccst.copy(),2).data
#    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same')
#    
#    fit_x, fit_y, fit_y_2d = fit_local_peaks(t,pcc_pws2)
#    fit_y_2d = np.convolve(fit_y_2d, np.ones(5)/5, 'same')
#    fit_y /= 0.000018
#    fit_y_2d /= 2e-5
#    pcc_pws2 /= 1.8e-5
##    fit_y_2d = np.convolve(fit_y_2d, np.ones(2)/2, 'same')/5.8e-5
##    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same')/0.000018
#    
##    plt.figure(figsize=(8,8))
#    fig,axes = plt.subplots(1,2,figsize=(11,9))
#    
#    plt.subplot(1,3,1)
#    plt.plot(pcc_pws2,t,'k', label = 'Coherency')
#    plt.plot(fit_y,fit_x,'g--', label = 'Envelope')
##    legend = plt.legend(fontsize=14, loc = 'lower right')
#    
#    plt.fill_betweenx(t,pcc_pws2, 0, where = (pcc_pws2 > 0), color='k')
##    plt.xlim([-0.000018,0.000018])
#    plt.xlim([-1.2,1.2])
#    plt.ylim([30,5])
##    plt.ylabel('Lag time (s)', fontsize=16, weight = 'bold')
##    plt.title('(a)', fontsize=16, weight = 'bold')
#    plt.xticks([-1,-0.5,0,0.5,1.0], size=16)
#    plt.yticks(size = 16)
#    plt.xlabel('Amplitude', fontsize=18, weight = 'bold') #Normalized
#    plt.ylabel('Lag time (s)', fontsize=18, weight = 'bold')
#    plt.title('(a) Phase autocorrelation', fontsize=18, weight = 'bold')
#
#
#    plt.subplot(1,3,2)
#    plt.plot(fit_y_2d,fit_x,'k', label = '2nd derivative')
#    legend = plt.legend(fontsize=16, loc = 'lower right')
#    legend.get_frame().set_alpha(1.0)
##    plt.fill_betweenx(t,fit_y_2d, 0, where = (pcc_lines > 0), color='k')
##    plt.xlim([-5.8e-5,5.8e-5])
#    plt.xlim([-1.2,1.2])
#    plt.ylim([30,5])
#    plt.title('(b) 2nd derivative', fontsize=18, weight = 'bold')
#    plt.xlabel('Amplitude', fontsize=18, weight = 'bold')  #Normalized 
##    plt.ylabel('Lag time (s)', fontsize=16, weight = 'bold')
##    plt.xticks([-0.006,-0.003,0,0.003,0.006],size=14)
#    plt.xticks([-1,-0.5,0,0.5,1.0], size = 16)
#    plt.yticks([],[], size = 16)
#
#    
#    plt.subplot(1,3,3)
#    ax = plt.plot(vs,z,'b',label='CRUST 1.0')
#    plt.xlim([2.8,5.2])
##    plt.ylim([-59.68+0.21,0.21])
##    plt.ylim([-53.53+0.21,0.21])
#    plt.ylim([-53.53+0.21,-8.96+0.21])
#    plt.ylabel('Depth(km)', size = 18, color ='b', weight = 'bold')
#    plt.xlabel('Vs(km/s)', size = 18, color ='b', weight = 'bold')
#    plt.tick_params(axis='y', colors='b')
#    plt.gca().yaxis.tick_right()
#    plt.gca().yaxis.set_label_position('right')
#    plt.xticks([3.0,3.5,4.0,4.5,5.0], size = 16)
#    plt.yticks( size = 16)
#    plt.title('(c) CRUST 1.0', fontsize=18, weight = 'bold')
#    
#    
#    fig.subplots_adjust(wspace=0)
#    
#    dmax,tmax = find_local_peaks(fit_x,fit_y_2d)
#    for i in range(len(tmax)):
#        if tmax[i] > 18.81 and tmax[i] < 20.79:
#            print('{} s'.format(tmax[i]))
#            plt.subplot(1,3,2)
#            plt.fill_betweenx([18.81,20.79], -1.5, 1.5, color='k', alpha = 0.075, edgecolor='none')
#            plt.axhline(y = 19.8, color='b', linestyle='-.')
#            plt.axhline(y = tmax[i], color='r', linestyle='-')
#            plt.plot(dmax[i],tmax[i],'ro',label='Maximum')
#            legend = plt.legend(fontsize = 14, loc = 'lower center')
#            legend.get_frame().set_alpha(1.0)
#            
#            plt.subplot(1,3,1)
#            plt.fill_betweenx([18.81,20.79], -1.5, 1.5, color='k', alpha = 0.075,  edgecolor='none')
#            plt.axhline(y = 19.8, color='b', linestyle='-.',label = 'Moho depth in CRUST 1.0')
#            plt.axhline(y = tmax[i], color='r', linestyle='-', label = 'Moho depth in this paper')
#            legend = plt.legend(fontsize = 14, loc = 'lower center')
#            legend.get_frame().set_alpha(1.0)
#            
#            plt.subplot(1,3,3)
#            plt.fill_betweenx([-33.65+0.21, -37.18+0.21], 0, 5.5, color='k', alpha = 0.075, edgecolor='none', label = 'Reference range')
#            plt.axhline(y = -35.18, color='b', linestyle='-.')
#            plt.axhline(y = 0.21-tmax[i]*3.568/2, color='r', linestyle='-')
#            legend = plt.legend(fontsize = 14, loc = 'lower center')
#            legend.get_frame().set_alpha(1.0)
#
#            print('-----------------------------------------------------')
#            print(k)
#            print(tmax[i]*3.568/2)
#            print('-----------------------------------------------------')
    

    
##=================================================西石门 四方向 自相关 二阶导数=============================================================================
#pccsts = [pccst9,pccst10,pccst11,pccst12]
pccsts = [pccst5,pccst6,pccst7,pccst8]#,pccst9,pccst10,pccst11,pccst12,pccst13,pccst14,pccst15,pccst16]
#pccsts = [pccst1,pccst2,pccst3,pccst4]#,pccst13,pccst14,pccst15,pccst16]
#pccsts = [pccst13,pccst14,pccst15,pccst16]

j = 1
fig,axes = plt.subplots(4, 2, figsize=(10,8))

svav = 3.568 
#h = 35.23
h = 34.93
h = 35.20
#h = 34.89

t1 = h / (svav*1.05) * 2
t2 = h / (svav*0.95) * 2
t0 = h / svav * 2

for pccst in pccsts: 
    
#    pcc_lines = pccst.copy().stack()[0].data
#    pcc_lines = np.convolve(pcc_lines, np.ones(2)/2, 'same')
    pcc_pws2 = PWS(pccst.copy(),2).data 
    pcc_pws2 = np.convolve(pcc_pws2, np.ones(3)/3, 'same')
    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same')
    
    fit_x, fit_y, fit_y_2d = fit_local_peaks(t,pcc_pws2)
    fit_y_2d = np.convolve(fit_y_2d, np.ones(4)/4, 'same')

    if j ==1:
        fit_y /= 0.000018*30/10
        fit_y_2d /= 1e-5
        pcc_pws2 /= 30*1.8e-5/10
    elif j == 2:
        fit_y /= 0.000018*25/10
        fit_y_2d /= 1e-5
        pcc_pws2 /= 25*1.8e-5/10
    elif j == 3 :
        fit_y /= 10*1.5e-5
        fit_y_2d /= 1e-5
        pcc_pws2 /= 10*1.5e-5
    elif j == 4 :
        fit_y /= 15*1.5e-5/10
        fit_y_2d /= 1e-5
        pcc_pws2 /= 15*1.5e-5/10
#    fit_y_2d = np.convolve(fit_y_2d, np.ones(2)/2, 'same') / 5.8e-5
#    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same') / 1.8e-5
    #--------------------------------------------------------------------------------------------
    
#    plt.figure(figsize=(8,8))
    plt.subplot(4,2,j*2-1)
    plt.plot(t, pcc_pws2, 'k')
    plt.plot(fit_x, fit_y, 'r--')
    plt.axvline(x = t0, color='k', linestyle='-')
    plt.fill_between(t, pcc_pws2, 0, where = (pcc_pws2 > 0), color='k')
    plt.fill_between([t1, t2], -1, 1, alpha = 0.3)
    plt.ylim([-1, 1 ])
    plt.xlim([5,  30])
#    plt.ylabel('Phase autucorrelation function', fontsize=16, weight = 'bold')
#    plt.ylabel('Lag time (s)', fontsize=16, weight = 'bold')
    if j == 1:
        plt.title('(a)  PAC: E; N; NW; NE', fontsize=16, weight = 'bold')
    if j == 4:
        plt.xticks(size = 14)
    else:
        plt.xticks([], [])
    plt.yticks(size = 14)
    plt.xlabel('Lag time (s)', fontsize = 16, weight = 'bold')
    #-------------------------------------------------------------------------------------------
    
    plt.subplot(4,2,j*2)
    plt.plot(fit_x,fit_y_2d,'k')
    plt.axvline(x = t0, color = 'k', linestyle = '-')
    plt.fill_between([t1, t2], -1, 1, alpha = 0.3)
#    plt.fill_betweenx(t,fit_y_2d, 0, where = (pcc_lines > 0), color='k')
    plt.ylim([-1, 1])
    plt.xlim([5,30])
#    plt.xlabel('Second derivative', fontsize=16, weight = 'bold')
    plt.xlabel('Lag time (s)', fontsize=16, weight = 'bold')
    if j == 1:
        plt.title('(b)  2nd derivative: E; N; NW; NE', fontsize=16, weight = 'bold')
#    plt.xticks([-0.006,-0.003,0,0.003,0.006],size=14)
    if j == 4:
        plt.xticks(size=14)
    else:
        plt.xticks([],[])
    plt.yticks(size=14)
    
    fig.subplots_adjust(hspace = 0.0)
#    fig.subplots_adjust(wspace = 0.0)

    
    dmax,tmax = find_local_peaks(fit_x,fit_y_2d)
    for i in range(len(tmax)):
#        plt.plot(tmax[i], dmax[i],'ro')
        if tmax[i] > t1 and tmax[i] < t2:
            plt.subplot(4,2,j*2)
            plt.plot(tmax[i], dmax[i],'ro')
            plt.axvline(x = tmax[i],color='b',linestyle='--')
            plt.subplot(4,2,j*2-1)
            plt.axvline(x = tmax[i],color='b',linestyle='--')
            print('-------------- {} -------------------'.format(j))
            print(tmax[i]/2*svav)
            
            break
    
    j+=1
fig.text(0.05, 0.5, 'Amplitude', va='center', rotation='vertical',weight = 'bold',fontsize=16)

#=======================================四台站 自相关 二阶导数================================================================================
#
##pccsts = [pccst1+pccst2+pccst3+pccst4, pccst5+pccst6+pccst7+pccst8, pccst9+pccst10+pccst11+pccst12, pccst13+pccst14+pccst15+pccst16]
#pccsts = [pccst1,pccst2,pccst3,pccst4]#,pccst5,pccst6,pccst7,pccst8,pccst9,pccst10,pccst11,pccst12,pccst13,pccst14,pccst15,pccst16]
##pccsts = [pccst5,pccst6,pccst7,pccst8]#,pccst9,pccst10,pccst11,pccst12,pccst13,pccst14,pccst15,pccst16]
##pccsts = [pccst9,pccst10,pccst11,pccst12]#,pccst13,pccst14,pccst15,pccst16]
##pccsts = [pccst13,pccst14,pccst15,pccst16]
#
#j = 1
#fig,axes = plt.subplots(4, 2, figsize=(10,8))
#
#for pccst in pccsts: 
#    
##    pcc_lines = pccst.copy().stack()[0].data
##    pcc_lines = np.convolve(pcc_lines, np.ones(2)/2, 'same')
##    pcc_pws2 = PWS(pccsts[j*4-1].copy(),2).data + PWS(pccsts[j*4-2].copy(),2).data + PWS(pccsts[j*4-3].copy(),2).data + PWS(pccsts[j*4-4].copy(),2).data
#    pcc_pws2 = PWS(pccst.copy(),2).data
#    pcc_pws2 = np.convolve(pcc_pws2, np.ones(3)/3, 'same')
##    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same')
#    
#    fit_x, fit_y, fit_y_2d = fit_local_peaks(t,pcc_pws2)
#    fit_y_2d = np.convolve(fit_y_2d, np.ones(5)/5, 'same')
#
#    fit_y /= 0.000018*5
#    fit_y_2d /= 5*1e-5
#    pcc_pws2 /= 5*1.8e-5
##    if j == 1:
##        fit_y /= 20*0.000018
##        fit_y_2d /= 5*1e-5
##        pcc_pws2 /= 20*1.8e-5
##    else:
##        fit_y /= 0.000018
##        fit_y_2d /= 1e-5
##        pcc_pws2 /= 1.8e-5
##    pcc_pws2 = np.convolve(pcc_pws2, np.ones(4)/4, 'same') 
##    fit_x, fit_y, fit_y_2d = fit_local_peaks(t,pcc_pws2)
##    fit_y_2d = np.convolve(fit_y_2d, np.ones(2)/2, 'same')
##    pcc_pws2 = np.convolve(pcc_pws2, np.ones(2)/2, 'same')
##------------------------------------------------------------------------------
##    plt.figure(figsize=(8,8))
#    plt.subplot(4,2,j*2-1)
#    plt.plot(t,     pcc_pws2,'k')
#    plt.plot(fit_x, fit_y,'r:')
#    plt.axvline(x = 19.8, color='k', linestyle='-')
#    plt.fill_between(t, pcc_pws2, 0, where = (pcc_pws2 > 0), color='k')
#    plt.fill_between([18.81,20.79], -1, 1, alpha = 0.3)
#    
#    plt.ylim([-1,1])
#    if j == 1:
#            plt.ylim([-1,1])
#    plt.xlim([5,30])
##    plt.ylabel('Phase autucorrelation function', fontsize=16, weight = 'bold')
##    plt.ylabel('Lag time (s)', fontsize=16, weight = 'bold')
#    if j == 1:
#        plt.title('PAC:  BYA; HJC; XSM; ZJW', fontsize=16, weight = 'bold')
#    if j == 4:
#        plt.xticks(size=14)
#    else:
#        plt.xticks([],[])
#    plt.yticks(size=14)
#    plt.xlabel('Lag time (s)', fontsize=16, weight = 'bold')
##------------------------------------------------------------------------------
#    plt.subplot(4,2,j*2)
#    plt.plot(fit_x, fit_y_2d, 'k')
#    plt.axvline(x = 19.8, color='k', linestyle='-')
#    plt.fill_between([18.81,20.79], -1, 1, alpha = 0.3)
##    plt.fill_betweenx(t,fit_y_2d, 0, where = (pcc_lines > 0), color='k')
#    plt.ylim([-1,1])
#    plt.xlim([5,30])
##    plt.xlabel('Second derivative', fontsize=16, weight = 'bold')
#    plt.xlabel('Lag time (s)', fontsize=16, weight = 'bold')
#    if j == 1:
#        plt.title('2nd derivative: BYA; HJC; XSM; ZJW', fontsize=16, weight = 'bold')
##    plt.xticks([-0.006,-0.003,0,0.003,0.006],size=14)
#    if j == 4:
#        plt.xticks(size=14)
#    else:
#        plt.xticks([],[])
#    plt.yticks(size=14)
#    
#    fig.subplots_adjust(hspace = 0)
##    fig.subplots_adjust(wspace = 0)
#    
#    dmax,tmax = find_local_peaks(fit_x,fit_y_2d)
#    for i in range(len(tmax)):
#        if tmax[i] > 18.81 and tmax[i] < 20.79:
#            plt.subplot(4,2,j*2)
#            plt.axvline(x = tmax[i],color='b',linestyle='--')
#            plt.plot(tmax[i], dmax[i],'ro')
#            plt.subplot(4,2,j*2-1)
#            plt.axvline(x = tmax[i],color='b',linestyle='--')
#            print(tmax[i]*3.568/2)
#    
#    j+=1
#fig.text(0.05, 0.5, 'Amplitude', va='center', rotation='vertical',weight = 'bold',fontsize=16)
#
#
#





##----------------- stack PAC -------------------------
#
##pcc = pccst.stack()[0].data
##pcc = tf_PWS(pccst,t,2,1,2).data
#pcc = PWS(pccst,2).data
#
##------------------   PAC    -------------------------
#
#plt.figure(figsize=(15,5))
#plt.subplot(1,2,1)
##pcc = pcc/numtxt
#pcc = np.convolve(pcc, np.ones(2)/2, 'same')
##plt.figure()
##plt.plot(t,pcc,label='PAC')
#plt.plot(t,pcc,'k',label='PAC')
#
##-------------  envelope  ---------------------------
###plt.figure()
##Hpcc =  hilbert(pcc)
##pcc_e = abs(Hpcc)
###plt.plot(t,pcc_e,':',label='envelope')
##plt.plot(t,pcc_e,':',label='envelope')
##---------------  envelope  --------------------------
#fit_x, fit_y, fit_y_2d = fit_local_peaks(t,pcc,17)
#
#plt.plot(fit_x,fit_y,'r-',label='envelope')
#
#plt.legend(fontsize = 16)
#
#plt.xlabel('Lag Time(s)',size=16)
#plt.ylabel('Amplitude',size=16)
#plt.xticks(size=14)
#plt.yticks(size=14)
##plt.ylim([-0.01,0.01])
##plt.xlim([0,30])
#
##--------------- The second derivative -------------------
#plt.subplot(1,2,2)
##plt.figure()
#fit_y_2d = np.convolve(fit_y_2d, np.ones(3)/3, 'same')
#plt.plot(fit_x,fit_y_2d,'k')
#plt.ylabel(' The second derivative ',size=16)
#plt.xlabel('Lag Time(s)',size=16)
#plt.xticks(size=14)
#plt.yticks(size=14)
##plt.ylim([-0.1,0.1])
##plt.ylim([-0.0002,0.0002])
##plt.xlim([0,30])
##---------------------- The END  ----------------------
##
##
#
#





