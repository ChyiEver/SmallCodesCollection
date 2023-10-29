# Read First
Here, I have loaded some programs or codes which are small but useful for my work so that I can find them at any time.  Of course, it would be my honor if there are a few lines of code which can help you in this repository.

# Small Codes

## 1 Extract formatted time series from existing strings

```python
import re
from datetime import datetime

def format_time(filename, pattern = r'\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}\.\d{3}'):
    '''
    Input:
        filename = 'SemiTDM-2.5KHz 2023-7-29-9-2-18.275.txt'
        pattern = r'\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}\.\d{3}'
    Output:
        formatted_time = '2023-07-29T09:02:18.275000Z'
    '''

    match = re.search(pattern, filename)
    if match:
        time_string = match.group()
    time_obj = datetime.strptime(time_string, "%Y-%m-%d-%H-%M-%S.%f")  #  Use **strptime** to parse the input time string
    formatted_time = time_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")        #  Format to target format using **strftime**
    return formatted_time
```

## 2 Find the position of the timeth occurrence of string substr in str

```python
# 找字符串substr在str中第time次出现的位置
    
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
```

## 3 Find the local maximums of a one-dimensional digital signal

```python
# 寻找信号的局部极大值
def find_local_peaks(t,signal):
    """
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

```

...

# Contact me
You can contact me via _qi.gh@outlook.com_ or _qigh@semi.ac.cn_  .
