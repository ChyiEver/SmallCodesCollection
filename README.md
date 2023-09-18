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

## 2.

...

## Updating

...

# Contact me
You can contact me via _qi.gh@outlook.com_ or _qigh@semi.ac.cn_  .
