import collections
from collections import deque

class RunningMean:
    def __init__(self, WIN_SIZE=15, n_size = 1):
        self.n = 0
        self.mean = np.zeros(n_size)
        self.cum_sum = 0
        self.past_value = 0
        self.WIN_SIZE = WIN_SIZE
        self.windows = collections.deque(maxlen=WIN_SIZE+1)
        
    def clear(self):
        self.n = 0
        self.windows.clear()

    def push(self, x):
        
        #currently fillna with past value, might want to change that
        x = fillna_npwhere(x, self.past_value)
        self.past_value = x
        
        self.windows.append(x)
        self.cum_sum += x
        
        if self.n < self.WIN_SIZE:
            self.n += 1
            self.mean = self.cum_sum / float(self.n)
            
        else:
            self.cum_sum -= self.windows.popleft()
            self.mean = self.cum_sum / float(self.WIN_SIZE)

    def get_mean(self):
        return self.mean if self.n else np.zeros(n_size)

    def __str__(self):
        return "Current window values: {}".format(list(self.windows))

######################################################
dict_MAclass ={}    
for assetID in range(0,14):
    dict_MAclass[assetID] = RunningMean(15)
######################################################    
    

def fillna_npwhere(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array
