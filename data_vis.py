import numpy as np
import matplotlib as plt

def data_vis (x):
    if type(x) == int:
        return('Sorry, this function requires more data!')
    if type(x) == str:
        return('Sorry, this function doesnt work on strings')
    if type(x) == list :
        if np.shape(x) == (2,):
            plt.pyplot.plot(x[0], x[1])
        for i in 0, 10000 :
            if np.shape(x) == (i, 2):
                plt.pyplot.plot(x[0], x[1])
    else :
        return(x)
    
    



