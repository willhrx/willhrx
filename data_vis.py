import numpy as np
import matplotlib as plt

def data_vis (x):
    if type(x) == int:
        return('Sorry, this function requires more data!')
    if type(x) == str:
        return('Sorry, this function doesnt work on strings')
    if type(x) == list :
        if np.shape(x) == (2,):
            plt.pyplot.plot(x[0], x[1], "x")
        for i in range(100000) :
            if np.shape(x) == (i, 2):
                sig_x = 0
                sig_y = 0
                sig_xy = 0
                sig_x2 = 0
                x_bar = 0
                y_bar = 0
                y = x[0]
                x_min = y[0]
                x_max = y[0]
                for n in range(i):
                    y = x[n]
                    x_bar += y[0]/i
                    y_bar +=y[1]/i
                    sig_x += y[0]
                    sig_y += y[1]
                    sig_xy += y[0]*y[1]
                    sig_x2 += y[0]**2
                    plt.pyplot.plot(y[0], y[1], "x", c = 'red')
                    if y[0] < x_min :
                        x_min = y[0]
                    if y[0] > x_max :
                        x_max = y[0]
                Sxy = sig_xy - (sig_x * sig_y)/i
                Sxx = sig_x2 - (sig_x**2)/i
                b1 = int(Sxy/Sxx)
                b0 = int(y_bar - b1*x_bar)
                lx = np.arange(x_min, x_max, (x_max - x_min)/300)
                ly = b0 + lx * b1
                plt.pyplot.plot(lx, ly, "-", c = 'blue')
                
    else :
        return(x)
    
    



