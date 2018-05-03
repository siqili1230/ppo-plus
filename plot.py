
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

def plot_line(x,y,img_path,name,game,num_frames):

    plt.plot(x, y,'r',label="{}".format(name))
    plt.xlim(0, num_frames* 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.savefig(img_path)


if __name__=='__main__':
    plot_line([1,2,3,4,5,6],[2,3,5,7,11,13],'./test.png','a2c','BreakOut')