import pandas as pd
import numpy as np
import scipy.optimize as sc_optim
import imageio
import matplotlib
import matplotlib.pyplot as plt
import time
from structs import *

class back_test():
    def __init__(self):
        self.start_index = START_INDEX
        self.end_index = END_INDEX

    def back_test(self, bl):
        start_index = self.start_index
        end_index = self.end_index
        ret_port_set = []
        weight_port_set = []
            
        for i in range(end_index - start_index - 2):
            cur_idx = start_index + i
            weight_bl, real_ret = bl.get_post_weight(cur_idx)
            ret_port = np.dot(weight_bl, real_ret.T)
            ret_port_set.append(ret_port)
            weight_port_set.append(weight_bl)
        
        acc_ret_port_set = self.get_accumulate_return(ret_port_set)
        eq_acc = bl.calculate_comparative_return(start_index, end_index)

        self.plot_return(acc_ret_port_set, eq_acc)

    def get_accumulate_return(self, ret_port_set):
        # Get accumulated log return all over time
        acc_ret_port_set = [0]
        for ret in ret_port_set:
            acc_ret_port_set.append(acc_ret_port_set[-1] + ret)

        return acc_ret_port_set

    def plot_return(self, acc_ret_port_set, eq_acc):
        x = np.arange(0,len(acc_ret_port_set),1)
        type_name = VIEW_TYPE_NAME[VIEW_TYPE]
        plt.plot(x, eq_acc[0:51], color='blue', label='Equal weight')
        plt.plot(x, acc_ret_port_set, color='red', label='Arbitrary View')
        plt.title('BL Return Back Test_'+str(type_name)+'_Year '+ BACK_TEST_PERIOD_NAME)
        plt.xlabel(BACK_TEST_X_LABEL)
        plt.ylabel(BACK_TEST_Y_LABEL)
        plt.legend()
        plt.savefig("plot/" + 'BL Return Back Test_'+str(type_name)+'_Year '+ BACK_TEST_PERIOD_NAME + ".png")
