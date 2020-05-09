import pandas as pd
import numpy as np
import scipy.optimize as sc_optim
import imageio
import matplotlib
import matplotlib.pyplot as plt
import time
from structs import *
from black_litterman import black_litterman
from back_test import back_test


if __name__ == "__main__":

    print("-" * 30, 'Initial Black Litterman Model', "-" * 30)
    type_name = VIEW_TYPE_NAME[VIEW_TYPE]
    print('Use view type: ', type_name)
    bl = black_litterman()
    bl.get_cc_return()
    bl.get_market_value_weight()

    print("-" * 30, 'Do Back Test', "-" * 30)
    bt = back_test()
    bt.back_test(bl)




