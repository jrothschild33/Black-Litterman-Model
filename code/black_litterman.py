import pandas as pd
import numpy as np
import scipy.optimize as sc_optim
import imageio
import matplotlib
import matplotlib.pyplot as plt
import time
from structs import *

class black_litterman():

    def __init__(self):
        self.price_filename = PRICE_FILENAME
        self.mv_filename = MV_FILENAME
        self.price_sheet_name = PRICE_SHEETNAME
        self.mv_sheet_name = MV_SHEETNAME
        self.index_num = INDEX_NUMBER
        self.back_test_T = BACK_TEST_T
        self.view_T = VIEW_T
        self.view_type = VIEW_TYPE
        self.tau = TAU
        self.index_cc_ret = 0
        self.stock_cc_ret = 0
        self.index_name = 0
        self.stock_names = 0
        self.stock_number = 0
        self.market_value_weight = 0

    def print_data(self):
        print(self.index_cc_ret)
        print(self.stock_cc_ret)
        print(self.index_name)
        print(self.stock_names)

    def read_data(self, filename, sheet_name):
        df = pd.read_excel(filename, sheet_name = sheet_name)
        df.set_index("Date", inplace=True)
        df.index = range(len(df))
        df = df.astype('float64')
        return df

    def get_cc_return(self):
        filename = self.price_filename
        sheet_name = self.price_sheet_name
        index_num = self.index_num
        df = self.read_data(filename, sheet_name)
        # calculate cc return
        log_ret = np.log(df/df.shift()) # shift(): Move one unit downward
        log_ret = log_ret.drop(index=[0])
        names = log_ret.columns.tolist()
        index_name = names[index_num]
        stock_names = names[3:]
        index_cc_ret = log_ret[index_name]
        stock_cc_ret = log_ret[stock_names]

        self.index_cc_ret = index_cc_ret
        self.stock_cc_ret = stock_cc_ret
        self.index_name = index_name
        self.stock_names = stock_names
        self.stock_number = len(stock_names)

    def get_market_value_weight(self):
        filename = self.mv_filename
        sheet_name = self.mv_sheet_name
        mv = self.read_data(filename, sheet_name)
        stock_names = mv.columns.tolist()[0:-1]
        # calculate market value weight
        for n in stock_names:
            mv[n] = mv[n] / mv["Total"]
        mv = mv.drop(index=[0])
        mv = mv[stock_names]
        self.market_value_weight = np.array(mv)

    def get_implied_excess_equilibrium_return(self, stock_cc_ret, w_mkt):
        rf = 0.0006132  # weekly risk-free cc return = ln(1+3.24%)/(365/7) = 0.0006132
        mkt_cov = np.array(stock_cc_ret.cov())

        # lambd: implied risk-aversion coefficient
        lambd = ((np.dot(w_mkt, stock_cc_ret.mean())) - rf) / np.dot(np.dot(w_mkt, mkt_cov), w_mkt.T)
        implied_ret = lambd * np.dot(mkt_cov, w_mkt)
        return implied_ret, lambd

    def get_post_weight(self, start_idx):
        T = self.back_test_T 
        view_type = self.view_type
        index_cc_ret, stock_cc_ret = self.index_cc_ret, self.stock_cc_ret
        real_ret = np.array(stock_cc_ret.iloc[start_idx])
        stock_cc_ret = stock_cc_ret.iloc[start_idx - T: start_idx]
        index_cc_ret = index_cc_ret.iloc[start_idx - T: start_idx]
        mkt_cov = np.array(stock_cc_ret.cov())
        
        # Get market value weight of these stock at current time
        mv_i = self.market_value_weight[start_idx - 1]

        implied_ret, lambd = self.get_implied_excess_equilibrium_return(stock_cc_ret, mv_i)
        
        P, Q = self.get_views_P_Q_matrix(view_type, stock_cc_ret)
        
        omega = self.get_views_omega(mkt_cov, P)
        
        posterior_ret = self.get_posterior_combined_return(implied_ret, mkt_cov, P, Q, omega)
        if(view_type == 0):
            # weight_type == 0: No views, use current market value weight to generate asset allocation
            weight_bl = np.array(mv_i)
        elif(view_type == 1 or view_type == 2 or view_type == 3):
            # weight_type == 1: Use views
            weight_bl = self.get_weight_bl(posterior_ret, mkt_cov, lambd)

        return weight_bl, real_ret

    def calculate_comparative_return(self, start_idx, end_index):
        stock_cc_ret = self.stock_cc_ret
        stock_names= self.stock_names
        stock_cc_ret = stock_cc_ret.iloc[start_idx: end_index-2]

        stock_cc_ret["mean"] = stock_cc_ret.loc[:,stock_names].mean(axis=1)
        eq_acc = [0]
        eq_ret = np.array(stock_cc_ret["mean"])
        for r in eq_ret:
            eq_acc.append(eq_acc[-1] + r)
        return eq_acc

    def get_weight_bl(self, posterior_ret, mkt_cov, lambd):
        return np.dot(np.linalg.inv(lambd * mkt_cov), posterior_ret)

    def get_posterior_combined_return(self, implied_ret, mkt_cov, P, Q, omega):
        tau = self.tau
        k = np.linalg.inv(np.linalg.inv(tau * mkt_cov) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
        posterior_ret = np.dot(k, np.dot(np.linalg.inv(tau * mkt_cov), implied_ret) + 
                            np.dot(np.dot(P.T, np.linalg.inv(omega)), Q))
        return posterior_ret

    def get_views_omega(self, mkt_cov, P):
        tau = self.tau
        K = len(P)   # K: number of views
        omega = np.identity(K)
        for i in range(K):
            P_i = P[i]
            omg_i = np.dot(np.dot(P_i, mkt_cov), P_i.T) * tau
            omega[i][i] = omg_i
        return omega

    def get_views_P_Q_matrix(self, index, stock_cc_ret):
        # T_near: use the mean of returns in nearest T periods as views
        T_near = self.view_T
        N = self.stock_number

        if(index == 0 or index == 1):
            # index = 0  Use Market value weight as asset allocation weight
            # index = 1: Assign arbitrary views
            P = np.array([[0. for i in range(N)] for j in range(3)])
            P[0, 8] = 1
            P[0, 9] = -1
            P[1,1] = 1
            P[1,3] = -1
            P[2,3] = 0.1
            P[2,4] = 0.9
            P[2,6] = -0.1
            P[2,7] = -0.9
            Q = np.array([0.0001, 0.00025, 0.0001])
        elif(index == 2):
            P = np.array([[0. for i in range(N)] for j in range(1)])
            P[0, 2] = 1
            P[0, 3] = -1
            Q = [0.017]
        elif(index == 3):
            P = np.identity(N)
            stock_cc_ret_near = stock_cc_ret.iloc[-T_near:]
            Q = np.array(stock_cc_ret_near.mean())
        else:
            print("There is no such kind of view type!")
        return P, Q