import pandas as pd
import numpy as np
import scipy.optimize as sc_optim
import imageio
import matplotlib
import matplotlib.pyplot as plt
import time
from structures import *

'''
BlackLitterman结构：
|
|---1. __init__(self)：传入初始化参数
|
|---2. print_data()：打印重要参数
|
|---3. read_data(filename, sheet_name)：读取数据excel表格，定义索引、转化数据格式
|
|---4*. get_cc_return()【*必须运行】：将初始数据进行处理，得到股票名称列表、收益率等数据
|
|---5*. get_market_value_weight()【*必须运行】：将初始市值数据进行处理，得到市值权重market_value_weight矩阵
|
|---6. get_implied_excess_equilibrium_return(stock_cc_ret, w_mkt)：计算风险厌恶系数lambda、先验预期收益率implied_ret（即mu_0)
|
|---7. get_views_P_Q_matrix(index, stock_cc_ret)：设定观点矩阵 P、相对收益率强度矩阵 Q（共3种Views可选）
|
|---8. get_views_omega(mkt_cov, P)：阵计算Omega矩阵
|
|---9. get_posterior_combined_return(implied_ret, mkt_cov, P, Q, omega)：计算后验期望收益率mu_p
|
|---10.get_weight_bl(posterior_ret, mkt_cov, lambd)：定义由BL模型得到的新权重weight_bl的计算公式
|
|---11.get_post_weight(start_idx)：计算指定时间窗口T和不同Views下得到的BL模型新权重、真实收益率
|
|---12.calculate_comparative_return(start_idx, end_index)：计算等权重累计收益率：eq_acc（作为对照组）

'''


class BlackLitterman:

    # Step1. 初始化参数
    def __init__(self):
        # 数据参数
        self.price_filename = PRICE_FILENAME
        self.price_sheet_name = PRICE_SHEETNAME
        self.mv_filename = MV_FILENAME
        self.mv_sheet_name = MV_SHEETNAME
        # 模型参数
        self.tau = TAU
        # 回测参数
        self.back_test_T = BACK_TEST_T
        # 观点参数
        self.view_type = VIEW_TYPE
        self.view_T = VIEW_T
        # 投资品种
        ## 股票参数
        self.stock_cc_ret = 0
        self.stock_names = 0
        self.stock_number = 0
        self.market_value_weight = 0
        ## 股指参数
        self.index_num = INDEX_NUMBER
        self.index_name = 0
        self.index_cc_ret = 0

    # Step2. 打印重要参数
    def print_data(self):
        print(self.index_cc_ret)                # 股指（标普500）收益率
        print(self.stock_cc_ret)                # 10只股票的收益率
        print(self.index_name)                  # 股指名称：标普500
        print(self.stock_names)                 # 股票名称列表

    # Step3. 读取数据
    def read_data(self, filename, sheet_name):  # 读取数据excel表格
        df = pd.read_excel(filename, sheet_name = sheet_name)
        df.set_index("Date", inplace=True)      # 按Date一列进行索引
        df.index = range(len(df))               # 将索引化作数字
        df = df.astype('float64')               # 将数据转换为float64格式
        return df

    # Step4. 将初始数据进行处理，得到股票名称列表、收益率等数据
    def get_cc_return(self):
        filename = self.price_filename
        sheet_name = self.price_sheet_name
        # 索引数值：此处为标普500的索引为“0”
        index_num = self.index_num
        df = self.read_data(filename, sheet_name)
        # 计算收益率（对数形式）
        '''
        shift(): Move one unit downward
        相当于每个单元格除以上一个单元格，得到收益率，再取对数形式，是(P2-P1)/P1的近似形式
        由于第一行没有可除的数据，所以删去，最终数据是525-1=524条
        '''
        log_ret = np.log(df/df.shift())
        log_ret = log_ret.drop(index=[0])
        # 将3个指数、10个股票名称转化为名称列表
        names = log_ret.columns.tolist()
        # index_name：标普500
        index_name = names[index_num]
        # 股票名称：第3个及往后的数据
        stock_names = names[3:]
        # index_cc_ret：标普500收益率
        index_cc_ret = log_ret[index_name]
        # stock_cc_ret：10只股票的收益率表格
        stock_cc_ret = log_ret[stock_names]

        # 对数据进行更新赋值
        self.index_cc_ret = index_cc_ret
        self.stock_cc_ret = stock_cc_ret
        self.index_name = index_name
        self.stock_names = stock_names
        self.stock_number = len(stock_names)

    # Step5. 将初始市值数据进行处理，得到市值权重market_value_weight矩阵
    def get_market_value_weight(self):
        filename = self.mv_filename
        sheet_name = self.mv_sheet_name
        mv = self.read_data(filename, sheet_name)
        # 由于最后一列包含“Total”一项，故需要切片[0:-1]
        stock_names = mv.columns.tolist()[0:-1]
        # calculate market value weight
        for n in stock_names:
            mv[n] = mv[n] / mv["Total"]
        # 去掉第一列，与收益率文件保持数量一致
        mv = mv.drop(index=[0])
        # 去掉“Total”列，只保留10只股票市值权重数据
        mv = mv[stock_names]
        # 将市值权重保存为矩阵形式
        self.market_value_weight = np.array(mv)

    # Step6. 计算风险厌恶系数lambda、先验预期收益率implied_ret
    def get_implied_excess_equilibrium_return(self, stock_cc_ret, w_mkt):
        '''
        :param stock_cc_ret: 指定T部分的10只股票收益率数据（维度：T * 10）
        :param w_mkt: 当前的市场权重（维度：1*10）
        :return: 风险厌恶系数lambd、先验预期收益率：implied_ret
        '''

        # weekly risk-free cc return = ln(1+3.24%)/(365/7) = 0.0006132
        rf = 0.0006132
        # 根据股票收益率计算得到协方差矩阵：mkt_cov
        mkt_cov = np.array(stock_cc_ret.cov())

        # lambd: implied risk-aversion coefficient（风险厌恶系数）
        lambd = ((np.dot(w_mkt, stock_cc_ret.mean())) - rf) / np.dot(np.dot(w_mkt, mkt_cov), w_mkt.T)
        # 计算先验预期收益率：implied_ret
        implied_ret = lambd * np.dot(mkt_cov, w_mkt)
        return implied_ret, lambd

    # Step7. 设定观点矩阵 P、相对收益率向量 Q（共3种Views可选）
    def get_views_P_Q_matrix(self, view_type, stock_cc_ret):
        N = self.stock_number
        if(view_type == 0 or view_type == 1):
            # view_type = 0: 投资者无观点，使用当前市值权重（即均衡状态下的权重）作为作为投资组合的权重
            # view_type = 1: 为投资者分配任意观点，这里随机分配了3个观点
            '''
            观点1. 伯克希尔哈撒韦比埃克森美孚的预期收益高0.01%；
            观点2. 微软比摩根大通的预期收益高0.025%；
            观点3. 10%摩根+90%VISA的投资组合比10%沃尔玛+90%美国银行的投资组合预期收益高0.01%
            '''
            P = np.zeros([3, N])
            P[0, 8] = 1
            P[0, 9] = -1
            P[1, 1] = 1
            P[1, 3] = -1
            P[2, 3] = 0.1
            P[2, 4] = 0.9
            P[2, 6] = -0.1
            P[2, 7] = -0.9
            Q = np.array([0.0001, 0.00025, 0.0001])
        elif(view_type == 2):
            # view_type = 2: Reasonable views
            P = np.zeros([1, N])
            P[0, 2] = 1
            P[0, 3] = -1
            Q = [0.017]
        elif(view_type == 3):
            # view_type = 3: 选用最近VIEW_T期的历史平均收益率作为预期收益率
            # T_near: 使用近期T_near期数据的历史平均收益率作为预期收益率
            T_near = self.view_T
            P = np.identity(N)
            stock_cc_ret_near = stock_cc_ret.iloc[-T_near:]
            Q = np.array(stock_cc_ret_near.mean())
        else:
            print("There is no such kind of view type!")
        return P, Q

    # Step8. 计算Omega矩阵
    def get_views_omega(self, mkt_cov, P):
        tau = self.tau
        # K: 投资者观点的数量
        K = len(P)
        # 生成K维度的对角矩阵（对角线上全为1）
        omega = np.identity(K)
        for i in range(K):
            # 逐行选取P（Views矩阵，维度：K*N，此处N=10）
            P_i = P[i]
            omg_i = np.dot(np.dot(P_i, mkt_cov), P_i.T) * tau
            # 将得到的结果赋值到矩阵对角线元素
            omega[i][i] = omg_i
        return omega

    # Step9. 计算后验期望收益率mu_p
    def get_posterior_combined_return(self, implied_ret, mkt_cov, P, Q, omega):
        # tau为缩放尺度
        tau = self.tau
        # 后验期望收益率mu_p的计算公式
        k = np.linalg.inv(np.linalg.inv(tau * mkt_cov) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
        posterior_ret = np.dot(k, np.dot(np.linalg.inv(tau * mkt_cov), implied_ret) +
                            np.dot(np.dot(P.T, np.linalg.inv(omega)), Q))
        return posterior_ret

    # Step10. 计算由BL模型得到的新权重weight_bl
    def get_weight_bl(self, posterior_ret, mkt_cov, lambd):
        weight_bl = np.dot(np.linalg.inv(lambd * mkt_cov), posterior_ret)
        return weight_bl

    # Step11. 计算指定时间窗口T和不同Views下得到的BL模型新权重、真实收益率
    def get_post_weight(self, start_idx):
        T = self.back_test_T                                                # T区间：200（个数据）
        view_type = self.view_type                                          # 三种观点类型：例如“0”，意味着['Market value as view']
        index_cc_ret, stock_cc_ret = self.index_cc_ret, self.stock_cc_ret   # 传入原始标普500和10只股票收益率
        real_ret = np.array(stock_cc_ret.iloc[start_idx])                   # 真实收益率：按行索引提取数据
        stock_cc_ret = stock_cc_ret.iloc[start_idx - T: start_idx]          # 提取指定T回测区间10只股票收益率数据(2014年底以前的200天）
        index_cc_ret = index_cc_ret.iloc[start_idx - T: start_idx]          # 提取指定T回测区间标普500收益率数据(2014年底以前的200天）
        mkt_cov = np.array(stock_cc_ret.cov())                              # 将T区间部分的股票收益率计算成协方差矩阵

        # Get market value weight of these stock at current time（取得这些股票当前的市场权重mv_i，日期：2014年的最后一行）
        mv_i = self.market_value_weight[start_idx - 1]

        # 得到T区间内的风险厌恶系数lambda、先验预期收益率implied_ret（即mu_0)
        implied_ret, lambd = self.get_implied_excess_equilibrium_return(stock_cc_ret, mv_i)
        P, Q = self.get_views_P_Q_matrix(view_type, stock_cc_ret)           # 根据选定的View类型，设置P和Q矩阵
        omega = self.get_views_omega(mkt_cov, P)                            # 根据选定的View类型，计算Omega矩阵

        posterior_ret = self.get_posterior_combined_return(implied_ret, mkt_cov, P, Q, omega)
        if (view_type == 0):
            # weight_type == 0: 无观点，使用当前市值权重作为BL模型的权重（即无需代入BL公式计算）
            weight_bl = np.array(mv_i)
        elif (view_type == 1 or view_type == 2 or view_type == 3):
            # weight_type == 1: 根据Views的类型，计算BL模型得到的新权重weight_bl
            weight_bl = self.get_weight_bl(posterior_ret, mkt_cov, lambd)

        return weight_bl, real_ret

    # Step12. 计算等权重累计收益率：eq_acc（作为对照组）
    def calculate_comparative_return(self, start_idx, end_index):
        stock_names = self.stock_names                                          # 传入10只股票名称列表
        stock_cc_ret = self.stock_cc_ret                                        # 传入10只股票收益率
        stock_cc_ret = stock_cc_ret.iloc[start_idx: end_index + 1]              # 选定2015年10只股票收益率数据
        stock_cc_ret["mean"] = stock_cc_ret.loc[:, stock_names].mean(axis=1)    # 新增一列mean：2015年10只股票每日平均收益率
        eq_acc = [0]
        eq_ret = np.array(stock_cc_ret["mean"])
        for r in eq_ret:
            eq_acc.append(eq_acc[-1] + r)                                       # 累加每日收益率，形成列表
        return eq_acc
