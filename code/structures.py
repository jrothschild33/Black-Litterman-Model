# 原始数据
## 股票价格数据
PRICE_FILENAME = './Data/Price_Data.xlsx'
PRICE_SHEETNAME= 'Weekly'

## 股票市值数据
MV_FILENAME = './Data/Market_Value.xlsx'
MV_SHEETNAME= 'Weekly'

# 模型参数
TAU = 0.3           # 后验期望收益率协方差矩阵的放缩尺度，取值在0~1之间

# 模型回测
## 回测参数
BACK_TEST_T = 200   # 回测时间T窗口：200期
START_INDEX = 273   # 开始日期：2015/1/2
END_INDEX = 324     # 结束日期：2015/12/25
INDEX_NUMBER = 0    # 股指数据索引：0.标普500,1.道琼斯，2.纳斯达克

## 绘图参数
BACK_TEST_X_LABEL = 'Week'
BACK_TEST_Y_LABEL = 'Accumulated Return(log)'
BACK_TEST_PERIOD_NAME = '2015'

# 观点参数
VIEW_TYPE = 2       # 对观点列表进行索引
VIEW_TYPE_NAME = ['Market value as view', "Arbitrary views", "Reasonable views", "Near period return as view"]
VIEW_T = 10         # 当观点为"Near period return as view"时，需要定义近期参数，即取VIEW_T期历史收益率求平均值，作为预期收益率