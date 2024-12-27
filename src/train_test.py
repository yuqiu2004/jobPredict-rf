import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def test(path):
    """
    回归任务测试
    :param path: 数据集路径
    :return: void
    """
    # 加载数据
    raw_data = pd.read_csv(path)  # 修改为实际数据路径
    x = raw_data.drop('target', axis=1)  # 假设 'target' 是你要预测的数值列
    y = raw_data['target']
    # 拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    # 创建回归模型
    rfr = RandomForestRegressor()
    # 训练模型
    rfr.fit(x_train, y_train)
    # 预测
    y_pre = rfr.predict(x_test)
    # 计算评估指标
    # mse = mean_squared_error(y_test, y_pre)
    # r2 = r2_score(y_test, y_pre)
    # 打印模型评估结果
    # print(f'Mean Squared Error: {mse}')
    # print(f'R^2 Score: {r2}')
    # 将预测结果添加到测试集的最后一列
    x_test['predictions'] = y_pre
    # 打印最后的测试集数据（包括预测值）
    print(x_test)
