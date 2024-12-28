import joblib
import pandas as pd
from src.utils.data_solve import handle_input

column_names = ['Qualifications', 'Salary', 'CompanySize', 'Preference', 'Skills', 'Responsibilities']

def predict(path, encoder, params=''):
    # 加载模型和 label_encoder
    rfc = joblib.load(path)
    label_encoder = joblib.load(encoder)

    # 将用户输入的数据转化为列表并转换为数字
    user_input = []
    if not params:
        user_input = ['M', '79000', '26000', 'Female', 'Social media platforms (e.g., Facebook, Twitter, Instagram) Content creation and scheduling Social media analytics and insights Community engagement Paid social advertising', 'Manage and grow social media accounts, create engaging content, and interact with the online community. Develop social media content calendars and strategies. Monitor social media trends and engagement metrics.']
    else:
        user_input = params
    # 将用户输入转换为 DataFrame，保持列名顺序
    input_data = pd.DataFrame([user_input], columns=column_names)
    # print(input_data)
    x = handle_input(input_data)
    # 进行预测
    y_pre = rfc.predict(x)
    # 解码预测结果
    y_pre_text = label_encoder.inverse_transform(y_pre)
    # 输出预测结果
    print("预测结果：", y_pre_text)
    return y_pre_text
