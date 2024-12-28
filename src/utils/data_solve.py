import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import joblib
import os

qualifications_mapping = {'b': 0, 'p': 1, 'm': 2}
preference_mapping = {'female': 0, 'male': 1, 'both': 2}
max_features=100
tfidf_path = './models/tfidf_vectorizer.joblib'

def tfidf_transform(column):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    if os.path.exists(tfidf_path):
        tfidf_vectorizer = joblib.load(tfidf_path)
    else:
        tfidf_vectorizer.fit(column)
        joblib.dump(tfidf_vectorizer, tfidf_path)
    tfidf_features = tfidf_vectorizer.transform(column)
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = tfidf_features.toarray()
    # 如果特征数不足 max_features，填充 0
    num_features = tfidf_features.shape[1]
    # if num_features < max_features:
    #     # 填充 0
    #     padding = max_features - num_features
    #     dense_matrix = np.hstack([dense_matrix, np.zeros((dense_matrix.shape[0], padding))])
    # elif num_features > max_features:
    #     # 如果特征数多于 max_features，裁剪掉多余的部分
    #     dense_matrix = dense_matrix[:, :max_features]
    # 将填充后的矩阵转换为稀疏矩阵（保持稀疏矩阵的高效性）
    tfidf_features = csr_matrix(dense_matrix)
    # print(f'features: {tfidf_features}')
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f"{column.name}_tfidf_{i}" for i in range(tfidf_features.shape[1])])
    return tfidf_df

def handle_input(raw_data):
    # 数据清洗
    raw_data['Qualifications'] = raw_data['Qualifications'].str.lower()
    raw_data['Preference'] = raw_data['Preference'].str.lower()
    raw_data['Skills'] = raw_data['Skills'].str.lower()
    raw_data['Responsibilities'] = raw_data['Responsibilities'].str.lower()

    # 文本列进行转换为数值 数值特征直接用 不做归一化
    raw_data['Qualifications'] = raw_data['Qualifications'].map(qualifications_mapping)
    raw_data['Preference'] = raw_data['Preference'].map(preference_mapping)
    # 分别对 Skills 和 Responsibilities 列进行 TF-IDF 转换
    skills_tfidf = tfidf_transform(raw_data['Skills'])
    responsibilities_tfidf = tfidf_transform(raw_data['Responsibilities'])
    # 合并 TF-IDF 特征和原始数据的其他列
    final_features = pd.concat(
        [raw_data[['Qualifications', 'Salary', 'CompanySize', 'Preference']], skills_tfidf, responsibilities_tfidf],
        axis=1)
    return final_features

