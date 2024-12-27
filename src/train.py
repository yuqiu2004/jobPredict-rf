import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.utils.data_solve import handle_input

target_col_name = 'Job'

def tfidf_transform(column, max_features=100):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_features = tfidf_vectorizer.fit_transform(column)
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f"{column.name}_tfidf_{i}" for i in range(tfidf_features.shape[1])])
    return tfidf_df

def load_and_train(path, output_path, encoder_path):
    raw_data = pd.read_csv(path)
    final_features = handle_input(raw_data)

    # 将目标列 (Job) 编码为数值
    label_encoder = LabelEncoder()
    raw_data['Job_Label'] = label_encoder.fit_transform(raw_data[target_col_name])

    # x = raw_data.drop(target_col_name, axis=1) # axis=1表示删除列 默认是0删除行
    x = final_features
    y = raw_data['Job_Label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    y_pre = rfc.predict(x_test)
    # 预测后解码标签
    y_pre_text = label_encoder.inverse_transform(y_pre)
    y_test_text = label_encoder.inverse_transform(y_test)
    df = pd.DataFrame({
        'Predicted': y_pre_text,
        'Actual': y_test_text
    })
    print(df)

    # 保存训练好的模型
    joblib.dump(rfc, output_path)
    joblib.dump(label_encoder, encoder_path)
    importances = rfc.feature_importances_
    # print(f'importances: {importances}')
    # print(f'classification report: {classification_report(y_test, y_pre, zero_division=1)}')
    print(f'=== accuracy score: {accuracy_score(y_test, y_pre)} ===')

