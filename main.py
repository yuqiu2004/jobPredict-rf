from src.train import load_and_train
from src.predict import predict
from flask import Flask, request, jsonify

data_path = './data/job_simplified.csv'
model_path = './models/random_forest_model.pkl'
encoder_path = './models/label_encoder.joblib'
app = Flask('job-predict-api')

def train_and_save():
    load_and_train(data_path, model_path, encoder_path)
    print(f'=== train completed! ===')

def load_and_test():
    predict(model_path, encoder_path)
    print(f'=== predict end... ===')

@app.route('/')
def hello_world():
    return 'Hello World'

#Post接口 用于远程调用
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        print(data)
        # 提取 JSON 数据中的各个字段
        qualification = data.get("qualification")
        salary = data.get("salary")
        company_size = data.get("company_size")
        preference = data.get("preference")
        skills = data.get("skills")
        responsibilities = data.get("responsibilities")
        # 检查输入数据是否完整
        if not all([qualification, salary, company_size, preference, skills, responsibilities]):
            return jsonify({"error": "Missing required fields in the input data."}), 400

        # 将提取的数据放入 user_input 列表
        user_input_list = [qualification, salary, company_size, preference, skills, responsibilities]
        result = predict(model_path, encoder_path, user_input_list)
        # print(result.tolist()[0])
        return jsonify({"job": result.tolist()[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081, debug=True)
    # load_and_test()
    # train_and_save()