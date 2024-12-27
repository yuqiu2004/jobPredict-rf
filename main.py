from src.train import load_and_train
from src.predict import predict

data_path = './data/job_simplified.csv'
model_path = './models/random_forest_model.pkl'
encoder_path = './models/label_encoder.joblib'

def train_and_save():
    load_and_train(data_path, model_path, encoder_path)
    print(f'=== train completed! ===')

def load_and_test():
    predict(model_path, encoder_path)
    print(f'=== predict end... ===')

if __name__ == "__main__":
    load_and_test()
    # train_and_save()