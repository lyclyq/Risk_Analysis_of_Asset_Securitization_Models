import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

#  Kaggle 创建 API 令牌
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

# 目标数据集（Kaggle）
KAGGLE_DATASET = "wordsforthewise/lending-club"

# 自动下载 LendingClub
def download_kaggle_dataset():
    print("Downloading LendingClub dataset from Kaggle...")

    # 确保 Kaggle API 配置文件存在
    if not os.path.exists("kaggle.json"):
        print("从 Kaggle 账户下载 kaggle.json 并放入当前目录！")
        return

    # 运行 Kaggle 命令行下载数据集
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} --unzip -p data")
    print("数据下载完成！")

# 2️⃣ **数据预处理**
def preprocess_data():
    print("处理数据...")

    chunks = pd.read_csv("data/accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv", chunksize=100000, low_memory=False)

    # 合并所有 chunks
    data = pd.concat(chunks)


    data = data[['loan_amnt', 'int_rate', 'fico_range_high', 'fico_range_low', 'annual_inc', 'dti', 'loan_status']]

    # 仅保留“Fully Paid”和“Charged Off”（即已还款和违约）
    data = data[(data['loan_status'] == 'Fully Paid') | (data['loan_status'] == 'Charged Off')]

    # 违约状态转换0 = 正常还款，1 = 违约
    data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

    # 填充
    data.fillna(data.mean(), inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['loan_status']))

    return data, scaled_data

# LSTM 训练集
def create_lstm_dataset(scaled_data, labels, time_steps=10):
    X, Y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        Y.append(labels.iloc[i+time_steps])

    return np.array(X), np.array(Y)

# 构建训练 LSTM
def train_lstm(X_train, Y_train, X_test, Y_test):
    print("训练 LSTM 违约预测模型")

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')  # 预测二分类0=正常
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

    return model, history

# 评估模型
def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(Y_test, predictions)
    print(f"模型准确率: {acc:.2f}")

    # 绘制
    plt.plot(history.history['accuracy'], label='训练集准确率')
    plt.plot(history.history['val_accuracy'], label='测试集准确率')
    plt.legend()
    plt.title("LSTM 训练过程")
    plt.show()

# main
if __name__ == "__main__":
    # 下载 LendingClub 贷款数据
    download_kaggle_dataset()
    # #
    #     # 处理数据
    raw_data, scaled_data = preprocess_data()

    # 创建 LSTM 训练数据
    X, Y = create_lstm_dataset(scaled_data, raw_data['loan_status'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 训练
    model, history = train_lstm(X_train, Y_train, X_test, Y_test)

    # 评估
    evaluate_model(model, X_test, Y_test)
