#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_test_split(x, y, random=0):
    if random is not None:
        np.random.seed(random)

    index = np.random.permutation(len(x))
    x = x[index]
    y = y[index]

    
    total_samples = len(x)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.1)
    test_size = total_samples - train_size - val_size

   
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_val = x[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    x_test = x[train_size+val_size:]
    y_test = y[train_size+val_size:]

    return x_train, x_val, x_test, y_train, y_val, y_test


data = pd.read_csv('D:/merge.csv')
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


x_train, x_val, x_test, y_train, y_val, y_test = train_test_split(x, y, random=None)


train_df = pd.DataFrame(x_train)
train_df.insert(0, 'class', y_train)
test_df = pd.DataFrame(x_test)
test_df.insert(0, 'class', y_test)
val_df = pd.DataFrame(x_val)
val_df.insert(0, 'class', y_val)


train_df.to_csv('train_data7.csv', index=False)
test_df.to_csv('test_data7.csv', index=False)
val_df.to_csv('val_data7.csv', index=False)


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_gaussian_basis(X: np.ndarray, mu: float, sigma: float) -> float:
    return np.exp(-((X - mu) ** 2) / (2 * sigma ** 2))


def get_feature_vector(features: np.ndarray, O_1: int, O_2: int, O_3: int, O_4: int, O_5: int) -> np.ndarray:
    X_1 = features[:, 0]
    X_2 = features[:, 1]
    X_3 = features[:, 2]
    X_4 = features[:, 3]
    X_5 = features[:, 4]
    X_6 = features[:, 5]
    X_7 = features[:, 6]
    
    s_1 = (np.max(X_1) - np.min(X_1)) / (O_1 - 1)
    s_2 = (np.max(X_2) - np.min(X_2)) / (O_2 - 1)
    s_3 = (np.max(X_3) - np.min(X_3)) / (O_3 - 1)
    s_4 = (np.max(X_4) - np.min(X_4)) / (O_4 - 1)
    s_5 = (np.max(X_5) - np.min(X_5)) / (O_5 - 1)
    
    phi = np.ones((features.shape[0], (O_1 * O_2 * O_3 * O_4 * O_5) + 7))
    phi[:, 0] = X_1
    phi[:, 1] = X_2
    phi[:, 2] = X_3
    phi[:, 3] = X_4
    phi[:, 4] = X_5
    phi[:, 5] = X_6
    phi[:, 6] = X_7
    
    for i in range(1, O_4 + 1):
        mu_i = s_4 * (i - 1) + np.min(X_4)
        k = O_4 * (i - 1) + 3
        phi[:, k + 4] = get_gaussian_basis(X_4, mu_i, s_4)
        
    for i in range(1, O_5 + 1):
        mu_i = s_5 * (i - 1) + np.min(X_5)
        k = O_5 * (i - 1) + 3 + O_4
        phi[:, k + 4] = get_gaussian_basis(X_5, mu_i, s_5)

    for i in range(1, O_1 + 1):
        for j in range(1, O_2 + 1):
            for k in range(1, O_3 + 1):
                mu_i = s_1 * (i - 1) + np.min(X_1)
                mu_j = s_2 * (j - 1) + np.min(X_2)
                mu_k = s_3 * (k - 1) + np.min(X_3)
                l = O_2 * O_3 * (i - 1) + O_3 * (j - 1) + k
                phi[:, l + 7 + O_4 + O_5] = get_gaussian_basis(X_1, mu_i, s_1) * get_gaussian_basis(X_2, mu_j, s_2) * get_gaussian_basis(X_3, mu_k, s_3)

    return phi


def BLR(train_data, test_data_feature, O1=2, O2=4, O3=6, O4=8, O5=10, regularization=0.0):
    train_data_feature = train_data[1:, 1:-1]
    train_data_label = train_data[1:, -1]

    train_phi = get_feature_vector(train_data_feature, O1, O2, O3, O4, O5)
    I = np.identity(train_phi.shape[1])
    weights = np.linalg.inv(I + train_phi.T @ train_phi) @ train_phi.T @ train_data_label
    y_BLR_prediction = get_feature_vector(test_data_feature, O1, O2, O3, O4, O5) @ weights
    return y_BLR_prediction


def MLR(train_data, test_data_feature, O1=2, O2=4, O3=6, O4=8, O5=10, regularization=0.0):
    train_data_feature = train_data[1:, 1:-1]
    train_data_label = train_data[1:, -1]

    train_phi = get_feature_vector(train_data_feature, O1, O2, O3, O4, O5)
    weights = np.linalg.inv(train_phi.T @ train_phi + regularization * np.identity(train_phi.shape[1])) @ train_phi.T @ train_data_label
    y_MLR_prediction = get_feature_vector(test_data_feature, O1, O2, O3, O4, O5) @ weights
    return y_MLR_prediction


def MSE(data, prediction):
    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean_squared_error = sum_squared_error / len(data)
    return mean_squared_error


def main():
    O_1 = 2
    O_2 = 4
    O_3 = 6
    O_4 = 8
    O_5 = 10
    
    # 讀取資料
    data_train = pd.read_csv('D:/train_data7.csv')
    data_test = pd.read_csv('D:/test_data7.csv')

    # 將 "male" 轉換為 1，"female" 轉換為 0
    data_train['0'] = data_train['0'].map({'male': 1, 'female': 0})
    data_test['0'] = data_test['0'].map({'male': 1, 'female': 0})

    # 將資料轉換為 NumPy 陣列
    data_train = data_train.values
    data_test = data_test.values
    data_test_feature = data_test[1:, 1:-1]  # 從第二欄開始取特徵
    data_test_label = data_test[1:, -1]

    regularization = 0.1  # 正規化參數

    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2, O3=O_3, O4=O_4, O5=O_5, regularization=regularization)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2, O3=O_3, O4=O_4, O5=O_5, regularization=regularization)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(
        e1=MSE(predict_BLR, data_test_label), e2=MSE(predict_MLR, data_test_label)))

    # 建立一個子圖格矩陣，用於繪製每個特徵值的擬合結果
    fig, axs = plt.subplots(nrows=data_test_feature.shape[1], ncols=1, figsize=(8, 6*data_test_feature.shape[1]))

    # 迴圈繪製每個特徵值的最佳擬合線
    for i in range(data_test_feature.shape[1]):
        # 繪製散點圖，顯示實際資料
        axs[i].scatter(data_test_feature[:, i], data_test_label, label='Actual')
    
        # 繪製 BLR 的最佳擬合線
        axs[i].plot(data_test_feature[:, i], predict_BLR, label='BLR', color='red')
    
        # 繪製 MLR 的最佳擬合線
        axs[i].plot(data_test_feature[:, i], predict_MLR, label='MLR', color='green')
    
        # 設定 x 軸標籤為 'X'，y 軸標籤為 'y'
        axs[i].set_xlabel(f'X{i+1}')
        axs[i].set_ylabel('y')
    
        # 設定圖的標題
        axs[i].set_title(f'Best-Fit Lines Comparison for Feature {i+1}')
    
        # 顯示圖例
        axs[i].legend()

    # 調整子圖格矩陣的間距和對齊
    fig.tight_layout()

    # 顯示圖形
    plt.show()


    
main()


# In[3]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data_train = pd.read_csv('D:/train_data7.csv')
data_test = pd.read_csv('D:/test_data7.csv')


data_train['0'] = data_train['0'].map({'male': 1, 'female': 0})
data_test['0'] = data_test['0'].map({'male': 1, 'female': 0})


X_train = data_train.iloc[:, 1:-1].values 
y_train = data_train.iloc[:, -1].values   

X_test = data_test.iloc[:, 1:-1].values    
y_test = data_test.iloc[:, -1].values      


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = XGBRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)


# In[4]:


get_ipython().system('jupyter nbconvert --to script HW3.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




