#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_test_split(x,y,random = 0):
    
    #設定隨機種子固定隨機排序列數
    if(random is not None):
        np.random.seed(random)
    
    #打亂原本排序列數
    index = np.random.permutation(len(x))
    x = x[index]
    y = y[index]
    
    #分類不同標籤
    all_test = 60
    class_type_array = np.unique(y)
    class_type = len(class_type_array)
    each_test_type = int(all_test/class_type)
    
    #開始分類test和train
    x_test = []
    x_train =[]
    y_test = []
    y_train =[]
    
    for s in class_type_array:
        index1 = np.where(y==s)[0]
        x_sum = x[index1]
        y_sum = y[index1]
       # 將 x_sum 和 y_sum 陣列中特定索引 each_test_type 以後的元素刪除，
        # 並將刪除後的新陣列分配給 x_train_part 和 y_train_part 變數
        x_test_part = np.delete(x_sum, slice(each_test_type, None), axis=0)
        y_test_part = np.delete(y_sum, slice(each_test_type, None), axis=0)

        # 將 x_sum 和 y_sum 陣列中特定索引 each_test_type 以前的元素刪除，
        # 並將刪除後的新陣列分配給 x_test_part 和 y_test_part 變數
        x_train_part = np.delete(x_sum, slice(0, each_test_type), axis=0)
        y_train_part = np.delete(y_sum, slice(0, each_test_type), axis=0)
        
        x_test.append(x_test_part)
        x_train.append(x_train_part)
        y_test.append(y_test_part)
        y_train.append(y_train_part)
    
    #合併資料
    x_test = np.concatenate(x_test)
    x_train = np.concatenate(x_train)
    y_test = np.concatenate(y_test)
    y_train = np.concatenate(y_train)
    
    return x_test,x_train,y_test,y_train


    
class MyGaussianNB:
    def __init__(self):
        self.classes = None
        self.prior = None
        self.means = None
        self.variances = None

    def fit(self, x, y, prior=None):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = x.shape[1]
        self.prior = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        if prior is not None:
            self.prior = prior
        else:
            for i, c in enumerate(self.classes):
                self.prior[i] = np.sum(y == c) / len(y)

        for i, c in enumerate(self.classes):
            x_c = x[y == c]
            self.means[i, :] = np.mean(x_c, axis=0)
            self.variances[i, :] = np.var(x_c, axis=0)


    def predict(self, x):
        n_samples = x.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))
        for i, c in enumerate(self.classes):
            prior = np.log(self.prior[i])
            mean = self.means[i, :]
            var = self.variances[i, :]
            likelihood = np.sum(
                -0.5 * ((x - mean) ** 2 / var) - 0.5 * np.log(2 * np.pi * var), axis=1)
            posteriors[:, i] =  likelihood + prior
        return self.classes[np.argmax(posteriors, axis=1)]
  
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
    
#讀取數據
data = pd.read_csv('D:/wine.csv')
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

#劃分為資料集與測試集
x_test,x_train,y_test,y_train = train_test_split(x, y, random=None)

clf = MyGaussianNB()
clf.fit(x_train, y_train)

#計算精確度
accuracy = clf.score(x_test, y_test)
print('test:', accuracy)

#將資料轉換成數據框
train_df = pd.DataFrame(x_train)
train_df.insert(0, 'class', y_train)
test_df = pd.DataFrame(x_test)
test_df.insert(0, 'class', y_test)

#將資料存取成csv檔
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#讀取數據
test_df = pd.read_csv('D:/test_data.csv')
x_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

#創建 2D PCA降維
pca = PCA(n_components=2)
#pca = PCA(n_components=3)
x_test_pca = pca.fit_transform(x_test)

#繪製降維後數據點
plt.figure(figsize=(8, 6))
for i in np.unique(y_test):
    plt.scatter(x_test_pca[y_test == i, 0], x_test_pca[y_test == i, 1], label=i)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()



#創建 3D PCA降維
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_test)

#將降維後數據與標籤合併
data_pca = np.column_stack((x_pca, y_test))

#分離不同標籤的數據
class_type_array = np.unique(y_test)
fig = plt.figure()
ax = Axes3D(fig)

#繪製散點圖
for s in class_type_array:
    index = np.where(data_pca[:, 3] == s)[0]
    ax.scatter(data_pca[index, 0], data_pca[index, 1], data_pca[index, 2], label='class {}'.format(int(s)))

#添加標籤
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('test_data - PCA 3D')

#添加圖例
ax.legend()

#顯示圖形
plt.show()


# In[34]:


import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_test_split(x,y,random = 0):
    
    #設定隨機種子固定隨機排序列數
    if(random is not None):
        np.random.seed(random)
    
    #打亂原本排序列數
    index = np.random.permutation(len(x))
    x = x[index]
    y = y[index]
    
    #分類不同標籤
    all_test = 60
    class_type_array = np.unique(y)
    class_type = len(class_type_array)
    each_test_type = int(all_test/class_type)
    
    #開始分類test和train
    x_test = []
    x_train =[]
    y_test = []
    y_train =[]
    
    for s in class_type_array:
        index1 = np.where(y==s)[0]
        x_sum = x[index1]
        y_sum = y[index1]
       # 將 x_sum 和 y_sum 陣列中特定索引 each_test_type 以後的元素刪除，
        # 並將刪除後的新陣列分配給 x_train_part 和 y_train_part 變數
        x_test_part = np.delete(x_sum, slice(each_test_type, None), axis=0)
        y_test_part = np.delete(y_sum, slice(each_test_type, None), axis=0)

        # 將 x_sum 和 y_sum 陣列中特定索引 each_test_type 以前的元素刪除，
        # 並將刪除後的新陣列分配給 x_test_part 和 y_test_part 變數
        x_train_part = np.delete(x_sum, slice(0, each_test_type), axis=0)
        y_train_part = np.delete(y_sum, slice(0, each_test_type), axis=0)
        
        x_test.append(x_test_part)
        x_train.append(x_train_part)
        y_test.append(y_test_part)
        y_train.append(y_train_part)
    
    #合併資料
    x_test = np.concatenate(x_test)
    x_train = np.concatenate(x_train)
    y_test = np.concatenate(y_test)
    y_train = np.concatenate(y_train)
    
    return x_test,x_train,y_test,y_train


    
class noprior:
    def __init__(self):
        self.classes = None
        self.prior = None
        self.means = None
        self.variances = None

    def fit(self, x, y, prior=None):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = x.shape[1]
        self.prior = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        if prior is not None:
            self.prior = prior
        else:
            for i, c in enumerate(self.classes):
                self.prior[i] = np.sum(y == c) / len(y)

        for i, c in enumerate(self.classes):
            x_c = x[y == c]
            self.means[i, :] = np.mean(x_c, axis=0)
            self.variances[i, :] = np.var(x_c, axis=0)


    def predict(self, x):
        n_samples = x.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))
        for i, c in enumerate(self.classes):
            prior = np.log(self.prior[i])
            mean = self.means[i, :]
            var = self.variances[i, :]
            likelihood = np.sum(
                -0.5 * ((x - mean) ** 2 / var) - 0.5 * np.log(2 * np.pi * var), axis=1)
            posteriors[:, i] =  likelihood
        return self.classes[np.argmax(posteriors, axis=1)]
  
    def score(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
    
# 读取数据
data = pd.read_csv('D:/wine.csv')
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random=None)

no = noprior()
no.fit(x_train, y_train)

# 计算测试集的预测分数
accuracy = no.score(x_test, y_test)
print('test:', accuracy)


# In[ ]:




