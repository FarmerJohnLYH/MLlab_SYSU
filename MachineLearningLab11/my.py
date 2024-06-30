import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
import scipy.stats as stats

# 下载并加载Adult数据集
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# 数据预处理
data = data.dropna()

# 将收入标签转换为二进制
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# 将离散型变量编码
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
print(data.columns)

# 将数据集拆分为训练集和测试集
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 对连续型变量标准化
scaler = StandardScaler()
# test_scaler = StandardScaler()
continuous_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
# continuous_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
X_test[continuous_cols] = scaler.fit_transform(X_test[continuous_cols])

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.feature_summaries = {}
        self.label_encoders = label_encoders

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.class_priors = {c: np.sum(y == c) / n_samples for c in self.classes}
        
        for feature in X.columns:
            self.feature_likelihoods[feature] = {}
            self.feature_summaries[feature] = {}
            for c in self.classes:
                feature_values = X[feature][y == c]
                if feature in continuous_cols:
                    self.feature_summaries[feature][c] = (np.mean(feature_values), np.std(feature_values))
                else:
                    values, counts = np.unique(feature_values, return_counts=True)
                    self.feature_likelihoods[feature][c] = {v: (counts[i] + 1) / (np.sum(counts) + len(values)) for i, v in enumerate(values)}

    def _calculate_likelihood(self, feature, value, class_label):
        if feature in continuous_cols:
            mean, std = self.feature_summaries[feature][class_label]
            likelihood = stats.norm(mean, std).pdf(value)
        else:
            likelihood = self.feature_likelihoods[feature][class_label].get(value, 1 / (len(X_train) + len(self.feature_likelihoods[feature][class_label])))
        return likelihood

    def _calculate_posterior(self, x, class_label):
        posterior = np.log(self.class_priors[class_label])
        for feature in X.columns:
            posterior += np.log(self._calculate_likelihood(feature, x[feature], class_label))
        return posterior

    def predict(self, X):
        y_pred = []
        for _, x in X.iterrows():
            posteriors = {c: self._calculate_posterior(x, c) for c in self.classes}
            y_pred.append(max(posteriors, key=posteriors.get))
        return y_pred

# 训练朴素贝叶斯模型
nb = NaiveBayes()
nb.fit(X_train, y_train)

# 预测
y_pred = nb.predict(X_test)

# 评估指标
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.4f}%")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
# 绘制可视化图形 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 使用 seaborn 绘制混淆矩阵的热力图
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.svg', dpi=600, bbox_inches='tight')
plt.show()