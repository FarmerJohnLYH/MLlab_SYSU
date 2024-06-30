import pandas as pd
import numpy as np
def dataclean(data_path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    # 读取具有适当列名的数据集
    data = pd.read_csv(data_path, names=columns, sep=',\s*', engine='python')
    # 删除由'?'表示的缺失值的行
    cleaned_data = data.replace('?', pd.NA).dropna()
    # 删除'native-country'列
    cleaned_data = cleaned_data.drop(columns=['native-country'])
    cleaned_data = cleaned_data.drop(columns=['fnlwgt'])
    return cleaned_data
def entropy(target_col):
    """
    计算数据集的熵。
    """
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(count / len(target_col)) * np.log2(count / len(target_col)) for count in counts])
    return entropy
def information_gain(data, split_attribute_name, target_name="income"):
    """
    计算数据集的信息增益。该函数接受三个参数：
    1. data：数据的 pandas DataFrame
    2. split_attribute_name：用于拆分数据的特征的名称
    3. target_name：需要预测的目标特征的名称
    """
    # 计算整个数据集的熵
    total_entropy = entropy(data[target_name])
    # 计算拆分属性的值和相应的计数
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    # print(split_attribute_name,"::vals,counts = ",np.size(vals) ,counts)
    # 计算加权熵
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    # 计算信息增益
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

cnt = 0
class DecisionTreeNode:
    """决策树节点类，表示每个节点在决策树中的结构。"""
    def __init__(self, feature_name=None, value=None, leaf=False, prediction=None):
        self.feature_name = feature_name  # 当前节点的特征名称
        self.children = {}  # 子节点字典，键为特征值，值为DecisionTreeNode
        self.value = value  # 父节点的特征值，用于标记路径
        self.leaf = leaf  # 是否是叶节点
        self.prediction = prediction  # 如果是叶节点，此属性存储预测结果
def build_decision_tree(data, features, target_name="income", depth=0, max_depth=10, min_samples_split=2, min_samples_leaf=1):
    """
    递归构建决策树，添加了剪枝参数。
    """
    # 检查样本数是否少于最小分割样本数，如果是，则停止分割，返回最常见的结果
    global cnt
    cnt = cnt + 1
    if(cnt % 100 == 0):
        print(cnt,features)
    if len(data) < min_samples_split or len(features) == 0 or depth == max_depth:
        most_common = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        return DecisionTreeNode(leaf=True, prediction=most_common)

    # 从特征中选择最高信息增益的特征
    igains = {feature: information_gain(data, feature, target_name) for feature in features if len(np.unique(data[feature])) > 1}
    if not igains:
        most_common = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        return DecisionTreeNode(leaf=True, prediction=most_common)
    
    best_feature = max(igains, key=igains.get)
    tree_node = DecisionTreeNode(feature_name=best_feature)
    depth += 1  # 增加树的深度
    
    # 对最佳特征的每个独特值递归构建子树
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value]
        if len(sub_data) < min_samples_leaf:
            prediction = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
            tree_node.children[value] = DecisionTreeNode(leaf=True, prediction=prediction)
        else:
            subtree = build_decision_tree(sub_data, [f for f in features if f != best_feature], target_name, depth, max_depth, min_samples_split, min_samples_leaf)
            tree_node.children[value] = subtree
            subtree.value = value
    
    return tree_node


def predict(tree, instance):
    """
    使用决策树对单个实例进行预测。
    """
    # 检查当前节点是否是叶节点
    if tree.leaf:
        return tree.prediction
    # 寻找当前特征在实例中的值
    value = instance[tree.feature_name]
    # 递归预测子树
    subtree = tree.children.get(value, None)
    # 如果没有子树，返回训练集中最常见的类别
    if subtree is None:
        return np.unique(adult_data['income'])[np.argmax(np.unique(adult_data['income'], return_counts=True)[1])]
    return predict(subtree, instance)

adult_data = dataclean('adult_train.txt')
print("adult_data=",adult_data) #=[30162,13]
overall_entropy = entropy(adult_data['income'])
print("overall_entropy = ", overall_entropy)
# 计算除目标之外的每个特征的信息增益
features = adult_data.columns[:-1]  # 排除目标特征'收入'
print(features)
information_gains = {feature: information_gain(adult_data, feature) for feature in features}
# 显示计算得到的信息增益
print("information_gains=",information_gains)

        
import numpy as np
import time
def evaluate(decision_tree):
    cleaned_test = dataclean('adult_test.txt')
    # 对测试集中的每一行数据使用决策树进行预测
    predictions = [predict(decision_tree, row) for index, row in cleaned_test.iterrows()]
    # 计算准确率
    actual = cleaned_test['income'].values
    for i in range(len(predictions)):
        predictions[i] = predictions[i].strip()+'.'
    accuracy = np.mean(predictions == actual) 
    with open('log.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")

        
    cleaned_test = dataclean('adult_train.txt')
    # 对测试集中的每一行数据使用决策树进行预测
    predictions = [predict(decision_tree, row) for index, row in cleaned_test.iterrows()]
    # 计算准确率
    actual = cleaned_test['income'].values
    for i in range(len(predictions)):
        predictions[i] = predictions[i].strip()+'.'
    accuracy = np.mean(predictions == actual) 
    with open('log.txt', 'a') as f:
        f.write(f"Train Accuracy: {accuracy*100:.2f}%\n")

    #将predictions写入文件,作为pred列
    # cleaned_test['pred'] = predictions
    # cleaned_test.to_csv('adult_test_pred.csv',index=False)
    # print("Predictions are written to adult_test_pred.txt")

if __name__ == '__main__':
    # 使用函数构建决策树
    # max_depths = [5, 10, 15]
    # min_samples_splits = [2, 5, 10, 20]
    # min_samples_leafs = [1, 3, 5, 10]
    max_depths = [5]
    min_samples_splits = [20]
    min_samples_leafs = [5]
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                now = time.time()
                decision_tree = build_decision_tree(adult_data, features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                with open('log.txt', 'a') as f:
                    f.write(f"Decision Tree with max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf} is built in {time.time()-now} seconds\n")
                evaluate(decision_tree) # evaluate


