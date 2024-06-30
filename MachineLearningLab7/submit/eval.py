from data import dataclean, predict, DecisionTreeNode
import numpy as np
def evaluate(decision_tree):
    cleaned_test = dataclean('adult_test.txt')
    # 对测试集中的每一行数据使用决策树进行预测
    predictions = [predict(decision_tree, row) for index, row in cleaned_test.iterrows()]
    print(predictions)
    print(cleaned_test['income'].values)
    # 计算准确率
    actual = cleaned_test['income'].values
    for i in range(len(predictions)):
        predictions[i] = predictions[i].strip()+'.'
    accuracy = np.mean(predictions == actual) 
    print(f"Accuracy: {accuracy*100:.2f}%")
    #将predictions写入文件,作为pred列
    cleaned_test['pred'] = predictions
    cleaned_test.to_csv('adult_test_pred.csv',index=False)
    print("Predictions are written to adult_test_pred.txt")
if __name__ == '__main__':
    import pickle
    with open('decision_tree.pkl', 'rb') as f:
        decision_tree = pickle.load(f)
    evaluate(decision_tree)