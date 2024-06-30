import numpy as np
from numpy import linalg as la

name = ['鳗鱼饭', '日式炸鸡排', '寿司饭', '烤牛肉', '三文鱼汉堡', '鲁宾三明治', '印度烤鸡', '麻婆豆腐', '宫保鸡丁', '印度奶酪咖喱', '俄式汉堡']
def loadExData():
    return np.array([
        [2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
        [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
        # [1, 1, 2, 1, 1, 2, 1, 1, 4, 5, 1]
        [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]
    ])


def eucSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def cosSim(inA, inB):
    num = float(inA.T @ inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def pearsonSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]

def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xformedItems = dataMat.T @ U[:, :4] @ Sig4.I

    for j in range(n):
        userRating = dataMat[user, j]

        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print(f'The {item} and {j} similarity is: {similarity}')
        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    print(dataMat[user, :])
    unratedItems = np.nonzero(dataMat[user, :] == 0)
    unratedItems = unratedItems[0]
    print("unratedItems: ", unratedItems)
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        # print("item=",item)
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

if __name__ == '__main__':
    data = loadExData()
    user = 1  # 示例用户，假设我们要为用户1推荐菜品
    # recommended_items = recommend(data, user)
    # print(f'Recommended items for user {user}: {recommended_items}')
    for user in range(0,1):
        print("user=",user)
        recommended_items = recommend(data, user)
        #用 name 代替recommended_items[0]
        recommended_items = [(name[recommended_items[i][0]],recommended_items[i][1]) for i in range(len(recommended_items))]
        print(f'Recommended items for user {user}: {recommended_items}')
