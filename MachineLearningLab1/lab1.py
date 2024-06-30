import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取CSV文件
df = pd.read_csv('IMDB-Movie-Data.csv')

# 分割电影类型
df['Genre'] = df['Genre'].str.split(',')

# 将每个电影的类型分开
df = df.explode('Genre')

# 统计每个类型的数量
genre_counts = df['Genre'].value_counts()

# 按照类型数量升序排列
genre_counts = genre_counts.sort_values(ascending=False)

# 绘制条形图
iyel = np.array([255,223,146])/256 # 奶黄色
iblue = np.array([144,190,224])/256 # 淡蓝色
plt.figure(figsize=(16,9))
bars = plt.bar(genre_counts.index, genre_counts.values,color = iyel, edgecolor = iblue)
# 升序展示

# 在条形图上添加数字
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')


plt.gca().invert_xaxis()
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of each Genre')
plt.xticks(rotation=90)
plt.savefig('genre_counts.png', dpi = 800)
plt.show()