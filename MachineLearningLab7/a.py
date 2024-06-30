

import pandas as pd
import io

data = """
Decision Tree with max_depth=5, min_samples_split=2, min_samples_leaf=1 is built in 154.65724396705627 seconds
Accuracy: 80.28%
Decision Tree with max_depth=5, min_samples_split=2, min_samples_leaf=3 is built in 170.38222908973694 seconds
Accuracy: 81.87%
Decision Tree with max_depth=5, min_samples_split=2, min_samples_leaf=5 is built in 135.40307664871216 seconds
Accuracy: 82.23%
Decision Tree with max_depth=5, min_samples_split=2, min_samples_leaf=10 is built in 104.51713132858276 seconds
Accuracy: 82.26%
Decision Tree with max_depth=5, min_samples_split=5, min_samples_leaf=1 is built in 135.96152019500732 seconds
Accuracy: 81.10%
Decision Tree with max_depth=5, min_samples_split=5, min_samples_leaf=3 is built in 146.12953972816467 seconds
Accuracy: 82.14%
Decision Tree with max_depth=5, min_samples_split=5, min_samples_leaf=5 is built in 147.78110718727112 seconds
Accuracy: 82.23%
Decision Tree with max_depth=5, min_samples_split=5, min_samples_leaf=10 is built in 114.40119314193726 seconds
Accuracy: 82.26%
Decision Tree with max_depth=5, min_samples_split=10, min_samples_leaf=1 is built in 113.90637016296387 seconds
Accuracy: 81.72%
Decision Tree with max_depth=5, min_samples_split=10, min_samples_leaf=3 is built in 111.86602926254272 seconds
Accuracy: 82.44%
Decision Tree with max_depth=5, min_samples_split=10, min_samples_leaf=5 is built in 109.14300441741943 seconds
Accuracy: 82.52%
Decision Tree with max_depth=5, min_samples_split=10, min_samples_leaf=10 is built in 111.66923332214355 seconds
Accuracy: 82.26%
Decision Tree with max_depth=5, min_samples_split=20, min_samples_leaf=1 is built in 72.26541757583618 seconds
Accuracy: 82.44%
Decision Tree with max_depth=5, min_samples_split=20, min_samples_leaf=3 is built in 70.5857424736023 seconds
Accuracy: 82.97%
Decision Tree with max_depth=5, min_samples_split=20, min_samples_leaf=5 is built in 69.99088072776794 seconds
Accuracy: 83.00%
Decision Tree with max_depth=5, min_samples_split=20, min_samples_leaf=10 is built in 71.02867031097412 seconds
Accuracy: 82.69%
Decision Tree with max_depth=10, min_samples_split=2, min_samples_leaf=1 is built in 277.5126419067383 seconds
Accuracy: 78.37%
Decision Tree with max_depth=10, min_samples_split=2, min_samples_leaf=3 is built in 231.10482335090637 seconds
Accuracy: 81.03%
Decision Tree with max_depth=10, min_samples_split=2, min_samples_leaf=5 is built in 182.48457431793213 seconds
Accuracy: 81.92%
Decision Tree with max_depth=10, min_samples_split=2, min_samples_leaf=10 is built in 127.49188899993896 seconds
Accuracy: 82.24%
Decision Tree with max_depth=10, min_samples_split=5, min_samples_leaf=1 is built in 182.84895610809326 seconds
Accuracy: 80.10%
Decision Tree with max_depth=10, min_samples_split=5, min_samples_leaf=3 is built in 181.0812873840332 seconds
Accuracy: 81.71%
Decision Tree with max_depth=10, min_samples_split=5, min_samples_leaf=5 is built in 181.9937653541565 seconds
Accuracy: 81.92%
Decision Tree with max_depth=10, min_samples_split=5, min_samples_leaf=10 is built in 126.78135013580322 seconds
Accuracy: 82.24%
Decision Tree with max_depth=10, min_samples_split=10, min_samples_leaf=1 is built in 126.46585130691528 seconds
Accuracy: 81.06%
Decision Tree with max_depth=10, min_samples_split=10, min_samples_leaf=3 is built in 126.26388263702393 seconds
Accuracy: 82.16%
Decision Tree with max_depth=10, min_samples_split=10, min_samples_leaf=5 is built in 126.63857388496399 seconds
Accuracy: 82.37%
Decision Tree with max_depth=10, min_samples_split=10, min_samples_leaf=10 is built in 127.13712215423584 seconds
Accuracy: 82.24%
Decision Tree with max_depth=10, min_samples_split=20, min_samples_leaf=1 is built in 91.32670140266418 seconds
Accuracy: 81.89%
Decision Tree with max_depth=10, min_samples_split=20, min_samples_leaf=3 is built in 92.16112971305847 seconds
Accuracy: 82.76%
Decision Tree with max_depth=10, min_samples_split=20, min_samples_leaf=5 is built in 91.22318530082703 seconds
Accuracy: 82.88%
Decision Tree with max_depth=10, min_samples_split=20, min_samples_leaf=10 is built in 91.53296875953674 seconds
Accuracy: 82.66%
Decision Tree with max_depth=15, min_samples_split=2, min_samples_leaf=1 is built in 272.0530173778534 seconds
Accuracy: 78.37%
Decision Tree with max_depth=15, min_samples_split=2, min_samples_leaf=3 is built in 223.39912509918213 seconds
Accuracy: 81.03%
Decision Tree with max_depth=15, min_samples_split=2, min_samples_leaf=5 is built in 175.36762690544128 seconds
Accuracy: 81.92%
Decision Tree with max_depth=15, min_samples_split=2, min_samples_leaf=10 is built in 121.77710700035095 seconds
Accuracy: 82.24%
Decision Tree with max_depth=15, min_samples_split=5, min_samples_leaf=1 is built in 176.04789757728577 seconds
Accuracy: 80.10%
Decision Tree with max_depth=15, min_samples_split=5, min_samples_leaf=3 is built in 175.87146544456482 seconds
Accuracy: 81.71%
Decision Tree with max_depth=15, min_samples_split=5, min_samples_leaf=5 is built in 175.60051131248474 seconds
Accuracy: 81.92%
Decision Tree with max_depth=15, min_samples_split=5, min_samples_leaf=10 is built in 122.16849851608276 seconds
Accuracy: 82.24%
Decision Tree with max_depth=15, min_samples_split=10, min_samples_leaf=1 is built in 122.55124115943909 seconds
Accuracy: 81.06%
Decision Tree with max_depth=15, min_samples_split=10, min_samples_leaf=3 is built in 122.53725361824036 seconds
Accuracy: 82.16%
Decision Tree with max_depth=15, min_samples_split=10, min_samples_leaf=5 is built in 122.42697858810425 seconds
Accuracy: 82.37%
Decision Tree with max_depth=15, min_samples_split=10, min_samples_leaf=10 is built in 122.56558537483215 seconds
Accuracy: 82.24%
Decision Tree with max_depth=15, min_samples_split=20, min_samples_leaf=1 is built in 88.21663403511047 seconds
Accuracy: 81.89%
Decision Tree with max_depth=15, min_samples_split=20, min_samples_leaf=3 is built in 88.30078315734863 seconds
Accuracy: 82.76%
Decision Tree with max_depth=15, min_samples_split=20, min_samples_leaf=5 is built in 88.42629647254944 seconds
Accuracy: 82.88%
Decision Tree with max_depth=15, min_samples_split=20, min_samples_leaf=10 is built in 88.13567805290222 seconds
Accuracy: 82.66%
"""

df = pd.read_csv(io.StringIO(data), sep="\s+", header=None)
# 合并df连续的两行
# df = df[0].str.cat(df[1].values, sep=" ").str.split(" ", expand=True)

print(df)
df.to_csv("log1.csv", index=False)
