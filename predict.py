# -*- coding: utf-8 -*-
import time
start_time = time.time()
import numpy as np
from gensim.models import word2vec

model = word2vec.Word2Vec.load("word2vec.model")    # 加载模型

# 计算两个句子相似度
# list1 = ['单发毛囊', '双发毛囊', '黑点征', '血管样']
# list2 = ['单发毛囊', '双发毛囊', '黑点征', '竹节样发', '火柴样发', '衰老性秃发', '切割性蜂窝织炎']
# print(model.n_similarity(list1, list2))

# 词向量化

# yinxie = {"单发毛囊": 1, "双发毛囊": 2, "黑点征": 3, "竹节样发": 4, "火柴样发": 5}
# keywords = jieba.analyse.extract_tags("单发毛囊双发毛囊黑点征", topK=50, withWeight=True,
#                                               allowPOS=('j', 'l', 'n', 'nt', 'nz', 'vn', 'eng'))  #这几个性质的词比较有用
# print(keywords)
keywords01 = ['单发毛囊', '黑点征', '白点征', '血管样', "盘状红斑狼疮"]
keywords02 = ['单发毛囊', '黑点征', '白点征', '血管样', "盘状红斑狼疮"]
vec1 = np.zeros(100)
for word1 in keywords01:
    vec1 = np.add(vec1, model[word1])
# vec1 = np.divide(sentenceVector, len(keywords01))# np.divide数组的除法运算
vec2 = np.zeros(100)
for word2 in keywords02:
    vec2 = np.add(vec2, model[word2])
# vec2 = np.divide(sentenceVector, len(keywords02))# np.divide数组的除法运算


def calcuDistance(vec1, vec2):
    """
    欧式距离
    :param vec1:
    :param vec2:
    :return:
    """
    return np.sqrt(sum(np.square(vec1 - vec2)))

# 方法一：欧式距离，值越小越相似
result = calcuDistance(vec1, vec2)
dist = np.linalg.norm(vec1 - vec2)
print(dist)


# 方法二：余弦相似度,值越大越相似
d1 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(d1)

# 方法三：调包，值越大学相似
print(model.n_similarity(keywords01, keywords02))

print(time.time()-start_time)