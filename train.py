# -*- coding: utf-8 -*-
import multiprocessing
from gensim.models import word2vec


inp = '/Users/xiaosongzi/Desktop/word2vec_train/word.txt'
sentences = word2vec.LineSentence(inp)  # 把目标文本读取出来

# 从0开始训练模型
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=6, size=100)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
model.save("word2vec.model")