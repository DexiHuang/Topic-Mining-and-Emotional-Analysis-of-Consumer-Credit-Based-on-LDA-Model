# encoding=utf-8
import pyLDAvis.gensim_models
import jieba.posseg as jp
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import xlrd
import pandas as pd
import nltk
import jieba
import pyLDAvis
import pyLDAvis.sklearn
from gensim.models.wrappers import LdaMallet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt


def main():
    # 读取需要处理的文本
    #doc1 = open('./data/text.txt', 'rb').read()
    content = []
    with open('./data/text.txt', 'r',encoding='utf-8') as f:
        content = f.read().splitlines()
        print(content)
    # 构建文本集

    # 词性标注条件
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd', 'vn', 'vd')
    # 停用词表
    stopwords = open("./stop_words.txt", "r", encoding='utf-8').read()
    # 分词
    words_ls = []
    for text in content:
        # 采用jieba进行分词
        words = [word.word for word in jp.cut(text) if
                 word.flag in flags and word.word not in stopwords and len(word.word) >= 2]
        words_ls.append(words)

    pd.Series(words_ls).to_excel('words.xls')
    # 构造词典
    dictionary = Dictionary(words_ls)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in words_ls]

    # lda模型，num_topics设置主题的个数
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, iterations=50)
    # U_Mass Coherence
    ldaCM = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print(ldaCM.get_coherence())


    # 打印所有主题，每个主题显示10个词
    for topic in lda.print_topics(num_words=10):
        print(topic)

    # 用pyLDAvis将LDA模式可视化
    plot = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    # 保存到本地html
    pyLDAvis.save_html(plot, './result/pyLDAvis.html')



if __name__ == '__main__':
    main()
