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

    print(words_ls)
    # 构造词典
    dictionary = Dictionary(words_ls)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in words_ls]
    model_list = []
    perplexity = []
    coherence_values = []

    for num_topics in range(2, 15, 1):
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             random_state=1,
                             num_topics=num_topics, update_every=1, chunksize=100, passes=10, alpha='auto',
                             per_word_topics=True)

        model_list.append(lda_model)

        # 计算困惑度
        perplexity_values = lda_model.log_perplexity(corpus)
        print('%d 个主题的Perplexity为: ' % (num_topics - 1),
              perplexity_values)  # a measure of how good the model is. lower the better.
        perplexity.append(perplexity_values)

        # 计算一致性
        coherencemodel = CoherenceModel(model=lda_model, texts=words_ls, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('%d 个主题的Coherence为: ' % (num_topics - 1), round(coherencemodel.get_coherence(), 3))

    # 用subplot()方法绘制多幅图形
    plt.figure(figsize=(16, 5), dpi=200)
    x = range(2, 15, 1)
    # 将画板划分为2行1列组成的区块，并获取到第一块区域
    ax1 = plt.subplot(1, 2, 1)
    # 在第一个子区域中绘图
    plt.plot(x, perplexity)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.xticks(range(1, 15, 2))  # 设置刻度
    plt.title('困惑度')
    plt.grid(True, alpha=0.5)

    # 选中第二个子区域，并绘图
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.xticks(range(1, 15, 2))  # 设置刻度
    plt.title('一致性')
    plt.grid(True, alpha=0.5)

    plt.savefig('./困惑度与一致性.png', dpi=450)


if __name__ == '__main__':
    main()