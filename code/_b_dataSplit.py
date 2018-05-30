# coding=utf-8
# @Time    : 2018/5/13 18:47
# @Author  : zhangdong
# @Email   : 18335101891@163.com
# @File    : _b_dataSplit.py
# @Software: PyCharm Community Edition


#把数据集trainText.txt分成训练集，验证集，测试集  80%/10%/10%
#trainText.txt是在原来数据集中去掉了17篇后的数据

#类别有
import pickle
import random
random.seed(10)
import jieba
from zhon.hanzi import punctuation
from string import punctuation as punctuation_en
labels={'人类作者':1, '机器作者':2, '自动摘要':3, '机器翻译':4}

pkfile=open('dataIndex.pk','rb')
dataIndex=pickle.load(pkfile)
print(dataIndex['1'])

global dataIndex

def splitData(l):
    shuffle_1=list(dataIndex[l])
    random.shuffle(shuffle_1)
    len_1=len(dataIndex[l])
    shuffletrain=set(shuffle_1[:int(len_1*0.9)]);shuffletest=set(shuffle_1[int(len_1*0.9):])
    return shuffletrain,shuffletest


shuffle_1_train,shuffle_1_test=splitData('1')
shuffle_2_train,shuffle_2_test=splitData('2')
shuffle_3_train,shuffle_3_test=splitData('3')
shuffle_4_train,shuffle_4_test=splitData('4')

trainIndexSet=shuffle_1_train|shuffle_2_train|shuffle_3_train|shuffle_4_train
testIndexSet=shuffle_1_test|shuffle_2_test|shuffle_3_test|shuffle_4_test



#得到停用此表
stopwords=[]
with open('stopword.txt','r',encoding='utf-8') as fstop:
    for line in fstop.readlines():
        word=line.strip()
        stopwords.append(word)

ftrain=open('smp_train.txt','w',encoding='utf-8')#训练集
ftest=open('smp_test.txt','w',encoding='utf-8')#测试集

f1=open('text_for_word2vec.txt','w',encoding='utf-8')#整个数据集语料，用来做词向量的训练

with open('../data/trainText.txt','r',encoding='utf-8') as fr:
    for id,line in enumerate(fr.readlines()):
        #print(id)
        newline = line.strip().split('\t')
        text = newline[0];
        label = newline[1]
        word_list = [word for word in jieba.lcut(text) if word not in stopwords and punctuation and punctuation_en]
        word_str = ' '.join(word_list)
        f1.write(word_str + '\n')
        if id in trainIndexSet:
            ftrain.write(word_str+'\t'+label+'\n')
        else:
            ftest.write(word_str+'\t'+label+'\n')


