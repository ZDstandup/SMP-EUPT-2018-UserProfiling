# coding=utf-8
# @Time    : 2018/5/14 20:19
# @Author  : zhangdong
# @Email   : 18335101891@163.com
# @File    : text_clean.py
# @Software: PyCharm Community Edition

import jieba
from zhon.hanzi import punctuation
from string import punctuation as punctuation_en

print(punctuation)
print(punctuation_en)

#得到停用此表
stopwords=[]
with open('stopword.txt','r',encoding='utf-8') as fstop:
    for line in fstop.readlines():
        word=line.strip()
        stopwords.append(word)


f_clean_test=open('clean_test.txt','w',encoding='utf-8')
with open('smp_test.txt','r',encoding='utf-8') as fr:
    for id ,line in enumerate(fr.readlines()):
        print(id)
        newline=line.strip().split('\t')
        text=newline[0];label=newline[1]
        word_list=[word for word in jieba.lcut(text) if word not in stopwords and punctuation and punctuation_en]
        word_str=' '.join(word_list)
        f_clean_test.write(word_str+'\n')



