#coding=utf-8
#author@zhangdong

# class DataAnalysis():
#     def __init__(self):
#         pass
#
import re
fw=open('../data/trainText.txt','w',encoding='utf-8')

#labels=set(['自动摘要','机器翻译','机器作者','人类作者'])
labels={'人类作者':'1', '机器作者':'2', '自动摘要':'3', '机器翻译':'4'}
dataIndex={'1':set([]),'2':set([]),'3':set([]),'4':set([])}
with open('../data/training.txt','r',encoding='utf-8') as fr:
    wrong_line=[]
    count=0
    for line in fr.readlines():
        d=eval(line)
        #print(d['id'],d)
        id=d['id'];text=d['内容'];label=d['标签']
        text=re.sub('\n','',text)
        text=re.sub('\t','',text)
        text=re.sub('\r','',text)
        if label in labels and text!='':
            try:
                fw.write(text+'\t'+labels[label]+'\n')
                dataIndex[labels[label]].add(count)
                count+=1
            except:
                wrong_line.append(d['id'])
    print(count)


print(len(wrong_line))
print(wrong_line)
print(count)

pkfile=open('dataIndex.pk','wb')
import pickle
pickle.dump(dataIndex,pkfile)
