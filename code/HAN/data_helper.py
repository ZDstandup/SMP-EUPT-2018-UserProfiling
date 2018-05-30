#coding=utf-8
#author@zhangdong


import os
from collections import defaultdict
import pickle
import json
import numpy as np


def build_vocab(vocab_path,allText_filename):

    if os.path.exists(vocab_path):
        vocab_file=open(vocab_path,'rb')
        vocab=pickle.load(vocab_file)
        print ('load vocab finished!')
    else:

        word_freq=defaultdict(int)
        with open(allText_filename,'r',encoding='utf-8') as fr:
            for line in fr:
                #review=json.loads(line)
                words=line.strip().split()
                #words=word_tokenizer.tokenize(review['text'])
                for word in words:
                    word_freq[word]+=1
            print ('load finished')


        #把词频小于5的word看为UNK,在词典中序号都为0
        vocab={}
        vocab['UNKNOW_TOKEN']=0
        i=1
        for word,fre in word_freq.items():
            if fre>3:
                vocab[word]=i
                i+=1

        #保存这个词典
        with open(vocab_path,'wb') as f:
            pickle.dump(vocab,f)
            print (len(vocab))
            print ('vocab saved !')

    return vocab


vocab=build_vocab('smp_contest_vocab.pk','./../text_for_word2vec.txt')

#print(len(vocab))

#训练集131760
#测试集14642

def load_dataset(fullfile='../text_for_word2vec.txt',trainOrtest='train',trainOrtestFile='',max_sent_in_doc=20,max_word_in_sent=20,vocab_path='smp_contest_vocab.pk'):
    data_path=trainOrtest+'_data.pk'
    #vocab_path='yelp_academic_dataset_review_vocab.pk'
    if trainOrtest=='train':
        doc_num=131760
    elif trainOrtest=='test':
        doc_num=14642

    if not os.path.exists(data_path):
        vocab=build_vocab(vocab_path,fullfile)
        num_classes=4
        UNKNOW=0
        data_x=np.ones([doc_num,max_sent_in_doc,max_word_in_sent])
        data_y=[]

        with open(trainOrtestFile,'r',encoding='utf-8') as fr:
            for line_index,line in enumerate(fr):
                newline=line.strip().split('\t')
                text=newline[0].split();label=int(newline[1])
                doc = np.zeros([max_sent_in_doc, max_word_in_sent])

                word_num_sent=int(len(text)/max_sent_in_doc)        #由于已经分词，所以将每个文章进行平均，平均到20个句子中

                #把一个文本text分成max_sent_in_doc个句子，每个句子中
                sents=[]#max_sent_in_doc个sent列表，每个列表中暂时有word_num_sent个词
                for i in range(max_sent_in_doc):
                    sent=text[i*word_num_sent:(i+1)*word_num_sent]
                    sents.append(sent)



                for i , sent in enumerate(sents):
                    sent=sents[i]
                    if i < max_sent_in_doc:
                        word_to_index=np.zeros([max_word_in_sent],dtype=int)      #每个句子的表示
                        for j,word in enumerate(sent):
                            if j < max_word_in_sent:
                                word_to_index[j]=vocab.get(word,UNKNOW)         #获取这个word的序号，没有的话就是默认的UNKNOW=0

                        doc[i]=word_to_index        #第i个句子就有word_to_index表示


                data_x[line_index]=doc      #data_x的数据集中的第line_index行就是这个doc表示了

                #---下面获取label
                label_vec=[0]*num_classes
                label_vec[label-1]=1        #用一个4维的向量表示标签

                data_y.append(label_vec)        #添加到data_y中

                print(line_index)

            #此时当前for循环已经构建好data_x与data_y
            #然后保存起来，保存到yelp_data_path
            pickle.dump((data_x,data_y),open(data_path,'wb'))
            print (trainOrtest+'样本数：',len(data_x))

    else:
        pkfile=open(data_path,'rb')
        data_x,data_y=pickle.load(pkfile)


    #划分数据集
    #进行shuffle
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(data_y)))
    print (shuffle_indices)
    x_shuffled = data_x[shuffle_indices]
    y_shuffled = np.array(data_y)[shuffle_indices]

    # split train/test set

    # dev_index=-1*int(dev_percent*len(data_y))
    # train_x, train_y = x_shuffled[:dev_index], y_shuffled[:dev_index]
    # dev_x, dev_y = x_shuffled[dev_index:], y_shuffled[dev_index:]

    return x_shuffled,y_shuffled


# train_x,train_y=load_dataset(fullfile='../text_for_word2vec.txt',trainOrtest='train',trainOrtestFile='../smp_train.txt',max_sent_in_doc=20,max_word_in_sent=20,vocab_path='smp_contest_vocab.pk')
# test_x,test_y=load_dataset(fullfile='../text_for_word2vec.txt',trainOrtest='test',trainOrtestFile='../smp_test.txt',max_sent_in_doc=20,max_word_in_sent=20,vocab_path='smp_contest_vocab.pk')


def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data

        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min(batch_size*(batch_num+1),data_size)
            yield shuffled_data[start_index:end_index]






