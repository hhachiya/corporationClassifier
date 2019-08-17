import pandas as pd
import pdb
import numpy as np
import tensorflow as tf
import MeCab
from gensim.models import word2vec
import pickle



if __name__ ==  '__main__':

    df = pd.read_csv("../data/corporation_sample.csv") #load csv file

    mt = MeCab.Tagger('')
    mt.parse('') 
  	 
    model = word2vec.Word2Vec.load("../data/model/wiki.model")

    title = df['title']
    description = df['description']
    y = df['class'] #0 or 1
    data_len = title.shape[0]

    out_title = []
    out_description = [] 
    
    
    for i in range(data_len):
        title_mecab = mt.parse(title[i])
        description_mecab = mt.parse(description[i])
        parts_title = []
        parts_description = []

        ##########Morphological analysis about title###########
        ##########and word2vec#################################
        for row in title_mecab.split("\n"):
            word = row.split("\t")[0]
            if word == "EOS":  
                break
            else:
                pos = row.split("\t")[1]
                slice = pos.split(",")
                if slice[0] in ["名詞","動詞","形容詞"]: #except special charactor
                    try:
                        word_vec = model.__dict__['wv'][word] #word to vec
                        parts_title.append(word_vec)
                    except:
                        pass

        ##########Morphological analysis about description#####
        ##########and word2vec#################################
        for row in description_mecab.split("\n"):
            word = row.split("\t")[0]
            if word == "EOS":
                break
            else:
                pos = row.split("\t")[1]
                slice = pos.split(",")
                if slice[0] in ["名詞","動詞","形容詞"]: #except special charactor
                    try:
                        word_vec = model.__dict__['wv'][word] #word to vec
                        parts_description.append(word_vec)
                    except:
                        pass
 
        out_title.append(parts_title)
        out_description.append(parts_description)
        #pdb.set_trace()

        with open('../data/out/title_vec.pickle','wb') as f:
            pickle.dump(out_title,f)
        with open('../data/out/description_vec.pickle','wb') as f:
            pickle.dump(out_description,f)
        
    
