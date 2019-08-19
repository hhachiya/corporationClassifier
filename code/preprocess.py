import pandas as pd
import pdb
import numpy as np
import tensorflow as tf
import MeCab
from gensim.models import word2vec
import pickle


if __name__ ==  '__main__':

    df = pd.read_csv("../data/corporation_sample.csv") #load csv file

    mt = MeCab.Tagger('') #mecabでのパラメーター選択
    mt.parse('')

    model = word2vec.Word2Vec.load("../data/model/wiki.model") #モデルの読み込み

    title = df['title'] #タイトルの全データ
    description = df['description'] #説明の全データ
    y = df['class'] #0 or 1
    data_len = title.shape[0]

    out_title = [] #タイトルの全データの特徴量が入る
    out_description = [] #内容の全データの特徴量が入る

    #------------------------------------
    #データ数分forループ
    for i in range(data_len):
        title_mecab = mt.parse(title[i]) #titl[i]を分かち書き
        description_mecab = mt.parse(description[i]) #description[i]を分かち書き
        parts_title = [] #タイトルごとの特徴量が入る
        parts_description = [] #内容ごとの特徴量が入る

        #-------------------------
        #タイトルに関して形態素解析を行い、word2vecを用いて特徴量を抽出
        for word_mecab in title_mecab.split("\n"): #改行ごとに単語を取得・・・（１）
            word = word_mecab.split("\t")[0] #タブごとに文章を分割
            if word == "EOS": #最後の単語になれば、（１）を終える
                break
            else:
                pos = word_mecab.split("\t")[1]
                slice = pos.split(",") #単語の品詞を取得
                if slice[0] in ["名詞","動詞","形容詞","固有名詞"]: #特定の品詞のみをword2vecにかける
                    try:
                        word_vec = model.__dict__['wv'][word] #単語をベクトルに変換　(200,)
                        parts_title.append(word_vec) #単語ごとに配列に追加
                    except: #辞書に存在しない言葉があった場合、ベクトルに変換できない
                        pass
        #-------------------------

        #-------------------------
        #タイトルに関して形態素解析を行い、word2vecを用いて特徴量を抽出
        for word_mecab in description_mecab.split("\n"): #改行ごとに単語を取得・・・（１）
            word = word_mecab.split("\t")[0] #タブごとに文章を分割
            if word == "EOS": #最後の単語になれば、（１）を終える
                break
            else:
                pos = word_mecab.split("\t")[1]
                slice = pos.split(",") #単語の品詞を取得
                if slice[0] in ["名詞","動詞","形容詞","固有名詞"]: #特定の品詞のみをword2vecにかける
                    try:
                        word_vec = model.__dict__['wv'][word] #単語をベクトルに変換　(200,)
                        parts_description.append(word_vec) #単語ごとに配列に追加
                    except: #辞書に存在しない言葉があった場合、ベクトルに変換できない
                        pass
        #-------------------------

        #-------------------------
        out_title.append(parts_title) #タイトルごとに配列に追加
        out_description.append(parts_description) #説明ごとに配列に追加
    #-------------------------
    #pdb.set_trace()
    #-------------------------
    X = np.array([np.hstack([np.mean(np.array(out_title[i]),axis=0),np.mean(np.array(out_description[i]),axis=0)]) for i in range(len(out_title))]) ##feature_title,feature_descriptionをまとめて出力(221(データ数),400(特徴ベクトル)))
    Y = np.array(y) #labelをnumpyに変換
    #-------------------------

    #-------------------------
    #------データをpickleで保存
    with open('../data/out/data_x.pickle','wb') as f:
        pickle.dump(X,f)
    with open('../data/out/data_y.pickle','wb') as f:
        pickle.dump(Y,f)
    #-------------------------

    """
    #pickleでout_title,out_out_descriptionを保存
    with open('../data/out/title_vec.pickle','wb') as f:
        pickle.dump(out_title,f)
    with open('../data/out/description_vec.pickle','wb') as f:
        pickle.dump(out_description,f)
    """
    #-------------------------

    #------------------------------------
