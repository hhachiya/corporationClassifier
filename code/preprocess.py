import pandas as pd
import pdb
import numpy as np
import tensorflow as tf
import MeCab
from gensim.models import word2vec
import pickle

def make_x_by_ave(title,description):
    data_len = len(title) #データの数
    #-------------------------
    #title,descriptionごとの処理
    for i in range(data_len): #データ数だけfor文
        title_len = len(title[i]) #title[i]の長さ
        description_len = len(description[i]) #description[i]の長さ

        feature_title_parts = np.zeros_like(title[0][0]) #title[i]の中の特徴量をすべて足し合わせ、最後にtitle[i]の長さで割る(平均)...(1)
        feature_description_parts = np.zeros_like(description[0][0]) #description[i]の中の特徴量をすべて足し合わせ、最後にdescription[i]の長さで割る(平均)...(2)

        #-------------------------
        #title[i]ごとにデータを処理
        for t in range(title_len): #title[i]の長さ分だけfor文
            feature_title_parts = np.add(feature_title_parts,title[i][t]) #(1)
        if title_len > 0: #0では割れないを避ける
            feature_title_parts = np.divide(feature_title_parts,title_len) #(1)
        #description[i]ごとにデータを処理
        for d in range(description_len): #description[i]の長さ分だけfor文
            feature_description_parts = np.add(feature_description_parts,description[i][d]) #(2)
        if description_len > 0: #0では割れないを避ける
            feature_description_parts = np.divide(feature_description_parts,description_len) #(2)
        #-------------------------

        #-------------------------
        #feature_title_parts,feature_description_partsをまとめる
        if i == 0:
            feature_title = feature_title_parts[np.newaxis]
            feature_description = feature_description_parts[np.newaxis]
        else:
            feature_title = np.append(feature_title,feature_title_parts[np.newaxis],axis=0)
            feature_description = np.append(feature_description,feature_description_parts[np.newaxis],axis=0)
        #-------------------------
    #-------------------------
    return np.append(feature_title[np.newaxis],feature_description[np.newaxis],axis=0) #feature_title,feature_descriptionをまとめて出力((2(title,description),221(データ数),200(特徴ベクトル)))



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
                if slice[0] in ["名詞","動詞","形容詞"]: #特定の品詞のみをword2vecにかける
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
                if slice[0] in ["名詞","動詞","形容詞"]: #特定の品詞のみをword2vecにかける
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

    #-------------------------
    X = make_x_by_ave(out_title,out_description) ##feature_title,feature_descriptionをまとめて出力((2(title,description),221(データ数),200(特徴ベクトル)))
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
