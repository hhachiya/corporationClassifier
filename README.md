# corporationClassifier
この `README.md` には、各コードの実行結果、各コードの説明を記載する。
実行ファイルは `test_corpolation_classifier.py`で、 データ作成は `preprocess.py`、 結果の画像作成・出力は `plot_solutions.py` で、行う。

## 項目 [Contents]

1. [データの作成（csvファイルから文章特徴量に変換するまでの流れ） : `preprocess.py`](#ID_1)
	1. [コードの説明](#ID_1-1)
	2. [コードの実行結果](#ID_1-2)

2. [実験方法 : `test_corpolation_classifier.py`](#ID_2)
	1. [コードの説明](#ID_2-1)


3. [実験結果の確認方法：`plot_solutions.py`](#ID_3)

4. [実行結果 : `plot_solutions.py`](#ID_4)



<a id="ID_1"></a>

## データの作成：`preprocess.py`

<a id="ID_1-1"></a>
### コードの説明
事前に用意しているcsvファイルを読み込み、文章特徴量に変換する。<br>

はじめに、csvファイルを読み込み、title,description,classに分ける。
ここで、word2vecの辞書はwikipediaをもとに学習させた辞書を用いている。
```preprocess.py
df = pd.read_csv("../data/corporation_sample.csv") #load csv file

mt = MeCab.Tagger('') #mecabでのパラメーター選択
mt.parse('')

model = word2vec.Word2Vec.load("../data/model/wiki.model") #モデルの読み込み

title = df['title'] #タイトルの全データ
description = df['description'] #説明の全データ
y = df['class'] #0 or 1
```

<br>

次にtitle,descriptionそれぞれに対して形態素解析を行い、word2vecを用いて特徴量に変換する。特定の品詞（名詞、動詞、形容詞、固有名詞）のみを用いて、特徴量を作成する。
```preprocess.py
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

```
<br>
最後にtitle,descriptionの特徴量の平均を求め、変数Xとする。
クラスはYとし、X,Yそれぞれを ../data/out にpickleファイルで保存する。

```preprocess.py
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
```
<a id="ID_1-2"></a>

### コードの実験結果
corporationClassifier/data/outに変数X,Yがそれぞれdata_x.pickle,data_y.pickleとして保存される。

<a id="ID_2"></a>

## 実験方法 : `test_corpolation_classifier.py`

<a id="ID_2-1"></a>

### コードの説明
Epoch数とバッチサイズを決め、先ほど用意したデータを読み込む。
```test_corpolation_classifier.py
#Epoch数
nEpo = 1000
# バッチデータ数
batchSize = 30

#======================================
# データ読み込み
X = pickle.load(open("../data/out/data_x.pickle","rb"))
Y = pickle.load(open("../data/out/data_y.pickle","rb"))
```
読み込んだデータをtrain,testに分割する。ここでは、train:test = 8:2としている。引数のtest_sizeによって割合が変化する。
```test_corpolation_classifier.py
(train_x,test_x,train_y,test_y) = train_test_split(X,Y,test_size = 0.2,random_state=0)
```
学習とテストに使用するプレイスホルダーを用意し、今回用いるネットワークを作成する。baseNNの引数であるratesをもとにfc層でdropoutを行っている。
```test_corpolation_classifier.py
x_train = tf.placeholder(tf.float32,shape=[None,400])
x_label = tf.placeholder(tf.float32,shape=[None,1])


x_test = tf.placeholder(tf.float32,shape=[None,400])
x_test_label = tf.placeholder(tf.float32,shape=[None,1])

## build model
train_pred = baseNN(x_train,rates=[0.2,0.5])

test_preds = baseNN(x_test,reuse=True,isTrain=False)

```
#### ニューラルネットワークのプログラム
```test_corpolation_classifier.py
def baseNN(x,reuse=False,isTrain=True,rates=[0.0,0.0]):
    node = [400,100,50,1]
    layerNum = len(node)-1
    f_size = 3

    with tf.variable_scope('baseCNN') as scope:
        if reuse:
            scope.reuse_variables()

        W = [weight_variable("convW{}".format(i),[node[i],node[i+1]]) for i in range(layerNum)]
        B = [bias_variable("convB{}".format(i),[node[i+1]]) for i in range(layerNum)]
        fc1 = fc_relu(x,W[0],B[0],rates[1])
        fc2 = fc_relu(fc1,W[1],B[1])
        fc3 = tf.matmul(fc2,W[2]) + B[2]
    return fc3

```

　
