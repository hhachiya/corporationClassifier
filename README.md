# corporationClassifier
この `README.md` には、各コードの実行結果、各コードの説明を記載する。
実行ファイルは `test_corpolation_classifier.py`で、 データ作成は `preprocess.py`、 結果の画像作成・出力は `plot_solutions.py` で、行う。

## 項目 [Contents]

1. [データの作成（csvファイルから文章特徴量に変換するまでの流れ） : `preprocess.py`](#ID_1)
	1. [コードの説明](#ID_1-1)
	2. [コードの実行結果](#ID_1-2)

2. [実験方法 : `test_corpolation_classifier.py`](#ID_2)
	1. [コードの説明](#ID_2-1)
	2. [コードの実行結果](#ID_2-2)

3. [実験結果の確認方法：`plot_solutions.py`](#ID_3)
	1. [コードの説明](#ID_3-1)
	2. [コードの実行結果](#ID_3-2)



<a id="ID_1"></a>

## データの作成：`preprocess.py`

<a id="ID_1-1"></a>
### コードの説明
事前に用意しているcsvファイル(corpolationClassifier/data/~.csv)を読み込み、文章特徴量に変換する。<br>

はじめに、csvファイルを読み込み、title,description,classに分ける。
ここで、word2vecの辞書はwikipediaをもとに学習させた辞書を用いている。
<br>
[modelの作成方法](http://bondo.hateblo.jp/entry/2018/05/14/085406)
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

次にtitle,descriptionそれぞれに対して形態素解析を行い、word2vecを用いて特徴量に変換する。特定の品詞（名詞、動詞、形容詞、固有名詞）のみを用いて、特徴量を作成する。この時の特徴ベクトルは(データ数、品詞数、特徴ベクトル(200))となっている。
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
最後にtitle,descriptionの特徴量（200次元 x 単語数の行列）を、単語数の軸でそれぞれ平均し結合することにより、400次元のベクトル（変数X）を作成する。

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
<a id="ID_2-2"></a>
### コードの実行結果
corpolationClassifier/data/outにtestしたデータがtest_result.csvとして保存される。
```test_result.csv
title	description	true class	predict class
176	簡単にオリジナルステッカー印刷 | 1000枚3,550円～低価格で製作	ステッカー・シール・ラベル印刷！自分だけのカスタムステッカー作成はステッカージャパンで！ 24時間365日注文受付・送料無料・低価格保証・豊富な素材・製品: アート紙ステッカー, ユポステッカー, 透明ステッカー, 屋外用ステッカー。	1	1
177	ステッカー・シールラベル印刷が激安 | 印刷通販【メガプリント】	メガプリントのステッカー印刷・シール印刷は写真も綺麗な高品質オフセット印刷なのに格安・激安で作成することが可能です。 ... カットの仕方も格安で制作することが出来る四角ステッカーから台紙までカットできる全抜きステッカー、シート状にハーフカットを配置 …	0	0
178	カッティングステッカー、切り文字ステッカー作成の激安専門店！	カッティングステッカー、切り文字ステッカー作成の事ならお任せ下さい！1枚から激安で作成します！自作では難しい、細かいデザインも最新機械でオーダー作成可能です！用途に合わせたシートも豊富にご用意！当店のシートは簡単に貼り付け、剥がせます！	0	1
179	ステッカー印刷-小ロット対応｜印刷通販【デジタ】	ステッカー印刷のネット通販デジタは驚きの激安価格で高品質なステッカー印刷を実現。屋外用フルカラーステッカーを、より自由に激安価格で制作することが可能になりました。長期間の使用にも耐えられる耐候インクを使用し、車やバイクに貼っても使える …	0	0
```
また、実験で得られたtrain loss,test_loss,train auc,test aucがcorpolationClassifier/data/out/logにtest_corpolation_classifier_log.pickleとして保存される。これは、最後に結果をplotするときに用いる。

<a id="ID_3"></a>
## 実験結果の確認方法：`plot_solutions.py`
<a id="ID_3-1"></a>
### コードの説明
`test_corpolation_classifier.py`で作成されたtest_corpolation_classifier_log.pickleを読み込み変数dataに格納する。
```plot_solutions.py
with open("../data/out/log/test_corpolation_classifier_log.pickle","rb") as f:
      for i in range(dataN*dataType):
          data.append(pickle.load(f))

```
変数dataをplotする。ここで、train loss,test lossのｙ軸は`plt.ylim([0,2])`で統一化している。同様に、aucをplotする際には`plt.ylim([0.5,1.1])`で軸の大きさを定めている。

```plot_solutions.py
for i in range(len(data_name)):
    plt.close()
    if i == 1 or i == 4:
        continue
    plt.plot(range(ite),data[i])
    if data_name[i] == "train loss" or data_name[i] == "test loss":
        plt.ylim([0,2])
    else:
        plt.ylim([0.5,1.1])
    plt.xlabel("iteration")
    plt.ylabel(data_name[i])
    plt.savefig("../data/out/{0}.png".format(data_name[i]))
```
<a id="ID_3-2"></a>
### コードの実行結果
以下のようにtrain loss,test loss,train auc,test aucがcorpolationClassifier/data/out保存される。
<br>
<img src ="https://user-images.githubusercontent.com/44080085/63733584-daf98e80-c8b3-11e9-8d45-2024e869105e.png" width="300">
<img src ="https://user-images.githubusercontent.com/44080085/63733599-ea78d780-c8b3-11e9-9cee-60e5f93ea1e6.png" width="300">
<img src ="https://user-images.githubusercontent.com/44080085/63733611-f9f82080-c8b3-11e9-8719-797f291e2f57.png" width="300">
<img src ="https://user-images.githubusercontent.com/44080085/63733609-f6fd3000-c8b3-11e9-9e31-e68d02b1e89a.png" width="300">
