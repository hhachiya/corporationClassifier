### word2vecとT-SNEを用いたライブドアのニュース記事の可視化

### 概要
http://ankaji92.hatenablog.com/entry/2016/11/27/212507
を参考に，wikipediaのアーカイブデータをダウンロードする．

## wikipediaデータからword2vecの辞書を学習する手順
1) python3 wikiextractor/WikiExtractor.py jawiki-latest-pages-articles.xml.bz2
2) cat text/*/* > jawiki_raw.txt
3) python wakati.py jawiki_raw.txt data_raw.txt
4) python train.py data_raw.txt ../wikipedia-data-model/word2vec.model

## text2titleの学習データ作成する手順
1) python3 wikiextractor/WikiExtractor.py jawiki-latest-pages-articles.xml.bz2 --json
2) cat text/*/* > jawiki_json.txt
3) python jsonExtractor.py

これにより、title_text.pklが生成される。

## title_text.pklの使い方
```python
>python
>>> import pickle 
>>> fp = open('title_text.pkl','rb') 
>>> title = pickle.load(fp) 
>>> text = pickle.load(fp) 
>>> title.index('徳川家康') # タイトルが徳川家康のindex 
3116 
>>> text[title.index('徳川家康')] # 徳川家康のtext 
```
