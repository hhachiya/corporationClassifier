import json
import pdb
import numpy as np
import pickle

with open('jawiki_json.txt','rt') as fp:
	jawiki = fp.readlines()


titles = []
texts = []

for a in np.arange(len(jawiki)):
	article = json.loads(jawiki[a])
	
	titles.append(article['title'])
	texts.append(article['text'])

with open('title_text.pkl','wb') as fp:
	pickle.dump(titles,fp)
	pickle.dump(texts,fp)	
