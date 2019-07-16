import numpy as np
import pandas as pd
# from collections import Counter, defaultdict
# import re
#
# from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import MultinomialNB
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import PCA
#
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import pos_tag
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
#
# from os import path
# #from PIL import image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#
# from keras.models import Sequential
# from keras import layers
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="darkgrid")
# sns.set(font_scale=1.3)

df=pd.read_csv("data/rotten_tomatoes_reviews.csv")

if __name__=="__main__":
    print(df.shape)
