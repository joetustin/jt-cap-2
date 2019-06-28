import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
#from TextCleaning import TextFormating

df = pd.read_csv("data/rotten_tomatoes_reviews.csv")

train, test = train_test_split(df, test_size =-.20, random_state=42)

#from future import unicode_literals
nlp = spacy.load("en_core_web_sm")
# for i, text in enumerate(df["Review"]):
#     corpus =TextFormating(text,nlp)(vectorize=True)
#     df_final["Rev"].iloc[i]=corpus
#doc =nlp(df.Review.values)
#doc =nlp("data/rotten_tomatoes_reviews.csv")

sentence = 'I am working'
document = nlp(sentence)
id_sequence = map(lambda x: x.orth, [token for token in document])
print(id_sequence)
text = map(lambda x: nlp.vocab[x].text, [id for id in id_sequence])
print(text)





