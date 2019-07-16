import numpy as np
import pandas as pd

df=pd.read_csv("data/rotten_tomatoes_reviews.csv")
df_quick = df[:10000]

if __name__=="__main__":
    print(df_quick.head())
    