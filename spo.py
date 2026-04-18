#print(df.loc[2:50])       # specific rows 
#print(df.iloc[0:3,1:3])    # specific rows + columns
#print(df.iloc[:,2:10])      #specific columns
#print(df["album_name"])     # specific column


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Spotify_Music.csv")

''' level1-'''
print(df.info())
print(df.columns)
print(df.head(10))
print(df.tail(10))
print(df.dtypes)
print(df.isnull().sum())
print(df.shape)
print(df['track_genre'].nunique())  # nunique give count and unique shows real values

'''level 2-'''

df = df.dropna()
df = df.drop(columns=['Unnamed: 0'])
print(df.isnull().sum())
#print(df.info())
#print(df[df.duplicated(keep=False)])
print(df.duplicated(keep=False).sum())
df = df.drop_duplicates()
df['duration_ms'] = (df['duration_ms'] / 60000).round(2)
print(df['duration_ms'])

'''level3--'''
print(df.sort_values(by='popularity',ascending=False)[['track_name','popularity']].head(10))
plt.figure()
sns.histplot(x=df['popularity'],bins=20,kde=True)
plt.title("popularity graph")
plt.savefig("A.png")
plt.show()


top_genres = df['track_genre'].value_counts().head(10).index
filtered_df = df[df['track_genre'].isin(top_genres)]
print(df.groupby('track_genre')['popularity'].mean().round(2))
plt.figure()
plt.title('genre vs popularity')
sns.violinplot(x='track_genre',y='popularity',data=filtered_df)
plt.xticks(rotation=45)
plt.savefig("B.png")
plt.show()


print(df.sort_values(by='energy',ascending=False)[['track_name','energy']].head(10))
print(df[['energy','danceability']].corr())
sample_df = df.sample(2000)
sns.regplot(x=sample_df['energy'],y=sample_df['danceability'],scatter_kws={'alpha':0.3},
                line_kws={'color':'red'})
plt.title("energy vs danceability")  
plt.savefig("C.png")              
plt.show()


explicit_count = df['explicit'].value_counts()
print(explicit_count)
plt.pie(explicit_count,labels=explicit_count.index,autopct='%1.1f%%',colors=['skyblue','orange'],explode=[0,0.1])
plt.title("Explicit vs non-Explicit")
plt.savefig("D.png")
plt.show()


print(df.loc[df['duration_ms'].idxmax()][['track_name','duration_ms']])
df = df[df['duration_ms'] > 0]
print(df.loc[df['duration_ms'].idxmin()][['track_name','duration_ms']])


numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

features = ['popularity', 'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
sns.heatmap(df[features].corr(),annot=True,cmap='coolwarm')
plt.show()

corr = df[features].corr()['popularity'].sort_values(ascending=False)
print(corr)










