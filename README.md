Importing neccesary libraryes 
import pandas as pd
import matplotlib .pyplot as plt
import numpy as np
import seaborn as sns

import nltk
read the dataset of business and user reviwes 
b_data = pd.read_json('yelp_academic_dataset_business.json',lines=True)
u_data = pd.read_json('yelp_academic_dataset_tip.json',lines=True)
b_data.head(10)
help of pandas merge function we merge the two dataset behalf of "business_id" for better anylesis of perticuleer user reviews on product
data = pd.merge(b_data,u_data,on='business_id')
data.head(20)

data['text'].values[0]
Understanding the how many stars is given by product and their counts 
ax = data['stars'].value_counts().sort_index().plot(kind='bar',title="Review counts in stars",figsize=(10,5))
ax.set_xlabel("Review Stars")
plt.show()
in this we use VADER(Valence Aware Dictionary and sEntiment Reasoner) for anylesis of each word behalf of polarity scores 
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sen  = SentimentIntensityAnalyzer()
help of polarity scores we understand the each user review is postive , negative or neutral that  basic we anylesis of the sentiment anylesis 
res = {}
for i, row in tqdm(data.iterrows(),total=len(data)):
    text  = row['text']
    myid = row['user_id']
    res[myid] = sen.polarity_scores(text)
    

vader = pd.DataFrame(res).T
vader = vader.reset_index().rename(columns={'index': 'user_id'})
merged_data = pd.merge(vader,data,on='user_id')
in above code our polarity scores data is get by user review and user id and this hole data file we merge with our main dataset
merged_data
ax = sns.barplot(data=merged_data, x='stars',y= 'compound')
ax.set_title("compound score for bussines reviwe")
plt.show()
fig,axs = plt.subplots(1,3,figsize = (12,3))
sns.barplot(data=merged_data, x= 'stars',y ='pos',ax = axs[0])
sns.barplot(data=merged_data, x= 'stars',y ='neu',ax = axs[1])
sns.barplot(data=merged_data, x= 'stars',y ='neg',ax = axs[2])
axs[0].set_title("positive")
axs[1].set_title("neutarl")
axs[2].set_title("Negative")
plt.show()
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')



# Initialize Matcher
matcher = Matcher(nlp.vocab)
# Define patterns to extract aspects (example patterns)




patterns = [
    {"label": "PRODUCT", "pattern": [{"POS": "NOUN"}, {"OP": "*"}]}  # Example pattern to match nouns and optional subsequent tokens
]
patterns
# Hackathon-AIML
: Aspect-Based Sentiment Analysis on Customer Reviews
