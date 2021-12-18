#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import json
import numpy as np
import pandas as pd
from langdetect import detect
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from textblob import TextBlob
import afinn
import plotly.express as px
import plotly.graph_objects as go
import langid
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import emoji


# ## Loading the Telegram data in JSON

# In[2]:


file = open('result.json', encoding='utf8')
data = json.load(file)


# ### Extracting only the messages component of data

# In[3]:


messages = data['messages']


# ### Converting JSON messages into pandas DataFrame

# In[4]:


messagesData = pd.DataFrame(messages)
messagesData.head()


# ### A look at the dataframe

# In[5]:


messagesData.head(5)
for index, dataPoint in messagesData.iterrows():
    if not pd.isna(dataPoint['sticker_emoji']):
        print(emoji.demojize(dataPoint['sticker_emoji'], delimiters=("", "")))
        dataPoint['text'] += emoji.demojize(dataPoint['sticker_emoji'], delimiters=("", ""))


# ### Dropping unnecessary features like ids of message, fromPerson, toPerson etc.

# In[6]:


messagesData = messagesData.loc[:, :'text']


# In[7]:


messagesData = messagesData.drop(['type','id', 'from', 'from_id'], axis = 1)


# ### See size of data and the only features remaining are date and text messages.

# In[8]:


messagesData


# ### isEnglish function takes into a string/text/message and returns true if it is in English and False otherwise
# #### Made use of detect from langid library

# In[9]:


def isEnglish(text):
    try:
#         b = TextBlob(text)
#         print(b.detect_language())
        return langid.classify(text)[0] == 'en'
#         return Detector(arabic_text).languag.code == 'en'
#         return b.detect_language()=='en'
#         return detect(text)=='en'
    except:
        return False


# ### Dropping all the non english text rows from dataframe using isEnglish function defined above
# #### Takes a few minutes to get executed, Progress is measured using tqdm

# In[ ]:


for index, dataPoint in tqdm(messagesData.iterrows(), total = len(messagesData), desc = 'Filtering non English chats'):
    if isEnglish(dataPoint['text']) == False:
        messagesData.drop(index, inplace = True)


# ### Need to reset indexes everytime we drop some rows from the dataFrame

# In[ ]:


messagesData = messagesData.reset_index(drop = True)
messagesData


# ### filterOnLetters function return True if the text/message/string contains "SHIB" or "DOGE", returns False otherwise

# In[ ]:


def filterOnLetters(text, word1 = "SHIB", word2="DOGE"):
#     words = text.split()
    if text.lower().find(word1.lower())!=-1 or text.lower().find(word2.lower())!=-1:
        return True
    return False


# ### Applying filterOnLetters function on the text column of our dataframe, dropping all the rows whose text doesn't contain "SHIB" or "DOGE"

# In[ ]:


# messagesData
messagesData = messagesData[messagesData['text'].apply(filterOnLetters)]


# ### Need to reset indexes everytime we drop some rows from the dataFrame

# In[ ]:


messagesData = messagesData.reset_index(drop = True)
messagesData


# ### Observation:
# #### Initially, we had 49436 rows/messages in our dataframe originally taken from telegram
# #### After removing non-english sentences, we had 40388 messages left with us in the dataframe
# #### Further, on applying the "DOGE" and "SHIB" filter on the dataFrame, we are left with only 2509 messages

# ### Unsupervised Sentiment Analysis -
# #### Using TextBlob lexicon to calculate the sentiment score of each message and further categorizing each message into "positive", "negative" and "neutral" according to the scores assigned by TextBlob

# In[ ]:


messagesData['Polarity_TextBlob'] = messagesData['text'].map(lambda text: TextBlob(text).sentiment.polarity)
listOfCategoryTextBlob = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' 
                                  for score in messagesData['Polarity_TextBlob']]
Category_TextBlob = pd.DataFrame(listOfCategoryTextBlob)
messagesData['Category_TextBlob'] = Category_TextBlob


# #### Using Afinn lexicon to calculate the sentiment score of each message and further categorizing each message into "positive", "negative" and "neutral" according to the scores assigned by Afinn

# In[ ]:


anf = afinn.Afinn()
messagesData['Polarity_Afinn'] = messagesData['text'].map(lambda text: anf.score(text))
Category_Afinn = pd.DataFrame(['positive' if score > 0 else 'negative' if score < 0 else 'neutral' 
                                  for score in messagesData['Polarity_Afinn']])
messagesData['Category_Afinn'] = Category_Afinn


# In[ ]:


sid = SentimentIntensityAnalyzer()
messagesData['Polarity_Vader'] = messagesData['text'].apply(lambda text: sid.polarity_scores(text)).apply(lambda score_dict: score_dict['compound'])
messagesData['Category_Vader']=''
messagesData.loc[messagesData.Polarity_Vader>0,'Category_Vader']='positive'
messagesData.loc[messagesData.Polarity_Vader==0,'Category_Vader']='neutral'
messagesData.loc[messagesData.Polarity_Vader<0,'Category_Vader']='negative'


# #### Above two lexicons are used in order to first compare which will suit better to our dataset.

# In[ ]:


messagesData


# ### Adding one more row to our dataset categorizing each message to know if it contains only "DOGE" or only "SHIB" or both.

# In[ ]:


messagesData['DOGE OR SHIB'] = pd.DataFrame(["DOGESHIB" if "doge" in words.lower() and "shib" in words.lower() else "DOGE" if "doge" in words.lower() else "SHIB"  for words in messagesData['text']])


# ### Grouping the data according to the new feature for comparison of the two lexicons being used

# In[ ]:


messagesData.groupby(by=['DOGE OR SHIB']).describe()


# ## Comparison of Afinn, TextBlob and Vader

# In[ ]:


# Picking the message index with most Afinn polarity i.e. most positive sentiment message according to Afinn.
pos_idx = messagesData[(messagesData.Polarity_Afinn == max(messagesData.Polarity_Afinn))].index[0]

# Picking the message index with least Afinn polarity i.e. most negative sentiment message according to Afinn.
neg_idx = messagesData[(messagesData.Polarity_Afinn == min(messagesData.Polarity_Afinn))].index[0]

# Scores +23 and -19 are taken from max and min values in the description of groupby above.

print(messagesData.iloc[pos_idx][['text']][0])
print()
print(messagesData.iloc[neg_idx][['text']][0])


# ### Observation:
# #### Most positive sentiment according to Afinn doesn't really seem like a positive message as the user is disappointed with the SHIB coin.
# #### Most negative sentiment according to Afinn doesn't really sound like a negative message as the user is being neutral

# In[ ]:


# Picking the message index with most TextBlob polarity i.e. most positive sentiment message according to TextBlob.
pos_idx = messagesData[(messagesData.Polarity_TextBlob == max(messagesData.Polarity_TextBlob))].index[0]

# Picking the message index with least TextBlob polarity i.e. most negative sentiment message according to TextBlob.
neg_idx = messagesData[(messagesData.Polarity_TextBlob == min(messagesData.Polarity_TextBlob))].index[0]

# Scores +1 and -1 are taken from max and min values in the description of groupby above, moreover, TextBlob gives a normalised score.

print(messagesData.iloc[pos_idx][['text']][0])
print()
print(messagesData.iloc[neg_idx][['text']][0])


# In[ ]:


# Picking the message index with most TextBlob polarity i.e. most positive sentiment message according to TextBlob.
pos_idx = messagesData[(messagesData.Polarity_Vader == max(messagesData.Polarity_Vader))].index[0]

# Picking the message index with least TextBlob polarity i.e. most negative sentiment message according to TextBlob.
neg_idx = messagesData[(messagesData.Polarity_Vader == min(messagesData.Polarity_Vader))].index[0]

# Scores +1 and -1 are taken from max and min values in the description of groupby above, moreover, TextBlob gives a normalised score.

print(messagesData.iloc[pos_idx][['text']][0])
print()
print(messagesData.iloc[neg_idx][['text']][0])


# ## Observation:
# ### While looking at most positive and most negative messages by Vader and others, we observe that Vader is indeed actually better scoring the text in terms of sentiments.
# ### Therefore, I will go with Vader and consider it's scores as more legit and accurate.

# ## Graphical Differences in Afinn, TextBlob and Vader scorings

# In[ ]:


fig = go.Figure()
x = ['Positive', 'Negative', 'Neutral']
fig.add_trace(go.Bar(
    x= x,
    y= [messagesData['Category_Afinn'].value_counts()[1], messagesData['Category_Afinn'].value_counts()[2], messagesData['Category_Afinn'].value_counts()[0]], 
    name='Afinn',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=x,
    y=[messagesData['Category_TextBlob'].value_counts()[1], messagesData['Category_TextBlob'].value_counts()[2], messagesData['Category_TextBlob'].value_counts()[0]],
    name='TextBlob',
    marker_color='green'
))
fig.add_trace(go.Bar(
    x=x,
    y=[messagesData['Category_Vader'].value_counts()[1], messagesData['Category_Vader'].value_counts()[2], messagesData['Category_Vader'].value_counts()[0]],
    name='Vader',
    marker_color='orange'
))


fig.update_layout(barmode='group',
                 title="Afinn v/s TextBlob v/s Vader polarity differences",
                xaxis_title="Sentiment",
                yaxis_title="Number of messages")
fig.show()
fig.write_image("AfinnVSTextBlobVSVader.png")


# ### Since, we are only observing dates 1-15 of May 2021, we don't really need month, year and time in the date column, we therefore slice our date to what's required from it.

# In[ ]:


messagesData['date']=messagesData['date'].str.slice(8,10)


# In[ ]:


messagesData


# In[ ]:


messagesData['date'].value_counts()


# In[ ]:


fig  = px.bar(
    x= messagesData['date'],
    title= "Total messages per day",
    labels={
            "x": "Date (May 2021)",
    }
)
fig.update_traces(marker_color='green')
fig.show()
fig.write_image("TotalMessagesPerDay.png")


# ### Observations:
# #### Number of messages increase suddenly on 8th and decrease post 11th, this might be an indicator of price change in either "DOGE" or "SHIB" or both, as people start discussing more about these in the given date range
# #### We will later observe if these messages indicate a positive or a negative sentiment towards these coins.

# In[ ]:


p = messagesData.groupby(['date','Category_Vader']).size()

def numberOfSentimentsPerDay(kindOfSentiment, y, p):
    for i in range(1,16):
        try:
            y.append(p["{:02d}".format(i)][kindOfSentiment])
        except:
            y.append(0)
    return y

yPositive = []
yNegative = []
yNeutral = []
yPositive = numberOfSentimentsPerDay('positive', yPositive, p)
yNegative = numberOfSentimentsPerDay('negative', yNegative, p)
yNeutral = numberOfSentimentsPerDay('neutral', yNeutral, p)


# In[ ]:


fig = go.Figure()
x = messagesData['date'].unique()
yPos = yPositive
yNeg = yNegative
yNeut = yNeutral

fig.add_trace(go.Bar(
    x= x,
    y= yPos,
    name='Positive',
    marker_color='green'
))

fig.add_trace(go.Bar(
    x=x,
    y= yNeg,
    name='Negative',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=x,
    y= yNeut,
    name='Neutral',
    marker_color='orange'
))

fig.update_layout(
    title="Per Day Sentiment Analysis",
    xaxis_title="Dates - May 2021",
    yaxis_title="Number of messages",
)

fig.show()
fig.write_image("averageSentimentPerDayPlot.png")


# ### Observations:
# #### In the date range [9-13], there are more positive sentiment messages in the group, which might be an indicator of people benefitting of either "DOGE" or "SHIB", it might also be the case that people are neutral for either one of "DOGE" or "SHIB" but benefitting from something else as there are comparable negative sentiments too.
# #### There's a high number of neutral messages on 11th and it might be the case that prices remained as expected or no profit no loss for on of the coins.

# In[ ]:


dogeData = messagesData[messagesData['DOGE OR SHIB'] == "DOGE"]
shibData = messagesData[messagesData['DOGE OR SHIB'] == "SHIB"]


# In[ ]:


p = dogeData.groupby(['date','Category_Vader']).size()

yDogePositive = []
yDogeNegative = []
yDogeNeutral = []
yDogePositive = numberOfSentimentsPerDay('positive', yDogePositive, p)
yDogeNegative = numberOfSentimentsPerDay('negative', yDogeNegative, p)
yDogeNeutral = numberOfSentimentsPerDay('neutral', yDogeNeutral, p)


# In[ ]:


fig = go.Figure()
x = dogeData['date'].unique()
yPos = yDogePositive
yNeg = yDogeNegative
yNeut = yDogeNeutral

fig.add_trace(go.Bar(
    x= x,
    y= yPos,
    name='Positive',
    marker_color='green'
))

fig.add_trace(go.Bar(
    x=x,
    y= yNeg,
    name='Negative',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=x,
    y= yNeut,
    name='Neutral',
    marker_color='orange'
))

fig.update_layout(
    title="Per Day Sentiment Analysis - DOGE",
    xaxis_title="Dates - May 2021",
    yaxis_title="Number of messages",
)

fig.show()
fig.write_image("averageSentimentPerDayPlot-DOGE.png")


# ### Observations : 
# #### Price of DOGE must have been risen on 11th or 12th as there are positive sentiments for DOGE on these dates, on prior days, there are comparable negative sentiments indicating that prices might have been falling before 11th.

# In[ ]:


p = shibData.groupby(['date','Category_Vader']).size()

yShibPositive = []
yShibNegative = []
yShibNeutral = []
yShibPositive = numberOfSentimentsPerDay('positive', yDogePositive, p)
yShibNegative = numberOfSentimentsPerDay('negative', yDogeNegative, p)
yShibNeutral = numberOfSentimentsPerDay('neutral', yDogeNeutral, p)


# In[ ]:


fig = go.Figure()
x = shibData['date'].unique()
yPos = yShibPositive
yNeg = yShibNegative
yNeut = yShibNeutral

fig.add_trace(go.Bar(
    x= x,
    y= yPos,
    name='Positive',
    marker_color='green'
))

fig.add_trace(go.Bar(
    x=x,
    y= yNeg,
    name='Negative',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=x,
    y= yNeut,
    name='Neutral',
    marker_color='orange'
))

fig.update_layout(
    title="Per Day Sentiment Analysis - SHIB",
    xaxis_title="Dates - May 2021",
    yaxis_title="Number of messages",
)

fig.show()
fig.write_image("averageSentimentPerDayPlot-SHIB.png")


# In[ ]:





# In[ ]:





# In[ ]:




