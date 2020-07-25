#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ayaanzhaque/Resurrect/blob/master/celebrityModel/realTwitterWebScraper.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[5]:


get_ipython().system('pip install GetOldTweets3')


# In[6]:


# Importing GetOldTweets3
import GetOldTweets3 as got
# Importing pandas
import pandas as pd

def get_tweets(username, top_only, start_date, end_date, max_tweets):
   
    # specifying tweet search criteria 
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)                          .setTopTweets(top_only)                          .setSince(start_date)                          .setUntil(end_date)                          .setMaxTweets(max_tweets)
    
    # scraping tweets based on criteria
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    
    # creating list of tweets with the tweet attributes 
    # specified in the list comprehension
    text_tweets = [[
                    tw.date,
                    tw.username,
                tw.text] for tw in tweet]
    
    # creating dataframe, assigning column names to list of
    # tweets corresponding to tweet attributes
    news_df = pd.DataFrame(text_tweets, 
                            columns = ['date', 'sender_name', 'text'])
    
    return news_df



# In[7]:


usernames = ["kobebryant"]


# In[12]:


for username in usernames:
  givenUsername = username
  trainTweetCount = 500

  tweet_df = get_tweets(givenUsername, 
                      top_only = True,
                      start_date = "2015-01-01", 
                      end_date = "2019-11-01",
                      max_tweets = trainTweetCount).sort_values('date', ascending=False)

  tweet_df.to_csv("/content/" + givenUsername + ".csv")
  print(tweet_df)

