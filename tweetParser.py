# ----------------------------------------------------------------------
# Step 1: Import necessary modules and environment (which contains
# Twitter API keys) and set up Twitter API authentication and the VADER
# Sentiment Analyzer
# ----------------------------------------------------------------------

# import environment, then import API keys from environment
import pandas as pd
import tweepy
import os

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_token = os.environ['twitter_access_token']
access_token_secret = os.environ['twitter_access_token_secret']

# import + initialize VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# ----------------------------------------------------------------------
# Step 2: Create function to parse tweets and return 
# ----------------------------------------------------------------------

def parseTweets(targetNewsOrg_list,numCycles):
    """
    This function takes in two arguments:
        1) Twitter handle (String), and 
        2) the number of (most recent) tweets you want analyzed. (int) 
    
    It returns a  list of dictionaries with the following key:value 
    pairs for each tweet:
        - "handle":"handle" (str)
        - "date":timestamp
        - "compound":value (float)
        - "positive":value (float)
        - "neutral":value (float)
        - "negative":value (float)
    """
    # variable to store oldest tweet
    oldest_tweet = None   
    
    # create an empty list to store dictionaries
    results_list = []
       
    # ----------------------------------------------------------------------
    # Step 3: 
    # - Iterate in increments of 10 until you get full numTweets.
    # - For each set of 10:
    #   - iterate through, 
    #   - analyze with VADER, then 
    #   - add a dictionary to an overall list
    # ----------------------------------------------------------------------
    
    # loop through each organization in the list to pull tweets
    for i in range(len(targetNewsOrg_list)):
        
        # select the current news org from the list
        handle = targetNewsOrg_list[i]
        
        # iterate the necessary number of times to get the requested numTweets
        for i in range(numCycles):
            
            # get list of tweets, then increment max id so no double-counting
            try:
                tweet_list = api.user_timeline(f"@{handle}", count=10, max_id=oldest_tweet)
            except Exception:
                raise
            
            for i in range(len(tweet_list)):
                # iterate over each tweet in the list to run analysis
                tweetAnalysis = analyzer.polarity_scores(tweet_list[i]['text'])
            
                # add dictionary holding results to results list
                results_list.append({"handle":handle,
                                     "date":tweet_list[i]['created_at'], 
                                     "compound":tweetAnalysis['compound'],
                                     "positive":tweetAnalysis['pos'],
                                     "neutral":tweetAnalysis['neu'],
                                     "negative":tweetAnalysis['neg']})
        
            # reduce max id by one so it doesn't skip a tweet next round
            oldest_tweet = int(tweet_list[i]['id_str']) - 1
   
    # ----------------------------------------------------------------------
    # Step 3: Convert list containing all results to a dataframe and return
    # ----------------------------------------------------------------------
    
    return pd.DataFrame(results_list)