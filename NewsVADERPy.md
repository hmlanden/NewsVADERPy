
# NewsVADERPy: News Mood Analysis 
This analysis was run at approximately 6:33 PM PST on 3/6/2018.

- Overall, according to VADER Sentiment, all the news organizations included in the analysis have neutral tweets, as interpreted by VADER Sentiment's recommended interpretation ranges (Positive: 0.5:1, Neutral: 0.5:-0.5, Negative: -0.5:-1).
- At first glance, Breitbart News appears to be the most neutral of all the news sources analyzed. However, upon review of the text of the tweets analyzed, it appears that Breitbart primarily tweets URLs to offsite articles, which likely defeats any attempt to analyze its tweets using VADER.
- The other candidate for the most neutral sentiments, the New York Times, does not appear to be nearly as neutral, at first glance. However, their account has the most neutral average sentiment overall, which seems to indicate that it has a good balance of positive and negative sentiments on their Twitter.
- Overall, as interpreted by VADER Sentiment's recommended ranges (Positive: 0.5:1, Neutral: 0.5:-0.5, Negative: -0.5:-1), all the news organizations fall well within the neutral range.

Opportunities for further investigation could include:
- Longer range analysis
- Analysis excluding tweets that are just URLs
- Looking at correlation between sentiment and Tweet velocity
- Comparing VADER Sentiment analysis with some kind of political bias analysis
- augmenting VADER Sentiment analysis with a custom library of hastags mapped to their sentiment (since it's unlikely that something like #CadetBoneSpurs would properly register as negative)


```python
# ----------------------------------------------------------------------
# Step 1: Import necessary modules and environment
# ----------------------------------------------------------------------

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# This file contains all Twitter-related actions, so no need to import here
import tweetParser as tp
```


```python
# ----------------------------------------------------------------------
# Step 2: Call API, get tweets, and parse tweets into a dataframe+CSV
# ----------------------------------------------------------------------

# create list of target news organizations' Twitter handles
targetNewsOrg_list = ["BBC", "FoxNews", "nytimes", "BreitbartNews",
                      "CBSNews", "USATODAY"]

# create and set color palette for all charts
orgPalette = sns.color_palette("bright", len(targetNewsOrg_list))
sns.set_palette(orgPalette)

# define number of tweets we want to pull from each org
numTweets = 150

# break into increments of 10
numCycles = int(round(numTweets/10))

# create dict to store dictionaries generated during analysis
completeResults_df = tp.parseTweets(targetNewsOrg_list, numCycles)

# rearrange columns to be more sensible
completeResults_df = completeResults_df[["handle", "count", "compound",
                                         "positive", "negative", "neutral",
                                         "date", "text"]]
completeResults_df.to_csv("TweetsAnalyzed.csv")

completeResults_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>handle</th>
      <th>count</th>
      <th>compound</th>
      <th>positive</th>
      <th>negative</th>
      <th>neutral</th>
      <th>date</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>1</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.520393e+09</td>
      <td>😂🙊 Who has a filthier mouth, Jennifer Lawrence...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC</td>
      <td>2</td>
      <td>-0.4215</td>
      <td>0.000</td>
      <td>0.219</td>
      <td>0.781</td>
      <td>1.520390e+09</td>
      <td>A diver swims through a thick soup of plastic ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC</td>
      <td>3</td>
      <td>0.4939</td>
      <td>0.127</td>
      <td>0.000</td>
      <td>0.873</td>
      <td>1.520388e+09</td>
      <td>Every day around the UK, an army of unpaid vol...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC</td>
      <td>4</td>
      <td>-0.1027</td>
      <td>0.165</td>
      <td>0.185</td>
      <td>0.650</td>
      <td>1.520386e+09</td>
      <td>Impressive and terrifying in equal measure. \n...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC</td>
      <td>5</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.520386e+09</td>
      <td>RT @bbcwritersroom: New #Brexit #comedy Soft B...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ----------------------------------------------------------------------
# Step 3: Generate first plot: scatterplot of last 100 tweets showing 
# compound sentiment and sorted by relative timestamp
# ----------------------------------------------------------------------
#set style to be seaborn
sns.set()

# generate overall plot
compoundSentByTime_plot = sns.lmplot(x="count", y="compound", 
                                data=completeResults_df, 
                                palette=orgPalette, hue='handle',
                                fit_reg=False, legend=True, size=8, 
                                scatter_kws={'s':175, 'alpha':0.75,
                                             'edgecolors':'face', 
                                             'linewidths':2})
plt.xlabel("Tweet Count (From Newest to Oldest)",size=14)
plt.ylabel("Compound Sentiment Score", size=14)
plt.title("Tweet Compound Sentiment Analysis Over Time by News Organization", 
          size=16)
plt.locator_params(nbins=5)
plt.xticks(rotation=45)
plt.savefig("CompoundSentimentAnalysis_Scatterplot.png")

#generate subplots
orgPalette = sns.color_palette("bright", len(targetNewsOrg_list))
compoundSentByTime_subplots = sns.lmplot(x="count", y="compound", 
                                data=completeResults_df, 
                                col="handle", col_wrap = 2,
                                palette=orgPalette, hue='handle',
                                fit_reg=False, legend=True, size=8,
                                scatter_kws={'s':175, 'alpha':0.5,
                                             'edgecolors':'face', 
                                             'linewidths':1})
plt.savefig("CompoundSentimentAnalysis_Subplots.png")


plt.locator_params(nbins=5)
plt.xticks(rotation=45)
plt.show(compoundSentByTime_plot)
plt.show(compoundSentByTime_subplots)
```


![png](output_3_0.png)



![png](output_3_1.png)



```python
# ----------------------------------------------------------------------
# Step 4: Generate second plot: bar plot showing overall compound 
# sentiment in the last 100 tweets
# ----------------------------------------------------------------------

# generate dataframe
meanCompoundSent_df = pd.DataFrame(completeResults_df.groupby("handle").mean()["compound"])
meanCompoundSent_df.reset_index(level=0, inplace=True)

# generate x + y
x_axis = np.arange(len(targetNewsOrg_list))
y_axis = meanCompoundSent_df["compound"]

# create bar plot
plt.bar(x_axis, y_axis, color=orgPalette, width=1, align='edge', 
        alpha=0.75, linewidth=1, edgecolor='black')
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, meanCompoundSent_df['handle'], rotation=35)
plt.xlabel('News Organization', size=14)
plt.ylabel('Mean Compound Sentiment', size=14)
plt.ylim(-0.15, 0.17)
plt.title('Average Compound Sentiment by News Organization', size=16)
plt.savefig("avgCompoundSentimentBarchart.png")
plt.show()
```


![png](output_4_0.png)

