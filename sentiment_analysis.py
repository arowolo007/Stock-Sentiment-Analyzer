"""
This is a sentiment analyzer for the various assests in my portfolio,

It gets the news headlines from finviz.com a website that shares information
on everything related to the financial markets.

I will be using beautiful soup library to scrap the news section from various
assest in my portfolio and stacking them up in a dataframe.

and finally use vader_sentiment from the popular NLP library NLTK to determine
the positivity or negativity.

Finaly, i will be ploting it on a graph to have a visual understanding of the 
result

"""


# Import required libraries

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Collect the news from finviv.com into tables

news_tables = {}

finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['AAL','AHT','AQMS','NIO']
period = 20

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent':'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response,'html.parser')

    news_table = html.find(id='news-table')
    news_tables[ticker]= news_table


# Extract the important variables from the data collected.

parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        title = row.a.get_text()
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

# Store as dataframe

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# carry out the sentiment analysis on the  data

vader = SentimentIntensityAnalyzer()


condition = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(condition)

df['date'] = pd.to_datetime(df['date']).dt.date

plt.figure(figsize=(10,8))
mean_df = df.groupby(['ticker','date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis='columns').transpose()

# Plot most recent specified period
mean_df.iloc[-period:].plot(kind='bar')
plt.show()

