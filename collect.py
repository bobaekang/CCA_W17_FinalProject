'''
--------------------------------------------------------------------------------
This is a Python script to collect news articles for the final project for
Computational Content Analysis course in Winter 2017.
--------------------------------------------------------------------------------
This script collects articles from the following newspapers:
    * The Guardian (using API)
    * The Daily Mail (scraping from the web)

In collecting the articles, I use search results for the following keywords:
    * artificial intelligence
    * automation
    * globalization
    * international trade
    * immigration
    * robot
--------------------------------------------------------------------------------
'''
# import packages
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas as pd #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import os #For looking through files
import os.path #For managing file paths
import json
import bs4
import re
from dateutil.parser import parse

'''
--------------------------------------------------------------------------------
For The Guardian articles
--------------------------------------------------------------------------------
In this part, I create a function `getGuardian` for collecting articles from
The Guardian by using the organization's web API.
--------------------------------------------------------------------------------
'''
guard_key = '0bd937fb-cf7e-4727-8698-5b69390c8cd3'
guard_begDate = '2016-01-01'
guard_endDate = '2016-12-31'

def getGuardian(api_key, search, from_date, to_date, pages = 5):

    searchDict = {  'date' : [], #The date the article was published
                    'section' : [], #The section of the article
                    'title' : [], #The title of the article
                    'url' : [], #The url to the article
                    'text' : [], #The text of the article
                    }

    for page in list(range(1, pages+1)):
        target = 'https://content.guardianapis.com/search?api-key={}&q={}&from-date={}&to-date={}&page={}'
        try:
            r = requests.get(target.format(api_key, search, from_date, to_date, page))
            response = json.loads(r.text)
            Docs = response['response']['results']
        except:
            pass

        for Doc in Docs:
            #These are provided by the directory
            # date = parse(Doc['webPublicationDate']).strftime('%Y-%m-%d')
            searchDict['date'].append(parse(Doc['webPublicationDate']).strftime('%Y-%m-%d'))
            # searchDict['date'].append(Doc['webPublicationDate'])
            searchDict['section'].append(Doc['sectionName'])
            searchDict['title'].append(Doc['webTitle'])
            searchDict['url'].append(Doc['webUrl'])
            #We need to download the text though
            try:
                text_raw = requests.get(Doc['webUrl']).text
            except:
                requests.ConnectionError
            soup = bs4.BeautifulSoup(text_raw, 'html.parser')
            pars = soup.body.findAll('p', class_= None)
            text_full = []
            for par in pars:
                text_full.append(par.text)
            text_clean = ' '.join(text_full)
            searchDict['text'].append(text_clean)

    searchDF = pd.DataFrame(searchDict)

    #Get tokens
    searchDF['tokenized_text'] = searchDF['text'].apply(lambda x: nltk.word_tokenize(x))
    searchDF['token_counts'] = searchDF['tokenized_text'].apply(lambda x: len(x))
    #Delete rows with no text due to the irregularity of the original html codes
    finalDF = searchDF[searchDF['text'] != '']
    return finalDF

guardian_art_intel_non_uk = getGuardian(guard_key, 'artificial intelligence', guard_begDate, guard_endDate, pages=3)
guardian_art_intel_uk = getGuardian(guard_key, 'artificial intelligence', guard_begDate, guard_endDate, pages=3)

# guardian_art_intel = getGuardian(guard_key, 'artificial intelligence', guard_begDate, guard_endDate, pages = 50)
# guardian_automation = getGuardian(guard_key, 'automation', guard_begDate, guard_endDate, pages = 50)
# guardian_globalization = getGuardian(guard_key, 'globalization', guard_begDate, guard_endDate, pages = 50)
# guaridan_immigration = getGuardian(guard_key, 'immigration', guard_begDate, guard_endDate, pages = 50)
# guardian_int_trade = getGuardian(guard_key, 'international trade', guard_begDate, guard_endDate, pages = 50)
# guardian_robot = getGuardian(guard_key, 'robot', guard_begDate, guard_endDate, pages = 50)
#
# guardian_art_intel.to_pickle('data/guardian_art_intel.pkl')
# guardian_automation.to_pickle('data/guardian_automation.pkl')
# guardian_globalization.to_pickle('data/guardian_globalization.pkl')
# guaridan_immigration.to_pickle('data/guardian_immigration.pkl')
# guardian_int_trade.to_pickle('data/guardian_int_trade.pkl')
# guardian_robot.to_pickle('data/guardian_robot.pkl')

'''
--------------------------------------------------------------------------------
For Daily Mail articles
--------------------------------------------------------------------------------
In this part, I create a function `getDailyMail` for collecting articles from
The Daily Mail by scraping the web search results.
--------------------------------------------------------------------------------
'''
def getDailyMail(search_term, startn=0, n=100):
    searchDict = {  'date' : [], # date of the article
                    'text' : [], # text of the article
                    'url' : [] # url to the article
                    }
    # Get the search result
    offsets = list(range(startn, n, 50))
    target = 'http://www.dailymail.co.uk/home/search.html?offset={}&size=50&sel=site&searchPhrase={}&sort=relevant&type=article&days=all'
    parsall = []
    for offset in offsets:
        r = requests.get(target.format(offset, search_term))
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        pars = soup.findAll('div', class_=r'sch-res-content')
        parsall.append(pars)
    pars = []
    for i in (range(len(parsall))):
            pars += parsall[i]

    # Get actual articles
    dates = []
    urls = []
    texts = []
    for par in pars:
        date = re.search('[A-Z][a-z]+ [0-9]+[a-z]+ 2016', par.text)
        if date == None:
            continue
        else:
            # Get data
            dates.append(parse(date.group(0)).strftime('%Y-%m-%d'))
            # dates.append(date.group(0))
            # Get urls
            a = par.findAll('a')
            url = 'http://www.dailymail.co.uk' + a[0].get('href')
            urls.append(url)

            # Get texts
            try:
                text_raw = requests.get(url).text
            except:
                requests.ConnectionError

            text_soup = bs4.BeautifulSoup(text_raw, 'html.parser')
            text_pars = text_soup.body.findAll('p', class_= None)
            # this is because DM articles are structured in two ways
            if re.search(r'^[A-Z]', text_pars[0].text) != None:
                text_full = []
                for text_par in text_pars:
                    text_full.append(text_par.text)
                text_clean = ' '.join(text_full)
                texts.append(text_clean)
            else:
                text_pars = text_soup.body.findAll('p', class_= 'mol-para-with-font')
                text_full = []
                for text_par in text_pars:
                    text_full.append(text_par.text)
                text_clean = ' '.join(text_full)
                texts.append(text_clean)

    searchDict['date'], searchDict['text'], searchDict['url'] = dates, texts, urls
    searchDF = pd.DataFrame(searchDict)

    #Get tokens
    searchDF['tokenized_text'] = searchDF['text'].apply(lambda x: nltk.word_tokenize(x))
    searchDF['token_counts'] = searchDF['tokenized_text'].apply(lambda x: len(x))

    return searchDF

# dailymail_art_intel = getDailyMail('artificial+intelligence', startn=0, n=1500)
# dailymail_automation = getDailyMail('automation', startn=0, n=1500)
# dailymail_globalization = getDailyMail('globalization', startn=0, n=1000)
# dailymail_immigration = getDailyMail('immigration', startn=0, n=2000)
# dailymail_int_trade = getDailyMail('international+trade', startn=0, n=2000)
# dailymail_robot = getDailyMail('robot', startn=0, n=2000)
#
# dailymail_art_intel.to_pickle('data/dailymail_art_intel.pkl')
# dailymail_automation.to_pickle('data/dailymail_automation.pkl')
# dailymail_globalization.to_pickle('data/dailymail_globalization.pkl')
# dailymail_immigration.to_pickle('data/dailymail_immigration.pkl')
# dailymail_int_trade.to_pickle('data/dailymail_int_trade.pkl')
# dailymail_robot.to_pickle('data/dailymail_robot.pkl')

'''
--------------------------------------------------------------------------------
Collect all articles and save into pickle files
--------------------------------------------------------------------------------
'''
# import guardian
guardian_art_intel = pd.read_pickle('data/guardian_art_intel.pkl')
guardian_automation = pd.read_pickle('data/guardian_automation.pkl')
guardian_globalization = pd.read_pickle('data/guardian_globalization.pkl')
guardian_immigration = pd.read_pickle('data/guardian_immigration.pkl')
guardian_int_trade = pd.read_pickle('data/guardian_int_trade.pkl')
guardian_robot = pd.read_pickle('data/guardian_robot.pkl')
# import dailymail
dailymail_art_intel = pd.read_pickle('data/dailymail_art_intel.pkl')
dailymail_automation = pd.read_pickle('data/dailymail_automation.pkl')
dailymail_globalization = pd.read_pickle('data/dailymail_globalization.pkl')
dailymail_immigration = pd.read_pickle('data/dailymail_immigration.pkl')
dailymail_int_trade = pd.read_pickle('data/dailymail_int_trade.pkl')
dailymail_robot = pd.read_pickle('data/dailymail_robot.pkl')

# add a topic columne:
# 'A' for automation and realted keywords
# 'I' for immigration and realted keywords
guardian_art_intel['category'] = '0'
guardian_automation['category'] = '0'
guardian_globalization['category'] = '1'
guardian_immigration['category'] = '1'
guardian_int_trade['category'] = '1'
guardian_robot['category'] = '0'

dailymail_art_intel['category'] = '0'
dailymail_automation['category'] = '0'
dailymail_globalization['category'] = '1'
dailymail_immigration['category'] = '1'
dailymail_int_trade['category'] = '1'
dailymail_robot['category'] = '0'

# create a merged corpus for each news outlet
# guardian
guardian_all = pd.concat([guardian_art_intel, guardian_automation, guardian_globalization,
                            guardian_immigration, guardian_int_trade, guardian_robot,],
                        ignore_index=True) # concatenate all
guardian_all.drop_duplicates(subset=['url'], inplace=True) # remove duplicates
guardian_all.reset_index(inplace=True) # reset index

# dailymail
dailymail_all = pd.concat([dailymail_art_intel, dailymail_automation, dailymail_globalization,
                            dailymail_immigration, dailymail_int_trade, dailymail_robot],
                        ignore_index=True) # concatenate all
dailymail_all.drop_duplicates(subset=['url'], inplace=True) # remove dubplicates
dailymail_all.reset_index(inplace=True) # reset index

# save both into pickle files
guardian_all.iloc[:,1:].to_pickle('data/guardian_all.pkl')
dailymail_all.iloc[:,1:].to_pickle('data/dailymail_all.pkl')
