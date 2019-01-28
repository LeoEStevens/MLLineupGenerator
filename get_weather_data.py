"""
get_weather_data.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class is just a basic web scraper. Each game on nflweather.com is on a different page.
Those pages are downloaded using urllib2 and processed using BeautifulSoup. 
The BeautifulSoup finds all p tags and passes them into a list, then it is searched through to get the tags we want.
A dummy average value is used for any data that doesnt exist or is incorrectly formatted.
The data is then cleaned and put into a dictionary to be returned.
"""
from BeautifulSoup import BeautifulSoup as BS
import urllib2

def get_weather_data(season, week, away, home):
    """
    This method downloads the weather data from nflweather.com.
    The data is grabbed using urllib2 and processed using beautifulsoup.
    The html tags and any units are then filtered out.
    """
    if int(season == 2010):
        url = 'http://www.nflweather.com/en/game/' + str(season) + '/week-' + str(week) + '-2/' + str(away) + '-at-' + str(home)
    else:
        url = 'http://www.nflweather.com/en/game/' + str(season) + '/week-' + str(week) + '/' + str(away) + '-at-' + str(home)
    try:
        html = urllib2.urlopen(url)
    except urllib2.HTTPError:
        print("URL ERROR: " + url)
        weather = {'conditions':'Clear', 'temp':72, 'wind':0,
                 'humidity':40, 'visibility':10, 'barometric_pressure':1013,
                 'dew_point':55}
        return weather
    soup = BS(html)
    weather_info = soup.findAll('p')
    if (int(season) == 2014 and week >= 14) or season > 2014:
        try:
            conditions = str(weather_info[3])
            conditions = conditions.replace('<p>', ' ').replace('</p>', ' ').strip()
        except:
            conditions = 'Clear'
        try: 
            temp = str(weather_info[4]).split('<b>')[1]
            temp = int(temp.replace('f.', ' ').replace('</p>', '').replace('</b>', ' ').strip())
        except:
            temp = 72
    
        try:
            wind = str(weather_info[6]).split('<b>')[1]
            wind = int(wind.split('mi')[0])
        except:
            wind = 0
        try:
            hum = str(weather_info[7]).split('<b>')[1]
            hum = int(hum.split('%')[0])
        except:
            hum = 40
        try:
            vis = str(weather_info[8]).split('<b>')[1]
            vis = int(vis.split('mi')[0])
        except:
            vis = 10
        try:
            bar = str(weather_info[9]).split('<b>')[1]
            bar = int(bar.split('"')[0])
        except:
            bar = 1013
        try:
            dew = str(weather_info[10]).split('<b>')[1]
            dew = int(dew.split('f.')[0])
        except:
            dew = 52

    else:
        conditions = str(weather_info[3])
        conditions = conditions.replace('<p>', ' ').replace('</p>', ' ').strip()
        try: 
            temp = str(weather_info[4]).split('<b>')[1]
            temp = int(temp.replace('f.', ' ').replace('</p>', '').replace('</b>', ' ').strip())
        except:
            temp = 72    
        try:
            wind = str(weather_info[5]).split('<b>')[1]
            wind = int(wind.split('mi')[0])
        except:
            wind = 0
        try: 
            hum = str(weather_info[6]).split('<b>')[1]
            hum = int(hum.split('%')[0])
        except:
            hum = 10
        try:
            vis = str(weather_info[7]).split('<b>')[1]
            vis = int(vis.split('mi')[0])
        except:
            vis = 10
        try: 
            bar = str(weather_info[8]).split('<b>')[1]
            bar = int(bar.split('"')[0])
        except:
            bar = 1013
        try:
            dew = str(weather_info[9]).split('<b>')[1]
            dew = int(dew.split('f.')[0])
        except:
            dew = 52
    weather = {'conditions':conditions, 'temp':temp, 'wind':wind,
            'humidity':hum, 'visibility':vis, 'barometric_pressure':bar,
            'dew_point':dew}
    return weather  
