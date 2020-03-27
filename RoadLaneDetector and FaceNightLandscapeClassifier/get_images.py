import requests
from bs4 import BeautifulSoup
import urllib

with requests.Session() as c:
    kind = input()
    url = input()
    page = c.get(url)	
    plain_text = page.text
    print(plain_text)
    soup = BeautifulSoup(plain_text, "lxml")
    i=0
    #for link in soup.findAll('img',{'class': 'rg_i Q4LuWd tx8vtf'}):
    for link in soup.findAll('img'):
        print(link.get('src'))
        print(i,"-"*80)
        try:
            urllib.request.urlretrieve(link.get('src'),'data/'+kind+'/'+kind+str(i+75)+'.jpeg')
        except:
            pass
        i+=1
