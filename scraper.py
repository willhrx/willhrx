import requests
from bs4 import BeautifulSoup

url = "https://www.imdb.com/chart/top-english-movies?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=470df400-70d9-4f35-bb05-8646a1195842&pf_rd_r=J49ZE8K2F2CM21XK9DR3&pf_rd_s=right-4&pf_rd_t=15506&pf_rd_i=moviemeter&ref_=chtmvm_ql_4"

response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')