import os
import requests
from bs4 import BeautifulSoup

def get_video_url(url):
    # URL에서 페이지 소스 가져오기
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # source 태그 찾기
    source_tag = soup.find('source')

    # source 태그가 있고 'src' 속성이 있는지 확인
    if source_tag and 'src' in source_tag.attrs:
        video_url = source_tag['src']
        return video_url
    else:
        return None


# 크롤링할 페이지 URL 설정
url = '해당 url (뭔가 유출하면 안되는 것 같음)'

# 비디오 URL 가져오기
video_url = get_video_url(url)

if video_url:
    print("비디오 URL:", video_url)

