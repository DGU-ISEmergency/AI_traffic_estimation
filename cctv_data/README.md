## **실시간 영동전화국 url 크롤링**

### **ffmpeg 사용해서 실시간 live streaming download**

api 및 키 받아옴 (도시교통정보센터 개방데이터 /  UTIC개방데이터 )

ffmpeg 다운로드 받은 폴더에서 bin 폴더 들어가서 (폴더) 검색창에 cmd열기

```
ffmpeg -i "------ /playlist.m3u8" -c copy test.mp4
```

아래 코드 작성

이후, 실시간 스트리밍 데이터 mp4로 변환해서 저장하기 (해당 시간동안 계속 돌려놓아야 함.)
자동으로 60초씩 해당 영상을 연결해서 저장해줌
