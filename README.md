# AIcalculateWebpage
# 머신러닝을 이용한 모바일 무인가게 웹 계산대 

______________
## 프로젝트 소개

무인가게에서 본인의 핸드폰을 이용하여 무인가게의 홈페이지에 접속해 구매하려는 상품들을 사진찍고 결제하는 시스템을 만들었다.

핸드폰 + 머신러닝(obejct detection + image classification) + 웹페이지 + 결제시스템


aihub을 이용한 데이터셋, pytorch을 이용한 계산모델, streamlit을 이용한 웹페이지 제작, ngrok을 이용한 웹페이지 배포, 아임포트 결제시스템 연동.
__________________

## 주요기능/사용방법 

1. 웹페이지 접속
2. 촬영버튼으로 상품 촬영
3. 계산된 가격을 결제

____________________________

## 상세설명

### 1. googleColab파트
### 머신러닝을_이용한_모바일_무인가게_웹_계산대의_yolov5와_Resnet 참고


a. 데이터셋 


  aihub에서 상품데이터 다운로드 

  https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=64


b. object detection

  yolov5을 이용해 한 사진에 찍힌 여러상품들을 탐지한다.

  
  https://pytorch.org/hub/ultralytics_yolov5/
  -> best.pt 로 저장

  
c. image classification


  Resnet을 이용해 탐지된 물품들을 식별한다.

  https://pytorch.org/hub/pytorch_vision_resnet/
  -> resnetBest.pt 로 저장

________________________________________
### 2. pythonCode파트
### app.py 참고


4. 아임포트

아임포트를 이용하여 결제 시스템을 연동한다.


https://developers.portone.io/docs/ko/readme

5. 웹페이지

streamlit을 이용하여 웹페이지 제작.

https://streamlit.io/


6. 배포

ngrok을 이용해 웹페이지 배포.

https://ngrok.com/
