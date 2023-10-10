import streamlit as st
from PIL import Image
import os
import time
from datetime import datetime
import torch
import tensorflow as tf
import json
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image,ImageOps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit.components.v1 as components
import pandas as pd


ts = datetime.timestamp(datetime.now())
stockslist={
    (1,"Bibim",1500,100,datetime(year=2019,month=11,day=6,hour=12,minute=33)),
    (2,"Japageti",1400,100,datetime(year=2019,month=11,day=6,hour=12,minute=31)),
    (3,"nuguri",1000,100,datetime(year=2019,month=11,day=6,hour=12,minute=32)),
    (4,"snack",900,100,datetime(year=2019,month=11,day=6,hour=12,minute=30))
}
price = int(0)

# DB Management
import sqlite3 
conn = sqlite3.connect('Stocks_data.db')
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS StocksTable(ID integer PRIMARY KEY,name text ,price integer,remaining integer,date text)')


#cur.executemany('INSERT INTO StocksTable(ID, name, price, remaining, date) VALUES (?,?,?,?,?)' ,stockslist)
#conn.commit()

def table_modify(num):
    cur.execute("SELECT *FROM Stockstable WHERE ID=:Id",{"Id":num})
    data=cur.fetchone()
    quantity=float(data[3])
    product_price=int(data[2])
    quantity=quantity-0.5

    if quantity==-0.5:
        st.error(data[1] + "의 재고가 다 떨어졌습니다.") 
    else:
        global price
        price += product_price
        cur.execute("UPDATE Stockstable SET remaining='%s' WHERE ID='%s'"% (quantity,num))
        cur.execute("UPDATE Stockstable SET date='%s' WHERE ID='%s'"% (datetime.now(),num))
        conn.commit()

device = torch.device("cpu") # device 객체

#데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()

data_dir = 'resnetDatasetSplit'
train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=4, shuffle=True, num_workers=4)

class_names = train_datasets.classes
print('클래스:', class_names)


st.markdown("<h1 style='text-align: center; color: orange;'>Payment</h1>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("")

image_file = st.file_uploader(' ', type=['jpg','png','jpeg'], label_visibility="collapsed")
st.markdown("")
st.markdown("")

if image_file is None:
    st.markdown("<h4 style='text-align: center;'>☘️Please upload your image file first☘️</h4>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("✅If you upload an image, the product will be recognized automatically", unsafe_allow_html=True)
    st.markdown("✅As this is demo version, the number of products in the image should be limited to less than four", unsafe_allow_html=True)
    st.markdown("✅Please upload clear image of products", unsafe_allow_html=True)
    st.markdown("✅Please double-check the total amount before proceeding with the payment, just in case", unsafe_allow_html=True)

else:
    image = Image.open(image_file)
    imgpath = os.path.join('inputImage', str(ts)+'.jpg')
    outputpath = os.path.join('outputImage', os.path.basename(imgpath))
    with open(imgpath, mode="wb") as f:
        f.write(image_file.getbuffer())
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
    pred = model(imgpath)
    pred.render()  # render box in image
    pred_json = json.loads(pred.pandas().xyxy[0].to_json(orient="records"))
    for im in pred.ims:
        im_base64 = Image.fromarray(im)
        im_base64.save(outputpath)
    img_ = Image.open(outputpath)
    st.markdown("<h5 style='text-align: center; color: orange;'>This is your uploaded image and product prediction</h5>", unsafe_allow_html=True)
    st.markdown("")
    st.image(img_, use_column_width='always')
    pred_json = json.loads(pred.pandas().xyxy[0].to_json(orient="records"))
    #st.success(pred_json)  ---> 물품좌표print
    detectnum = str(len(pred_json))+" items have been detected"
    st.success("❗"+detectnum)


    ## 감지물품 3개
    if len(pred_json) ==3:
        xmin0 =int(pred_json[0]["xmin"])
        ymin0= int(pred_json[0]["ymin"])
        xmax0 =int(pred_json[0]["xmax"])
        ymax0 = int(pred_json[0]["ymax"])
        xmin1 =int(pred_json[1]["xmin"])
        ymin1= int(pred_json[1]["ymin"])
        xmax1 =int(pred_json[1]["xmax"])
        ymax1 = int(pred_json[1]["ymax"])
        xmin2 =int(pred_json[2]["xmin"])
        ymin2= int(pred_json[2]["ymin"])
        xmax2 =int(pred_json[2]["xmax"])
        ymax2 = int(pred_json[2]["ymax"])
        xmin2 =int(pred_json[2]["xmin"])
        ymin2= int(pred_json[2]["ymin"])
        xmax2 =int(pred_json[2]["xmax"])
        ymax2 = int(pred_json[2]["ymax"])
        #st.success('비빔면 1500원, 짜파게티1400원, 너구리 1000원, 스낵면 900원')

        #crop0 = 첫번째 인식제품
        croppedImage0=img_.crop((xmin0,ymin0,xmax0,ymax0))
        croppedImage0_resize = croppedImage0.resize((300, 200))
        #st.image(croppedImage0_resize, use_column_width='True')
        croppedImage0 = transforms_test(croppedImage0).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        price = 0
        with torch.no_grad():
            outputs = model(croppedImage0)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage0.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)

        price_memory_1=price
        st.success("1️⃣ "+results+" 💸"+str(price_memory_1)+"￦")
        
        
        #crop1
        croppedImage1=img_.crop((xmin1,ymin1,xmax1,ymax1))
        croppedImage1_resize = croppedImage1.resize((300, 200))
        #st.image(croppedImage1_resize, use_column_width='True')
        croppedImage1 = transforms_test(croppedImage1).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        with torch.no_grad():
            outputs = model(croppedImage1)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage1.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)

        price_memory_2 = price - price_memory_1
        st.success("2️⃣ "+results+" 💸"+str(price_memory_2)+"￦")
        
        #crop2
        croppedImage2=img_.crop((xmin2,ymin2,xmax2,ymax2))
        croppedImage2_resize = croppedImage2.resize((300, 200))
        #st.image(croppedImage2_resize, use_column_width='True')
        croppedImage2 = transforms_test(croppedImage2).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        with torch.no_grad():
            outputs = model(croppedImage2)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage1.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)

        price_memory_3=price-price_memory_1-price_memory_2
        st.success("3️⃣ "+results+ " 💸"+str(price_memory_3)+"￦")
        
        #총가격
        st.error("✔️ Totalprice is 💰"+str(price)+"￦")


    ## 감지물품 2개
    if len(pred_json) ==2:
        xmin0 =int(pred_json[0]["xmin"])
        ymin0= int(pred_json[0]["ymin"])
        xmax0 =int(pred_json[0]["xmax"])
        ymax0 = int(pred_json[0]["ymax"])
        xmin1 =int(pred_json[1]["xmin"])
        ymin1= int(pred_json[1]["ymin"])
        xmax1 =int(pred_json[1]["xmax"])
        ymax1 = int(pred_json[1]["ymax"])
        #st.success('비빔면 1500원, 짜파게티1400원, 너구리 1000원, 스낵면 900원')

        #crop0
        croppedImage0=img_.crop((xmin0,ymin0,xmax0,ymax0))
        croppedImage0_resize = croppedImage0.resize((300, 200))
        #st.image(croppedImage0_resize, use_column_width='True')
        croppedImage0 = transforms_test(croppedImage0).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        price = 0
        with torch.no_grad():
            outputs = model(croppedImage0)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage0.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)
            
        price_memory_1=price
        st.success("1️⃣ "+results+" 💸"+str(price_memory_1)+"￦")
        
        #crop1
        croppedImage1=img_.crop((xmin1,ymin1,xmax1,ymax1))
        croppedImage1_resize = croppedImage1.resize((300, 200))
        #st.image(croppedImage1_resize, use_column_width='True')
        croppedImage1 = transforms_test(croppedImage1).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        with torch.no_grad():
            outputs = model(croppedImage1)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage1.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)

        price_memory_2 = price - price_memory_1
        st.success("2️⃣ "+results+" 💸"+str(price_memory_2)+"￦")

        #총가격
        st.error("✔️ Totalprice is 💰"+str(price)+"￦")

    ##감지물품 1개
    if len(pred_json) ==1:
        xmin0 =int(pred_json[0]["xmin"])
        ymin0= int(pred_json[0]["ymin"])
        xmax0 =int(pred_json[0]["xmax"])
        ymax0 = int(pred_json[0]["ymax"])
        #st.success('비빔면 1500원, 짜파게티1400원, 너구리 1000원, 스낵면 900원')

        #crop0
        croppedImage0=img_.crop((xmin0,ymin0,xmax0,ymax0))
        croppedImage0_resize = croppedImage0.resize((300, 200))
        #st.image(croppedImage0_resize, use_column_width='True')
        croppedImage0 = transforms_test(croppedImage0).unsqueeze(0).to(device)
        #resnet
        model = torch.load('resnetBest.pt',map_location=torch.device('cpu'))
        model.eval()
        results=''
        price = 0
        with torch.no_grad():
            outputs = model(croppedImage0)
            _, preds = torch.max(outputs, 1)
        imshow(croppedImage0.cpu().data[0], title='predication: ' + class_names[preds[0]])
        results = str(class_names[preds[0]])

        if class_names[preds[0]]=='Bibim':
            table_modify(1)

        if class_names[preds[0]]=='Japageti':
            table_modify(2)

        if class_names[preds[0]]=='nuguri':
            table_modify(3)

        if class_names[preds[0]]=='snack':
            table_modify(4)
            
        price_memory_1=price
        st.success("1️⃣ "+results+" 💸"+str(price_memory_1)+"￦")
        
        ##총가격
        st.error("✔️ Totalprice is 💰"+str(price)+"￦")

    pay_name = str(results+" 외 "+str(int(len(pred_json))-1)+"건")

    html_text = """
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />

            <script src="http://code.jquery.com/jquery-1.12.4.min.js" ></script>
            <script src="http://service.iamport.kr/js/iamport.payment-1.1.5.js"></script>
            <script>
            (function() {
                var IMP = window.IMP;
                var code = "imp54356007";  // FIXME: 가맹점 식별코드
                IMP.init(code);

                // 결제요청
                IMP.request_pay({
                    // name과 amount만 있어도 결제 진행가능
                    pg : 'html5_inicis', // pg 사 선택
                    pay_method : 'card',
                    merchant_uid : 'merchant_' + new Date().getTime(),
                name : '%s',
                    amount : %d,
                    buyer_email : 'iamport@siot.do',
                    buyer_name : '구매자이름',
                    buyer_tel : '010-1234-5678',
                    buyer_addr : '서울특별시 강남구 삼성동',
                    buyer_postcode : '123-456',
                    m_redirect_url : 'https://www.yourdomain.com/payments/complete'
                }, function(rsp) {
                    if ( rsp.success ) {
                        var msg = '결제가 완료되었습니다.';
                        msg += '고유ID : ' + rsp.imp_uid;
                        msg += '상점 거래ID : ' + rsp.merchant_uid;
                        msg += '결제 금액 : ' + rsp.paid_amount;
                        msg += '카드 승인번호 : ' + rsp.apply_num;
                }
                    else {
                        var msg = '결제에 실패하였습니다. 에러내용 : ' + rsp.error_msg
                    }
                    alert(msg);
                });
            })();
            </script>

        </head>

            <body>
            </body>

        </html>
        """%(pay_name,price)


    html_file = open('html_file1500.html', 'w')
    html_file.write(html_text)
    html_file.close()

    path_to_html = "html_file1500.html" 

    # Read file and keep in variable
    with open(path_to_html,'r') as f: 
        html_data = f.read()

    ## Show in webpage
    st.success("❗ Please double-check your total price. Then, click button below!")
    st.markdown("")
    if st.button("Make Payment"):
      st.components.v1.html(html_data,height=600,width=840)


































































































