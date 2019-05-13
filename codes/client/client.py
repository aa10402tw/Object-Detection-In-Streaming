import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd_new import build_ssd
from data import VOC_CLASSES as labels



from selenium import webdriver
from io import BytesIO
import PIL  
import base64
import time
import  matplotlib.pyplot as plt


def getVideoImage():
    return driver.execute_script("""var canvas = document.createElement('canvas');
                                    var video = arguments[0];
                                    canvas.height = video.videoHeight;
                                    canvas.width = video.videoWidth;
                                    var ctx = canvas.getContext('2d');
                                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                    return canvas.toDataURL();"""
                                    , driver.find_element_by_xpath("//*[@id=\"myVideo\"]"))

def cleanCanvas():
    driver.execute_script("""var canvas = arguments[0];
                        var ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.beginPath();
                        """
                        , driver.find_element_by_xpath("//*[@id=\"myCanvas\"]"))
    
def drawRect(name, topLeft, topRight, bottomLeft, bottomRight):
    driver.execute_script("""var canvas = arguments[0];
                                    var ctx = canvas.getContext('2d');
                                    ctx.font = "24px Arial";
                                    ctx.fillStyle = 'red';
                                    ctx.fillText("{}",canvas.width*{},canvas.height*{});
                                    ctx.strokeStyle="red";
									ctx.lineWidth = 5;
                                    ctx.rect(canvas.width*{}, canvas.height*{}, canvas.width*{}, canvas.height*{});
                                    ctx.stroke();""".format(name ,topLeft, topRight, topLeft, topRight, bottomLeft, bottomRight)
                                    , driver.find_element_by_xpath("//*[@id=\"myCanvas\"]"))


net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_new_115000.pth')

try:
    driver = webdriver.Chrome('D:\programs\chromedriver\chromedriver')
except:
    driver = webdriver.Chrome()
    
#driver.get("http://localhost/dash_test.html")
driver.get("http://192.168.1.78/dash.html")
video = driver.find_element_by_xpath("//*[@id=\"myVideo\"]")

keyboardBreak = False
while True:
    while True:
        try:
            time.sleep(0.05)
            base64Image = getVideoImage()
            image = BytesIO(base64.b64decode(base64Image.split("base64,")[1]))
        except KeyboardInterrupt:
            print("interrupt")
            keyboardBreak = True
            break
        except:
            continue
        else:
            break
    if(keyboardBreak):
        break
    img = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(img,cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    

    top_k=10

#     plt.figure(figsize=(10,10))
#     plt.imshow(rgb_image)
#     currentAxis = plt.gca()

    checkSet = set()

    checkedBoxXpathList = ["//*[@id=\"chooseObj\"]/input[1]",
                           "//*[@id=\"chooseObj\"]/input[2]",
                           "//*[@id=\"chooseObj\"]/input[3]",
                           "//*[@id=\"chooseObj\"]/input[4]",
                           "//*[@id=\"chooseObj\"]/input[5]"]
    

    checkBoxNameList = ['person',
                       'bottle',
                       'chair',
                       'diningtable',
                       'tvmonitor']

    for i in range(len(checkedBoxXpathList)):
         if(driver.find_element_by_xpath(checkedBoxXpathList[i]).get_attribute('checked') == 'true'):
            checkSet.add(checkBoxNameList[i])


    detections = y.data
    
    allDrawRect = []
    #print(checkSet)
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        if(not (labels[i-1] in checkSet)):
            continue 
        while detections[0,i,j,0] >= 0.3:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
    #         print(label_name)
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
    #         coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
    #         color = colors[i]
    #         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            h, w, _ = rgb_image.shape
#             drawRect(display_txt,pt[0]/w, pt[1]/h, (pt[2]-pt[0]+1)/w, (pt[3]-pt[1]+1)/h)
            allDrawRect.append((display_txt,pt[0]/w, pt[1]/h, (pt[2]-pt[0]+1)/w, (pt[3]-pt[1]+1)/h))
    #         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
    cleanCanvas()
    for i in allDrawRect:
        drawRect(*i)

