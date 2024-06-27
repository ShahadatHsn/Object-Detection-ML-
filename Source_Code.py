import cv2
#import matplotlib.pyplot as plt
config='Dataset.pbtxt'
frozen='frozen.pb'
model=cv2.dnn_DetectionModel(frozen,config)
class_name = []
#img=cv2.imread("photo1.jpg")

file='labels.txt'
with open(file,'rt') as fp:
    class_name=fp.read().rstrip('\n').split('\n')
print(class_name)
print(len(class_name))

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("can't open the video")
cap.set(3,640)
cap.set(4,480)

font_scale = 1
font=cv2.FONT_HERSHEY_COMPLEX

while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox= model.detect(frame,confThreshold = 0.55)
    print(ClassIndex)
    
    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame, boxes,(0,0,255),1)
                cv2.putText(frame,class_name[ClassInd-1].upper(), (boxes[0]+10, boxes[1]+40),font, fontScale=font_scale, color=(0,0,255),thickness=1)
    #cv2.imshow("Image",img)
    cv2.imshow("Object Detection",frame)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()