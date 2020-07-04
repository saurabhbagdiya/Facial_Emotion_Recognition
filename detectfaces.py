import numpy as np
import cv2
import time as time
import DetectFacialExpression as objmodel
import MySQLdb
db = MySQLdb.connect(user='root', passwd='', db='Saurabh_Bagdiya',
                             host='127.0.0.1', port=3306)
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)
tim=12
r=0
while r<5:
    #ret, img = cap.read()
    img=cv2.imread('classroom'+str(r)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print (face_cascade)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    i=0
    arr=[0,0,0,0,0,0]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite('./Prediction_data/img'+str(i)+'.jpg',roi_color)
        emotion=objmodel.fnShowAndDetect(roi_color)
        print("inside")
        if emotion=="Angry":
            arr[0]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)

        if emotion=="Fear":
            arr[1]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)
        if emotion=="Happy":
            arr[2]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)
        if emotion=="Sad":
            arr[3]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)
        if emotion=="Surprise":
            arr[4]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)
        if emotion=="Neutral":
            arr[5]+=1
            cv2.imwrite("./emotion/"+emotion+"/"+str(i)+'.jpg',roi_color)
        print(emotion)
        i=i+1
    '''cv2.imshow("screen",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''
   # print (all(faces))
    if(not faces==()):
        t=str(tim)+'min'
        cursor=db.cursor()
        cursor.execute('''UPDATE `17th_april_CN` SET `Angry`=%s,`Fear`=%s,`Happy`=%s,`Sad`=%s,`Surprise`=%s,`Neutral`=%s WHERE `Interval` LIKE %s''',(str(arr[0]),str(arr[1]),str(arr[2]),str(arr[3]),str(arr[4]),str(arr[5]),t,))   
        db.commit()
        
        tim=tim+12
        print("heres")
    #time.sleep(300)
    r=r+1

cv2.destroyAllWindows()

# ret, img = cap.read()
# #img=cv2.imread("grp.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# i=0;
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     cv2.imshow('img',roi_color)
#     cv2.imwrite('./Prediction_data/img'+str(i)+'.jpg',roi_color)
#     i=i+1


# cap.release()
# cv2.destroyAllWindows()