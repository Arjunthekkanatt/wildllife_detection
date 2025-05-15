from fileinput import filename

import cv2
import datetime
import os
import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s',force_reload=False)

os.makedirs("detections",exist_ok=True)
cap=cv2.VideoCapture("Sample_video.mp4")
ret,frame=cap.read()
ret,frame2=cap.read()
while cap.isOpened():
    ret,frame=cap.read()
    diff=cv2.absdiff(frame,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=3)
    con,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    motion_detected=False

    for c in con:
        if cv2.contourArea(c)<1000:
            continue
        motion_detected=True

    if motion_detected:
        results=model(frame)
        labels,cords=results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        for i in range(len(labels)):
            row=cords[i]
            print(row)
            if row[4]>=0.4:
                x1,y1,x2,y2 = int(row[0]*frame.shape[1]),int(row[1]*frame.shape[0]),int(row[2]*frame.shape[1]),int(row[3]*frame.shape[0])
                label = model.names[int(labels[i])]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame,label,(x1,y1,),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"detections/motion_{timestamp}.jpg"
        cv2.imwrite(filename,frame)
        print(f"[info] Motion detected and saved:{filename}")

    cv2.imshow("motion detection",frame)
    frame=frame2
    ret,frame2=cap.read()
    if not ret:
     break
    if cv2.waitKey(30)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()