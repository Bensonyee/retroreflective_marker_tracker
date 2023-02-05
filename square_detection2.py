#import PySpin
#import EasyPySpin
import cv2
import time
import numpy as np
from numpy.linalg import norm
#import pupil_apriltags as apriltag


def cosine_sim(o, a, b):
    cosine = np.dot(a - o, b - o) / (norm(a - o)*norm(b - o))
    theta = np.arccos(cosine)
    return abs(theta - np.pi / 2)
    
def main():
    ANG_DIS_THR = 0.25
    
    
    cap = cv2.VideoCapture("230131_marker/square.avi")
    cnt = 0
    prev_timestamp = time.time()
    while True:
        ret, frame = cap.read()
        cnt += 1
        if cnt % 3 != 1:
            continue
        frame = cv2.resize(frame, (1024, 520), interpolation=cv2.INTER_AREA)
        #frame = cv2.resize(frame, (512, 260), interpolation=cv2.INTER_AREA)
        #print(frame.shape)
        
        
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)  # for RGB camera demosaicing
        # blur
        blur = cv2.GaussianBlur(frame, (0,0), sigmaX=1, sigmaY=1)

        # divide
        #divide = cv2.divide(frame, blur, scale=255)

        # otsu threshold
       # ret, thresh = cv2.threshold(frame, 180, 255, cv2.THRESH_BINARY)

        # apply morphology
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #dst =  cv2.fastNlMeansDenoisingColored(frame,None,3,3,7,21) 
        #img_show = cv2.resize(frame, None, fx=1, fy=1)
        #out_gray=cv2.divide(frame, bg, scale=255)
        #out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[0] 
        #tags = at_detector.detect(frame)
        #print(len(tags))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        lower_red = np.array([0, 0, 205])
        upper_red = np.array([255, 255 , 255])
        #mask = cv2.inRange(hsv, lower_red, upper_red)
        #mask = cv2.inRange(hsv[:,:,2], 200,255)
        #mask = cv2.inRange(hsv[:,:,1], 0,80)
        #mask = cv2.inRange(hsv[:,:,0], 100,140)
        mask = cv2.inRange(hsv,lower_red, upper_red)
        #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

        
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
        #cv2.imshow("press q to quit", mask)
        filter = stats[:,4] > 200
        stats = stats[filter].astype('int32')[1:]
        centroids = centroids[filter].astype('int32')[1:]
        
        #draw check point
        for stat in stats:
            #print(c)
            cv2.rectangle(frame, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (0,0,255), 4)
        for c in centroids:
            cv2.circle(frame, c, 3, (0, 0, 255), 2, cv2.LINE_AA)
            
        if len(centroids) == 4:
            c = np.mean(centroids, axis=0).astype("int32")
            cv2.circle(frame, c, 10, (0, 255, 0), 2, cv2.LINE_AA)
            
        elif len(centroids) == 3:
            sim1 = cosine_sim(centroids[0], centroids[1], centroids[2])
            sim2 = cosine_sim(centroids[1], centroids[0], centroids[2])
            sim3 = cosine_sim(centroids[2], centroids[0], centroids[1])
            if sim1 < ANG_DIS_THR:
                frame = cv2.line(frame, centroids[0], centroids[1], (0,255,255), 3)
                frame = cv2.line(frame, centroids[0], centroids[2], (0,255,255), 3)
                c = np.mean([centroids[1], centroids[2]], axis=0).astype("int32")
                cv2.circle(frame, c, 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame,'Sim: '+str(sim1) , (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            if sim2 < ANG_DIS_THR:
                frame = cv2.line(frame, centroids[1], centroids[0], (0,255,255), 3)
                frame = cv2.line(frame, centroids[1], centroids[2], (0,255,255), 3)
                c = np.mean([centroids[0], centroids[2]], axis=0).astype("int32")
                cv2.circle(frame, c, 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame,'Sim: '+str(sim2) , (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            if sim3 < ANG_DIS_THR:
                frame = cv2.line(frame, centroids[2], centroids[0], (0,255,255), 3)
                frame = cv2.line(frame, centroids[2], centroids[1], (0,255,255), 3)
                c = np.mean([centroids[0], centroids[1]], axis=0).astype("int32")
                cv2.circle(frame, c, 10, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame,'Sim: '+str(sim3) , (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            
        cv2.putText(frame, 'Num Pt: '+str(len(centroids)), (700, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(frame, 'FPS: '+str(1/(time.time() - prev_timestamp)), (700, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        prev_timestamp = time.time()
        cv2.imshow("press q to quit", frame)
        
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()