import PySpin
import EasyPySpin
import cv2
import numpy as np
import pupil_apriltags as apriltag

def main():
    cap = EasyPySpin.VideoCapture(0)

    if not cap.isOpened():
        print("Camera can't open\nexit")
        return -1
    
    cap.cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)
    cap.set_pyspin_value("AcquisitionMode", PySpin.AcquisitionMode_Continuous)
    
    #print(cap.get_pyspin_value("StreamBufferHandlingMode"))
    
    cap.set_pyspin_value("BinningVertical", 2)
    #cap.set_pyspin_value("PixelFormat", PySpin.PixelFormat_Mono8)
    cap.set_pyspin_value("PixelFormat", PySpin.PixelFormat_BayerRG8)
    cap.set_pyspin_value("Width", 2048)
    cap.set_pyspin_value("Height", 1080)
    #cap.set_pyspin_value("OffsetX", 768)
    #cap.set_pyspin_value("OffsetY", 284)
    
    cap.set_pyspin_value("ExposureAuto", PySpin.ExposureAuto_Continuous)
    #cap.set_pyspin_value("ExposureAuto", PySpin.ExposureAuto_Off)
    #cap.set_pyspin_value("ExposureTime", 600)
    cap.set_pyspin_value("GainAuto", PySpin.GainAuto_Continuous)
    
    #cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # -1 sets exposure_time to auto
    #cap.set(cv2.CAP_PROP_GAIN, -1)  # -1 sets gain to auto
    
    # bg_noises = []
    # for _ in range(100):
        # ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)  # for RGB camera demosaicing
        # bg_noises.append(frame)
    # bg_noises = np.array(bg_noises)
    # bg_noise = np.mean(bg_noises, axis=0).astype(np.uint8)
    # cv2.imshow("press q to quit", cv2.resize(bg_noise, None, fx=1, fy=1))
    # cv2.waitKey(0)
    # print(bg_noise)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    at_detector = apriltag.Detector(families = 'tag36h11')
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)  # for RGB camera demosaicing
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
        hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

        lower_red = np.array([0, 0, 255])
        upper_red = np.array([160, 255 , 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

        cv2.circle(frame, maxLoc, 20, (0, 0, 255), 2, cv2.LINE_AA)
    

        cv2.imshow("press q to quit", mask)
        
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()