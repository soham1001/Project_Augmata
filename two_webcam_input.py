import cv2 
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
while(1):

        _, frame1 = cap1.read()
        resized_frame1=cv2.resize(frame1,(750,780))        
        cv2.imshow('EYE1', resized_frame1)
        _, frame2 = cap2.read()
        resized_frame2=cv2.resize(frame2,(750,780))
        cv2.imshow('EYE2', resized_frame2)
        if cv2.waitKey(1) == ord('q'):
        
            break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
