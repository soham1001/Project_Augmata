#Project Augmata - Robotics Club
#14-01-19
#Import libraries
import cv2

#Main function
def main():
    cap = cv2.VideoCapture(0)
    while True:
        #Get camera data
        _, frame = cap.read()
        #Stereo vision set up
        #Add date and time
        #Add notifications
        #Display results
        cv2.imshow("Input", frame)
        #Transmit data
        #Wait for ESC to be pressed
        key = cv2.waitKey(5) & 0xFF
        if key == 27: break
    cv2.destroyAllWindows()

#Date and time
def date_time():
    return 0

def notif():
    return 0

#Run
if __name__ == "__main__":
    main()
