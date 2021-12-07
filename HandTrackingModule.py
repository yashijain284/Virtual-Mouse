import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy

class handDetector():
    cap = cv2.VideoCapture(0)
    initHand = mp.solutions.hands  # Initializing mediapipe
    # Object of mediapipe with "arguments for the hands module"
    mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    draw = mp.solutions.drawing_utils  # Object to draw the connections between each finger index
    wScr, hScr = autopy.screen.size()  # Outputs the high and width of the screen (1920 x 1080)
    pX, pY = 0, 0  # Previous x and y location
    cX, cY = 0, 0  # Current x and y location

    def handLandmarks(colorImg):
        landmarkList = []  # Default values if no landmarks are tracked

        landmarkPositions = mainHand.process(colorImg)  # Object for processing the video input
        landmarkCheck = landmarkPositions.multi_hand_landmarks  # Stores the out of the processing object (returns False on empty)
        if landmarkCheck:  # Checks if landmarks are tracked
            for hand in landmarkCheck:  # Landmarks for each hand
                for index, landmark in enumerate(
                        hand.landmark):  # Loops through the 21 indexes and outputs their landmark coordinates (x, y, & z)
                    draw.draw_landmarks(img, hand,
                                        initHand.HAND_CONNECTIONS)  # Draws each individual index on the hand with connections
                    h, w, c = img.shape  # Height, width and channel on the image
                    centerX, centerY = int(landmark.x * w), int(
                        landmark.y * h)  # Converts the decimal coordinates relative to the image for each index
                    landmarkList.append([index, centerX, centerY])  # Adding index and its coordinates to a list

        return landmarkList

    def fingers(landmarks):
        fingerTips = []  # To store 4 sets of 1s or 0s
        tipIds = [4, 8, 12, 16, 20]  # Indexes for the tips of each finger

        # Check if thumb is up
        if landmarks[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)

        # Check if fingers are up except the thumb
        for id in range(1, 5):
            if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][
                2]:  # Checks to see if the tip of the finger is higher than the joint
                fingerTips.append(1)
            else:
                fingerTips.append(0)

        return fingerTips

    while True:
        check, img = cap.read()  # Reads frames from the camera
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Changes the format of the frames from BGR to RGB
        lmList = handLandmarks(imgRGB)
        # cv2.rectangle(img, (75, 75), (640 - 75, 480 - 75), (255, 0, 255), 2)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Gets index 8s x and y values (skips index value because it starts from 1)
            x2, y2 = lmList[12][1:]  # Gets index 12s x and y values (skips index value because it starts from 1)
            finger = fingers(lmList)  # Calling the fingers function to check which fingers are up

            if finger[1] == 1 and finger[
                2] == 0:  # Checks to see if the pointing finger is up and thumb finger is down
                x3 = np.interp(x1, (75, 640 - 75),
                               (0, wScr))  # Converts the width of the window relative to the screen width
                y3 = np.interp(y1, (75, 480 - 75),
                               (0, hScr))  # Converts the height of the window relative to the screen height

                cX = pX + (x3 - pX) / 7  # Stores previous x locations to update current x location
                cY = pY + (y3 - pY) / 7  # Stores previous y locations to update current y location

                autopy.mouse.move(wScr - cX,
                                  cY)  # Function to move the mouse to the x3 and y3 values (wSrc inverts the direction)
                pX, pY = cX, cY  # Stores the current x and y location as previous x and y location for next loop

            if finger[1] == 0 and finger[
                0] == 1:  # Checks to see if the pointer finger is down and thumb finger is up
                autopy.mouse.click()  # Left click

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    initHand = mp.solutions.hands  # Initializing mediapipe
    # Object of mediapipe with "arguments for the hands module"
    mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    draw = mp.solutions.drawing_utils  # Object to draw the connections between each finger index
    wScr, hScr = autopy.screen.size()  # Outputs the high and width of the screen (1920 x 1080)
    pX, pY = 0, 0  # Previous x and y location
    cX, cY = 0, 0  # Current x and y location

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):

                    h, w, c= img.shape
                    cx, cy = int(lm.x*w), int(lm. y*h)
                    print(id, cx, cy)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        def findHands(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img
        def findPosition(self, img, handno=0, draw=True):

            lmlist = []
            xList = []
            yList = []
            bbox = []
            if self.results.multi_hand_landmarks:
                myhand = self.results.multi_hand_landmarks[handno]

                for id, lm in enumerate(myhand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)
                    lmlist.append([id, cx, cy])
                    # if id == 0:
                    if draw:

                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
            return self.lmlist, bbox


        def fingersUp(self):
            fingers =[]
            # Thumb
            if self.lmList [self.tipIds[0]][1]>self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # totalFingers = fingers.count(1)

            return fingers
        def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1+x2)//2, (y1+y2)//2

            if draw:
                cv2.line(img, (x1, y1)(x2, y2), (255, 0, 255),t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
            length= math.hypot(x2-x1, y2-y1)

            return length, img, [x1, y1, x2, y2, cx, cy]




        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)