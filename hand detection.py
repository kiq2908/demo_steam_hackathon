from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        for hand in hands:
            # Process each hand individually
            lmList = hand["lmList"]  # List of 21 Landmark points
            bbox = hand["bbox"]  # Bounding box info x,y,w,h
            centerPoint = hand['center']  # center of the hand cx,cy
            handType = hand["type"]  # Handtype Left or Right

            fingers = detector.fingersUp(hand)
# Count the number of fingers
            finger_count = fingers.count(1)

            # Display the finger count and hand type on the image
            display_text = f"{handType}: {finger_count} fingers"
            cv2.putText(img, display_text, (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            print(finger_count)
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()