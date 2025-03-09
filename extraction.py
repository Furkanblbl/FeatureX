import cv2
import numpy as np

cap = cv2.VideoCapture('images/balloons.avi')

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame,(480,480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Masking red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Apply erode and dilate
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Finding the smallest circle enclosed by the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        print("x: ",x,"  y: ",y,"  rad: ",radius)
        center = (int(x), int(y))
        radius = int(radius)

        # Checking if there is a circle
        if radius > 10:
            cv2.circle(frame, center, radius, (0, 255, 0), 3)

    cv2.imshow("shapes", frame)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()