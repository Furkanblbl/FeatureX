import cv2
import numpy as np

img = cv2.imread('images/shapes.png')
img = cv2.resize(img,(480,480))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color_ranges = {
    "red": [(0, 120, 70), (10, 255, 255)],
    "blue": [(90, 50, 50), (130, 255, 255)],
    "green": [(35, 50, 50), (85, 255, 255)]
}

def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    if sides == 3:
        return "triangle"
    elif sides == 4:
        return "square"
    else:
        return "circle"

selected_color = "red"
selected_shape = "triangle"

lower, upper = color_ranges[selected_color]
mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

# Kenar tespiti
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Belirlenen objeleri bul
for contour in contours:
    shape = detect_shape(contour)
    if shape == selected_shape:
        # Seçilen objeye dikdörtgen çiz
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("shapes", img)

cv2.waitKey(0)
cv2.destroyAllWindows()