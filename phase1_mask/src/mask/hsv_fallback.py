import cv2, numpy as np

def find_crosswalk_polygon(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,0,180), (180,60,255))
    edges = cv2.Canny(white, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1,2)
    area = cv2.contourArea(cnt) / (bgr.shape[0]*bgr.shape[1] + 1e-6)
    conf = float(min(1.0, area * 12))
    return poly.tolist(), conf
