import cv2
import numpy as np

# ---------- LOAD ----------
img = cv2.imread("attendance.png")
img = cv2.resize(img, (900, 1200))

# ---------- PREPROCESS ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15, 2
)

# ---------- DETECT LINES ----------
vertical = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
)

horizontal = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
)

# ---------- FIND INTERSECTIONS ----------
intersections = cv2.bitwise_and(vertical, horizontal)

# find intersection points
cnts, _ = cv2.findContours(intersections, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

points = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w//2
    cy = y + h//2
    points.append((cx, cy))

# ---------- CLUSTER INTO GRID ----------
points = sorted(points, key=lambda p: (p[1], p[0]))

# cluster rows
rows = []
current = []
last_y = -1

for p in points:
    x, y = p
    
    if last_y == -1:
        last_y = y
    
    if abs(y - last_y) > 20:
        rows.append(current)
        current = []
        last_y = y
    
    current.append(p)

if current:
    rows.append(current)

# sort each row left → right
for r in rows:
    r.sort(key=lambda p: p[0])

# ---------- BLUE DETECTION ----------
def is_absent(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(
        hsv,
        np.array([100,100,50]),
        np.array([140,255,255])
    )
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            return True
    return False

# ---------- EXTRACT CELLS USING GRID ----------
attendance = []

# rows-1 because intersections include boundaries
for i in range(1, len(rows)-1):  # skip header
    
    absent = 0
    
    for j in range(1, 6):  # columns 1–5
        
        x1, y1 = rows[i][j]
        x2, y2 = rows[i+1][j+1]
        
        cell = img[y1:y2, x1:x2]
        
        if is_absent(cell):
            absent += 1
    
    attendance.append(5 - absent)

# ---------- OUTPUT ----------
print("\nFINAL PERFECT ATTENDANCE:\n")

for i, val in enumerate(attendance):
    print(f"Student {i+1} → {val}/5")

# ---------- DEBUG ----------
debug = img.copy()

for i in range(1, len(rows)-1):
    for j in range(1, 6):
        
        x1, y1 = rows[i][j]
        x2, y2 = rows[i+1][j+1]
        
        cell = img[y1:y2, x1:x2]
        
        if is_absent(cell):
            cv2.rectangle(debug, (x1,y1), (x2,y2), (0,0,255), 2)
        else:
            cv2.rectangle(debug, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("FINAL PERFECT RESULT", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()