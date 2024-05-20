import cv2

path = "C:/Users/user/PycharmProjects/emergency/yolov7/inference/test1/frame0.jpg"
image = cv2.imread(path, cv2.IMREAD_COLOR)


# 1 -> 2번에서 우회전하는거
area1_pointA = (412, 450)
area1_pointB = (408, 465)
area1_pointC = (381, 450)
area1_pointD = (378, 465)


# 2
area2_pointA = (422, 485)
area2_pointB = (418, 505)
area2_pointC = (391, 485)
area2_pointD = (388, 505)

# 3
area3_pointA = (422, 505)
area3_pointB = (418, 523)
area3_pointC = (391, 505)
area3_pointD = (388, 523)

# 4
area4_pointA = (422, 523)
area4_pointB = (418, 550)
area4_pointC = (391, 523)
area4_pointD = (388, 550)

# 5 가장 오른쪽 차선인데 아래에서 우회전 하는거
# area5_pointA = (866, 525)
# area5_pointB = (905, 525)
# area5_pointC = (866, 550)
# area5_pointD = (900, 550)

area6_pointA = (880, 425)
area6_pointB = (880, 445)
area6_pointC = (860, 425)
area6_pointD = (860, 450)


# 6 초록
area7_pointA = (880, 450)
area7_pointB = (880, 468)
area7_pointC = (860, 450)
area7_pointD = (860, 468)

# 8 파랑
area8_pointA = (880, 468)
area8_pointB = (880, 485)
area8_pointC = (860, 468)
area8_pointD = (860, 485)


cv2.line(image, area1_pointA, area1_pointB, (0, 255, 0), 2)
cv2.line(image, area1_pointC, area1_pointD, (0, 255, 0), 2)
cv2.line(image, area2_pointA, area2_pointB, (0, 255, 255), 2)
cv2.line(image, area2_pointC, area2_pointD, (0, 255, 255), 2)
cv2.line(image, area3_pointA, area3_pointB, (255, 255, 0), 2)
cv2.line(image, area3_pointC, area3_pointD, (255, 255, 0), 2)
cv2.line(image, area4_pointA, area4_pointB, (255, 255, 255), 2)
cv2.line(image, area4_pointC, area4_pointD, (255, 255, 255), 2)
# cv2.line(image, area5_pointA, area5_pointB, (0, 255, 0), 2)
# cv2.line(image, area5_pointC, area5_pointD, (0, 255, 0), 2)


cv2.line(image, area6_pointA, area6_pointB, (0, 255, 0), 2)
cv2.line(image, area6_pointC, area6_pointD, (0, 255, 0), 2)
cv2.line(image, area7_pointA, area7_pointB, (255, 255, 0), 2)
cv2.line(image, area7_pointC, area7_pointD, (255, 255, 0), 2)
cv2.line(image, area8_pointA, area8_pointB, (0, 255, 255), 2)
cv2.line(image, area8_pointC, area8_pointD, (0, 255, 255), 2)

# cv2.line(image, area9_pointA, area9_pointB, (0, 255, 0), 2)
# cv2.line(image, area9_pointC, area9_pointD, (0, 255, 0), 2)
#
# cv2.line(image, area10_pointA, area10_pointB, (0, 255, 0), 2)
# cv2.line(image, area10_pointC, area10_pointD, (0, 255, 0), 2)
#
# cv2.line(image, area11_pointA, area11_pointB, (0, 255, 0), 2)
# cv2.line(image, area11_pointC, area11_pointD, (0, 255, 0), 2)
#
# cv2.line(image, area0_pointA, area0_pointB, (0, 255, 0), 2)
# cv2.line(image, area0_pointC, area0_pointD, (0, 255, 0), 2)



cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2

# 마우스 콜백 함수
def draw_line(event, x, y, flags, param):
    global pointA, pointB, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            pointA = (x, y)
        else:
            drawing = False
            pointB = (x, y)
            cv2.line(image, pointA, pointB, (0, 255, 0), 2)
            print(f"Line drawn from {pointA} to {pointB}")
            cv2.imshow("image", image)


# 변수 초기화
drawing = False
pointA = (0, 0)
pointB = (0, 0)

# 마우스 콜백 함수 설정
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_line)

# 이미지 표시
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

