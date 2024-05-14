import cv2

path = "C:/Users/user/PycharmProjects/emergency/yolov7/inference/images/frame2.jpg"
image = cv2.imread(path, cv2.IMREAD_COLOR)
# 1
# area1_pointA = (538, 350)
# area1_pointB = (565, 350)
# area1_pointC = (535, 380)
# area1_pointD = (560, 380)

# 2
# area1_pointA = (565, 350)
# area1_pointB = (590, 350)
# area1_pointC = (560, 380)
# area1_pointD = (585, 380)

# 3
# area1_pointA = (595, 350)
# area1_pointB = (620, 350)
# area1_pointC = (590, 370)
# area1_pointD = (620, 370)

# 4
# area1_pointA = (620, 350)
# area1_pointB = (648, 350)
# area1_pointC = (615, 370)
# area1_pointD = (648, 370)

# 5
# area1_pointA = (648, 350)
# area1_pointB = (670, 350)
# area1_pointC = (643, 370)
# area1_pointD = (670, 370)

# 6
# area1_pointA = (376, 430)
# area1_pointB = (372, 444)
# area1_pointC = (366, 430)
# area1_pointD = (362, 444)
#
# Line drawn from (388, 441) to (386, 460)
# Line drawn from (360, 438) to (360, 463)
# area6_pointA = (422, 407)
# area6_pointB = (415, 438)
# area6_pointC = (391, 407)
# area6_pointD = (382, 437)

# 6-1
# 6
area6_pointA = (422, 407)
area6_pointB = (415, 438)
area6_pointC = (391, 407)
area6_pointD = (382, 437)

# 7
area7_pointA = (412, 439)
area7_pointB = (405, 450)
area7_pointC = (381, 439)
area7_pointD = (372, 450)

# 8
area8_pointA = (408, 450)
area8_pointB = (400, 460)
area8_pointC = (376, 450)
area8_pointD = (368, 460)


# area1_pointA = (318, 454)
# area1_pointB = (315, 464)
# area1_pointC = (298, 456)
# area1_pointD = (297, 464)

# area1_pointC = (316, 453)
# area1_pointD = (314, 465)

# Line drawn from (318, 454) to (315, 464)
# Line drawn from (298, 456) to (297, 464)
# Line drawn from (318, 442) to (316, 453)
# Line drawn from (317, 455) to (314, 465)

# Line drawn from (376, 427) to (366, 446)
# Line drawn from (370, 446) to (364, 460)
# Line drawn from (362, 468) to (356, 481)
# Line drawn from (354, 487) to (354, 498)
# Line drawn from (349, 501) to (339, 516)
# Line drawn from (335, 526) to (331, 532)
cv2.line(image, area6_pointA, area6_pointB, (0, 255, 0), 2)
cv2.line(image, area6_pointC, area6_pointD, (0, 255, 0), 2)
cv2.line(image, area7_pointA, area7_pointB, (255, 255, 0), 2)
cv2.line(image, area7_pointC, area7_pointD, (255, 255, 0), 2)
cv2.line(image, area8_pointA, area8_pointB, (0, 255, 255), 2)
cv2.line(image, area8_pointC, area8_pointD, (0, 255, 255), 2)


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

