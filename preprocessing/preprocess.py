# import json
# import csv
#
#
# def get_object_params(i_width: int, i_height: int, xmin, ymin, xmax, ymax):
#     image_width = 1.0 * i_width
#     image_height = 1.0 * i_height
#
#     center_x = xmin + 0.5 * (xmax - xmin)
#     center_y = ymin + 0.5 * (ymax - ymin)
#
#     absolute_width = xmax - xmin
#     absolute_height = ymax - ymin
#
#     l_x = center_x / image_width
#     l_y = center_y / image_height
#     l_width = absolute_width / image_width
#     l_height = absolute_height / image_height
#
#     return l_x, l_y, l_width, l_height
#
# with open('test/가평오거리(가평)_20210422102825_0000016.jpg.json',encoding="utf-8") as w:
#     data = json.load(w)
#
# filename = "test/test.txt"
# w = open(filename,'w')
#
# for i in range(len(data['row'])):
#     if data['row'][i]['attributes1'] in ["일반차량","목적차량(특장차)","이륜차"]:
#         points1 = data['row'][i]['points1']
#         points3 = data['row'][i]['points3']
#         height = data['row'][i]['height']
#         width = data['row'][i]['width']
#         xmin, ymin, xmax, ymax = int(points1.split(",")[0]), int(points1.split(",")[1]),  int(points3.split(",")[0]), int(points3.split(",")[1])
#         bbox = get_object_params(int(width), int(height),xmin, ymin, xmax, ymax)
#         w.write("car\t")
#         w.write(str(bbox[0]))
#         w.write("\t")
#         w.write(str(bbox[1]))
#         w.write("\t")
#         w.write(str(bbox[2]))
#         w.write("\t")
#         w.write(str(bbox[3]))
#         w.write("\n")



import os
import json


def get_object_params(i_width: int, i_height: int, xmin, ymin, xmax, ymax):
    image_width = 1.0 * i_width
    image_height = 1.0 * i_height

    center_x = xmin + 0.5 * (xmax - xmin)
    center_y = ymin + 0.5 * (ymax - ymin)

    absolute_width = xmax - xmin
    absolute_height = ymax - ymin

    l_x = center_x / image_width
    l_y = center_y / image_height
    l_width = absolute_width / image_width
    l_height = absolute_height / image_height

    return l_x, l_y, l_width, l_height


# JSON 파일이 있는 폴더 경로
json_folder = 'data/valid/label'

# txt 파일 저장 경로
output_folder = 'data/valid/processing_label'

# JSON 폴더 내의 모든 파일에 대해 반복
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        # JSON 파일 읽기
        with open(os.path.join(json_folder, filename), encoding="utf-8") as file:
            data = json.load(file)

        # 출력 파일 이름 설정
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(output_folder, output_filename)


        with open(output_path, 'w') as output_file:
            for i in range(len(data['row'])):
                if data['row'][i]['attributes1'] in ["일반차량", "목적차량(특장차)", "이륜차"]:
                    points1 = data['row'][i]['points1']
                    points3 = data['row'][i]['points3']
                    height = data['row'][i]['height']
                    width = data['row'][i]['width']
                    xmin, ymin, xmax, ymax = int(points1.split(",")[0]), int(points1.split(",")[1]), \
                                             int(points3.split(",")[0]), int(points3.split(",")[1])
                    bbox = get_object_params(int(width), int(height), xmin, ymin, xmax, ymax)
                    output_file.write("car\t")
                    output_file.write(str(bbox[0]))
                    output_file.write("\t")
                    output_file.write(str(bbox[1]))
                    output_file.write("\t")
                    output_file.write(str(bbox[2]))
                    output_file.write("\t")
                    output_file.write(str(bbox[3]))
                    output_file.write("\n")

