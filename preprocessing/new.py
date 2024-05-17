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
# json_folder = 'D:/교통/label/Training/LG전자 부천랜드점 부근/BC2000201'
#
# # txt 파일 저장 경로
# output_folder = 'D:/교통/label/Training/LG전자 부천랜드점 부근/BC2000201_txt'
#
# with open('D:/교통/label/Training/LG전자 부천랜드점 부근/BC2000201/LG전자 부천랜드점 부근_BC2000201.json', encoding="utf-8") as w:
#     data = json.load(w)
#     print(data)
#
# file_name = data["images"][0]["file_name"]
# # 'BC2000201/20201017_140001_S_23166.jpg'
# bbox = data['annotations'][0]['bbox']
# # [[1058.20068359375, 355.29833984374994, 1110.9539794921875, 399.44107055664057], [1503.923217773437, 458.4909667968753, 1626.4052305008659, 543.5307013286157], [1205.2662697260423, 298.5004924052246, 1240.8599901833934, 322.17485087627523], [626.1760253906252, 533.053955078125, 660.8404541015627, 625.0039672851562], [1194.117385601163, 465.47065134621533, 1295.1500994468242, 564.2287777111485], [939.7453002929686, 334.36456298828125, 987.3761116635403, 372.9399719238281], [977.5584106445307, 278.4782409667963, 1000.3755054684898, 297.1411743164057], [1045.9282007536917, 312.15728759765574, 1091.4313795755834, 348.5766234722775], [702.5313414564639, 810.2594604492189, 989.987032243505, 1085.7707485305928], [292.9383850097656, 918.8872680664064, 374.1304626464844, 1078.80712890625], [1155.9615159623659, 823.6261682242983, 1478.837808436056, 1079.9999999999986], [32.30283911671924, 948.785046728972, 135.26813880126184, 1077.981308411215], [1397.0977917981056, 793.002888444752, 1684.1284349705488, 1084.037383177567], [1149.0277957224757, 257.95153128447737, 1201.1291719687765, 295.16245245371033], [1048.3317397103904, 279.6546570940669, 1081.2503739826122, 315.9511020982982], [1059.8307146958923, 260.94282519747617, 1090.945588186075, 290.4759574679998], [623.1905696618896, 476.3644519682268, 660.8891535877858, 553.6858466055938], [1129.8280233515104, 273.1547635345086, 1150.2348709591154, 289.95843946516317], [1113.3224848453594, 261.1521378697553, 1128.3275198509514, 275.855354309078], [1093.8159393380902, 262.9525317194683, 1106.7202694428993, 275.2552230258404], [1130.428224751734, 259.0516783784235, 1145.433259757326, 273.4548291761274], [1366.4847345534029, 343.0556548808987, 1376.0879569569815, 370.36162826821237], [1288.7586532244375, 311.8488281525402, 1297.7616742277926, 327.7523071583383], [1558.5137589784172, 366.92746059817256, 1570.7947976650453, 403.76627255571316], [1663.3122283764508, 451.76121678120177, 1683.1281506964724, 505.40293431087525]]
# category_id = data['annotations'][0]['category_id']
# # [1, 1, 1, 7, 1, 1, 1, 1, 1, 7, 1, 7, 1, 3, 4, 4, 7, 1, 1, 1, 1, 7, 7, 7, 7]

import json
import os

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

# 최상위 폴더 경로
base_folder = 'D:\교통\label\Validation'

# 하위 폴더 리스트
sub_folders = ['LG전자 부천랜드점 부근', '과학공원네거리', '국민은행 심곡점 앞', '김가네 상동점 앞 (홈플러스 진입로 앞)', '넘말사거리', '디아뜨갤러리 앞 (포도마을 맞은편)', '멀뫼사거리', '범박터널-범박1교방향', '보강센트럴빌 2차 앞', '부천IC 자동차매매단지 앞', '부천남부역 공영주차장', '사단사거리', '삼정고가교', '소방서사거리', '소사역앞', '소사회주로삼거리', '소신여객 앞', '신한은행 중동점 부근', '아이쇼핑 앞', '역곡남부역사거리', '연구단지네거리', '오정물류단지(하행)', '용반네거리', '원종 철골공영주차장 (마사회)', '웨스턴파크 (부천역전우체국 건너편)', '중동IC하부(판교진출)', '중동사거리', '진터지하차도', '충대서문네거리', '현대블루핸즈 소사점 부근']

# 각 하위 폴더에 대해 반복
for sub_folder in sub_folders:
    json_folder = os.path.join(base_folder, sub_folder)

    # JSON 폴더 및 하위 폴더 내의 모든 파일에 대해 반복
    for root, dirs, files in os.walk(json_folder):
        for filename in files:
            if filename.endswith('.json'):
                # JSON 파일 읽기
                with open(os.path.join(root, filename), 'r',encoding='utf-8') as file:
                    data = json.load(file)

                # 이미지의 너비와 높이
                image_width = 1920
                image_height = 1080

                # 각 이미지에 대해 반복
                for j in range(len(data["images"])):
                    # YOLO 형식의 bounding box를 저장할 리스트 초기화
                    yolo_format_bboxes = []

                    # 각 bounding box에 대해
                    for i in range(len(data['annotations'][j]['bbox'])):
                        # bounding box의 좌표 가져오기
                        xmin, ymin, xmax, ymax = data['annotations'][j]['bbox'][i]

                        # bounding box의 중심, 너비, 높이 계산
                        x_center = (xmin + xmax) / 2
                        y_center = (ymin + ymax) / 2
                        bbox_width = xmax - xmin
                        bbox_height = ymax - ymin

                        # 이미지의 너비와 높이로 bounding box의 차원 정규화
                        x_center /= image_width
                        y_center /= image_height
                        bbox_width /= image_width
                        bbox_height /= image_height

                        # 카테고리 ID 가져오기
                        class_id = data['annotations'][j]['category_id'][i]

                        # YOLO 형식의 bounding box를 생성하고 리스트에 추가
                        yolo_format_bboxes.append([class_id, x_center, y_center, bbox_width, bbox_height])

                    # 출력 파일 이름 설정
                    output_filename = data["images"][j]["file_name"].replace('/', '_') + '.txt'
                    output_path = os.path.join(json_folder, output_filename)

                    # YOLO 형식의 bounding box를 txt 파일에 쓰기
                    with open(output_path, 'w') as output_file:
                        for yolo_bbox in yolo_format_bboxes:
                            output_file.write(" ".join(map(str, yolo_bbox)) + "\n")

# JSON 폴더 내의 모든 파일에 대해 반복
# for filename in os.listdir(json_folder):
#     if filename.endswith('.json'):
#         # JSON 파일 읽기
#         with open(os.path.join(json_folder, filename), encoding="utf-8") as file:
#             data = json.load(file)
#
#         # 출력 파일 이름 설정
#         output_filename = os.path.splitext(filename)[0] + '.txt'
#         output_path = os.path.join(output_folder, output_filename)
#
#
#         with open(output_path, 'w') as output_file:
#             for i in range(len(data['row'])):
#                 if data['row'][i]['attributes1'] in ["일반차량", "목적차량(특장차)", "이륜차"]:
#                     points1 = data['row'][i]['points1']
#                     points3 = data['row'][i]['points3']
#                     height = data['row'][i]['height']
#                     width = data['row'][i]['width']
#                     xmin, ymin, xmax, ymax = int(points1.split(",")[0]), int(points1.split(",")[1]), \
#                                              int(points3.split(",")[0]), int(points3.split(",")[1])
#                     bbox = get_object_params(int(width), int(height), xmin, ymin, xmax, ymax)
#                     output_file.write("car\t")
#                     output_file.write(str(bbox[0]))
#                     output_file.write("\t")
#                     output_file.write(str(bbox[1]))
#                     output_file.write("\t")
#                     output_file.write(str(bbox[2]))
#                     output_file.write("\t")
#                     output_file.write(str(bbox[3]))
#                     output_file.write("\n")
#
