# -*- coding: utf-8 -*-
# import os
#
# # 최상위 폴더 경로
# base_folder = 'D:/traffic_data/image/valid'
#
# # 하위 폴더 리스트
# sub_folders = ['[원천]LG전자 부천랜드점 부근', '[원천]갑천교네거리', '[원천]계룡대교네거리', '[원천]과학공원네거리', '[원천]국민은행 심곡점 앞', '[원천]김가네 상동점 앞 (홈플러스 진입로 앞)', '[원천]넘말사거리', '[원천]디아뜨갤러리 앞 (포도마을 맞은편)', '[원천]멀뫼사거리', '[원천]범계사거리', '[원천]범계역사거리', '[원천]범박터널-범박1교방향', '[원천]보강센트럴빌 2차 앞', '[원천]부천IC 자동차매매단지 앞', '[원천]부천남부역 공영주차장', '[원천]사단사거리', '[원천]삼정고가교', '[원천]소방서사거리', '[원천]소사역앞', '[원천]소사회주로삼거리', '[원천]소신여객 앞', '[원천]신한은행 중동점 부근', '[원천]아이쇼핑 앞', '[원천]역곡남부역사거리', '[원천]연구단지네거리', '[원천]오정물류단지(하행)', '[원천]용반네거리', '[원천]원골네거리', '[원천]원종 철골공영주차장 (마사회)', '[원천]웨스턴파크 (부천역전우체국 건너편)', '[원천]유성네거리', '[원천]중동IC하부(판교진출)', '[원천]중동사거리', '[원천]진터지하차도', '[원천]충대서문네거리', '[원천]충대오거리', '[원천]현대블루핸즈 소사점 부근']
#
# # 각 하위 폴더에 대해 반복
# for sub_folder in sub_folders:
#     image_folder = os.path.join(base_folder, sub_folder)
#
#     # 폴더 내의 모든 파일에 대해 반복
#     for root, dirs, files in os.walk(image_folder):
#         for filename in files:
#             if filename.endswith('.jpg'):
#                 # 폴더 이름 가져오기
#                 folder_name = os.path.basename(root)
#
#                 # 새 파일 이름 설정 (폴더 이름 + 원래 파일 이름)
#                 new_filename = folder_name + '_' + filename
#                 new_filepath = os.path.join(root, new_filename)
#
#                 # 원래 파일 경로
#                 original_filepath = os.path.join(root, filename)
#
#                 # 파일 이름 변경
#                 os.rename(original_filepath, new_filepath)

#
# import os
# import shutil
#
# # 원본 폴더 경로
# src_folder = 'D:/traffic_data/image/valid'
#
# # 대상 폴더 경로
# dst_folder = 'D:/traffic_data/image/valid'
#
# # 원본 폴더 및 하위 폴더 내의 모든 파일에 대해 반복
# for root, dirs, files in os.walk(src_folder):
#     for filename in files:
#         if filename.endswith('.jpg'):
#             # 원본 파일 경로
#             src_filepath = os.path.join(root, filename)
#
#             # 대상 파일 경로
#             dst_filepath = os.path.join(dst_folder, filename)
#
#             # 파일 이동
#             shutil.move(src_filepath, dst_filepath)


# import os
#
# # 대상 폴더 경로
# folder_path = 'D:/traffic_data/label/Validation'
#
# # 폴더 내의 모든 파일에 대해 반복
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.endswith('.txt'):
#             # 파일 경로
#             file_path = os.path.join(root, filename)
#
#             # 파일 읽기
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
#
#             # 각 줄의 첫 번째 숫자(레이블)에서 1 빼기
#             for i in range(len(lines)):
#                 line = lines[i].split()
#                 line[0] = str(int(line[0]) - 1)
#                 lines[i] = ' '.join(line) + '\n'
#
#             # 결과를 동일한 파일에 다시 쓰기
#             with open(file_path, 'w') as file:
#                 file.writelines(lines)

import os

# 폴더 경로
folder1 = 'D:/traffic_data/label/valid'
folder2 = 'D:/traffic_data/image/valid'

# 각 폴더의 파일 목록 가져오기
files_in_folder1 = set(file.split(".")[0] for file in os.listdir(folder1))
files_in_folder2 = set(file.split(".")[0] for file in os.listdir(folder2))

# 두 폴더의 파일 목록 비교
files_only_in_folder1 = files_in_folder1 - files_in_folder2
files_only_in_folder2 = files_in_folder2 - files_in_folder1

# 결과 출력
print("Files only in folder 1:", files_only_in_folder1)
print("Files only in folder 2:", files_only_in_folder2)