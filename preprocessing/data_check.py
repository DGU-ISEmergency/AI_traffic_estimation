# 폴더 unzip하기

# import os
# import zipfile
#
#
# def unzip_folders_in_directory(directory):
#     # 디렉토리 내의 모든 항목 가져오기
#     items = os.listdir(directory)
#
#     # 각 항목에 대해 반복
#     for item in items:
#         item_path = os.path.join(directory, item)
#
#         # zip 파일인지 확인하고 unzip
#         if item.endswith('.zip'):
#             unzip_folder = os.path.splitext(item_path)[0]
#
#             # unzip 폴더 생성
#             os.makedirs(unzip_folder, exist_ok=True)
#
#             # zip 파일 압축 해제
#             with zipfile.ZipFile(item_path, 'r') as zip_ref:
#                 zip_ref.extractall(unzip_folder)
#
#             print(f"{item_path}를 압축 해제하여 {unzip_folder}에 저장했습니다.")
#
#
# # unzip할 폴더의 경로 설정
# folder_path = 'D:/교통 image/Validation/교통안전(Bbox)/'
#
#
# # unzip 폴더 실행
# unzip_folders_in_directory(folder_path)\

