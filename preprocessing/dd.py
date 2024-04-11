# import os
# import shutil
#
# # 이동할 파일들이 있는 최상위 디렉토리
# top_directory = "data/valid/raw"
#
# # 각 폴더를 반복하며 파일을 상위 디렉토리로 이동
# for root, dirs, files in os.walk(top_directory):
#     for file_name in files:
#         file_path = os.path.join(root, file_name)
#         # 상위 디렉토리로 이동할 파일 경로 설정
#         destination_path = os.path.join(top_directory, file_name)
#         # 파일을 상위 디렉토리로 이동
#         shutil.move(file_path, destination_path)


# import os
#
# # 해당 폴더 경로 설정
# folder_path = 'dddd'
#
# # 폴더 내의 모든 파일에 대해 반복
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)
#
#         # 파일 열기 및 내용 읽기
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#
#         # 'car' 문자열을 '0'으로 바꾸기
#         modified_lines = [line.replace('car', '0') for line in lines]
#
#         # 파일 열기 및 수정된 내용 쓰기
#         with open(file_path, 'w') as file:
#             file.writelines(modified_lines)



import zipfile
import os

# def unzip_files_in_folder(folder_path):
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.zip'):
#                 zip_file_path = os.path.join(root, file)
#                 extract_to_path = os.path.join(root, file[:-4])  # Remove '.zip' extension
#                 os.makedirs(extract_to_path, exist_ok=True)
#                 with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#                     zip_ref.extractall(extract_to_path)
#                 print(f"Extracted '{zip_file_path}' to '{extract_to_path}'")
#
# # 사용 예시
# folder_path = 'D:/교통문제 해결을 위한 CCTV 교통 영상(시내도로)/Training/교통안전(Segmantation)'
# unzip_files_in_folder(folder_path)

