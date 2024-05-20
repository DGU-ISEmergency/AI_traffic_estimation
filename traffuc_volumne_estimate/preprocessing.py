import os
import pandas as pd

# csv 파일이 있는 디렉토리 경로
directory = "counting_data/row"

# DataFrame을 저장할 리스트
df_list = []

# 디렉토리 내의 모든 파일을 순회
for filename in os.listdir(directory):
    # 파일이 csv 확장자를 가지면
    if filename.endswith(".csv"):
        # 파일의 전체 경로
        file_path = os.path.join(directory, filename)

        # pandas를 사용하여 csv 파일 읽기
        df = pd.read_csv(file_path)

        # DataFrame을 리스트에 추가
        df_list.append(df)

# 모든 DataFrame을 하나로 합치기
merged_df = pd.concat(df_list)

# 합쳐진 DataFrame을 새로운 csv 파일로 저장
merged_df.to_csv("merge_row.csv", index=False)