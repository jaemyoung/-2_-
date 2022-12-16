# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:35:13 2022

@author: user
"""

import numpy as np
import pandas as pd


#%% 데이터 불러오기
data = pd.read_excel("data/플레도AI블록_아이정보포함_학습데이터_220628.xlsx",header= 1)
user = pd.read_excel("data/아이정보_데이터_220628.xlsx",header= 1)

data["아이 나이"] = data["아이 생년월일"].apply(lambda x : 2023 - int(str(x)[:4])) # 생년월일로 나이 계산 후 저장
data["아이 성별"] = data["아이 성별"].apply(lambda x : 1 if x == "MALE" else 0)

# 컨텐츠 별 data 저장

#한글
korean_data = data[(data["컨텐츠 분류1"] =="한글") &(data["문제풀이 소요시간"]< 60) & (data["정오답"] == "정답")].sort_values(by=["아이 고유 식별 값"])
korean_data["누적 문제풀이 소요시간"] = korean_data.groupby(by=["아이 고유 식별 값"])["문제풀이 소요시간"].apply(lambda x : x.expanding().mean()) # 누적 평균 문제풀이 소요시간
K_train= korean_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][:8000]
K_val= korean_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][8000:10000]
K_test= korean_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][10000:]

#데이터 저장
np.savetxt("korean/aibloc.data.train", K_train, delimiter=' ')
np.savetxt("korean/aibloc.data.val", K_val, delimiter=' ')
np.savetxt("korean/aibloc.data.test", K_test, delimiter=' ')

#수학
math_data = data[(data["컨텐츠 분류1"] =="수학") &(data["문제풀이 소요시간"]< 60) & (data["정오답"] == "정답")].sort_values(by=["아이 고유 식별 값"])
math_data["누적 문제풀이 소요시간"] = math_data.groupby(by=["아이 고유 식별 값"])["문제풀이 소요시간"].apply(lambda x : x.expanding().mean()) # 누적 평균 문제풀이 소요시간
m_train= math_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][:8000]
m_val= math_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][8000:10000]
m_test= math_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][10000:]

#데이터 저장
np.savetxt("math/aibloc.data.train", m_train, delimiter=' ')
np.savetxt("math/aibloc.data.val", m_val, delimiter=' ')
np.savetxt("math/aibloc.data.test", m_test, delimiter=' ')

#영어
english_data = data[(data["컨텐츠 분류1"] =="영어") &(data["문제풀이 소요시간"]< 60) & (data["정오답"] == "정답")].sort_values(by=["아이 고유 식별 값"])
english_data["누적 문제풀이 소요시간"] = english_data.groupby(by=["아이 고유 식별 값"])["문제풀이 소요시간"].apply(lambda x : x.expanding().mean()) # 누적 평균 문제풀이 소요시간
e_train= english_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][:8000]
e_val= english_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][8000:10000]
e_test= english_data[["아이 나이","아이 성별","문제풀이 소요시간","누적 문제풀이 소요시간"]][10000:]

#데이터 저장
np.savetxt("english/aibloc.data.train", e_train, delimiter=' ')
np.savetxt("english/aibloc.data.val", e_val, delimiter=' ')
np.savetxt("english/aibloc.data.test", e_test, delimiter=' ')
