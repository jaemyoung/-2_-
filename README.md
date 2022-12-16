# DeepLearning2_implementation
[Yang et al., 2021] Delving into deep imbalanced regression. ICML 구현

사용 데이터 : 플레도 AI 학습블록 사용 로그데이터
총 데이터 수 : 63,366개
Columns :  [아이 고유 식별값, 아이 생년월일, 아이 성별, 단계, 컨텐츠 분류1, 컨텐츠 분류2, 컨텐츠 분류3, 단계, 단계별 문제, 문제 세부 번호, 문제 고유 식별 값, 학습 시각, 문제풀이 소요시간, 문제 정답, 아이 블록 입력데이터, 정오답, 통계 메인 키, 향상 능력]
  
![image](https://user-images.githubusercontent.com/35715977/208093433-9377cc8d-2485-4fcc-8641-8410c6b87a20.png)

Goal
 “User 데이터를 이용하여 적정 문제풀이 시간 예측” -> 학습 능력 측정에 대한 정량적 지표로 활용이 가능
설명 변수 : [User 나이, User 성별, User가 이전에 해결한 누적 평균 문제풀이 소요시간(초)]
종속 변수 : [문제풀이 소요시간(초)]
적정 문제풀이시간 > User의 문제풀이 시간 -> Good!
적정 문제풀이시간 < User의 문제풀이 시간 -> Bad! 
![image](https://user-images.githubusercontent.com/35715977/208093470-7a9538fc-4a7f-45ae-b9a6-c176f70b39ce.png)

결과 해석
연속형 데이터의 불균형을 해결하여 예측 정확도를 개선
특히, Few된 데이터 분포에서 상당히 예측성능이 좋아지는 것을 볼 수 있음
설명변수에 새로운 User 데이터를 추가하여 사용 가능
![image](https://user-images.githubusercontent.com/35715977/208093406-fd055f0f-c673-4b4b-a034-e26b83cb0364.png)
