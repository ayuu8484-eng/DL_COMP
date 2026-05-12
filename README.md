# IMBK8기_DL_COMPETITION
## 신용점수분류 DL 모델 구축


- 기간: 2026년 05월 12일
- 데이터 출처: 데이터 클리닝한 Credit score classification 캐글 데이터 (ROW: 100000, COL: 28)


### 1. 데이터 전처리

- 분석 효율을 위해 고유식별값 'ID', 'Customer_ID', 'SSN' 컬럼 제거
- 시간순을 나타내는 'month' 컬럼 제거
- 다중공산성 문제를 고려하기 위해 수치형 컬럼간의 상관계수 확인
  <img width="947" height="853" alt="image" src="https://github.com/user-attachments/assets/f5fb54ef-40a3-456e-a0b7-5f03c826ca2b" />
  - 상관계수 높은 컬럼(0.7이상): 
    - 'Monthly_Inhand_Salary' - 'Annual_Income'
    - 'Amount_invested_monthly' - (Annual_Income, Monthly_Inhand_Salary)
  - 이 중 'Annual_Income', 'Amount_invested_monthly'컬럼 제거하기로 결정
- 실제 데이터를 직접 반영하기 위해 이상치 처리는 따로 하지 않았음



### 2. EDA

1) 신용점수 분포 확인

  <img width="713" height="470" alt="image" src="https://github.com/user-attachments/assets/92458ae3-15d1-4db3-aea0-7a2cf113d547" />
  - Standard가 가장 높은 분포를 가지고 있고 다음은 Poor, Good 순으로 분포


2) 미지급 잔액과 신용카드 이자율에 따른 신용점수 분포

  <img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/887cbc83-9350-4a2b-ab57-4f5963dba5d9" />
  -  신용점수가 낮을수록 미지급 금액과 신용카드 이자율의 중앙값이 높아지는 것으로 확인
  - 미지급 금액과, 신용카드 이자율이 신용등급에 영향을 주는 요인일 것으로 추측

 
3) 고객납부행태별 신용등급 분포

  <img width="1037" height="728" alt="image" src="https://github.com/user-attachments/assets/7bb92186-96a9-4106-a97e-94772e7a1859" />
  - 세가지 등급 모두 '저지출, 소액결제' 그룹의 인원 수가 가장 많은 것으로 확인
  - 이를 봤을 때 소비 방식에서 신용등급을 나뉘어지는데에 거리가 멀어보이는 것으로 확인


### 3. 모델링 방법

1) 모델 선정 (TabTransformer)
- 수치형 + 범주형 데이터이므로 범주형 변수간의 복잡도를 잘 반영하고 분류모델의 최고 성능을 보이는 TabTransformer를 선정
- 100000개의 데이터이기 때문에 batch_size를 설정해 데이터셋을 나눠 학습


2) 범주형 컬럼, 수치형 컬럼 나누기
- 범주형 컬럼 : ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
- 수치형 컬럼 : ['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month', 'Monthly_Balance']


3) 데이터 분할
- test_size = 0.2
- 클래스 불균형 방지 => stratify = y


4) 연속형 변수 스케일링
- StandardScaler


5) 범주형, 수치형 나눠서 텐서 변형

```
# 범주형에 'Credit_Score' 빼기
obj_cols_ny = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']


# 텐서 변형
X_train_obj = torch.tensor(X_train[obj_cols_ny].values, dtype=torch.int64)
X_train_num = torch.tensor(X_train[num_cols].values, dtype=torch.float32)
X_test_obj = torch.tensor(X_test[obj_cols_ny].values, dtype=torch.int64)
X_test_num = torch.tensor(X_test[num_cols].values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
```


6) 100,000개의 데이터이므로 batch_size 설정

- batch_size = 256


7) 모델 설정

```
model = TabTransformer(
    categories=(15, 3, 3, 6),
    num_continuous=X_train_num.shape[1],
    dim=32,
    dim_out=3,
    depth=8,
    heads=6,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
).to(device)
```


8) 손실 함수 및 최적화 도구
- CrossEntropyLoss
- Adam: 학습률 0.001


9) epochs 50으로 반복학습


### 4. 성능 결과

1) Accuracy
<img width="444" height="301" alt="image" src="https://github.com/user-attachments/assets/dc141316-2307-4f46-9b5f-c62bb9699907" />


2) Confusion Matrix
<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/51da0e05-f10e-4283-bb33-58dde45b3e14" />
