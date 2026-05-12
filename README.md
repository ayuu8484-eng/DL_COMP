# IMBK8기_DL_COMPETITION
## 신용점수분류 DL 모델 구축

- 기간: 2026년 05월 12일
- 데이터 출처: 데이터 클리닝한 Credit score classification 캐글 데이터 (ROW: 100000, COL: 28)

### 1. 데이터 전처리
- 분석 효율을 위해 고유식별값 `ID`, `Customer_ID`, `SSN` 컬럼 제거
- 시간순을 나타내는 `month` 컬럼 제거
- 다중공산성 문제를 고려하기 위해 수치형 컬럼간의 상관계수 확인
<img width="947" height="853" alt="image" src="https://github.com/user-attachments/assets/f5fb54ef-40a3-456e-a0b7-5f03c826ca2b" />
  - 상관계수 높은 컬럼: 
  'Monthly_Inhand_Salary' - 'Annual_Income'
  'Amount_invested_monthly' - (Annual_Income, Monthly_Inhand_Salary)


### 2. EDA
### 3. 모델링 방법
### 4. 성능 결과
