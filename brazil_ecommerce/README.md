# 배송 시간 예측 워크샵
이 워크샵은 아래와 같은 모회사의 배송 예측에 대한 정보 입니다. "Kaggle의 Brazilian E-Commerce Public Dataset by Olist" 를 가지고 이 문제를 풀어 보겠습니다.

![naver_delivery_prediction](img/naver_delivery_prediction.png)

## 데이터 정보
- Brazilian E-Commerce Public Dataset by Olist
    - https://www.kaggle.com/olistbr/brazilian-ecommerce

## 문제 접근 방법
- 데이터가 여러개 항목의 CSV 로 구성 되어 있습니다.
- CSV에서 주문 확정 시간, 배송 도착 시간의 차이를 계산하여 레이블 데이터를 계산 합니다. 
    - 이 배송시간을 5개의 구간으로 나누어서 분류 문제로 만듧니다.
- CSV에 위의 레이블을 예측하기 위한 데이터 컬럼 값들을 조인 및 추출 합니다.
- 추출된 데이터 컬럼 값을 탐색하여, 어떤 컬럼 값이 레이블의 영향을 주는지를 확인 합니다.
- 피쳐 엔지니어링을 통하여, 새로운 피쳐를 생성 합니다.
- 최종 피쳐를 통하여 아래 세가지 알고리즘을 갖고 훈련 및 모델 평가를 합니다.
    - CatBoost
    - XGBoost
    - Amazon AutoGluon


## 피쳐 엔지니얼링 및 알고리즘 평가 
아래는 피쳐 엔지니어링의 및 세가지의 알고리즘 적용 효과를 요약 했습니다. 가장 하단 부터 보시면 시간 순의 실험한 것들을 볼 수 있고, 가장 최상단은 마지막 실험 결과를 의미 합니다.


### AutoGluon with target encoding with Smoothing, product_id/classes의 mean_smoothed, error_smoothed 피쳐 추가
```
accuracy: 0.92%
f1_score: 0.92%
```
![autogluon-target-en-smooth-conf](img/autogluon-target-en-smooth-conf.png)

![autogluon-target-en-smooth-fe-imp](img/autogluon-target-en-smooth-fe-imp.png)

![autogluon-target-en-smooth-leaderboard](img/autogluon-target-en-smooth-leaderboard.png)


### XGBoost with target encoding with Smoothing, product_id/classes의 mean_smoothed, error_smoothed 피쳐 추가
```
accuracy: 0.77%
f1_score: 0.77%
```
![xgboost-target-en-smooth.png](img/xgboost-target-en-smooth.png)



### CatBoost with target encoding with Smoothing, product_id/classes의 mean_smoothed, error_smoothed 피쳐 추가
```
accuracy: 0.90%
f1_score: 0.89%
```
![catboost-target-en-smoothe-w20](img/catboost-target-en-smoothe-w20.png)



### CatBoost with target encoding, product_id/classes의 mean, count, error 피쳐 추가
```
accuracy: 0.95%
f1_score: 0.95%
```
![catboost-target-enconf](img/catboost-target-en-conf.png)


![catboost-target-en-feimp](img/catboost-target-en-fe-imp.png)



### CatBoost with new features (Month, Day, WeekOfDay)
```
accuracy: 0.46%
f1_score: 0.44%
```
![catboost-date-feature](img/catboost-date-feature.png)



### CatBoost with new features, customer_seller_state, custom_seller_city, custom_seller_zipcode
```
accuracy: 0.39%
f1_score: 0.32%
```
![catboost-all-cate](img/catboost-all-cate.png)
![catboost-target-en-fe-imp](img/catboost-target-en-fe-imp..png)


### AutoGluon with new features, customer_seller_state, custom_seller_city, custom_seller_zipcode
```
accuracy: 0.40%
f1_score: 0.29%
```
![autogluon-all-cate](img/autogluon-all-cate.png)

### XGBoost with new features, customer_seller_state, custom_seller_city, custom_seller_zipcode
```
accuracy: 0.39%
f1_score: 0.34%
```




### AutoGluon with new feature, customer_seller_state
    - cate_cols = ['customer_state','product_category_name_english','seller_state',customer_seller_state]
```
accuracy: 0.37%
f1_score: 0.30%
```
![alutogluon-4-cate](img/alutogluon-4-cate.png)


### CatBoost with new feature, customer_seller_state
    - cate_cols = ['customer_state','product_category_name_english','seller_state',customer_seller_state]
```
accuracy: 0.39%
f1_score: 0.28%
```



### XGBoost with new feature, customer_seller_state
    - cate_cols = ['customer_state','product_category_name_english','seller_state',customer_seller_state]
```
accuracy: 0.38%
f1_score: 0.31%
```
![xgboost-4-cate](img/xgboost-4-cate.png)



### CatBoost
    - cate_cols = ['customer_state','product_category_name_english','seller_state']
    - Encoding
        - Label Encoding    

```
accuracy: 0.38%
f1_score: 0.31%
```
![xgboost-4-coate](img/catboost-3-cate.png)


### AutoGluon with medium_quality_faster_train
    - cate_cols = ['customer_state','product_category_name_english','seller_state']
    - Encoding
        - No Encoding    

```
accuracy: 0.38%
f1_score: 0.27%
```
![autogluon-medium-quality-3-cate](img/autogluon-medium-quality-3-cate.png)

### XGBoost with Label-Encoding: 
    - cate_cols = ['customer_state','product_category_name_english','seller_state']
    - Encoding
        - Label-Encoding    
```
accuracy: 0.37%
f1_score: 0.29%
```
![xgboost-label-en-3-cate](img/xgboost-label-en-3-cate.png)

### XGBoost with One-Hot-Encoding: 
    - cate_cols = ['customer_state','product_category_name_english','seller_state']
    - Encoding
        - One-Hot-Encoding    
```
accuracy: 0.38%
f1_score: 0.31%
```

# Reference:

- AutoGluon Tabular Prediction
    - 오토글루온 Tabular 공식 페이지
    - https://autogluon.mxnet.io/stable/tutorials/tabular_prediction/index.html
- SageMaker XGBoost Algorithm
    - SageMaker 내장 알고리즘 설명
    - https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
- XGBoost for Multi-class Classification
    - 블로그: SK Learn XGBoost 알고리즘을 end-to-end 설명
    - https://towardsdatascience.com/xgboost-for-multi-class-classification-799d96bcd368
    - Git Repo
        - https://github.com/ernestng11/touchpoint-prediction/blob/master/model-building.ipynb
- XGBoost Parameters
    - SK Learn XGBoost 파라미터
    - https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
- CatBoost vs. Light GBM vs. XGBoost
    - CatBoost, Light GBM, XGBoost 를 비행기 지연 예측을 통해서 비교
    - https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
- Feature Engineering
    - RecSys 2020 Tutorial: Feature Engineering for Recommender Systems
        - https://www.youtube.com/watch?v=uROvhp7cj6Q
    - Git Repo
        - https://github.com/rapidsai/deeplearning/tree/main/RecSys2020Tutorial