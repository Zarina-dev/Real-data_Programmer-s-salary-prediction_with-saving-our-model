# Real-data_Programmer-s-salary-prediction_with-saving-our-model
Real data_Programmer's salary prediction_with saving our model. Building model part of  WebApp 


## 추가: 23년 7월 21일
saved_steps.pkl 을 다시 사용하여 predict해봄

### 1. 모델 pickle했을때의 환경과 똑같은 환경 만듦
unpickle을 위해 ```python3.9```, ```scikit-learn==1.0``` 이 두개 준비

### 2. load 및 predict 코드
Note! ```.pd```보다 ```.pkl```로 저장하는게 better인 거 같음 (다른 저장된 ```.pd```을 다시 활용하려고 했는데 잘 안됐음)

```
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def unpickle_model(filename):
    with open(filename,'rb')as file:
        data = pickle.load(file)
        print('MODEL UNPICKLED: ', data, '\n')
    model = data['model']
    encoder_country = data['encoder_Country']
    encoder_education = data['encoder_education']

    return (model, encoder_country, encoder_education)


def encode_labels(encoder_Country, encoder_Education, country, educ_level, prof_year):
    X = np.array([[country, educ_level, prof_year]])

    X[:,0] = encoder_Country.transform(X[:,0])
    X[:,1] = encoder_Education.transform(X[:,1])
    X = X.astype('float')
    return X

def main():
    model, encoder_country, encoder_education = unpickle_model('saved_steps.pkl')
    print(encoder_country.classes_)
    print(encoder_education.classes_)



    # predict
    X = encode_labels(encoder_country, encoder_education, 'Spain', "Bachelor’s degree", 5) # "'"의 폰트가 다르면 못알아봄
    Y_pred = model.predict(X)
    print('input: ', X)
    print('output: ', Y_pred)

    '''
    따라서, 
    1. pickle했을때 encoder_country.classes_의 출력값과 같이 training시 labeled된 text들도 같이 저장됨
    2. 모델이 training시 안 본 label들이 있으면 Predict 할 때 'y contains previously unseen label' 에러가 뜸
        (그러면 나중에 새로운 label추가할 필요가 생길 경우는???)
    3. 모델 저장했을때의 환경만 맞으면 언제든지 다시 활용가능

    '''

if __name__ =="__main__":
    main()
```
### 3. 콘솔창 출력
```
MODEL UNPICKLED:  {'model': DecisionTreeRegressor(max_depth=10, random_state=0), 'encoder_Country': LabelEncoder(), 'encoder_education': LabelEncoder()} 

['Australia' 'Brazil' 'Canada' 'France' 'Germany' 'India' 'Israel' 'Italy'
 'Netherlands' 'Norway' 'Poland' 'Russian Federation' 'Spain' 'Sweden'
 'Switzerland' 'Turkey'
 'United Kingdom of Great Britain and Northern Ireland'
 'United States of America']

['Bachelor’s degree' 'Less than Bachelors' 'Master’s degree' 'Post grad']

input:  [[12.  0.  5.]]
output:  [37583.36754177]
```

