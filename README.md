<details>
<summary>ENG (English Version)</summary>

# Week 04 — Machine Learning Basics: Classification and Clustering

Classification and clustering of the Iris dataset with scikit-learn. Model evaluation through a confusion matrix and per-class metrics, followed by a comparison of k-NN and Random Forest on the same test set.

## 1. Classification and Clustering

| Category | Classification | Clustering |
| --- | --- | --- |
| Learning type | Supervised learning | Unsupervised learning |
| Training labels | Required | Unused |
| Objective | Prediction of predefined classes | Discovery of groups of similar samples |
| Example | Prediction of Iris species | Grouping of Iris samples by flower measurements |
| Algorithms | k-NN, SVM, Decision Tree, Random Forest | k-means, DBSCAN |

---

## 2. k-Nearest Neighbors (k-NN)

- **Principle** — Selection of the `k` closest training samples and majority voting under uniform weighting.
- **Advantages** — Intuitive prediction based on nearby examples and no iterative parameter optimization.
- **Limitations** — Storage of training samples and neighbor search at prediction time.
- **Choice of `k`** — Greater sensitivity to noise with small values and weaker local distinctions with large values.
- **Feature scaling** — Standardization to reduce the influence of differences in feature scales on distances.

### Dog Breed Classification Example

Classification of Dachshunds and Samoyeds using body length and height. Example label encoding of `0` for Dachshund and `1` for Samoyed, with `k=3` for majority voting among three neighbors. Prediction dependent on the measurements and labels in the training set.

---

## 3. Iris Classification with k-NN

### Practice Setup

```bash
python -m pip install scikit-learn pandas
```

- **Dataset** — 150 samples across three species: setosa, versicolor, and virginica.
- **Features** — Sepal length, sepal width, petal length, and petal width.
- **Data split** — 70% for training and 30% for testing, with stratification by species.
- **Pipeline** — Standardization fitted on training data and applied to test data before prediction.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42,
    stratify=iris.target,
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3),
)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred))
```

---

## 4. Classification Performance Evaluation

| Metric | Meaning |
| --- | --- |
| Accuracy | Proportion of correct predictions among all samples |
| Precision | Proportion of actual positives among predicted positives |
| Recall | Proportion of correctly identified positives among actual positives |
| F1 score | Harmonic mean of precision and recall |
| Support | Number of actual samples in each class |

- **Confusion matrix** — Actual classes in rows, predicted classes in columns, and correct predictions on the diagonal.
- **Class imbalance** — Inspection of per-class metrics alongside overall accuracy to detect errors on minority classes.
- **Macro and weighted averages** — Equal weight per class for the macro average and weighting by support for the weighted average.

Continuation of the Iris classification example from Section 3.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)
```

---

## 5. Ensemble Learning and Random Forest

Combination of predictions from multiple estimators. Diversity through different training samples, feature subsets, or learning algorithms.

| Method | Main idea |
| --- | --- |
| Bagging | Training on samples drawn with replacement and aggregation of predictions |
| Pasting | Training on samples drawn without replacement and aggregation of predictions |
| Boosting | Sequential addition of models to address errors or residuals from earlier stages |
| Random Forest | Aggregation of decision trees trained on bootstrap samples, with a random subset of features considered at each split |

### Iris Classification with Random Forest

Reuse of the training and test splits from Section 3 for comparison with k-NN under the same split.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)
print("Random forest accuracy:", accuracy_score(y_test, forest_pred))
```

---

## 6. Iris Clustering with k-Means

- **Principle** — Alternation between assignment to the nearest centroid and recalculation of centroids from assigned samples.
- **Objective** — Reduction of the sum of squared distances between samples and their assigned centroids.
- **Configuration** — Three clusters for the exercise and ten centroid initializations.
- **Limitations** — Sensitivity to feature scaling, outliers, and centroid initialization.
- **Interpretation** — Cluster IDs as identifiers for discovered groups. Species labels excluded from model inputs.

Standalone example using only the four Iris measurements.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

cluster_model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, n_init=10, random_state=42),
)
clusters = cluster_model.fit_predict(iris.data)

result = pd.DataFrame(iris.data, columns=iris.feature_names)
result["cluster"] = clusters

print(result.head())
print(result["cluster"].value_counts().sort_index())
```

---

## 7. k-NN and k-Means Comparison

| Category | k-NN Classification | k-Means |
| --- | --- | --- |
| Learning type | Supervised learning | Unsupervised learning |
| Meaning of `k` | Number of neighbors used for prediction | Number of clusters |
| Decision rule | Vote among the nearest labeled samples | Assignment to the nearest centroid |
| Output | Predicted class | Assigned cluster ID |

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# Week 04 — 머신러닝 기초와 분류·군집화

scikit-learn을 활용한 붓꽃(Iris) 데이터 분류 및 군집화 실습. 혼동행렬과 클래스별 지표를 활용한 성능 평가 및 동일한 테스트 데이터에서의 k-NN과 랜덤포레스트 비교.

## 1. 분류와 군집화

| 구분 | 분류(Classification) | 군집화(Clustering) |
| --- | --- | --- |
| 학습 방식 | 지도학습 | 비지도학습 |
| 학습용 정답 라벨 | 필요 | 사용하지 않음 |
| 목적 | 미리 정의된 클래스 예측 | 유사한 데이터 집단 탐색 |
| 예시 | 붓꽃 품종 예측 | 꽃의 측정값을 기준으로 붓꽃 데이터 그룹화 |
| 대표 알고리즘 | k-NN, SVM, 결정 트리, 랜덤포레스트 | k-means, DBSCAN |

---

## 2. k-최근접 이웃(k-NN)

- **동작 원리** — 가장 가까운 학습 데이터 `k`개를 선택한 뒤 동일 가중치 기준으로 다수결 분류.
- **장점** — 주변 사례에 기반한 직관적인 예측과 반복적인 파라미터 최적화가 필요 없는 구조.
- **한계** — 학습 데이터 보관에 필요한 메모리와 예측 시 이웃 탐색 비용.
- **`k` 값 선택** — 작은 값에서 잡음에 대한 민감도 증가, 큰 값에서 국소적 차이 감소.
- **특성 스케일링** — 특성별 값 범위 차이가 거리 계산에 미치는 영향을 줄이기 위한 표준화.

### 강아지 품종 분류 예제

몸 길이와 키를 활용한 닥스훈트와 사모예드 분류. 닥스훈트를 `0`, 사모예드를 `1`로 설정하고 `k=3`에서 가장 가까운 이웃 세 개의 라벨로 다수결 적용. 학습 데이터의 측정값과 라벨에 따른 예측 결과 변화.

---

## 3. k-NN을 활용한 붓꽃 분류

### 실습 환경 구성

```bash
python -m pip install scikit-learn pandas
```

- **데이터셋** — setosa, versicolor, virginica 세 품종의 표본 150개.
- **입력 특성** — 꽃받침 길이와 너비, 꽃잎 길이와 너비.
- **데이터 분할** — 학습 70%, 테스트 30% 분할 및 품종별 비율 유지.
- **파이프라인** — 학습 데이터에서 표준화 기준을 계산하고 같은 기준으로 테스트 데이터 변환 후 예측.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=42,
    stratify=iris.target,
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3),
)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred))
```

---

## 4. 분류 모델 성능 평가

| 지표 | 의미 |
| --- | --- |
| 정확도(Accuracy) | 전체 표본 중 올바르게 예측한 비율 |
| 정밀도(Precision) | 양성으로 예측한 표본 중 실제 양성의 비율 |
| 재현율(Recall) | 실제 양성 표본 중 양성으로 예측한 비율 |
| F1 점수 | 정밀도와 재현율의 조화 평균 |
| 표본 수(Support) | 각 클래스의 실제 표본 개수 |

- **혼동행렬** — 행은 실제 클래스, 열은 예측 클래스, 대각선은 올바르게 예측한 개수.
- **클래스 불균형** — 소수 클래스의 예측 오류를 확인하기 위한 전체 정확도와 클래스별 지표 비교.
- **매크로 평균과 가중 평균** — 클래스별로 동일한 비중을 적용하는 매크로 평균과 실제 표본 수를 반영하는 가중 평균.

3절의 붓꽃 분류 코드에 이어서 실행.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)
```

---

## 5. 앙상블 학습과 랜덤포레스트

여러 모델의 예측을 결합하는 학습 방식. 학습 표본, 특성 조합 또는 학습 알고리즘의 차이를 통한 모델 다양성 확보.

| 기법 | 핵심 개념 |
| --- | --- |
| 배깅(Bagging) | 복원 추출한 표본으로 여러 모델 학습 및 예측 결합 |
| 페이스팅(Pasting) | 비복원 추출한 표본으로 여러 모델 학습 및 예측 결합 |
| 부스팅(Boosting) | 앞 단계의 오류나 잔차를 보완하도록 모델을 순차적으로 추가 |
| 랜덤포레스트(Random Forest) | 복원 추출한 표본으로 여러 결정 트리를 학습하고, 각 분기에서 일부 특성을 무작위로 선택해 분기 후보 검토 |

### 랜덤포레스트를 활용한 붓꽃 분류

3절과 동일한 학습·테스트 데이터를 재사용해 같은 분할 조건에서 k-NN과 정확도 비교.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)
print("Random forest accuracy:", accuracy_score(y_test, forest_pred))
```

---

## 6. k-means를 활용한 붓꽃 군집화

- **동작 원리** — 각 표본을 가장 가까운 중심점에 배정하고, 배정된 표본을 기준으로 중심점을 갱신하는 과정의 반복.
- **목적** — 각 표본과 배정된 군집 중심점 사이의 거리 제곱합 감소.
- **실습 설정** — 군집 수 3개 지정 및 초기 중심점 설정 10회 시도.
- **한계** — 특성의 척도, 이상치, 초기 중심점에 따른 결과 변화.
- **결과 해석** — 군집 번호는 발견된 집단의 식별자. 군집화 입력에서 품종 라벨 제외.

붓꽃의 네 가지 측정값만 사용하는 독립 실행 예제.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

cluster_model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, n_init=10, random_state=42),
)
clusters = cluster_model.fit_predict(iris.data)

result = pd.DataFrame(iris.data, columns=iris.feature_names)
result["cluster"] = clusters

print(result.head())
print(result["cluster"].value_counts().sort_index())
```

---

## 7. k-NN과 k-means 비교

| 구분 | k-NN 분류 | k-means 군집화 |
| --- | --- | --- |
| 학습 방식 | 지도학습 | 비지도학습 |
| `k`의 의미 | 예측에 참고할 이웃 수 | 생성할 군집 수 |
| 판단 기준 | 가장 가까운 학습 표본의 라벨 투표 | 가장 가까운 중심점에 배정 |
| 출력 | 예측 클래스 | 배정된 군집 번호 |

</details>
