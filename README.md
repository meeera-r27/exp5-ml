# MLE and MAP Estimation using Naive Bayes

## Aim

To implement **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP) Estimation** for a Multinomial Naive Bayes classifier and compare their performance using different smoothing techniques on the **20 Newsgroups dataset**.

---

## Dataset

The experiment uses the **20 Newsgroups dataset** from Scikit-learn.

Four categories are selected:

* `alt.atheism`
* `soc.religion.christian`
* `comp.graphics`
* `sci.med`

The dataset is divided into:

| Dataset  | Samples |
| -------- | ------: |
| Training |    2257 |
| Testing  |    1502 |

The text is converted into numerical word-count features using `CountVectorizer`. A maximum of **5000 features** is used, with English stop words removed.  

---

## Objectives

* To implement a Multinomial Naive Bayes classifier.
* To calculate word probabilities using **MLE**.
* To implement **MAP estimation** using different priors.
* To study the effect of smoothing on classification accuracy.
* To compare MLE and MAP-based predictions.
* To identify the best-performing smoothing technique.

---

## Technologies Used

* **Python**
* **NumPy**
* **Scikit-learn**
* **Jupyter Notebook / Google Colab**

### Python Libraries

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
```

These libraries and functions are used directly in the experiment. 

---

# Methodology

The experiment follows these steps:

```text
Load 20 Newsgroups Dataset
          ↓
Select Four Categories
          ↓
Preprocess Text
          ↓
CountVectorizer
          ↓
Generate 5000 Word Features
          ↓
Calculate Class Priors
          ↓
Aggregate Word Counts per Class
          ↓
Calculate MLE Probabilities
          ↓
Predict Using Naive Bayes
          ↓
Apply Different MAP Priors
          ↓
Compare Test Accuracies
```

---

## 1. Dataset Loading

The 20 Newsgroups dataset is loaded with four selected categories. Headers, footers and quotations are removed during loading. 

```python
categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]

train_data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

test_data = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
```

---

## 2. Text Vectorization

The text documents are converted into word-count vectors using `CountVectorizer`.

```python
vectorizer = CountVectorizer(
    stop_words='english',
    max_features=5000
)

X_train = vectorizer.fit_transform(train_data.data).toarray()
X_test = vectorizer.transform(test_data.data).toarray()

y_train = train_data.target
y_test = test_data.target
```

The resulting shapes are:

```text
Train shape: (2257, 5000)
Test shape: (1502, 5000)
```

 

---

# 3. Class Prior Probability

The class prior probability is calculated from the number of training samples belonging to each class.

$$
P(c)=\frac{N_c}{N}
$$

where:

* \(N_c\) = number of samples in class \(c\)
* \(N\) = total number of training samples

The notebook calculates these probabilities using class counts. 

---

# 4. Maximum Likelihood Estimation (MLE)

For MLE, the probability of each word given a class is calculated using its observed frequency.

$$
P(w|c)=\frac{N_{cw}}{\sum_w N_{cw}}
$$

where:

* \(N_{cw}\) = count of word \(w\) in class \(c\)
* \(\sum_w N_{cw}\) = total word count in class \(c\)

The notebook calculates:

```python
theta_mle = N_c / N_c.sum(axis=1, keepdims=True)
```



### MLE Test Accuracy

**77.10%**



---

# 5. Naive Bayes Prediction

The classifier uses log probabilities to avoid numerical underflow.

The log-posterior is calculated as:

$$
\log P(c|X)
\propto
X\cdot\log(\theta)^T+\log P(c)
$$

The class having the maximum posterior probability is selected as the predicted class.

The notebook uses `argmax` to select the class with the highest log-posterior probability. 

---

# 6. MAP Estimation

MAP estimation incorporates prior information into the parameter estimation.

The notebook evaluates four different priors/smoothing methods:

1. Lidstone Smoothing
2. Laplace Smoothing
3. Strong Dirichlet Prior
4. Non-Uniform Prior

The MAP probability is calculated using:

$$
\theta_{MAP}
=
\frac{N_c+(\alpha-1)}
{\sum(N_c+(\alpha-1))}
$$

The implementation follows:

```python
numerator = N_c + (alpha - 1)
denominator = numerator.sum(axis=1, keepdims=True)
theta_map = numerator / denominator
```



---

# 7. MAP Priors Used

### A. Lidstone Smoothing

$$
\alpha = 1.01
$$

A small amount of smoothing is applied to the word probabilities.

### B. Laplace Smoothing

$$
\alpha = 2.0
$$

This provides stronger smoothing than the Lidstone configuration used in this experiment.

### C. Strong Dirichlet Prior

$$
\alpha = 10.0
$$

A strong prior is applied to the word probabilities.

### D. Non-Uniform Prior

The prior is based on the empirical frequency of words in the training data.

The notebook defines these four priors directly in the `priors_to_test` dictionary. 

---

# Results

| S.No. | Method                              | Alpha | Test Accuracy |
| ----: | ----------------------------------- | ----: | ------------: |
|     1 | MLE                                 |     — |    **0.7710** |
|     2 | Lidstone Smoothing                  |  1.01 |    **0.8103** |
|     3 | Laplace Smoothing                   |   2.0 |    **0.8182** |
|     4 | Strong Dirichlet Prior              |  10.0 |    **0.7863** |
|     5 | Non-Uniform Prior (Empirical Basis) |     — |    **0.8063** |

The MAP results are obtained directly from the notebook output. 

---

# Result Analysis

The results show that:

* MLE achieved an accuracy of **77.10%**.
* All tested MAP approaches except the strong Dirichlet prior improved over MLE.
* Lidstone smoothing achieved **81.03%**.
* Laplace smoothing achieved the highest accuracy of **81.82%**.
* The strong Dirichlet prior achieved **78.63%**.
* The non-uniform empirical prior achieved **80.63%**.

### Best Model

**Laplace Smoothing with α = 2.0**

$$
\boxed{\text{Test Accuracy}=81.82\%}
$$

It improved the accuracy from **77.10% with MLE to 81.82% with MAP**, an improvement of **4.72 percentage points**.

---

# Comparison

```text
MLE                         77.10%
     ↓
Strong Dirichlet Prior      78.63%
     ↓
Non-Uniform Prior           80.63%
     ↓
Lidstone Smoothing          81.03%
     ↓
Laplace Smoothing           81.82%  ← BEST
```

---

# Conclusion

This experiment implemented **MLE and MAP estimation using a Multinomial Naive Bayes classifier** for text classification on four categories of the 20 Newsgroups dataset.

The MLE approach achieved **77.10% test accuracy**. MAP estimation was then evaluated using different smoothing techniques. Among the tested methods, **Laplace Smoothing with α = 2.0 achieved the highest accuracy of 81.82%**.

Therefore, for this experiment, MAP estimation with Laplace smoothing performed better than the basic MLE approach.

---

# Project Structure

```text
exp5_ml/
│
├── exp5_ml.ipynb
└── README.md
```

---

# How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
exp5_ml.ipynb
```

using **Jupyter Notebook** or **Google Colab**.

### 3. Install required libraries

```bash
pip install numpy scikit-learn
```

### 4. Run all cells

Execute the cells sequentially to:

* Load the dataset
* Vectorize the text
* Calculate MLE probabilities
* Perform Naive Bayes prediction
* Apply MAP smoothing
* Compare the test accuracies

---

# Requirements

```text
Python 3.x
NumPy
Scikit-learn
Jupyter Notebook / Google Colab
```

---

## Author

**Meera R**

B.Tech Computer Science and Engineering
