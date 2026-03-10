# Student Performance Prediction (CMPUT 466 Project)

This project explores the use of machine learning models to predict whether a student will **pass or fail a course** based on demographic, academic, and social features.  

The project was completed as part of **CMPUT 466 (Machine Learning)** at the **University of Alberta**.

The models are trained on the **UCI Student Performance Dataset** and evaluated using validation and test sets to compare predictive performance.

**Models implemented:**
- Logistic Regression (MLE and MAP)
- Neural Network
- Gaussian Naive Bayes

Best Accuracy: 94.94%

---

## Dataset

The dataset used in this project is the **Student Performance Dataset** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/320/student+performance  
- File used: `student-mat.csv`
- Total samples: **395 students**

Each student record includes features such as:

- Prior grades (`G1`, `G2`)
- Study time
- Number of past class failures
- Absences
- Family and school support indicators
- Demographic information (school, gender, guardian, etc.)

Categorical variables are **one-hot encoded** before training.

### Prediction Task

This project formulates the problem as a **binary classification task**:

- **1** → Student passes (final grade `G3 ≥ 10`)
- **0** → Student fails (`G3 < 10`)

---

## Models Implemented

Three machine learning models were implemented and compared:

### Logistic Regression
- Implemented using **binary cross-entropy loss**
- Optimized using **gradient descent**
- Explored both:
  - **MLE (Maximum Likelihood Estimation)**
  - **MAP (Maximum A Posteriori)** with L2 regularization

### Neural Network
- Fully connected network with:
  - **1 hidden layer**
  - **ReLU activation**
  - **Sigmoid output**
- Hyperparameters tuned:
  - learning rate
  - number of epochs
  - hidden layer size

### Gaussian Naive Bayes
- Implemented using **scikit-learn**
- Assumes:
  - conditional independence between features
  - Gaussian distribution for each feature

This model serves as a **probabilistic baseline** for comparison.

---

## Data Split

The dataset is split into:

| Set | Percentage |
|----|----|
| Training | 60% |
| Validation | 20% |
| Test | 20% |

This allows for consistent model comparison and hyperparameter tuning.

---

## Evaluation

Models were evaluated primarily using **accuracy**, defined as the proportion of correctly predicted outcomes.

Confusion matrices were used to visualize model performance and compare prediction behavior.

---

## Results

| Model | Accuracy | Errors |
|------|------|------|
| Logistic Regression (MLE) | **94.94%** | 4 |
| Logistic Regression (MAP) | 94.94% | 4 |
| Neural Network | 88.61% | 11 |
| Naive Bayes | 87.34% | 10 |

### Key Findings

- **Logistic regression achieved the best overall performance**
- Regularization (MAP) did not significantly improve results
- Neural networks performed well but produced more false positives
- Naive Bayes provided a competitive baseline despite minimal tuning

---

## Installation

Clone the repository and install the required Python packages.

```bash
git clone https://github.com/dwhaling/student-performance-prediction.git
cd student-performance-prediction
pip install numpy pandas scikit-learn matplotlib
```

---

## Running the Project

Each model is implemented in a separate script and can be run independently.

### Logistic Regression (MLE)
```bash
python mle.py
```

### Logistic Regression (MAP)
```bash
python map.py
```

### Gaussian Naive Bayes
```bash
python bayes.py
```

### Neural Network
```bash
python nn.py
```

### What the Scripts Do

Each script will:

1. Load and preprocess the dataset  
2. Train the specified machine learning model  
3. Evaluate performance on validation and test sets  
4. Generate output files containing accuracy results and confusion matrices  

Sample output files and evaluation results are included in the `evaluation/` folder.

---


## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Author

**Dawson Whaling**  
BSc Computing Science  
University of Alberta


