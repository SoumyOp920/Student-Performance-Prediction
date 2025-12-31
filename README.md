🎓 Student Performance Prediction using Machine Learning

📌 Project Overview

This project implements an end-to-end Machine Learning pipeline to predict whether a student will Pass or Fail based on academic and attendance-related features.The goal of this project is to understand and apply core Machine Learning concepts such as data preprocessing, supervised learning, classification, model training, and evaluation.

🎯 Problem Statement

Predict student performance (Pass / Fail) using the following factors:

◦ Study Hours
◦ Attendance Percentage
◦ Previous Exam Score
◦ This is a Supervised Learning Classification Problem, as the output labels are known.

🧠 Machine Learning Concepts Used

◦ Supervised Learning
◦ Classification
◦ Logistic Regression
◦ Train–Test Split
◦ Confusion Matrix
◦ Accuracy, Precision, Recall
◦ Probability-based prediction

📊 Dataset Description

The dataset contains the following columns:

◦ Column-Name	              Description
◦ study_hours:	      Number of hours studied
◦ attendance:	        Attendance percentage
◦ previous_score:	    Previous exam score
◦ pass_fail	          Target variable (0 = Fail, 1 = Pass)

The dataset was synthetically generated with realistic constraints.

Noise was added to simulate real-world variability.

🔍 Data Preprocessing

Before building the model, the following preprocessing steps were performed:

◦ Data inspection (head, tail, info, describe)
◦ Missing value analysis and handling
◦ Feature selection
◦ Separation of features (X) and target (y)

🔄 Train–Test Split

The dataset was split into:

◦ 80% Training Data
◦ 20% Testing Data

This ensures the model is evaluated on unseen data, preventing overfitting.

🤖 Model Used
Logistic Regression

◦ Chosen for binary classification (Pass / Fail)
◦ Simple, interpretable, and effective baseline model
◦ Outputs probability scores used for final classification

📈 Model Evaluation Metrics

The model was evaluated using:

◦ Accuracy – Overall correctness
◦ Confusion Matrix – Detailed error analysis
◦ Precision – Reliability of Pass predictions
◦ Recall – Ability to identify all Pass cases

These metrics provide a comprehensive understanding of model performance.

🧪 Prediction on New Data

The model supports prediction for new student data:

◦ Accepts user input for features
◦ Outputs:

      Pass / Fail prediction
      Probability of passing (rounded to 2 decimals)

🛠 Technologies Used

Python

◦ NumPy
◦ Pandas
◦ Matplotlib
◦ Seaborn
◦ Scikit-learn

🚀 How to Run the Project

1. Clone the repository
2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the Jupyter Notebook:
```bash
jupyter notebook Student_Performance_Prediction.ipynb
```

✅ Key Learnings

◦ Built a complete ML workflow from scratch

◦ Understood how classifiers learn and predict

◦ Learned why accuracy alone is not sufficient

◦ Gained hands-on experience with model evaluation metrics

🔮 Future Improvements

◦ Compare Logistic Regression with KNN and Decision Tree

◦ Use a real-world dataset (Kaggle / UCI)

◦ Add feature scaling and hyperparameter tuning

◦ Deploy model using Flask

👨‍💻 Author

Soumy Mittal
B.Tech | AI / ML Enthusiast

⭐ If you like this project, feel free to star the repository!
