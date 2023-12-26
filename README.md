# Pipelines with Pandas
# and a lot of
# Machine Learning

Tech stack: Python, Numpy, Pandas, Scikit-learn, LightGBM, Matplotlib, Pytest

#### The Business problem: Analyzing Creditworthiness in Fintech

The challenge proposed by Kaggle for the detection of clients at risk of defaulting on credit payments is a binary Classification task.
The propousal of this project is to develop some Machine Learning and Deep Learning models, with the aim of putting into practice the Python fundamentals related to Object-Oriented Programming. To achieve this, the focus was on designing Pipelines, constructing Estimators, and creating Machine Learning Models that can fit within Scikit-Learn Pipelines.

The Pipelines have been designed to internally work with numpy, but their outputs are Pandas DataFrames. In this way, a step called "CustomBackup" is created to provide data to those activities related to Data Science that require Pandas DataFrames.

Kaggle Competition: https://www.kaggle.com/competitions/home-credit-default-risk/overview

Some files are not included in the repository for privacy reasons.

---

# Summary

- All the implementation are in the LGBM_NN_pipeline.ipynb notebook.
- The evaluation of each model is not included in the README.md file, but it is in the notebook.
- Each pipeline has the same first steps, but the last steps are different.
- The first pipelines is based on LightGBM
- The second pipeline has 3 models on the head of the pipeline
- NN y DNN where made with Numpy and structured with Scikit-Learn Estimators

#### Pipeline 1: LightGBM Predictor

model: LightGBM

![pipe_1](https://github.com/jackonedev/pipelines_and_machine_learning/blob/main/image/pipe1.png?raw=true)

#### Pipeline 2: MBFs Assembly Model

models: LightGBM (as feature selector), Logistic Regression, NN and DNN

![pipe_2](https://github.com/jackonedev/pipelines_and_machine_learning/blob/main/image/pipe2.png?raw=true)

---

# Technical Conclusions

#### Regarding the implementation of "Pipeline_1 LightGBM Predictor", the following conclusions can be drawn:
- As long as the data structure remains stable, the model can be used in production.
- It provides an end-to-end solution from data reading to prediction.
- Straightforward implementation.
- Good performance in terms of time and accuracy.
- Easy to optimize hyperparameters.

#### Regarding the implementation of "Pipeline_2 MBFs Assembly Model", the following conclusions can be drawn:
- It is a practical solution because it has the same interface as a single model.
- The solution requires previous partial executions to obtain future attributes such as the dimensionality of the selected features.
- It is a solution that runs only on CPU.
- The 3 models in the ensemble run sequentially and not in parallel processes.
- The base models of NN and DNN are formalized as Scikit-Learn estimators.
- Practical methods are created for NN and DNN models (plotting, scoring, etc.).
- It could become an end-to-end solution if a meta-data component is implemented on the local disk.

#### Additional clarifications about both models:
- Prior to training the model, another step "CustomBackup" could be added to obtain the features selected by the feature selection model.
- Implementation of unsupervised clustering models to facilitate data understanding.