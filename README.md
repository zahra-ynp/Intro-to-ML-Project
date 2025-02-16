# 🏀 NBA Match Outcome Prediction - Machine Learning Project  

##  Project Overview  
This project predicts the outcome of NBA matches using **machine learning**. Given **historical match data**, including team performance metrics, win percentages, and field goal percentages, the model estimates the probability of the **home team winning**.  

**Dataset:** [Kaggle - NBA Games](https://www.kaggle.com/datasets/nathanlauga/nba-games?select=players.csv)  

##  Machine Learning Models  
Three models were trained and optimized using **Grid Search** and **5-fold cross-validation**:  
-  **Random Forest Classifier**  
-  **K-Nearest Neighbors (KNN)**  
-  **Support Vector Machine (SVM)**  

##  Evaluation  
The models were evaluated using the following metrics:  
✔ **Accuracy**  
✔ **Precision**  
✔ **Recall**  
✔ **F1 Score**  
✔ **ROC-AUC Score**  

 **Comparison Chart**: The performance evaluation is available in the **Jupyter Notebook (`notebook.ipynb`)**.  
 **Final Model Selection**: Random Forest and SVM performed similarly, but **Random Forest was chosen as the final model** due to its better efficiency.  

##  Prediction App  
A simple **prediction script (`app.py`)** allows users to input **home and away team names**, and the app predicts the **winning team and probability**.  
