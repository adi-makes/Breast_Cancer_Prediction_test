
# Breast Cancer Prediction 🧬🎯

This project leverages machine learning to predict breast cancer using clinical and diagnostic features. The model is trained on the popular Wisconsin Breast Cancer dataset to classify tumors as **benign** or **malignant**.

## 📁 Project Structure

- `Breast_Cancer_Prediction_test.ipynb` – Jupyter Notebook containing code for:
  - Data loading and preprocessing
  - Data visualization
  - Model training and evaluation (using Logistic Regression)
  - Final predictions and accuracy analysis

## 📊 Dataset

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, available via `sklearn.datasets`. It includes:

- 569 samples
- 30 numerical features (e.g., radius, texture, perimeter, area, etc.)
- Target values: `0` (Malignant), `1` (Benign)

## 🔧 Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Data visualization
- **Scikit-learn** – Model building (Logistic Regression, train/test split, evaluation)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. Open the notebook:
   ```bash
   jupyter notebook Breast_Cancer_Prediction_test.ipynb
   ```

3. Run the cells sequentially to train the model and view results.

## 📈 Model Performance

- Model used: **Logistic Regression**
- Accuracy: ~96% on test data
- Evaluation includes:
  - Confusion matrix
  - Classification report
  - Accuracy score

## 🧠 Key Learnings

- Feature scaling using `StandardScaler`
- Importance of train-test splitting
- Logistic Regression for binary classification
- Visualization of correlation and distributions

## 📌 Future Improvements

- Try advanced models like Random Forest, SVM, or Neural Networks
- Hyperparameter tuning using GridSearchCV
- Deploy as a web app using Flask/Streamlit

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).
