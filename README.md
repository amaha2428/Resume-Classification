# Resume Classification Project

## Overview
This project involves building a machine learning model to classify resumes into different job categories. The dataset contains resumes along with their respective job categories. The primary goal is to preprocess the text data, build various classification models, and evaluate their performance to identify the best model for this task.

## Dataset
The dataset used for this project is named `UpdatedResumeDataSet.csv`. It contains two columns:
- `Resume`: The text content of the resume.
- `Category`: The job category associated with the resume.

## Project Steps
1. **Data Exploration and Visualization**:
   - Load and inspect the dataset.
   - Visualize the distribution of job categories to understand the class balance.

2. **Data Preprocessing**:
   - Clean the resume text to remove noise such as hashtags, URLs, non-ASCII characters, punctuations, mentions, and extra whitespaces.
   - Transform the text data into numerical features using TF-IDF Vectorization.

3. **Model Building**:
   - Split the dataset into training and testing sets.
   - Build and evaluate multiple classification models:
     - Random Forest Classifier
     - Logistic Regression
     - K-Neighbors Classifier
     - Support Vector Machine (SVM)
   - Compare the models based on their F1 score and accuracy.

4. **Model Evaluation**:
   - Evaluate the performance of each model on the test data.
   - Select the best model based on consistency and overall performance.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- re (Regular Expressions)

## Results
- All models performed well, achieving an accuracy between 96-100%.
- The Random Forest Classifier was chosen as the final model due to its consistency across multiple runs.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/resume-classification.git
   cd resume-classification
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:
   - Load the dataset and preprocess the text data.
   - Split the dataset and train the models.
   - Evaluate the models and select the best one.

## Conclusion
This project demonstrates the process of building and evaluating machine learning models for text classification. The Random Forest Classifier was selected as the best model for classifying resumes into job categories due to its high accuracy and consistency.

## Contact
For any questions or suggestions, please contact Amaha God'spower at amahagodspower@gmail.com.

---
