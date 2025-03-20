   Nexthike-Project-Mobile-Prize
    Project Report: Mobile Price Prediction 

---

    1. Introduction 
      1.1 Project Overview 
The  Mobile Price Prediction  project aims to develop a machine learning model that accurately predicts the price of mobile phones based on their specifications. This project leverages  data preprocessing, feature engineering, and predictive modeling  to extract insights and build a reliable model.

      1.2 Objectives 
- Analyze mobile phone specifications and their impact on pricing.
- Perform data cleaning and preprocessing.
- Apply feature selection techniques to improve model accuracy.
- Train and evaluate machine learning models for price prediction.

---

    2. Dataset Overview 
      2.1 Data Source 
-  Dataset Name:  Processed_Flipdata.csv
-  -  Features Included:  RAM, Battery, Processor, Display Size, Camera, Brand, and Price.

      2.2 Data Preprocessing 
-  Handling Missing Values: 
  - Numerical columns: Filled with mean values.
  - Categorical columns: Filled with the most frequent values.
-  Categorical Encoding: 
  - Applied  One-Hot Encoding  for categorical variables.
-  Outlier Detection & Treatment: 
  - Used  Interquartile Range (IQR)  method to detect and cap outliers.

---

    3. Feature Engineering 
      3.1 Correlation Analysis 
- Used  heatmap visualization  to analyze feature relationships.
- Identified the strongest predictors of mobile phone price.

      3.2 Feature Importance 
- Implemented  Random Forest  to rank feature importance.
- Extracted  top 10 influential features  affecting price.

---

    4. Model Development 
      4.1 Model Selection 
-  Baseline Model:  Linear Regression
-  Feature Selection Model:  Random Forest

      4.2 Data Splitting 
-  Training Set:  80%
-  Testing Set:  20%

      4.3 Model Training 
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

   Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   Training Model
model = LinearRegression()
model.fit(X_train, y_train)
```

---

    5. Model Evaluation 
      5.1 Evaluation Metrics 
-  Mean Absolute Error (MAE) 
-  Root Mean Squared Error (RMSE) 

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

   Model Predictions
y_pred = model.predict(X_test)

   Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}, RMSE: {rmse}")
```

---

    6. Results and Insights 
      6.1 Key Findings 
-  RAM, Battery, Processor, and Display Size  are the most influential factors affecting price.
- Brands with high-end specifications command  higher prices .

      6.2 Business Recommendations 
- Focus marketing strategies on high-impact features like  RAM and Processor .
- Optimize pricing models based on market trends and brand perception.
- Implement further  data-driven strategies  for competitive analysis.

---

    7. Future Enhancements 
- Implement  Deep Learning models  for better accuracy.
- Integrate  real-time pricing trends  using web scraping.
- Develop an  interactive dashboard  for real-time analysis.

---

    8. Conclusion 
This project successfully developed a machine learning model to predict mobile phone prices based on feature extraction. The model's performance was evaluated, and key insights were derived for business decision-making.

---

    9. References 
-  Python Libraries:  Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
-  Machine Learning Algorithms:  Linear Regression, Random Forest
-  Statistical Methods:  Feature Correlation, Outlier Detection

