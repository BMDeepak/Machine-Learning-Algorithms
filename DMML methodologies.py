
#Multi linear regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

financial_path = "C:\\Users\\shiva\\Documents\\DMML Project\\numeric\\financial_regression.csv"  
financial_data = pd.read_csv(financial_path)

target_column = "sp500 close"
numerical_columns = financial_data.select_dtypes(include=["float64"]).columns
predictors = numerical_columns.drop(target_column, errors='ignore')

financial_cleaned = financial_data[[target_column] + list(predictors)].dropna()


X = financial_cleaned[predictors]
y = financial_cleaned[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) / y_test <= 0.1) * 100

print(f"Accuracy within ±10%: {accuracy:.2f}%")
print(f"R-squared(multiple linear regression): {r2}")
print(f"Mean Squared Error(multiple linear regression): {mse}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.show()




#Random Forest
from sklearn.ensemble import RandomForestRegressor
financial_path = "C:\\Users\\shiva\\Documents\\DMML Project\\numeric\\financial_regression.csv"  
financial_data = pd.read_csv(financial_path)

target_column = "sp500 close"
numerical_columns = financial_data.select_dtypes(include=["float64"]).columns
predictors = numerical_columns.drop(target_column, errors='ignore')

financial_cleaned = financial_data[[target_column] + list(predictors)].dropna()


X = financial_cleaned[predictors]
y = financial_cleaned[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier_rf = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=5,n_estimators=100, oob_score=True)

classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) / y_test <= 0.1) * 100

print(f"Accuracy within ±10%: {accuracy:.2f}%")
print(f"R-squared(Random forest regression): {r2}")
print(f"Mean Squared Error(Random forest regression): {mse}")
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color="green", alpha=0.7, edgecolor="black")
plt.axvline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()



#Gradient Boosting Algorithms 
housing_path = "C:\\Users\\shiva\\Documents\\DMML Project\\numeric\\housing.csv"
housing_data = pd.read_csv(housing_path)
target_column = "median_house_value"
numerical_columns = housing_data.select_dtypes(include=["float64", "int64"]).columns
predictors = numerical_columns.drop(target_column, errors='ignore')

housing_cleaned = housing_data[[target_column] + list(predictors)].dropna()

X = housing_cleaned[predictors]
y = housing_cleaned[target_column]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1],
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1
)


grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

best_rf_model.fit(X_train, y_train)
y_pred = best_rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) / y_test <= 0.1) * 100

print(f"Optimized Accuracy within ±10%: {accuracy:.2f}%")
print(f"Optimized R-squared(Decision Tree): {r2}")
print(f"Optimized Mean Squared Error(Decision Tree): {mse}")

grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)
importances = best_xgb_model.feature_importances_
features = predictors

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="skyblue", edgecolor="black")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.grid(True)
plt.show()

#Decision Tree


housing_path = "C:\\Users\\shiva\\Documents\\DMML Project\\numeric\\housing.csv"
housing_data = pd.read_csv(housing_path)


target_column = "median_house_value"
numerical_columns = housing_data.select_dtypes(include=["float64", "int64"]).columns
predictors = numerical_columns.drop(target_column, errors='ignore')
housing_cleaned = housing_data[[target_column] + list(predictors)].dropna()

X = housing_cleaned[predictors]
y = housing_cleaned[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_

best_dt.fit(X_train, y_train)

y_pred = best_dt.predict(X_test)


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy = np.mean(np.abs(y_pred - y_test) / y_test <= 0.1) * 100

print(f"Accuracy within ±10%: {accuracy:.2f}%")
print(f"R-squared (Decision Tree): {r2}")
print(f"Mean Squared Error (Decision Tree): {mse}")
print(f"Mean Absolute Error (Decision Tree): {mae}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot: Actual vs Predicted")
plt.grid(True)
plt.show()


#Clustering/Topic Modeling
import re
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation



# Load SMS data
sms_data_path = "C:\\Users\\shiva\\Documents\\DMML Project\\text\\SMSSpamCollection"
sms_data = pd.read_csv(sms_data_path, sep='\t', header=None, names=["Label", "Message"])

# Load spaCy model (English)
nlp = spacy.load("en_core_web_sm")

def preprocess_message(message):
    # Convert to lowercase
    message = message.lower()
    
    # Remove punctuation and non-alphabetic characters
    message = re.sub(r"[^a-z\s]", "", message)
    
    # Process text using spaCy for tokenization, stopword removal, and lemmatization
    doc = nlp(message)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    
    # Join tokens back into a single string
    processed_message = " ".join(tokens)
    
    # Correct spelling using TextBlob
    corrected_message = str(TextBlob(processed_message).correct())
    
    return corrected_message

# Apply preprocessing
sms_data["Processed_Message"] = sms_data["Message"].apply(preprocess_message)

#checking sample data if data is cleaned
print(sms_data[["Message", "Processed_Message"]].head(10))

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(sms_data["Processed_Message"])
print("Bag of Words shape:", X_bow.shape)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(sms_data["Processed_Message"])
print("TF-IDF shape:", X_tfidf.shape)

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit KMeans to the TF-IDF data
kmeans.fit(X_tfidf)

# Assign clusters to each message
sms_data["Cluster"] = kmeans.labels_
print(sms_data[["Message", "Processed_Message", "Cluster"]].head(10))


# Reduce dimensions for visualization
pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(X_tfidf.toarray())

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.title("Clustering of SMS Messages")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X_tfidf)

# Display the top words for each topic
def display_topics(model, feature_names, num_words):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))

display_topics(lda, tfidf_vectorizer.get_feature_names_out(), 10)