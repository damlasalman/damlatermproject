import pandas as pd
step_length_file = '/Users/damlasalman/Desktop/untitled folder 2/HKQuantityTypeIdentifierWalkingStepLength.csv'
asymmetry_percentage_file = '/Users/damlasalman/Desktop/untitled folder 2/HKQuantityTypeIdentifierWalkingAsymmetryPercentage.csv'
active_energy_burned_file = '/Users/damlasalman/Desktop/untitled folder 2/HKQuantityTypeIdentifierActiveEnergyBurned.csv'
combined_data_file = '/Users/damlasalman/Desktop/untitled folder 2/apple_health_walking_data.csv'
# For HKQuantityTypeIdentifierWalkingStepLength.csv
with open(step_length_file, 'r') as file:
    sample_data = file.readlines(1000)

# For HKQuantityTypeIdentifierWalkingAsymmetryPercentage.csv
with open(asymmetry_percentage_file, 'r') as file:
    sample_data_asymmetry = file.readlines(1000)

# For HKQuantityTypeIdentifierActiveEnergyBurned.csv
with open(active_energy_burned_file, 'r') as file:
    sample_data_energy = file.readlines(1000)

# For apple_health_walking_data.csv
with open(combined_data_file, 'r') as file:
    sample_data_combined = file.readlines(1000)
# Loading the datasets with semicolon delimiter
step_length_df = pd.read_csv(step_length_file, delimiter=';')
asymmetry_percentage_df = pd.read_csv(asymmetry_percentage_file, delimiter=';')
active_energy_burned_df = pd.read_csv(active_energy_burned_file, delimiter=';')

# Loading the dataset with comma delimiter
combined_data_df = pd.read_csv(combined_data_file)
step_length_df.head()
asymmetry_percentage_df.head()
active_energy_burned_df.head()
combined_data_df.head()
# Descriptive Statistics and Data Quality Check for each dataset

# Walking Step Length
step_length_desc = step_length_df.describe()
step_length_missing = step_length_df.isnull().sum()

# Walking Asymmetry Percentage
asymmetry_percentage_desc = asymmetry_percentage_df.describe()
asymmetry_percentage_missing = asymmetry_percentage_df.isnull().sum()

# Active Energy Burned
active_energy_burned_desc = active_energy_burned_df.describe()
active_energy_burned_missing = active_energy_burned_df.isnull().sum()

# Combined Walking Data
combined_data_desc = combined_data_df.describe()
combined_data_missing = combined_data_df.isnull().sum()

(step_length_desc, step_length_missing, asymmetry_percentage_desc, asymmetry_percentage_missing, active_energy_burned_desc, active_energy_burned_missing, combined_data_desc, combined_data_missing)
import matplotlib.pyplot as plt

# Investigating missing values in Walking Step Length dataset
step_length_missing_columns = step_length_df.columns[step_length_df.isnull().any()].tolist()
step_length_missing_data = step_length_df[step_length_missing_columns].isnull()

# Visualizing the distribution of missing values
plt.figure(figsize=(10, 6))
plt.title("Distribution of Missing Values in Walking Step Length Dataset")
step_length_missing_data.sum().plot(kind='bar')
plt.ylabel('Number of Missing Values')
plt.xlabel('Columns with Missing Values')
plt.show()
# For Walking Step Length
step_length_df = pd.read_csv(step_length_file, delimiter=';', header=1)

# For Walking Asymmetry Percentage
asymmetry_percentage_df = pd.read_csv(asymmetry_percentage_file, delimiter=';', header=1)

# For Active Energy Burned
active_energy_burned_df = pd.read_csv(active_energy_burned_file, delimiter=';', header=1)
# Walking Step Length
plt.figure(figsize=(10, 6))
plt.hist(step_length_df['value'].dropna(), bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Walking Step Length')
plt.xlabel('Step Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Walking Asymmetry Percentage
plt.figure(figsize=(10, 6))
plt.hist(asymmetry_percentage_df['value'].dropna(), bins=50, color='green', alpha=0.7)
plt.title('Distribution of Walking Asymmetry Percentage')
plt.xlabel('Asymmetry Percentage (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Active Energy Burned
plt.figure(figsize=(10, 6))
plt.hist(active_energy_burned_df['value'].dropna(), bins=50, color='red', alpha=0.7)
plt.title('Distribution of Active Energy Burned')
plt.xlabel('Energy Burned (kcal)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Converting 'startdate' to datetime format for each dataset
step_length_df['startdate'] = pd.to_datetime(step_length_df['startdate'])
asymmetry_percentage_df['startdate'] = pd.to_datetime(asymmetry_percentage_df['startdate'])
active_energy_burned_df['startdate'] = pd.to_datetime(active_energy_burned_df['startdate'])

# Time-Series Plot for Active Energy Burned
plt.figure(figsize=(12, 6))
plt.plot(active_energy_burned_df['startdate'], active_energy_burned_df['value'], color='red', alpha=0.7)
plt.title('Time Series of Active Energy Burned')
plt.xlabel('Date')
plt.ylabel('Energy Burned (kcal)')
plt.grid(True)
plt.show()

# Time-Series Plot for Walking Step Length
plt.figure(figsize=(12, 6))
plt.plot(step_length_df['startdate'], step_length_df['value'], color='blue', alpha=0.7)
plt.title('Time Series of Walking Step Length')
plt.xlabel('Date')
plt.ylabel('Step Length (cm)')
plt.grid(True)
plt.show()
# Aligning the datasets by day and aggregating the values
# Aggregating by mean might be the most appropriate for asymmetry percentage and step length
# For active energy burned, summing up the daily values could be more meaningful

# Grouping and aggregating the data by day
daily_step_length = step_length_df.groupby(step_length_df['startdate'].dt.date)['value'].mean()
daily_asymmetry_percentage = asymmetry_percentage_df.groupby(asymmetry_percentage_df['startdate'].dt.date)['value'].mean()
daily_active_energy_burned = active_energy_burned_df.groupby(active_energy_burned_df['startdate'].dt.date)['value'].sum()

# Combining the aggregated data into a single DataFrame
combined_daily_data = pd.DataFrame({
    'Step Length': daily_step_length,
    'Asymmetry Percentage': daily_asymmetry_percentage,
    'Active Energy Burned': daily_active_energy_burned
}).dropna()  # Dropping NaN values to ensure clean correlation analysis

# Calculating correlation matrix
correlation_matrix = combined_daily_data.corr()

correlation_matrix
import seaborn as sns

# Heatmap for Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

# Selecting the 'Active Energy Burned' data for decomposition
active_energy_series = combined_daily_data['Active Energy Burned']

# Performing seasonal decomposition
# Assuming daily data, the frequency is set to a typical week (7 days)
decomposition = seasonal_decompose(active_energy_series, model='additive', period=7)

# Plotting the decomposition
plt.figure(figsize=(14, 10))

# Trend
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend', color='blue')
plt.legend(loc='upper left')
plt.title('Trend Component of Active Energy Burned')

# Seasonal
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.title('Seasonal Component of Active Energy Burned')

# Residual
plt.subplot(413)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend(loc='upper left')
plt.title('Residuals of Active Energy Burned')

# Observed
plt.subplot(414)
plt.plot(active_energy_series, label='Observed', color='black')
plt.legend(loc='upper left')
plt.title('Observed Active Energy Burned')

plt.tight_layout()
plt.show()
# Box Plot for Walking Step Length
plt.figure(figsize=(8, 6))
sns.boxplot(data=step_length_df, x='value')
plt.title('Box Plot of Walking Step Length')
plt.xlabel('Step Length (cm)')
plt.show()
# Box Plot for Walking Asymmetry Percentage
plt.figure(figsize=(8, 6))
sns.boxplot(data=asymmetry_percentage_df, x='value')
plt.title('Box Plot of Walking Asymmetry Percentage')
plt.xlabel('Asymmetry Percentage (%)')
plt.show()
# Box Plot for Active Energy Burned
plt.figure(figsize=(8, 6))
sns.boxplot(data=active_energy_burned_df, x='value')
plt.title('Box Plot of Active Energy Burned')
plt.xlabel('Energy Burned (kcal)')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data Preparation
X = combined_daily_data[['Step Length', 'Asymmetry Percentage']]  # Independent variables
y = combined_daily_data['Active Energy Burned']  # Dependent variable

# Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting on the Test Set
y_pred = regressor.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

(mse, r2)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Data Preparation for Clustering
clustering_data = combined_daily_data[['Step Length', 'Asymmetry Percentage', 'Active Energy Burned']]

# Scaling the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Determining the Optimal Number of Clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

from sklearn.preprocessing import LabelEncoder

# Defining the Target Variable
# For simplicity, we'll use the median of 'Active Energy Burned' as the threshold for categorization
energy_burned_threshold = combined_daily_data['Active Energy Burned'].median()
combined_daily_data['Energy Burned Category'] = np.where(combined_daily_data['Active Energy Burned'] >= energy_burned_threshold, 'High', 'Low')

# Preparing the Data for Classification
X_classification = combined_daily_data[['Step Length', 'Asymmetry Percentage']]  # Features
y_classification = combined_daily_data['Energy Burned Category']  # Target

# Encoding the Target Variable
le = LabelEncoder()
y_classification_encoded = le.fit_transform(y_classification)

# Splitting the Data into Training and Testing Sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification_encoded, test_size=0.2, random_state=42)

# Displaying the first few entries of the encoded target variable
y_classification_encoded[:5]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Training the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

# Predicting on the Test Set
y_pred_class = rf_classifier.predict(X_test_class)

# Model Evaluation
accuracy = accuracy_score(y_test_class, y_pred_class)
classification_rep = classification_report(y_test_class, y_pred_class)
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

(accuracy, classification_rep, conf_matrix)

# Visualization of the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Feature Importance Visualization
feature_importances = rf_classifier.feature_importances_
features = X_classification.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importances in Random Forest Classifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

