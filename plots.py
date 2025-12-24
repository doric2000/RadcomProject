import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data (use app train set; it's the most challenging)
print("Loading data...")
df = pd.read_csv('data/APP-1/radcom_app_train.csv')

# 2. Check class balance (Class Imbalance)
# This is critical! If the model only sees YouTube, it will always predict YouTube.
plt.figure(figsize=(12, 6))
class_counts = df['label'].value_counts()
top_20_classes = class_counts.head(20)

sns.barplot(x=top_20_classes.values, y=top_20_classes.index, palette='viridis')
plt.title('Top 20 Most Frequent Applications')
plt.xlabel('Number of Samples')
plt.show()

print(f"Total number of classes: {len(class_counts)}")
print(f"Most frequent class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
print(f"Least frequent class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")

# 3. Check correlation (Correlation Matrix)
# See which features co-vary. If two features convey the same information, drop one.
# Filter non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])
# Drop identifier columns
numeric_df = numeric_df.drop(columns=['Source_port', 'Destination_port', 'Timestamp'], errors='ignore')

plt.figure(figsize=(10, 8))
# Take a small sample of interesting columns to avoid overwhelming the plot
cols_to_check = [
    'fwd_packets_amount', 'bwd_packets_amount', 
    'fwd_packets_length', 'bwd_packets_length',
    'min_fwd_inter_arrival_time', 'mean_fwd_inter_arrival_time'
]
corr_matrix = numeric_df[cols_to_check].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()