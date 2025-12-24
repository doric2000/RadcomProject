import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. טעינת הנתונים (נשתמש ב-Train של האפליקציות כי הוא הקשה ביותר)
print("Loading data...")
df = pd.read_csv('data/APP-1/radcom_app_train.csv')

# 2. בדיקת איזון מחלקות (Class Imbalance)
# זה קריטי! אם המודל רואה רק יוטיוב, הוא ינחש תמיד יוטיוב.
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

# 3. בדיקת קורלציה (Correlation Matrix)
# נראה אילו פיצ'רים הולכים ביחד. אם שני פיצ'רים אומרים אותו דבר, אפשר לזרוק אחד.
# נסנן עמודות לא מספריות
numeric_df = df.select_dtypes(include=[np.number])
# נזרוק עמודות מזהות
numeric_df = numeric_df.drop(columns=['Source_port', 'Destination_port', 'Timestamp'], errors='ignore')

plt.figure(figsize=(10, 8))
# ניקח רק מדגם קטן של עמודות מעניינות כדי לא לפוצץ את הגרף
cols_to_check = [
    'fwd_packets_amount', 'bwd_packets_amount', 
    'fwd_packets_length', 'bwd_packets_length',
    'min_fwd_inter_arrival_time', 'mean_fwd_inter_arrival_time'
]
corr_matrix = numeric_df[cols_to_check].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()