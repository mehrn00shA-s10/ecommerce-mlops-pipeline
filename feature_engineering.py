import pandas as pd
import numpy as np
import os

file_path = os.path.join('data','ecommerce_small.csv')
df = pd.read_csv(file_path)
#print(df.head())

df['event_time'] = pd.to_datetime(df['event_time'].str.replace(' UTC', ''))
df = df.dropna(subset=['user_session'])

dummies = pd.get_dummies(df['event_type'], prefix='is', dtype=int)
df = pd.concat([df, dummies], axis=1)

session_df = df.groupby('user_session').agg(
    view_count=('is_view', 'sum'),
    cart_count=('is_cart', 'sum'),
    is_purchased=('is_purchase', 'max'),
    unique_products=('product_id', 'nunique'),
    avg_price=('price', 'mean'),
        start_time=('event_time', 'min'),
    end_time=('event_time', 'max')
).reset_index()

session_df['session_duration'] = (session_df['end_time'] - session_df['start_time']).dt.total_seconds()

session_df = session_df.drop(columns=['start_time', 'end_time'])



# 5. Display engineered data summary
print("\n--- Engineered Data Summary ---")
print(f"Total Sessions: {len(session_df)}")
print(f"Total Purchases: {session_df['is_purchased'].sum()}")
print(f"Conversion Rate: {round((session_df['is_purchased'].sum() / len(session_df)) * 100, 2)}%")

# 6. Save final dataset for Machine Learning
output_path = os.path.join('data', 'session_features.csv')
session_df.to_csv(output_path, index=False)
print(f"\nDataset saved successfully to: {output_path}")