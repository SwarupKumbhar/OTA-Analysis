# OTA-Analysis

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv(r'C:\Users\swaru\Downloads\python_test_dataset_flights_6months.csv')
df.shape
df.head()
# Summary statistics
df.describe()
# Convert booking_date  to datetime
df['booking_date'] = pd.to_datetime(df['booking_date'])

### Observation 1: Top-performing suppliers and buyers ###
# Group by suppliers and buyers
supplier_performance = df.groupby('supplier_id').size().reset_index(name='bookings').sort_values(by='bookings', ascending=False)
buyer_performance = df.groupby('buyer_id').size().reset_index(name='bookings').sort_values(by='bookings', ascending=False)


# Visualize top 10 suppliers
plt.figure(figsize=(10, 6))
sns.barplot(x='supplier_id', y='bookings', data=supplier_performance.head(10))
plt.title('Top 10 Suppliers by Number of Bookings')
plt.xlabel('Supplier ID')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# Visualize top 10 buyers
plt.figure(figsize=(10, 6))
sns.barplot(x='buyer_id', y='bookings', data=buyer_performance.head(10))
plt.title('Top 10 Buyers by Number of Bookings')
plt.xlabel('Buyer ID')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
### Observation 2: Seasonal or Time-Based Trends ###
# Assuming 'booking_date' is in the dataset, convert to datetime
df['booking_date'] = pd.to_datetime(df['booking_date'])
# Extract month and visualize trends
df['month'] = df['booking_date'].dt.month
monthly_trends = df.groupby('month').size().reset_index(name='bookings')

# Visualize booking trends by month
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='bookings', data=monthly_trends, marker='o')
plt.title('Monthly Booking Trends')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()
# Correlation Analysis
correlation = df[['costprice', 'markup', 'selling_price', 'refund_amount']].corr()
# Comparative Analysis of Suppliers
supplier_performance = df.groupby('supplier_id')[['selling_price', 'refund_amount']].agg(['sum', 'mean'])

# Visualization of Supplier Performance
plt.figure(figsize=(10, 5))
sns.barplot(x=supplier_performance.index, y=supplier_performance[('selling_price', 'sum')])
plt.title('Total Revenue by Supplier')
plt.xlabel('Supplier ID')
plt.ylabel('Total Revenue')
plt.show()

### Short-term observations ###
# 1. Observation 1: Recent spike or drop in the number of bookings
df['booking_week'] = df['booking_date'].dt.to_period('W').apply(lambda r: r.start_time)  # Group by week
weekly_bookings = df['booking_week'].value_counts().sort_index()
# Plot for weekly bookings trend
plt.figure(figsize=(10, 5))
plt.plot(weekly_bookings.index, weekly_bookings.values, marker='o')
plt.title('Weekly Booking Trends')
plt.xlabel('Week')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.show()
# Short-term observation: Check the most recent change in booking numbers
last_two_weeks_bookings = weekly_bookings[-2:]
booking_change = last_two_weeks_bookings.diff().iloc[-1]
if booking_change > 0:
    print(f"Short-term observation 1: Bookings have increased by {booking_change} in the last week.")
else:
    print(f"Short-term observation 1: Bookings have decreased by {abs(booking_change)} in the last week.")
# 2. Observation 2: Recent increase in total revenue (selling price)
weekly_revenue = df.groupby('booking_week')['selling_price'].sum()
# Plot for weekly revenue trend
plt.figure(figsize=(10, 5))
plt.plot(weekly_revenue.index, weekly_revenue.values, marker='o', color='green')
plt.title('Weekly Revenue Trends')
plt.xlabel('Week')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()
# Short-term observation: Check the most recent change in revenue
last_two_weeks_revenue = weekly_revenue[-2:]
revenue_change = last_two_weeks_revenue.diff().iloc[-1]
if revenue_change > 0:
    print(f"Short-term observation 2: Revenue has increased by {revenue_change:.2f} in the last week.")
else:
    print(f"Short-term observation 2: Revenue has decreased by {abs(revenue_change):.2f} in the last week.")
# 3. Observation 3: Recent trends in refunds (refund_amount)
weekly_refunds = df.groupby('booking_week')['refund_amount'].sum()
# Plot for weekly refund trend
plt.figure(figsize=(10, 5))
plt.plot(weekly_refunds.index, weekly_refunds.values, marker='o', color='red')
plt.title('Weekly Refund Trends')
plt.xlabel('Week')
plt.ylabel('Total Refunds')
plt.xticks(rotation=45)
plt.show()
# Short-term observation: Check the most recent change in refund amount
last_two_weeks_refunds = weekly_refunds[-2:]
refund_change = last_two_weeks_refunds.diff().iloc[-1]
if refund_change > 0:
    print(f"Short-term observation 3: Refunds have increased by {refund_change:.2f} in the last week.")
else:
    print(f"Short-term observation 3: Refunds have decreased by {abs(refund_change):.2f} in the last week.")
### Long-term observations ###
# Extract Year-Month for long-term analysis
df['year_month'] = df['booking_date'].dt.to_period('M')
# 1. Observation 1: Long-term trend in revenue (selling price) over time
monthly_revenue = df.groupby('year_month')['selling_price'].sum()
# Plotting the long-term revenue trend
plt.figure(figsize=(10, 5))
plt.plot(monthly_revenue.index.to_timestamp(), monthly_revenue.values, marker='o', color='green')
plt.title('Long-Term Revenue Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Long-term observation 1
print("Long-term observation 1: The total revenue has shown the following trend over the months:")

# 2. Observation 2: Long-term trend in number of bookings over time
monthly_bookings = df.groupby('year_month')['buyer_id'].count()
# Plotting the long-term booking trend
plt.figure(figsize=(10, 5))
plt.plot(monthly_bookings.index.to_timestamp(), monthly_bookings.values, marker='o', color='blue')
plt.title('Long-Term Booking Trend')
plt.xlabel('Year-Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Long-term observation 2
print("Long-term observation 2: The number of bookings has shown the following trend over the months:")

# 3. Observation 3: Long-term trend in refund amounts over time
monthly_refunds = df.groupby('year_month')['refund_amount'].sum()
# Plotting the long-term refund trend
plt.figure(figsize=(10, 5))
plt.plot(monthly_refunds.index.to_timestamp(), monthly_refunds.values, marker='o', color='red')
plt.title('Long-Term Refund Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Refund Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Long-term observation 3
print("Long-term observation 3: The total refund amount has shown the following trend over the months:")
