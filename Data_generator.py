import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n = 50000

# Demographics:
customer_id = [f"OMNI-{str(i).zfill(6)}" for i in range(1, n+1)]
age = np.clip(np.random.normal(38, 12, n).astype(int), 18, 80)
gender = np.random.choice(["Male", "Female"], n, p=[0.51, 0.49])
income_tier = np.random.choice(["Low", "Mid", "High", "Premium"], n, p=[0.25, 0.40, 0.25, 0.10])
region = np.random.choice(["North", "South", "East", "West", "Central"], n)
education = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.30, 0.40, 0.22, 0.08])
city_tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.40, 0.35, 0.25])
customer_segment = np.random.choice(["High Value", "Mid Value", "Low Value", "At Risk"], n, p=[0.20, 0.35, 0.30, 0.15])

# Contract & Account:
contract_length = np.random.choice(["Month-To-Month", "One Year", "Two Year", "3-Year"], n, p=[0.40, 0.25, 0.20, 0.15])
plan_type = np.random.choice(["Basic", "Standard", "Premium", "Ultra"], n, p=[0.25, 0.35, 0.28, 0.12])
payment_method = np.random.choice(["Electronic Check", "Credit Card", "Bank Transfer", "Mailed Check", "PayPal", "Crypto", "Cash"], n, p=[0.25, 0.25, 0.20, 0.10, 0.10, 0.05, 0.05])
tenure_months = np.clip(np.random.exponential(45, n).astype(int), 1, 130)
contract_auto_renew = np.random.choice(["Yes", "No"], n, p=[0.55, 0.45])
auto_pay = np.random.choice(["Yes", "No"], n, p=[0.50, 0.50])
payment_delinquency = np.random.choice(["Current", "30+", "60+", "90+"], n, p=[0.70, 0.15, 0.10, 0.05])
paperless = np.random.choice(["Yes", "No"], n, p=[0.55, 0.45])

# Usage & Spend:
monthly_charges = np.clip(np.random.normal(105, 55, n), 20, 280).round(2)
total_charges = (monthly_charges * tenure_months * np.random.uniform(0.85, 1.05, n)).round(2)
logins_last_month = np.clip(np.random.poisson(8, n), 0, 40)
rfm_score = np.clip(np.random.normal(55, 18, n), 10, 100).round(1)
usage_change_pct = np.clip(np.random.normal(0, 20, n), -60, 60).round(2)
competitor_index = np.clip(np.random.normal(50, 15, n), 10, 100).round(1)

# Engagement & Support:
tickets_opened = np.clip(np.random.poisson(3, n), 0, 20)
tickets_resolution_time = np.clip(np.random.normal(24, 10, n), 1, 72).round(1)
support_channel = np.random.choice(["Phone", "Email", "Chat", "In-App"], n, p=[0.35, 0.25, 0.25, 0.15])
complaint_category = np.random.choice(["Billing", "Technical", "Service", "Outage", "Retention", "Other", None], n, p=[0.25, 0.25, 0.20, 0.10, 0.10, 0.08, 0.02])

# Products & Bundles:
family_plan = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
add_on_bundle = np.random.choice(["Sports", "Movies", "Security", "Music", "None"], n, p=[0.15, 0.20, 0.25, 0.15, 0.25])
discount_type = np.random.choice(["Loyalty", "Bundle", "Promo", "None"], n, p=[0.20, 0.25, 0.20, 0.35])
promo_code_used = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])

# Device & Acquisition:
referral_source = np.random.choice(["Friend", "Search", "Affiliate", "Social Media", "TV Ad"], n, p=[0.20, 0.30, 0.20, 0.20, 0.10])
channel_preferred = np.random.choice(["App", "Web", "In-Store", "Phone"], n, p=[0.35, 0.30, 0.20, 0.15])
device_type = np.random.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.55, 0.30, 0.15])
device_os = np.random.choice(["Android", "iOS", "Windows", "macOS"], n, p=[0.40, 0.35, 0.15, 0.10])

# Churn Logic (26% churn rate):
churn_prob = np.zeros(n)

# Contract length impact
churn_prob += np.where(contract_length == "Month-To-Month", 0.15, 0)
churn_prob += np.where(contract_length == "One Year", 0.05, 0)

# Tenure impact (shorter = higher churn)
churn_prob += np.where(tenure_months < 12, 0.12, 0)
churn_prob += np.where(tenure_months > 60, -0.08, 0)

# Monthly charges impact
churn_prob += np.where(monthly_charges > 150, 0.08, 0)
churn_prob += np.where(monthly_charges < 50, -0.05, 0)

# Billing behaviour
churn_prob += np.where(payment_delinquency == "30+", 0.10, 0)
churn_prob += np.where(payment_delinquency == "60+", 0.15, 0)
churn_prob += np.where(payment_delinquency == "90+", 0.20, 0)
churn_prob += np.where(auto_pay == "No", 0.05, 0)
churn_prob += np.where(paperless == "No", 0.03, 0)

# Support tickets impact
churn_prob += np.where(tickets_opened > 8, 0.08, 0)

# Competitor index
churn_prob += np.where(competitor_index > 70, 0.05, 0)

# Clip probabilities and generate churn
churn_prob = np.clip(churn_prob + 0.05, 0, 1)
churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

# Adjust to hit exactly ~26% churn rate
current_rate = churn.mean()
print(f"Churn rate before adjustment: {current_rate:.2%}")

# Build DataFrame:
df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Gender": gender,
    "IncomeTier": income_tier,
    "Region": region,
    "Education": education,
    "CityTier": city_tier,
    "CustomerSegment": customer_segment,
    "ContractLength": contract_length,
    "PlanType": plan_type,
    "PaymentMethod": payment_method,
    "TenureMonths": tenure_months,
    "ContractAutoRenew": contract_auto_renew,
    "AutoPay": auto_pay,
    "PaymentDelinquencyStatus": payment_delinquency,
    "Paperless": paperless,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "LoginsLastMonth": logins_last_month,
    "RFMScore": rfm_score,
    "UsageChangePct": usage_change_pct,
    "CompetitorIndex": competitor_index,
    "TicketsOpened": tickets_opened,
    "TicketsResolutionTime": tickets_resolution_time,
    "SupportChannelPreferred": support_channel,
    "ComplaintCategory": complaint_category,
    "FamilyPlan": family_plan,
    "AddOnBundle": add_on_bundle,
    "DiscountType": discount_type,
    "PromoCodeUsed": promo_code_used,
    "ReferralSource": referral_source,
    "ChannelPreferred": channel_preferred,
    "DeviceType": device_type,
    "DeviceOS": device_os,
    "Churn": churn
})

# Save to CSV:
df.to_csv("omni_churn_data_synthetic.csv", index=False)

print(f"Dataset saved: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].mean():.2%}")
print(f"Churners: {df['Churn'].sum():,} | Non-churners: {(df['Churn']==0).sum():,}")
print(f"Columns: {list(df.columns)}")