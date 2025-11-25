import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📊 AI Customer Churn Prediction")
st.write("Dự đoán khách hàng có rời bỏ hay không dựa trên mô hình Machine Learning")

# Load model, scaler và feature_names
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # ← Thêm dòng này

uploaded_file = st.file_uploader("📥 Tải file CSV Telco Customer Churn", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dữ liệu đầu vào:")
    st.dataframe(df.head())

    # Chuyển đổi dữ liệu
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df_processed = pd.get_dummies(df, drop_first=True)

    # Đồng bộ với cột của model
    missing_cols = set(feature_names) - set(df_processed.columns)  # ← Sửa dòng này
    for c in missing_cols:
        df_processed[c] = 0

    df_processed = df_processed[feature_names]  # ← Sửa dòng này

    # Scale
    X_scaled = scaler.transform(df_processed)

    # Predict
    proba = model.predict_proba(X_scaled)[:, 1]

    df["Churn_Score"] = proba

    st.subheader("🔍 Kết quả dự đoán:")
    st.dataframe(df.sort_values(by="Churn_Score", ascending=False))

    st.subheader("🔥 Khách hàng có nguy cơ cao (Churn > 0.7):")
    st.dataframe(df[df["Churn_Score"] > 0.7])

    st.bar_chart(df["Churn_Score"])
    import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Giả lập Dữ liệu & Mô hình ---

@st.cache_data
def load_and_predict_data():
    """
    Giả lập dữ liệu và kết quả dự đoán (Churn Score) cho khách hàng Softbank Corp.
    Thực tế: Dữ liệu này sẽ được tải từ DB và Churn Score sẽ được tính bằng mô hình ML đã huấn luyện.
    """
    np.random.seed(42)
    N = 1000  # Số lượng khách hàng mẫu
    
    data = {
        'CustomerID': [f'SB{i:04d}' for i in range(1, N + 1)],
        'Tenure': np.random.randint(1, 72, N), # Thời gian sử dụng (tháng)
        'MonthlyCharges': np.random.uniform(20, 150, N).round(2), # Cước hàng tháng
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], N, p=[0.55, 0.25, 0.20]),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], N, p=[0.4, 0.3, 0.3]),
        'Churn_Score': np.random.beta(a=0.5, b=5, size=N) # Giả lập Churn Score (ngẫu nhiên)
    }
    df = pd.DataFrame(data)
    
    # Điều chỉnh Churn Score để tạo mối quan hệ giả lập thực tế hơn
    df.loc[df['Contract'] == 'Month-to-month', 'Churn_Score'] *= 1.5
    df.loc[df['MonthlyCharges'] > 100, 'Churn_Score'] *= 1.2
    
    # Chuẩn hóa lại score về [0, 1]
    df['Churn_Score'] = df['Churn_Score'].clip(0, 1)
    df['Churn_Score'] = (df['Churn_Score'] - df['Churn_Score'].min()) / (df['Churn_Score'].max() - df['Churn_Score'].min())

    return df

df_churn = load_and_predict_data()

# --- 2. Định nghĩa các Giải pháp Giữ chân (Retention Strategies) ---

def suggest_retention_strategy(row):
    """Đưa ra giải pháp giữ chân dựa trên các đặc điểm của khách hàng."""
    score = row['Churn_Score']
    contract = row['Contract']
    internet = row['InternetService']
    charges = row['MonthlyCharges']
    tenure = row['Tenure']
    
    if score >= 0.8:
        if contract == 'Month-to-month' and internet == 'Fiber optic':
            return "Ưu đãi đặc biệt: Nâng cấp miễn phí lên gói 1 năm (giảm 15% cước) + Tặng thêm 5GB Data. (Chủ động gọi điện)"
        elif charges > 100 and tenure < 12:
            return "Giảm cước tháng 20% trong 3 tháng đầu. (Gửi SMS cá nhân hóa)"
        else:
            return "Gói bảo hiểm thiết bị miễn phí 6 tháng. (Tiếp cận qua Email cá nhân)"
    elif 0.6 <= score < 0.8:
        if contract == 'Month-to-month':
            return "Đề xuất chuyển đổi sang Hợp đồng 1 năm với ưu đãi data tăng gấp đôi. (Tự động hóa qua App)"
        else:
            return "Khảo sát ngắn (CSAT) về chất lượng dịch vụ Internet hiện tại. (Pop-up trong ứng dụng)"
    else:
        return "Theo dõi định kỳ. Không cần can thiệp khẩn cấp."

# Áp dụng hàm để tạo cột giải pháp
df_churn['Retention_Strategy'] = df_churn.apply(suggest_retention_strategy, axis=1)

# --- 3. Giao diện Streamlit ---

st.set_page_config(page_title="Softbank AI Retention Dashboard", layout="wide")

st.title("🛰️ Giải Pháp Giữ Chân Khách Hàng AI - Softbank Corp.")
st.markdown("---")
st.markdown("Dashboard này hiển thị kết quả dự đoán nguy cơ rời bỏ (Churn Score) và các giải pháp giữ chân được cá nhân hóa cho từng nhóm khách hàng.")

## Phần 1: Tổng quan và Phân tích Nguy cơ

st.header("1. Phân Tích Nguy Cơ Tổng Quan")

# Định nghĩa ngưỡng rủi ro
RISK_THRESHOLD = 0.60
high_risk_customers = df_churn[df_churn['Churn_Score'] >= RISK_THRESHOLD]

col1, col2, col3 = st.columns(3)

col1.metric(label="Tổng Khách Hàng", value=len(df_churn))
col2.metric(label="Khách Hàng Rủi Ro Cao (Score > 60%)", 
            value=len(high_risk_customers),
            delta=f"{len(high_risk_customers) / len(df_churn) * 100:.2f}%")
col3.metric(label="Nguy Cơ Chịu Ảnh Hưởng Cao Nhất", value=high_risk_customers['Contract'].mode()[0])

st.markdown("---")

# Biểu đồ phân phối Churn Score
st.subheader("Phân Phối Churn Score")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_churn['Churn_Score'], bins=30, kde=True, ax=ax)
ax.axvline(RISK_THRESHOLD, color='red', linestyle='--', label=f'Ngưỡng Rủi Ro ({RISK_THRESHOLD})')
ax.set_title('Phân phối Xác suất Rời bỏ Khách hàng')
ax.set_xlabel('Churn Score (0.0 - 1.0)')
ax.legend()
st.pyplot(fig)
# 

## Phần 2: Danh Sách Khách Hàng Cần Can Thiệp

st.header("2. Danh Sách Khách Hàng Rủi Ro Cao & Giải Pháp")

# Sắp xếp và lọc khách hàng rủi ro
display_cols = ['CustomerID', 'Churn_Score', 'Tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'Retention_Strategy']
top_risk_df = high_risk_customers.sort_values(by='Churn_Score', ascending=False)

st.dataframe(top_risk_df[display_cols], height=350, use_container_width=True,
             column_config={
                 "Retention_Strategy": st.column_config.TextColumn("Giải Pháp Giữ Chân Đề Xuất", width="large")
             })

## Phần 3: Phân tích Giải pháp

st.header("3. Phân Bổ Các Giải Pháp Đề Xuất")

# Đếm số lượng giải pháp được đề xuất
strategy_counts = top_risk_df['Retention_Strategy'].value_counts().reset_index()
strategy_counts.columns = ['Strategy', 'Count']

# Biểu đồ cột ngang
fig_strat, ax_strat = plt.subplots(figsize=(10, 5))
sns.barplot(x='Count', y='Strategy', data=strategy_counts, palette="viridis", ax=ax_strat)
ax_strat.set_title('Tần suất các Giải pháp Giữ chân được AI đề xuất')
ax_strat.set_xlabel('Số lượng Khách hàng')
ax_strat.set_ylabel('Giải pháp')
st.pyplot(fig_strat)
#
