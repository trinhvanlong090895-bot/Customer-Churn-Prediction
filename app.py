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
    import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# --- 1. Tải và Làm sạch Dữ liệu ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Xử lý cột TotalCharges: chuyển sang số và điền NaN (từ khách hàng mới) bằng 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Loại bỏ customerID và cột 'gender' (vì ít tác động trong mô hình này)
    df.drop(['customerID', 'gender'], axis=1, inplace=True) 
    
    return df

# --- 2. Tiền xử lý Dữ liệu (Encoding) ---
def preprocess_data(df):
    # Sao chép để tránh cảnh báo SettingWithCopyWarning
    df_processed = df.copy()

    # Mã hóa nhị phân (Yes/No và SeniorCitizen)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'Churn']
    for col in binary_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            # Xử lý trường hợp có 'No phone service' hoặc 'No internet service'
            unique_vals = df_processed[col].unique()
            if 'No phone service' in unique_vals:
                df_processed[col] = df_processed[col].replace('No phone service', 'No')
            if 'No internet service' in unique_vals:
                df_processed[col] = df_processed[col].replace('No internet service', 'No')
                
            df_processed[col] = le.fit_transform(df_processed[col])

    # Mã hóa One-Hot cho các biến phân loại còn lại
    categorical_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

# --- 3. Huấn luyện Mô hình ---
def train_model(df_processed):
    # Chia dữ liệu
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Chuẩn hóa biến số
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Huấn luyện Random Forest Classifier (sử dụng class_weight để xử lý mất cân bằng lớp)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                                   class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Đánh giá mô hình
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, X.columns, scaler

# --- Thực thi và Lưu trữ ---
if __name__ == '__main__':
    # Đảm bảo file CSV đã được tải lên
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    
    df_clean = load_and_clean_data(file_path)
    df_preprocessed = preprocess_data(df_clean)
    
    # Lưu lại DataFrame đã xử lý (cần cho Streamlit để dự đoán trên toàn bộ tập dữ liệu)
    df_preprocessed.to_csv('processed_data.csv', index=False)
    
    model, features, scaler = train_model(df_preprocessed)
    
    # Lưu mô hình, các tên cột và scaler
    with open('retention_model.pkl', 'wb') as file:
        pickle.dump({
            'model': model,
            'features': features.tolist(),
            'scaler': scaler
        }, file)
    
    print("Huấn luyện mô hình và lưu file 'retention_model.pkl' thành công.")
