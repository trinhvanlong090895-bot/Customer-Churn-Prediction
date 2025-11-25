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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import StringIO

# Thiết lập chế độ trang (tùy chọn)
st.set_page_config(layout="wide")

# --- Mô phỏng Dữ liệu và Tiền xử lý  ---
@st.cache_data
def load_and_preprocess_data():
    # Giả lập dữ liệu Telco Churn CSV 
    data = {
        'customerID':,
        'gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Male'],
        'SeniorCitizen': ,
        'Partner':,
        'Dependents':,
        'tenure': ,
        'PhoneService':,
        'MultipleLines': ['No phone service', 'No', 'No', 'No phone service', 'No', 'No phone service', 'No'],
        'InternetService':,
        'Contract':,
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.3, 70.7, 52.55, 20.25],
        'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', '151.65', ' ', ' '], # Mô phỏng giá trị trống
        'Churn':
    }
    df = pd.DataFrame(data)

    # Xử lý TotalCharges: Thay thế khoảng trắng bằng NaN và chuyển đổi sang số
    df = df.replace(' ', np.nan).astype(float)
    # Xử lý giá trị thiếu (Imputation - ví dụ: thay bằng giá trị trung bình)
    df.fillna(df.mean(), inplace=True)
    
    # Mã hóa biến mục tiêu 'Churn'
    df['Churn_Label'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Chọn các đặc trưng để mã hóa (bao gồm cả các biến được phân tích)
    categorical_features =
    
    # Lấy tên cột chỉ số (Tenure, Charges)
    numerical_features =

    # Xây dựng Pipeline cho tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )
    
    X = df.drop(, axis=1)
    y = df['Churn_Label']
    
    # Tách tập huấn luyện (vì đây là ví dụ minh họa, không cần tách test/train nghiêm ngặt)
    X_processed = preprocessor.fit_transform(X)
    
    # Lấy tên các đặc trưng sau khi mã hóa
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = list(cat_feature_names) + numerical_features
    
    return X_processed, y, final_feature_names

X_data, y_labels, feature_names = load_and_preprocess_data()

@st.cache_resource
def train_model(X, y):
    """Huấn luyện mô hình Random Forest cơ bản."""
    # Khởi tạo và huấn luyện mô hình [13]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

clf_model = train_model(X_data, y_labels)

def plot_feature_importance(model, feature_names, top_n=10):
    """Tính toán và trực quan hóa Gini Importance.[13]"""
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='#f63366')
    ax.set_xlabel('Điểm Quan trọng Gini (Gini Importance Score)')
    ax.set_title(f'Top {top_n} Đặc trưng Quan trọng Dự đoán Churn')
    ax.invert_yaxis()
    st.pyplot(fig)

# --- Giao diện Streamlit cho Feature Importance ---
st.header("1. Phân tích Động lực Churn (AI Diagnostics)")
st.subheader("Trực quan hóa Tầm quan trọng của Đặc trưng (Random Forest)")

# Slider chọn số lượng đặc trưng hiển thị
top_n_features = st.slider("Chọn số lượng đặc trưng quan trọng hiển thị", 5, len(feature_names), 10)

plot_feature_importance(clf_model, feature_names, top_n_features)
st.markdown("""
Sự trực quan hóa này cho phép các nhà quản lý nhanh chóng xác định các yếu tố thúc đẩy mô hình dự đoán churn.
Các đặc trưng có điểm Gini Importance cao nhất, như `tenure` và các biến liên quan đến `Contract`, 
được xác nhận là các đòn bẩy chính trong mô hình phân loại (như đã giả định trong phân tích dữ liệu mẫu ).
""")
