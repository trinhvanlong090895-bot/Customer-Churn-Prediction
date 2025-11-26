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
# --- BẮT ĐẦU PHẦN CODE MỚI (Dán tiếp theo dòng 49) ---

st.markdown("---")
st.title("🛡️ Chiến lược Giữ chân Khách hàng (SoftBank Action Center)")

# 1. Lọc danh sách khách hàng rủi ro cao (Churn Score > 70%)
# Lưu ý: Cột 'Churn_Score' đã được tạo ở dòng 41 trong code cũ của bạn
# Cần có DataFrame chứa Churn_Score và các cột khác đã được tiền xử lý
# Giả định: 'df_with_churn_score' là DataFrame đã có cột 'Churn_Score' và các đặc trưng
# Vì không có 'df_with_churn_score' được tạo ở đây, tôi sẽ sử dụng 'df' giả định từ cell trước và tạo cột 'Churn_Score' mẫu

# --- BỔ SUNG: Tạo Churn_Score giả định và DataFrame 'df' nếu chưa có --- 
# Dựa trên kernel state, 'df' và 'Churn_Score' chưa tồn tại trực tiếp trong cell này.
# Để code chạy được, ta cần tạo 'df' và 'Churn_Score' từ 'X_data' và 'clf_model' đã huấn luyện.
# Tái cấu trúc lại để lấy df từ context hoặc tạo df giả định nếu đây là một phần độc lập

# Lấy dữ liệu mẫu ban đầu để tạo lại DataFrame
# (Giả sử bạn đã có df_original từ bước 2 của notebook đầu tiên)
# Nếu không, cần load lại hoặc truyền vào từ các cell trước

# Để đơn giản và làm cho phần này chạy được, tôi sẽ mô phỏng lại df và Churn_Score
# THAY THẾ BẰNG CÁCH LẤY CHURN_SCORE THẬT TỪ MÔ HÌNH CỦA BẠN!

# Lấy các biến từ môi trường global nếu đã được định nghĩa ở các cell trước
# Giả định X_data, y_labels, clf_model, feature_names đã được định nghĩa

# Tạo lại DataFrame tương tự df ban đầu để sử dụng các cột string
# Đây là một giải pháp tạm thời, cần thay thế bằng DataFrame gốc với các cột gốc

# Lấy dữ liệu mẫu từ cell xuPLtbD6VpKh
data_sample = {
    'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK', '7795-CFOCW', '9237-HQITU', '9305-CDSKC', '2809-LSDNY'],
    'gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Male'],
    'SeniorCitizen': [0, 0, 0, 0, 0, 0, 0],
    'Partner': ['Yes', 'No', 'No', 'No', 'No', 'No', 'No'],
    'Dependents': ['No', 'No', 'No', 'No', 'No', 'No', 'No'],
    'tenure': [1, 34, 2, 45, 2, 8, 22],
    'PhoneService': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'MultipleLines': ['No phone service', 'No', 'No', 'No phone service', 'No', 'No phone service', 'No'],
    'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'DSL'],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Two year'],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.3, 70.7, 52.55, 20.25],
    'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', '151.65', '405.35', '458.55'],
    'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}
df_streamlit = pd.DataFrame(data_sample)

# Chuyển đổi TotalCharges sang số
df_streamlit['TotalCharges'] = pd.to_numeric(df_streamlit['TotalCharges'], errors='coerce')
df_streamlit.fillna(df_streamlit.mean(numeric_only=True), inplace=True)

# Sử dụng preprocessor từ cell xuPLtbD6VpKh để xử lý df_streamlit
# Cần phải tạo lại preprocessor nếu không có sẵn trong global scope
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

preprocessor_for_inference = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop'
)

# Fit preprocessor on dummy data to get consistent columns (or on training data originally)
# For this example, we fit it on the sample data itself
X_processed_sample = preprocessor_for_inference.fit_transform(df_streamlit.drop(['customerID', 'Churn'], axis=1))

# Dự đoán churn score từ mô hình đã huấn luyện (clf_model)
# Chú ý: clf_model cần phải có sẵn trong kernel state
if 'clf_model' in globals():
    churn_proba = clf_model.predict_proba(X_processed_sample)[:, 1]
    df_streamlit['Churn_Score'] = churn_proba
else:
    # Fallback nếu clf_model không được tìm thấy (chỉ để code chạy)
    df_streamlit['Churn_Score'] = np.random.rand(len(df_streamlit)) # Mẫu ngẫu nhiên
    print("Cảnh báo: Không tìm thấy 'clf_model', sử dụng Churn_Score ngẫu nhiên.")

high_risk_customers = df_streamlit[df_streamlit['Churn_Score'] > 0.7].copy()

if high_risk_customers.empty:
    st.success("Tuyệt vời! Hiện tại không có khách hàng nào trong nhóm rủi ro cao.")
else:
    st.warning(f"⚠️ Cảnh báo: Tìm thấy **{len(high_risk_customers)}** khách hàng có nguy cơ rời bỏ SoftBank.")

    # 2. Hàm logic đề xuất ưu đãi (SoftBank Recommendation Engine)
    def generate_softbank_offer(row):
        offers = [] # Khởi tạo danh sách offers

        # Kịch bản 1: Giá cước cao + Hợp đồng ngắn hạn -> Đề xuất gói cước rẻ hơn (LINEMO/Y!mobile)
        if row['MonthlyCharges'] > 80 and row['Contract'] == 'Month-to-month':
            offers.append("📉 Chuyển sang **LINEMO** (20GB) hoặc **Y!mobile**")
            offers.append("💰 Tặng 3,000 điểm **PayPay** nếu gia hạn")

        # Kịch bản 2: Dùng Fiber Optic -> Tăng gắn kết bằng hệ sinh thái (Điện + Net)
        # Sửa lỗi: Cần kiểm tra cột 'InternetService' chứ không phải 'row' trực tiếp
        elif row['InternetService'] == 'Fiber optic':
            offers.append("🏠 Kích hoạt **Ouchi Wari** (Giảm giá Combo Điện/Net)")
            offers.append("🎁 Tặng Yahoo! Premium miễn phí 6 tháng")

        # Kịch bản 3: Có gọi hỗ trợ kỹ thuật -> Cần chăm sóc đặc biệt
        # Sửa lỗi: Cần một cột cụ thể để kiểm tra việc gọi hỗ trợ kỹ thuật, ví dụ 'TechSupport'
        # Giả sử có cột 'TechSupport' và giá trị 'Yes' biểu thị có hỗ trợ
        # Nếu không có, cần bổ sung cột này vào dữ liệu hoặc dùng logic khác
        elif 'TechSupport' in row and row['TechSupport'] == 'Yes': # Thay 'row == 'Yes'' bằng logic hợp lệ
            offers.append("📞 **Priority Call:** CSKH gọi lại hỗ trợ trong 1h")
            offers.append("🔧 Kiểm tra thiết bị miễn phí tại SoftBank Shop")

        # Kịch bản 4: Khách hàng lâu năm (> 2 năm) -> Tri ân
        elif row['tenure'] > 24:
            offers.append("💎 Nâng hạng **SoftBank Premium**")
            offers.append("🎟️ Tặng vé xem bóng chày (SoftBank Hawks)")

        # Mặc định cho các nhóm còn lại
        else:
            offers.append("📩 Tặng Coupon 500 Yên qua ứng dụng My SoftBank")

        return " + ".join(offers)

    # Áp dụng hàm trên vào dữ liệu
    # Sử dụng st.spinner để báo hiệu đang xử lý
    with st.spinner('Đang phân tích hành vi và tạo đề xuất...'):
        # Cần tạo một cột mới để lưu trữ các đề xuất
        high_risk_customers['Offer_Recommendation'] = high_risk_customers.apply(generate_softbank_offer, axis=1)

    # 3. Hiển thị bảng hành động
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Danh sách hành động cụ thể")
        # Chỉ hiện các cột quan trọng để nhân viên dễ nhìn
        display_cols = ['customerID', 'Churn_Score', 'Offer_Recommendation', 'tenure', 'MonthlyCharges', 'Contract', 'InternetService']
        # Kiểm tra xem các cột có tồn tại không trước khi hiển thị để tránh lỗi
        valid_cols = [c for c in display_cols if c in high_risk_customers.columns]
        st.dataframe(high_risk_customers[valid_cols])

    with col2:
        st.subheader("📊 Thống kê giải pháp")
        # Đếm số lượng từng loại giải pháp chính
        # Cần đếm trên cột 'Offer_Recommendation'
        action_counts = high_risk_customers['Offer_Recommendation'].value_counts().head(5)
        st.bar_chart(action_counts)

    # 4. Tính năng GenAI (Mô phỏng soạn Email)
    st.markdown("### 📧 Soạn thảo Email tự động (GenAI Simulation)")

    # Chọn khách hàng từ danh sách rủi ro
    # Sửa lỗi: selectbox cần một list các giá trị để chọn
    selected_cust_id = st.selectbox("Chọn ID khách hàng để gửi ưu đãi:", high_risk_customers['customerID'].tolist())

    if selected_cust_id:
        # Lấy thông tin dòng dữ liệu của khách hàng đó
        # Sửa lỗi: Lấy dòng dựa trên customerID và .iloc[0] để có Series
        cust_info = high_risk_customers[high_risk_customers['customerID'] == selected_cust_id].iloc[0]

        # Soạn nội dung email
        email_content = f"""
        ----------------------------------------------------
        **To:** {cust_info['customerID']}@softbank.ne.jp
        **Subject:** Ưu đãi đặc biệt dành riêng cho bạn!

        Kính gửi Quý khách,

        Cảm ơn bạn đã gắn bó với SoftBank suốt {cust_info['tenure']} tháng qua.
        Hệ thống nhận thấy bạn đang gặp một số bất tiện (Điểm rủi ro: {cust_info['Churn_Score']:.2f}).

        Chúng tôi xin gửi tặng bạn gói ưu đãi được thiết kế riêng:
        👉 {cust_info['Offer_Recommendation']}

        Vui lòng mở ứng dụng PayPay để nhận ngay.
        ----------------------------------------------------
        """
        st.code(email_content, language='text')

        if st.button("🚀 Gửi Email Giữ Chân"):
            st.success(f"Đã gửi ưu đãi thành công tới khách hàng {selected_cust_id}!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle

# --- 1. Tải và Làm sạch Dữ liệu ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Xử lý TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Mã hóa biến mục tiêu Churn
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn']) # Yes=1, No=0
    
    # Loại bỏ customerID và gender 
    df.drop(['customerID', 'gender'], axis=1, inplace=True) 
    
    return df

# --- 2. Định nghĩa Quy trình Tiền xử lý (Preprocessor) ---
def create_preprocessor(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor, categorical_cols, numerical_cols

# --- 3. Huấn luyện Mô hình Cuối cùng (Pipeline) ---
def train_full_pipeline(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    preprocessor, categorical_cols, numerical_cols = create_preprocessor(X)
    
    # Xây dựng Pipeline hoàn chỉnh
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'))
    ])
    
    full_pipeline.fit(X, y)
    
    return full_pipeline, categorical_cols, numerical_cols, X.columns.tolist()

# --- Thực thi và Lưu trữ ---
if __name__ == '__main__':
    # ĐƯỜNG DẪN CỐ ĐỊNH CHO MỤC ĐÍCH HUẤN LUYỆN
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    
    df_clean = load_and_clean_data(file_path)
    
    pipeline_model, categorical_cols, numerical_cols, original_features = train_full_pipeline(df_clean)
    
    # Lưu Pipeline và các biến cần thiết
    with open('full_churn_pipeline.pkl', 'wb') as file:
        pickle.dump({
            'pipeline': pipeline_model,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'original_features': original_features
        }, file)
    
    print("\n[THÀNH CÔNG] Đã lưu mô hình Pipeline hoàn chỉnh vào 'full_churn_pipeline.pkl'.")
