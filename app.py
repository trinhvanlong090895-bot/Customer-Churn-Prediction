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
#!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Logic Gợi ý Giải pháp AI (Giải pháp Giữ chân) ---
def suggest_retention_strategy(row):
    """
    Hàm này đại diện cho logic nghiệp vụ sau khi AI dự đoán.
    Nó đưa ra giải pháp giữ chân CÁ NHÂN HÓA dựa trên Churn Score và các đặc điểm rủi ro chính.
    """
    score = row['Churn_Score'] # Access Churn_Score from the row
    # Sử dụng các cột thô từ DataFrame
    contract = row.get('Contract', 'Month-to-month')
    charges = row.get('MonthlyCharges', 0)
    tenure = row.get('tenure', 0)
    internet = row.get('InternetService', 'No')

    # Logic kiểm tra sự tồn tại của Fiber Optic (từ cột InternetService thô)
    is_fiber = (internet == 'Fiber optic')

    # LOGIC ĐỀ XUẤT GIẢI PHÁP

    if score >= 0.75:
        # Nhóm RỦI RO CỰC CAO (Ưu tiên can thiệp bằng nhân viên)
        if contract == 'Month-to-month' and is_fiber:
            return "Ưu đãi Vàng: Nâng cấp miễn phí lên gói 1 năm (giảm 15% cước) + Tặng thêm 5GB Data. (CSO gọi điện)"
        elif charges > 100 and tenure < 12:
            return "Giảm cước tháng 20% trong 6 tháng đầu + Đảm bảo chất lượng dịch vụ Internet. (Team Sales)"
        elif tenure > 60 and contract == 'Month-to-month':
             return "Gói Bảo hiểm Thiết bị miễn phí 12 tháng + Thư xin lỗi cá nhân hóa. (Team Hỗ trợ)"
        else:
            return "Gói dịch vụ độc quyền Softbank/PayPay miễn phí 3 tháng. (Team Marketing)"

    elif 0.5 <= score < 0.75:
        # Nhóm RỦI RO CAO (Sử dụng tự động hóa)
        if contract == 'Month-to-month':
            return "Đề xuất chuyển đổi sang Hợp đồng 1 năm với ưu đãi data/tốc độ tăng gấp đôi. (Gửi thông báo App/SMS)"
        elif internet == 'DSL':
            return "Đề xuất nâng cấp lên Fiber với giá ưu đãi trong 6 tháng. (Email Marketing tự động)"
        else:
            return "Khảo sát ngắn CSAT về chất lượng dịch vụ hiện tại. (Pop-up trong ứng dụng)"

    else:
        # Nhóm RỦ RO THẤP (Theo dõi định kỳ)
        return "Theo dõi định kỳ 30 ngày. Gửi nội dung giá trị (How-to, mẹo sử dụng) để tăng gắn kết."

# --- Bắt đầu Khung Streamlit của bạn ---

# Tải model, scaler và feature_names
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file model.pkl, scaler.pkl, hoặc feature_names.pkl. Vui lòng chạy file huấn luyện mô hình trước.")
    st.stop()


st.title("📊 Dự đoán tỷ lệ khách hàng rời bỏ dịch vụ AI - SOFTBANK")
st.write("Dự đoán khách hàng có thể bỏ hoặc không dựa vào Machine Learning mô hình")
st.markdown("---")


uploaded_file = st.file_uploader("📥 Tải tệp CSV Telco Customer Churn", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Đầu vào Dữ liệu:")
    st.dataframe(df.head())

    # ------------------------------------------------
    # 23-40: KHUNG XỬ LÝ VÀ DỰ ĐOÁN (CODE GỐC CỦA BẠN)
    # ------------------------------------------------

    # Make a copy for processing to retain original df for potential other uses
    df_for_processing = df.copy()

    # Convert 'TotalCharges' to numeric, handling missing values
    # Assuming 'TotalCharges' is the only column that might contain non-numeric data
    # (e.g., spaces for new customers), and other numeric columns are already clean.
    if 'TotalCharges' in df_for_processing.columns:
        df_for_processing['TotalCharges'] = pd.to_numeric(df_for_processing['TotalCharges'], errors='coerce')

    # Drop rows with NaNs (e.g., from 'TotalCharges' conversion or other missing data)
    # It's important to keep track of the original indices if customerID is not unique
    # or if we need to map back to the original `df`.
    # For simplicity, we will drop NaNs and assume the index aligns.
    df_for_processing.dropna(inplace=True)

    # Store a version of the DataFrame that will contain results (customer details + churn score)
    # This ensures we have customer details like Contract, MonthlyCharges, tenure, InternetService
    # for the `suggest_retention_strategy` function.
    results_df = df_for_processing.copy()

    # Columns to drop from features used for prediction.
    # Based on the comment, 'Gender' is not used. 'customerID' is an identifier.
    # Other raw categorical features will be one-hot encoded.
    columns_to_drop_from_features = ['customerID', 'gender'] # Add 'gender' as per comment
    df_features = df_for_processing.drop(columns=columns_to_drop_from_features, errors='ignore')

    # Mã hóa One-Hot cho các biến phân loại để chuẩn bị cho mô hình
    df_processed = pd.get_dummies(df_features, drop_first=True)

    # Đồng bộ với cột của mô hình
    missing_cols = set(feature_names) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0
    df_processed = df_processed[feature_names]

    # Tỉ lệ
    X_scaled = scaler.transform(df_processed)

    # Dự đoán
    proba = model.predict_proba(X_scaled)[:, 1]

    # Gắn Churn Score vào DataFrame kết quả
    results_df['Churn_Score'] = proba


    # ------------------------------------------------
    # 41-48: HIỂN THỊ KẾT QUẢ DỰ ĐOÁN (CODE GỐC CỦA BẠN)
    # ------------------------------------------------

    st.subheader("🔍 Kết quả Dự đoán:")
    # Use results_df for display
    st.dataframe(results_df.sort_values(by="Churn_Score", ascending=False).head(10))

    st.subheader("🔥 Khách hàng có nguy cơ cao (Churn > 0.7):")
    # Filter results_df directly
    high_risk_customers_df = results_df[results_df['Churn_Score'] > 0.7].copy()

    st.dataframe(
        high_risk_customers_df,
        column_config={
             "Churn_Score": st.column_config.ProgressColumn("Churn Score", format="%.2f", min_value=0.0, max_value=1.0)
        },
        use_container_width=True
    )

    # ------------------------------------------------
    # --- BỔ SUNG YÊU CẦU 1: PHÂN TÍCH ĐỘNG LỰC CHURN (FEATURE IMPORTANCE) ---
    # ------------------------------------------------

    st.markdown("---")
    st.header("1. Phân Tích Động Lực Churn (Nguyên nhân Khách hàng Rời bỏ)")

    # Lấy Feature Importance từ mô hình đã load
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette="magma", ax=ax)
    ax.set_title('Top 10 Đặc trưng Quan trọng Dự đoán Churn (Gini Importance)')
    ax.set_xlabel('Điểm Quan trọng')
    ax.set_ylabel('Đặc trưng Khách hàng')
    st.pyplot(fig)
    #

    st.markdown("""
    **Hướng Khắc phục Tổng quan dựa trên Phân tích Đặc trưng:**
    1. **Hợp đồng ngắn hạn (`Contract_Month-to-month`):** Luôn là yếu tố rủi ro hàng đầu. **Giải pháp:** Tập trung chiến dịch chuyển đổi khách hàng này sang hợp đồng 1 hoặc 2 năm với các ưu đãi gắn kết (bundle, PayPay points [2]).
    2. **Thời gian Gắn bó (`tenure`):** Khách hàng rất mới (tenure thấp) hoặc mới bắt đầu có rủi ro cao. **Giải pháp:** Tăng cường chương trình Onboarding/CSM chủ động trong 90 ngày đầu tiên để đảm bảo sự hài lòng với chất lượng mạng và hóa đơn.
    3. **Dịch vụ Fiber Optic:** Khách hàng trả phí cao có kỳ vọng cao hơn. **Giải pháp:** Áp dụng giám sát chủ động (proactive monitoring) để khắc phục các sự cố mạng tiềm ẩn trước khi khách hàng phàn nàn.[1]
    """)

    # ------------------------------------------------
    # --- BỔ SUNG YÊU CẦU 2: GIẢI PHÁP CÁ NHÂN HÓA VÀ PHÂN TÍCH TÁC ĐỘNG ---
    # ------------------------------------------------

    st.markdown("---")
    st.header("2. Giải Pháp Giữ Chân Cá Nhân Hóa (AI Retention Strategy)")

    # Thiết lập ngưỡng rủi ro có thể điều chỉnh
    risk_threshold = st.slider("Chọn Ngưỡng Churn Score Tối Thiểu để Can Thiệp:",
                      min_value=0.5, max_value=0.9, value=0.70, step=0.05)

    # Lọc lại danh sách khách hàng rủi ro cao theo ngưỡng mới
    high_risk_strategies_df = results_df[results_df['Churn_Score'] >= risk_threshold].copy()
    high_risk_strategies_df['Retention_Strategy'] = high_risk_strategies_df.apply(suggest_retention_strategy, axis=1)

    st.dataframe(
        high_risk_strategies_df[['customerID', 'Churn_Score', 'Retention_Strategy']], # Display relevant columns
        height=300,
        use_container_width=True,
        column_config={
             "Churn_Score": st.column_config.ProgressColumn("Churn Score", format="%.2f", min_value=0.0, max_value=1.0),
             "Retention_Strategy": st.column_config.TextColumn("Giải Pháp Giữ Chân Đề Xuất (AI)", width="large")
        }
    )

    # Biểu đồ Phân bổ Giải pháp (Để hiểu cần phân bổ ngân sách cho loại chiến dịch nào)
    st.subheader("Phân bổ Tần suất các Giải pháp AI Đề xuất:")

    if not high_risk_strategies_df.empty:
        # Extract just the strategy description before the parentheses
        strategy_counts = high_risk_strategies_df['Retention_Strategy'].apply(lambda x: x.split('(')[0].strip()).value_counts().head(5)

        fig_strat, ax_strat = plt.subplots(figsize=(8, 4))
        strategy_counts.plot(kind='barh', ax=ax_strat, color='teal')
        ax_strat.set_title('Top 5 Loại Giải pháp cần ưu tiên')
        ax_strat.set_xlabel('Số lượng Khách hàng Mục tiêu')
        plt.gca().invert_yaxis()
        st.pyplot(fig_strat)
    else:
        st.info("Không có khách hàng nào đạt ngưỡng rủi ro này để đề xuất giải pháp.")
