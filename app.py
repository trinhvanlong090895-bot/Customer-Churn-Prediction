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
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns # Cần thiết cho các biểu đồ phân tích

# --- 1. Hàm Gợi ý Giải pháp AI Cá nhân hóa ---

def suggest_retention_strategy(row):
    """
    Hàm này đại diện cho logic nghiệp vụ hậu mô hình, đưa ra giải pháp giữ chân 
    CÁ NHÂN HÓA dựa trên Churn Score và các đặc điểm rủi ro chính.
    """
    score = row['Churn_Score']
    
    # Lấy các đặc điểm rủi ro chính từ dữ liệu thô
    contract = row.get('Contract', 'Month-to-month') 
    charges = row.get('MonthlyCharges', 0)
    tenure = row.get('tenure', 0)
    internet_service = row.get('InternetService', 'No')
    payment_method = row.get('PaymentMethod', 'Mailed check')
    
    is_fiber = (internet_service == 'Fiber optic')
    
    # LOGIC ĐỀ XUẤT GIẢI PHÁP
    
    if score >= 0.75:
        # Nhóm RỦI RO CỰC CAO (Ưu tiên can thiệp bằng nhân viên)
        if contract == 'Month-to-month' and is_fiber:
            return "Ưu đãi Vàng: Nâng cấp miễn phí lên gói 1 năm (giảm 15% cước) + Tặng thêm 5GB Data. (CSO gọi điện)"
        elif charges > 100 and tenure < 12:
            return "Giảm cước tháng 20% trong 6 tháng đầu + Đảm bảo chất lượng dịch vụ. (Team Sales)"
        elif payment_method == 'Electronic check':
             return "Chuyển đổi phương thức thanh toán sang Bank Transfer + Tặng Coupon 3,000 Yên."
        else:
            return "Ưu đãi Bí mật: Gói dịch vụ độc quyền (Streaming/Game) miễn phí 3 tháng. (Team Marketing)"
            
    elif 0.5 <= score < 0.75:
        # Nhóm RỦI RO CAO (Sử dụng tự động hóa)
        if contract == 'Month-to-month':
            return "Đề xuất chuyển đổi sang Hợp đồng 1 năm với ưu đãi data/tốc độ tăng gấp đôi. (Thông báo App/SMS)"
        elif charges > 90:
            return "Tối ưu hóa gói cước: Tự động đề xuất gói rẻ hơn với tính năng tương đương. (Email Marketing tự động)"
        else:
            return "Khảo sát ngắn CSAT về chất lượng dịch vụ hiện tại để tìm kiếm vấn đề. (Pop-up trong ứng dụng)"
            
    else:
        # Nhóm RỦI RO THẤP (Theo dõi định kỳ)
        return "Theo dõi định kỳ 30 ngày. Gửi nội dung giá trị để tăng gắn kết."

# --- 2. Tải Tài nguyên (Đã được gói gọn trong file .pkl) ---
# Tải model, scaler và feature_names
try:
    # Thay đổi để tải retention_model.pkl (chứa tất cả model, scaler, features)
    with open('retention_model.pkl', 'rb') as file:
        model_assets = pickle.load(file)
    MODEL = model_assets['model']
    SCALER = model_assets['scaler']
    FEATURES = model_assets['features'] # Tên các cột đầu vào mô hình
    
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file 'retention_model.pkl'. Vui lòng chạy 'train_model.py' trước.")
    st.stop()


st.title("📊 Dự đoán tỷ lệ khách hàng rời bỏ dịch vụ AI - SOFTBANK")
st.write("Dự đoán và đưa ra giải pháp giữ chân khách hàng dựa trên mô hình Machine Learning.")
st.markdown("---")


uploaded_file = st.file_uploader("📥 Tải tệp CSV Telco Customer Churn", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📄 Đầu vào Dữ liệu:")
    st.dataframe(df_raw.head())
    
    # --- 3. TIỀN XỬ LÝ DỮ LIỆU ĐỂ DỰ ĐOÁN (Đồng bộ với train_model.py) ---
    
    df = df_raw.copy()
    
    # Loại bỏ các hàng có TotalCharges rỗng
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=['TotalCharges'], inplace=True) 
    
    # Lấy các cột cho dự đoán
    df_predict = df.drop(['customerID', 'gender', 'Churn'], axis=1, errors='ignore') 
    
    # Mã hóa One-Hot
    df_processed = pd.get_dummies(df_predict, drop_first=True)
    
    # Đồng bộ với cột của mô hình (tạo các cột bị thiếu và sắp xếp lại)
    missing_cols = set(FEATURES) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0
    df_processed = df_processed[FEATURES]
    
    # Tỉ lệ (Scaling) - Cần loại bỏ các cột đã được One-Hot trước khi Scale
    numerical_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[numerical_cols_to_scale] = SCALER.transform(df_processed[numerical_cols_to_scale])

    # Dự đoán
    proba = MODEL.predict_proba(df_processed)[:, 1]
    
    # Gắn Churn Score vào DataFrame kết quả (df_result)
    df_result = df.copy()
    df_result["Churn_Score"] = proba

    # --- 4. ÁP DỤNG LOGIC AI ĐỂ TẠO GIẢI PHÁP & HIỂN THỊ CÁ NHÂN HÓA ---
    
    df_result['Retention_Strategy'] = df_result.apply(suggest_retention_strategy, axis=1)

    st.subheader("🔍 Kết quả Dự đoán & Giải pháp Cá nhân hóa:")
    
    # Hiển thị Top khách hàng rủi ro cao
    risk_threshold = 0.70
    high_risk_df = df_result[df_result["Churn_Score"] > risk_threshold].sort_values(by="Churn_Score", ascending=False)
    
    st.dataframe(
        high_risk_df[['customerID', 'Churn_Score', 'tenure', 'MonthlyCharges', 'Contract', 'Retention_Strategy']].head(15),
        column_config={
             "Churn_Score": st.column_config.ProgressColumn(
                 "Churn Score",
                 format="%.2f",
                 min_value=0.0,
                 max_value=1.0,
             ),
             "Retention_Strategy": st.column_config.TextColumn("Giải Pháp Giữ Chân Đề Xuất (AI)", width="large")
        },
        use_container_width=True
    )

    st.markdown("---")
    
    # --- 5. PHÂN TÍCH TỔNG QUAN (Phần Bổ sung cho Báo cáo) ---
    st.header("5. Phân Tích Tổng Quan Nguy Cơ Rời Bỏ & Hướng Khắc Phục")
    
    # TÍNH CHỈ SỐ
    churn_risk_group = df_result[df_result["Churn_Score"] > risk_threshold]
    
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    col_metric_1.metric("Tổng Khách Hàng Rủi Ro Cao", len(churn_risk_group))
    col_metric_2.metric("Tỷ Lệ Rủi Ro (Score > 0.7)", f"{len(churn_risk_group) / len(df_result) * 100:.2f}%")
    col_metric_3.metric("Rủi Ro Cao Nhất Đến Từ Hợp đồng", churn_risk_group['Contract'].mode()[0])

    st.markdown("#### Biểu đồ phân tích Nguyên nhân cốt lõi:")
    
    col_chart_1, col_chart_2 = st.columns(2)
    
    # Biểu đồ 1: Phân tích Rủi ro theo Hợp đồng
    with col_chart_1:
        st.subheader("Rủi ro theo Loại Hợp đồng")
        fig, ax = plt.subplots(figsize=(6, 4))
        # Chỉ lấy dữ liệu Hợp đồng từ nhóm Rủi ro Cao
        sns.countplot(x='Contract', data=churn_risk_group, ax=ax, palette='Set1', order=churn_risk_group['Contract'].value_counts().index)
        ax.set_title('Phân bổ Rủi ro theo Loại Hợp đồng')
        ax.set_xlabel('Loại Hợp đồng')
        ax.set_ylabel('Số lượng Khách hàng Rủi ro')
        st.pyplot(fig)
        # 

    # Biểu đồ 2: Phân tích Rủi ro theo hình thức Thanh toán
    with col_chart_2:
        st.subheader("Rủi ro theo Hình thức Thanh toán")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        payment_order = churn_risk_group['PaymentMethod'].value_counts().index
        sns.countplot(y='PaymentMethod', data=churn_risk_group, ax=ax2, order=payment_order, palette='Set2')
        ax2.set_title('Phân bổ Rủi ro theo Hình thức Thanh toán')
        ax2.set_xlabel('Số lượng Khách hàng Rủi ro')
        ax2.set_ylabel('Hình thức Thanh toán')
        st.pyplot(fig2)
        # 

    st.markdown("### 📝 Hướng Khắc phục Tổng Quan (Dựa trên Phân tích):")
    
    st.error("1. Tập trung vào Khách hàng Hợp đồng **'Month-to-month'**: Phân khúc này chiếm tỷ lệ rủi ro cao nhất. Cần thiết lập các chương trình khuyến mãi chuyển đổi (Migration offers) hấp dẫn để kéo họ sang hợp đồng dài hạn.")
    st.warning("2. Tối ưu hóa Thanh toán **'Electronic Check'**: Hình thức này luôn đi kèm với rủi ro cao. Softbank nên thúc đẩy các phương thức thanh toán tự động khác (Bank Transfer/Credit Card) bằng các ưu đãi để giảm sự phụ thuộc vào Electronic Check.")
    st.info("3. **Can thiệp Sớm vào Phí Hàng tháng (Monthly Charges):** Sử dụng danh sách khách hàng rủi ro để xác định những người có cước cao nhưng ít sử dụng dịch vụ giá trị gia tăng, từ đó đề xuất gói cước tối ưu hơn để giảm cảm giác 'bị đắt'.")
import sys
!{sys.executable} -m pip install streamlit
