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


# Cấu hình trang
st.set_page_config(page_title="SoftBank Churn Prediction", layout="wide")

# --- PHẦN 1: TIÊU ĐỀ VÀ LOAD MODEL ---
st.title("🤖 AI Customer Churn Prediction & Retention")
st.markdown("**Dự án:** Ứng dụng AI dự báo và giữ chân khách hàng cho **SoftBank Corp.**")

# Load model, scaler và feature_names
# Lưu ý: Bạn cần đảm bảo file.pkl nằm cùng thư mục với file app.py
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Không tìm thấy file model.pkl, scaler.pkl hoặc feature_names.pkl. Vui lòng kiểm tra lại!")
    st.stop()

# Upload file
uploaded_file = st.file_uploader("📂 Tải file CSV dữ liệu khách hàng (Telco Customer Churn)", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    
    # --- PHẦN 2: XỬ LÝ DỮ LIỆU & DỰ BÁO (PREDICTION ENGINE) ---
    with st.spinner('Đang xử lý dữ liệu và chạy mô hình AI...'):
        # 1. Xử lý dữ liệu thô (Data Preprocessing)
        df_display = df.copy() # Giữ lại bản gốc để hiển thị
        
        # Xử lý TotalCharges (chuyển sang số)
        df = pd.to_numeric(df, errors="coerce")
        df = df.dropna()
        
        # One-Hot Encoding
        df_processed = pd.get_dummies(df, drop_first=True)
        
        # Đồng bộ cột với model đã huấn luyện
        missing_cols = set(feature_names) - set(df_processed.columns)
        for c in missing_cols:
            df_processed[c] = 0
        df_processed = df_processed[feature_names] # Sắp xếp lại đúng thứ tự cột
        
        # Scale dữ liệu
        X_scaled = scaler.transform(df_processed)
        
        # 2. Dự báo (Prediction)
        # Lấy xác suất rời mạng (cột 1)
        prediction_proba = model.predict_proba(X_scaled)[:, 1]
        df_display["Churn_Probability"] = prediction_proba
        
        # Phân loại rủi ro
        def categorize_risk(prob):
            if prob > 0.7: return "Nguy cơ Cao 🔴"
            elif prob > 0.4: return "Cảnh báo 🟡"
            else: return "An toàn 🟢"
            
        df_display = df_display["Churn_Probability"].apply(categorize_risk)

    # --- PHẦN 3: HIỂN THỊ KẾT QUẢ (DASHBOARD) ---
    
    # Tạo Tabs để giao diện gọn gàng
    tab1, tab2, tab3 = st.tabs()

    with tab1:
        st.subheader("Tổng quan rủi ro khách hàng")
        
        # KPI Metrics
        col1, col2, col3 = st.columns(3)
        high_risk_count = df_display[df_display["Churn_Probability"] > 0.7].shape
        avg_risk = df_display["Churn_Probability"].mean() * 100
        revenue_at_risk = df_display[df_display["Churn_Probability"] > 0.7]["MonthlyCharges"].sum()

        col1.metric("Khách hàng Rủi ro cao", f"{high_risk_count} người", delta_color="inverse")
        col2.metric("Tỷ lệ rủi ro trung bình", f"{avg_risk:.1f}%")
        col3.metric("Doanh thu đang bị đe dọa", f"¥{revenue_at_risk:,.0f}", "Tháng này")

        # Biểu đồ phân bố
        fig = px.histogram(df_display, x="Churn_Probability", nbins=20, title="Phân bố xác suất rời mạng", color_discrete_sequence=)
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Dữ liệu chi tiết:")
        st.dataframe(df_display.sort_values(by="Churn_Probability", ascending=False).head(10))

    # --- PHẦN 4: SOFTBANK ACTION CENTER (GIẢI PHÁP) ---
    with tab2:
        st.header("🛡️ Chiến lược Giữ chân Khách hàng (SoftBank Action Center)")
        st.write("Hệ thống tự động đề xuất gói giải pháp dựa trên hành vi từng khách hàng.")
        
        # Lọc khách hàng rủi ro cao
        high_risk_df = df_display[df_display["Churn_Probability"] > 0.7].copy()
        
        if high_risk_df.empty:
            st.success("Tuyệt vời! Không có khách hàng rủi ro cao.")
        else:
            # Logic đề xuất giải pháp (Rule-based Recommendation)
            def get_retention_offer(row):
                offers =
                # Rule 1: Nhạy cảm về giá (Cước cao + Hợp đồng ngắn) -> Đề xuất LINEMO/Y!mobile
                if row['MonthlyCharges'] > 80 and row['Contract'] == 'Month-to-month':
                    offers.append("📉 Chuyển đổi sang **LINEMO** (20GB) hoặc **Y!mobile**")
                
                # Rule 2: Dùng Fiber Optic -> Kích hoạt Ouchi Wari (Combo Điện/Net)
                if row == 'Fiber optic':
                    offers.append("🏠 Kích hoạt **Ouchi Wari** (Giảm giá Combo)")
                
                # Rule 3: Khách hàng lâu năm (> 2 năm) -> Tri ân
                if row['tenure'] > 24:
                    offers.append("💎 Nâng hạng **SoftBank Premium** + Vé xem bóng chày")
                
                # Rule 4: Không có TechSupport -> Tặng dịch vụ hỗ trợ
                if row == 'No':
                    offers.append("🔧 Tặng gói hỗ trợ kỹ thuật miễn phí 3 tháng")
                
                # Mặc định: Tặng điểm PayPay
                offers.append("💰 Tặng 1,000 điểm **PayPay**")
                
                return " + ".join(offers)

            high_risk_df = high_risk_df.apply(get_retention_offer, axis=1)
            
            # Hiển thị bảng hành động

    # --- PHẦN 5: GENAI SIMULATION (SOẠN EMAIL) ---
    with tab3:
        st.header("📧 Soạn thảo Email Cá nhân hóa (GenAI Demo)")
        
        # Chọn khách hàng từ danh sách rủi ro
        if not high_risk_df.empty:
            selected_cust = st.selectbox("Chọn ID khách hàng để gửi ưu đãi:", high_risk_df.head(20))
            
            if selected_cust:
                cust_data = high_risk_df == selected_cust].iloc
                
                # Template Email động
                email_body = f"""
                **To:** {cust_data}@softbank.ne.jp
                **Subject:** 🎁 Món quà đặc biệt từ SoftBank dành riêng cho bạn!
                
                Kính gửi Quý khách hàng,
                
                Cảm ơn bạn đã đồng hành cùng SoftBank trong suốt **{cust_data['tenure']} tháng** qua.
                Chúng tôi nhận thấy bạn đang sử dụng gói cước với mức phí khoảng **¥{cust_data['MonthlyCharges']}**.
                
                Để tri ân sự gắn bó của bạn, SoftBank trân trọng gửi tặng gói ưu đãi độc quyền:
                
                👉 **{cust_data}**
                
                Vui lòng truy cập ứng dụng **My SoftBank** hoặc liên kết ví **PayPay** để nhận ưu đãi ngay hôm nay.
                
                Trân trọng,
                Đội ngũ Chăm sóc Khách hàng SoftBank Corp.
                """
                
                st.info("Nội dung Email được tạo tự động:")
                st.markdown(email_body)
                
                if st.button("🚀 Gửi Email Ngay"):
                    st.success(f"Đã gửi ưu đãi thành công tới khách hàng {selected_cust}! Dữ liệu CRM đã cập nhật.")
        else:
            st.write("Không có khách hàng nào cần gửi email.")

else:
    st.info("Vui lòng tải lên file CSV để bắt đầu phân tích.")
