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
# --- PHẦN 2: GIẢI PHÁP GIỮ CHÂN KHÁCH HÀNG (SOFTBANK RETENTION ACTION) ---
st.markdown("---")
st.title("🛡️ Chiến lược Giữ chân Khách hàng (SoftBank Action Center)")
st.write("Hệ thống tự động đề xuất giải pháp dựa trên hành vi khách hàng và hệ sinh thái SoftBank.")

# 1. Lọc danh sách khách hàng rủi ro cao để xử lý
# Ngưỡng 0.7 là khách hàng có xác suất rời mạng trên 70%
high_risk_customers = df[df > 0.7].copy()

if high_risk_customers.empty:
    st.success("Tuyệt vời! Hiện tại không có khách hàng nào trong nhóm rủi ro cao.")
else:
    st.warning(f"⚠️ Cảnh báo: Tìm thấy **{len(high_risk_customers)}** khách hàng có nguy cơ rời bỏ SoftBank.")

    # 2. Xây dựng Logic Đề xuất Giải pháp (Recommendation Engine)
    # Hàm này sẽ gán các ưu đãi cụ thể của SoftBank dựa trên đặc điểm khách hàng
    def generate_softbank_offer(row):
        offers =[]
        
        # Kịch bản A: Nhạy cảm về giá (Cước cao + Hợp đồng ngắn hạn)
        # -> Đề xuất chuyển xuống thương hiệu giá rẻ hơn của SoftBank
        if row['MonthlyCharges'] > 80 and row['Contract'] == 'Month-to-month':
            offers.append("📉 Chuyển đổi sang **LINEMO** (20GB/tháng) hoặc **Y!mobile**")
            offers.append("💰 Tặng 3,000 điểm **PayPay** nếu gia hạn")

        # Kịch bản B: Khách hàng dùng Internet cáp quang (Fiber optic)
        # -> Tăng tính gắn kết bằng gói Combo (Mobile + Điện + Net)
        elif row == 'Fiber optic':
            offers.append("🏠 Kích hoạt gói **Ouchi Wari** (Giảm giá Combo Điện/Net)")
            offers.append("🎁 Tặng gói Yahoo! Premium miễn phí 6 tháng")

        # Kịch bản C: Khách hàng gặp vấn đề kỹ thuật (Có gọi TechSupport)
        # -> Cần chăm sóc con người (Human touch)
        elif row == 'Yes':
            offers.append("📞 **Priority Call:** CSKH gọi lại hỗ trợ kỹ thuật trong 1h")
            offers.append("🔧 Kiểm tra thiết bị/SIM miễn phí tại SoftBank Shop")

        # Kịch bản D: Khách hàng lâu năm (Tenure > 24 tháng)
        # -> Tri ân lòng trung thành
        elif row['tenure'] > 24:
            offers.append("💎 Nâng hạng thành viên **SoftBank Premium**")
            offers.append("🎟️ Tặng vé xem bóng chày (SoftBank Hawks)")

        # Mặc định
        else:
            offers.append("📩 Gửi khảo sát hài lòng & Tặng Coupon 500 Yên")

        return " + ".join(offers)

    # Áp dụng logic vào DataFrame
    with st.spinner('Đang phân tích hành vi và tạo đề xuất giữ chân...'):
        high_risk_customers = high_risk_customers.apply(generate_softbank_offer, axis=1)

    # 3. Hiển thị Dashboard hành động cho nhân viên
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Danh sách hành động cụ thể")
        # Hiển thị các cột quan trọng để nhân viên nắm bắt nhanh
        st.dataframe(high_risk_customers)
    
    with col2:
        st.subheader("📊 Thống kê giải pháp")
        # Biểu đồ phân bố các loại giải pháp
        # Lấy action đầu tiên trong chuỗi để thống kê
        action_counts = high_risk_customers.apply(lambda x: x.split('+').strip()).value_counts()
        st.bar_chart(action_counts)

    # 4. Mô phỏng Gửi Email tự động (GenAI Simulation)
    st.markdown("### 📧 Gửi Email Cá nhân hóa (GenAI Preview)")
    st.write("Hệ thống tự động soạn thảo email dựa trên lý do rời mạng của từng khách hàng.")
    
    # Widget chọn khách hàng để demo
    selected_cust_id = st.selectbox("Chọn ID khách hàng để xem trước Email:", high_risk_customers.head(10))
    
    if selected_cust_id:
        # Lấy thông tin khách hàng được chọn
        cust_info = [high_risk_customers == selected_cust_id].iloc
        
        # Template Email mô phỏng
        email_content = f"""
        **To:** {cust_info}@softbank.ne.jp
        **Subject:** 🎁 Món quà đặc biệt dành riêng cho bạn từ SoftBank!
        
        Kính gửi Quý khách hàng,
        
        Cảm ơn bạn đã đồng hành cùng SoftBank trong suốt {cust_info['tenure']} tháng qua. 
        Chúng tôi hiểu rằng bạn có thể đang cân nhắc về dịch vụ (Dự báo rủi ro: {cust_info:.1%}).
        
        Để tri ân và hỗ trợ bạn tốt hơn, SoftBank trân trọng gửi tặng bạn ưu đãi độc quyền:
        
        👉 **{cust_info}**
        
        Vui lòng mở ứng dụng **My SoftBank** hoặc liên kết ví **PayPay** để nhận ưu đãi này ngay hôm nay.
        
        Trân trọng,
        Đội ngũ Chăm sóc Khách hàng SoftBank Corp.
        """
        
        # Hiển thị nội dung email trong khung thông báo
        st.info(email_content)
        
        # Nút giả lập gửi
        if st.button(f"🚀 Gửi ưu đãi ngay cho {selected_cust_id}"):
            st.success(f"Đã gửi email thành công đến {selected_cust_id}! Dữ liệu đã được cập nhật vào hệ thống CRM.")
