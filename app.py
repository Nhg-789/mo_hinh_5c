import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from io import BytesIO

# ============================================================================
# PAGE CONFIG - PHẢI CHẠY TRƯỚC TẤT CẢ
# ============================================================================
st.set_page_config(
    page_title="📊 Phân loại Rủi ro Khách hàng",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================
@st.cache_data
def load_data(file_bytes):
    """
    Tải dữ liệu từ file CSV
    """
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {str(e)}")
        return None

# ============================================================================
# COMPONENT 1: SIDEBAR - CẤU HÌNH VÀ TẢI DỮ LIỆU
# ============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải file dữ liệu
    st.subheader("📁 Tệp dữ liệu")
    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=['csv'],
        help="Tải lên file CSV chứa dữ liệu huấn luyện"
    )
    
    # Tham số mô hình
    st.subheader("🤖 Tham số mô hình Logistic Regression")
    
    with st.expander("⚙️ Tham số nâng cao"):
        max_iter = st.slider(
            "Max iterations",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Số lần lặp tối đa để hội tụ"
        )
        
        random_state = st.number_input(
            "Random state",
            value=23,
            help="Giá trị seed cho tái tạo kết quả"
        )
        
        test_size = st.slider(
            "Tỷ lệ kiểm định (test_size)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Phần trăm dữ liệu dùng để kiểm định"
        )
    
    st.divider()
    
    # Nút huấn luyện
    col1, col2 = st.columns(2)
    with col1:
        train_button = st.button(
            "🎯 Huấn luyện mô hình",
            type="primary",
            use_container_width=True,
            help="Bấm để bắt đầu huấn luyện"
        )
    
    with col2:
        reset_button = st.button(
            "🔄 Đặt lại",
            use_container_width=True,
            help="Xóa kết quả huấn luyện"
        )
    
    if reset_button:
        st.session_state.model_trained = False
        st.session_state.model = None
        st.session_state.results = None
        st.session_state.x_test = None
        st.session_state.y_test = None
        st.session_state.yhat_test = None
        st.rerun()


# ============================================================================
# COMPONENT 2: HEADER - ĐỊNH HƯỚNG & KIỂM TRA DỮ LIỆU
# ============================================================================
st.title("📊 Ứng dụng Phân loại Rủi ro Khách hàng")
st.caption(
    "Ứng dụng sử dụng mô hình Logistic Regression để dự báo rủi ro của khách hàng "
    "dựa trên các chỉ số đánh giá chất lượng dịch vụ, năng lực, độc lập, và tính số."
)

# Kiểm tra file đã được tải
if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file CSV từ sidebar để bắt đầu")
    st.stop()

# Tải dữ liệu
file_bytes = uploaded_file.getvalue()
df = load_data(file_bytes)

if df is None:
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | {len(df)} dòng × {len(df.columns)} cột")
st.divider()


# ============================================================================
# TRAINING LOGIC
# ============================================================================
# Định nghĩa biến đầu vào X và biến mục tiêu y từ notebook
feature_columns = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3',
                   'NL4', 'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5',
                   'V6', 'TS1', 'TS2', 'TS3', 'TS4']
target_column = 'PD'

# Kiểm tra cột tồn tại
missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
if missing_cols:
    st.error(f"❌ Cột thiếu trong dữ liệu: {', '.join(missing_cols)}")
    st.stop()

# Khởi tạo session_state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.results = None
    st.session_state.x_test = None
    st.session_state.y_test = None
    st.session_state.yhat_test = None
    st.session_state.x_train = None
    st.session_state.y_train = None
    st.session_state.scaler = None

# Huấn luyện khi nút được bấm
if train_button:
    try:
        from sklearn.model_selection import train_test_split
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Kiểm tra dữ liệu
        if len(X) == 0:
            st.error("❌ Dữ liệu rỗng")
            st.stop()
        
        # Chia train/test
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=int(random_state)
        )
        
        # Huấn luyện mô hình
        model = LogisticRegression(max_iter=max_iter, random_state=int(random_state))
        model.fit(x_train, y_train)
        
        # Dự báo trên tập test
        yhat_test = model.predict(x_test)
        yhat_proba = model.predict_proba(x_test)
        
        # Tính toán chỉ tiêu
        cm = confusion_matrix(y_test, yhat_test)
        accuracy = accuracy_score(y_test, yhat_test)
        precision = precision_score(y_test, yhat_test, average='binary', zero_division=0)
        recall = recall_score(y_test, yhat_test, average='binary', zero_division=0)
        f1 = f1_score(y_test, yhat_test, average='binary', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, yhat_proba[:, 1])
        except:
            roc_auc = None
        
        class_report = classification_report(y_test, yhat_test, output_dict=True, zero_division=0)
        
        # Lưu vào session_state
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test
        st.session_state.yhat_test = yhat_test
        st.session_state.yhat_proba = yhat_proba
        st.session_state.x_train = x_train
        st.session_state.y_train = y_train
        
        st.session_state.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'train_size': len(x_train),
            'test_size': len(x_test)
        }
        
        st.success("✅ Huấn luyện thành công!")
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Lỗi huấn luyện: {str(e)}")


# ============================================================================
# TABS: NỘI DUNG CHÍNH
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tổng quan dữ liệu",
    "📈 Trực quan hóa dữ liệu",
    "🎯 Kết quả huấn luyện",
    "🔮 Sử dụng mô hình"
])

# ============================================================================
# TAB 1: TỔNG QUAN DỮ LIỆU
# ============================================================================
with tab1:
    st.header("📋 Tổng quan dữ liệu")
    
    # Kích thước dữ liệu
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Số dòng", len(df))
    with col2:
        st.metric("📋 Số cột", len(df.columns))
    with col3:
        file_size = len(file_bytes) / 1024
        st.metric("💾 Dung lượng file", f"{file_size:.2f} KB")
    
    st.divider()
    
    # Xem dữ liệu thô
    st.subheader("📄 Dữ liệu thô (10 dòng đầu)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Thống kê mô tả (chỉ các biến của mô hình)
    st.subheader("📊 Thống kê mô tả (Biến đầu vào & mục tiêu)")
    model_cols = feature_columns + [target_column]
    desc_df = df[model_cols].describe().T
    st.dataframe(desc_df, use_container_width=True)
    
    st.divider()
    
    # Phân phối biến mục tiêu
    st.subheader("🎯 Phân phối biến mục tiêu (PD)")
    target_counts = df[target_column].value_counts().reset_index()
    target_counts.columns = ['Nhãn', 'Số lượng']
    target_counts['Tỷ lệ %'] = (target_counts['Số lượng'] / len(df) * 100).round(2)
    st.dataframe(target_counts, use_container_width=True)


# ============================================================================
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ============================================================================
with tab2:
    st.header("📈 Trực quan hóa dữ liệu")
    
    # Chọn biến để vẽ (mặc định các biến ưu tiên)
    selected_features = st.multiselect(
        "Chọn biến để trực quan hóa",
        options=feature_columns,
        default=feature_columns[:4],
        help="Chọn tối đa 4 biến để hiển thị trên 1 màn hình"
    )
    
    if len(selected_features) == 0:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 biến")
    else:
        # Hiển thị tối đa 4 biểu đồ
        selected_features = selected_features[:4]
        
        # Tạo lưới 2x2
        cols = st.columns(2)
        
        for idx, feature in enumerate(selected_features):
            with cols[idx % 2]:
                # Histogram cho biến liên tục
                fig = px.histogram(
                    df,
                    x=feature,
                    nbins=20,
                    title=f"📊 Phân phối {feature}",
                    labels={feature: feature},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_title=feature,
                    yaxis_title="Tần số"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Biểu đồ phân phối biến mục tiêu
        st.divider()
        st.subheader("🎯 Phân phối biến mục tiêu PD")
        fig_target = px.bar(
            df[target_column].value_counts().reset_index(),
            x=target_column,
            y='count',
            title="Số lượng khách hàng có/không rủi ro",
            labels={target_column: 'Nhãn', 'count': 'Số lượng'},
            color=target_column,
            color_discrete_map={0: '#00CC96', 1: '#EF553B'}
        )
        fig_target.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_target, use_container_width=True)


# ============================================================================
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH
# ============================================================================
with tab3:
    st.header("🎯 Kết quả huấn luyện & Kiểm định mô hình")
    
    if not st.session_state.model_trained:
        st.info("👈 Vui lòng huấn luyện mô hình từ sidebar trước")
    else:
        results = st.session_state.results
        
        # Kích thước tập huấn luyện/kiểm định
        st.subheader("📊 Kích thước tập dữ liệu")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Tập huấn luyện", f"{results['train_size']} dòng ({results['train_size']/(results['train_size']+results['test_size'])*100:.1f}%)")
        with col2:
            st.metric("🧪 Tập kiểm định", f"{results['test_size']} dòng ({results['test_size']/(results['train_size']+results['test_size'])*100:.1f}%)")
        with col3:
            st.metric("📊 Tổng cộng", f"{results['train_size']+results['test_size']} dòng")
        
        st.divider()
        
        # Chỉ tiêu hiệu suất (Accuracy, Precision, Recall, F1)
        st.subheader("🏆 Chỉ tiêu hiệu suất")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Accuracy", f"{results['accuracy']:.4f}", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("🎯 Precision", f"{results['precision']:.4f}", f"{results['precision']*100:.2f}%")
        with col3:
            st.metric("🔍 Recall", f"{results['recall']:.4f}", f"{results['recall']*100:.2f}%")
        with col4:
            st.metric("⚖️ F1-Score", f"{results['f1']:.4f}", f"{results['f1']*100:.2f}%")
        
        if results['roc_auc'] is not None:
            st.metric("📈 ROC-AUC", f"{results['roc_auc']:.4f}", f"{results['roc_auc']*100:.2f}%")
        
        st.divider()
        
        # Ma trận nhầm lẫn
        st.subheader("🔲 Ma trận nhầm lẫn (Confusion Matrix)")
        cm = results['confusion_matrix']
        
        # Vẽ heatmap ma trận nhầm lẫn
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Dự báo: Không rủi ro', 'Dự báo: Có rủi ro'],
            y=['Thực tế: Không rủi ro', 'Thực tế: Có rủi ro'],
            text=cm,
            textposition='auto',
            colorscale='Blues'
        ))
        fig_cm.update_layout(height=400, width=500)
        st.plotly_chart(fig_cm, use_container_width=False)
        
        # Giải thích ma trận
        with st.expander("📖 Giải thích ma trận nhầm lẫn"):
            st.write(f"""
            - **TN (True Negative):** {cm[0,0]} - Dự báo không rủi ro, thực tế không rủi ro ✅
            - **FP (False Positive):** {cm[0,1]} - Dự báo có rủi ro, thực tế không rủi ro ❌
            - **FN (False Negative):** {cm[1,0]} - Dự báo không rủi ro, thực tế có rủi ro ❌
            - **TP (True Positive):** {cm[1,1]} - Dự báo có rủi ro, thực tế có rủi ro ✅
            """)
        
        st.divider()
        
        # Báo cáo chi tiết (Classification Report)
        st.subheader("📋 Báo cáo chi tiết")
        class_report = results['classification_report']
        
        report_df = pd.DataFrame(class_report).T
        report_df = report_df.round(4)
        st.dataframe(report_df, use_container_width=True)


# ============================================================================
# TAB 4: SỬ DỤNG MÔ HÌNH
# ============================================================================
with tab4:
    st.header("🔮 Sử dụng mô hình để dự báo")
    
    if not st.session_state.model_trained:
        st.info("👈 Vui lòng huấn luyện mô hình từ sidebar trước")
    else:
        model = st.session_state.model
        X_train = st.session_state.x_train
        
        # Chọn chế độ dự báo
        mode = st.radio(
            "Chọn chế độ dự báo",
            ["🎯 Nhập trực tiếp", "📊 Tải file dự báo"],
            horizontal=True
        )
        
        # ====== CHẾ ĐỘ 1: NHẬP TRỰC TIẾP ======
        if mode == "🎯 Nhập trực tiếp":
            st.subheader("📝 Nhập thông tin khách hàng")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                input_values = {}
                
                # Tạo input cho từng biến
                for idx, feature in enumerate(feature_columns):
                    # Tính trung vị và khoảng từ tập train
                    median_val = X_train[feature].median()
                    min_val = X_train[feature].min()
                    max_val = X_train[feature].max()
                    
                    # Chọn cột cho widget
                    if idx % 3 == 0:
                        col = col1
                    elif idx % 3 == 1:
                        col = col2
                    else:
                        col = col3
                    
                    with col:
                        input_values[feature] = st.slider(
                            feature,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(median_val),
                            help=f"Khoảng: {min_val:.0f}-{max_val:.0f}"
                        )
                
                # Nút dự báo
                submit_button = st.form_submit_button(
                    "🔮 Dự báo",
                    use_container_width=True,
                    type="primary"
                )
                
                if submit_button:
                    # Tạo DataFrame từ input
                    X_new = pd.DataFrame([input_values])
                    
                    # Dự báo
                    prediction = model.predict(X_new)[0]
                    proba = model.predict_proba(X_new)[0]
                    
                    # Hiển thị kết quả
                    st.divider()
                    st.subheader("📊 Kết quả dự báo")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 0:
                            st.success("✅ **Khách hàng: KHÔNG RỦI RO**")
                        else:
                            st.error("⚠️ **Khách hàng: CÓ RỦI RO**")
                    
                    with col2:
                        st.metric("Xác suất", f"{max(proba)*100:.2f}%")
                    
                    # Hiển thị chi tiết xác suất
                    st.divider()
                    st.write("#### 📈 Xác suất chi tiết")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🟢 Không rủi ro", f"{proba[0]*100:.2f}%")
                    with col2:
                        st.metric("🔴 Có rủi ro", f"{proba[1]*100:.2f}%")
                    
                    # Biểu đồ xác suất
                    fig_prob = go.Figure(data=[
                        go.Bar(
                            x=['Không rủi ro', 'Có rủi ro'],
                            y=proba,
                            marker_color=['#00CC96', '#EF553B'],
                            text=[f'{p*100:.2f}%' for p in proba],
                            textposition='auto'
                        )
                    ])
                    fig_prob.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_prob, use_container_width=True)
        
        # ====== CHẾ ĐỘ 2: TẢI FILE ======
        else:
            st.subheader("📊 Tải file dữ liệu để dự báo hàng loạt")
            
            uploaded_test_file = st.file_uploader(
                "Chọn file CSV để dự báo",
                type=['csv'],
                key='test_file_uploader'
            )
            
            if uploaded_test_file is not None:
                try:
                    X_test_new = pd.read_csv(BytesIO(uploaded_test_file.getvalue()))
                    
                    # Kiểm tra cột
                    missing_cols = [col for col in feature_columns if col not in X_test_new.columns]
                    extra_cols = [col for col in X_test_new.columns if col not in feature_columns]
                    
                    if missing_cols:
                        st.error(f"❌ Cột thiếu: {', '.join(missing_cols)}")
                    elif len(X_test_new) == 0:
                        st.error("❌ File rỗng")
                    else:
                        # Chỉ giữ các cột cần thiết
                        X_test_new = X_test_new[feature_columns]
                        
                        # Dự báo
                        predictions = model.predict(X_test_new)
                        probabilities = model.predict_proba(X_test_new)
                        
                        # Tạo kết quả
                        result_df = X_test_new.copy()
                        result_df['Dự báo'] = predictions.map({0: 'Không rủi ro', 1: 'Có rủi ro'})
                        result_df['Xác suất (%)'] = (probabilities.max(axis=1) * 100).round(2)
                        result_df['Xác suất không rủi ro (%)'] = (probabilities[:, 0] * 100).round(2)
                        result_df['Xác suất có rủi ro (%)'] = (probabilities[:, 1] * 100).round(2)
                        
                        # Hiển thị kết quả
                        st.success(f"✅ Dự báo thành công cho {len(result_df)} dòng dữ liệu")
                        
                        st.divider()
                        st.subheader("📋 Bảng kết quả dự báo")
                        
                        # Hiển thị bảng (cuộn được)
                        st.dataframe(result_df, use_container_width=True)
                        
                        st.divider()
                        
                        # Thống kê
                        st.subheader("📊 Thống kê kết quả")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            not_risk = (predictions == 0).sum()
                            st.metric("🟢 Không rủi ro", not_risk, f"{not_risk/len(predictions)*100:.1f}%")
                        with col2:
                            risk = (predictions == 1).sum()
                            st.metric("🔴 Có rủi ro", risk, f"{risk/len(predictions)*100:.1f}%")
                        with col3:
                            st.metric("📊 Tổng cộng", len(predictions))
                        
                        # Nút tải về kết quả
                        st.divider()
                        csv_result = result_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải kết quả (CSV)",
                            data=csv_result,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý file: {str(e)}")
