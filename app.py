with st.form("predict_form"):
                st.markdown("Nhập điểm khảo sát cho từng tiêu chí (thang điểm 1-5):")
                input_values = {}
                for group_name, cols in FEATURE_GROUPS.items():
                    st.markdown(f"**{group_name}**")
                    g_cols = st.columns(len(cols))
                    for gc, col in zip(g_cols, cols):
                        default_val = int(round(df[col].median()))
                        input_values[col] = gc.number_input(
                            col,
                            min_value=int(df[col].min()),
                            max_value=int(df[col].max()),
                            value=default_val,
                            step=1,
                            key=f"input_{col}",
                        )
                submitted = st.form_submit_button("Dự báo", type="primary", use_container_width=True)

            if submitted:
                x_new = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)
                pred = model.predict(x_new)[0]
                proba = model.predict_proba(x_new)[0]

                if pred == 1:
                    st.error(f"⚠️ Dự báo: **{TARGET_LABELS[pred]}**")
                else:
                    st.success(f"✅ Dự báo: **{TARGET_LABELS[pred]}**")

                p1, p2 = st.columns(2)
                p1.metric("Xác suất không có rủi ro", f"{proba[0] * 100:.2f}%")
                p2.metric("Xác suất có rủi ro", f"{proba[1] * 100:.2f}%")

        else:
            batch_file = st.file_uploader(
                "Tải lên tệp CSV chứa các cột biến đầu vào (giống cấu trúc X_test)",
                type=["csv"],
                key="batch_predict_uploader",
                help="Tệp cần chứa đầy đủ 24 cột biến đầu vào: " + ", ".join(feature_cols),
            )
            if batch_file is not None:
                try:
                    new_df = load_data(batch_file.getvalue())
                    missing = validate_columns(new_df, feature_cols)
                    if missing:
                        st.error("Tệp thiếu các cột bắt buộc: " + ", ".join(missing))
                    else:
                        X_new = new_df[feature_cols]
                        preds = model.predict(X_new)
                        probas = model.predict_proba(X_new)[:, 1]
                        result_df = new_df.copy()
                        result_df["Dự báo"] = [TARGET_LABELS[p] for p in preds]
                        result_df["Xác suất rủi ro (%)"] = (probas * 100).round(2)

                        st.markdown("**Kết quả dự báo**")
                        with st.container(height=350):
                            st.dataframe(result_df, use_container_width=True)

                        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải kết quả (CSV)",
                            data=csv_bytes,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Không thể xử lý tệp: {e}")
