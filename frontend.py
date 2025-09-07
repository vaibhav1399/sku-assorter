import streamlit as st  # type: ignore
import pandas as pd
import json
import io
import matplotlib.pyplot as plt
from logic import *
import time
import nltk
from logic import classify_items, generate_item_vectors

# Page setup
st.set_page_config(page_title="Product Assortment Pro", page_icon="📦", layout="wide")

# Hide Streamlit's default file size limit message
hide_file_limit_style = """
    <style>
    [data-testid="stFileUploader"] small {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_file_limit_style, unsafe_allow_html=True)

# Title
st.title("📦 Product Assortment Pro")

# Initialize analysis level state
if "analysis_level" not in st.session_state:
    st.session_state.analysis_level = "SKU Level"

# Sidebar
with st.sidebar:
    st.header("### 🧭 Choose Analysis Level")
    selected_level = st.radio(
        "Select Analysis Level",
        ["SKU Level", "Parent Level"],
        index=0 if st.session_state.analysis_level == "SKU Level" else 1,
        horizontal=True,
        key="analysis_selector"
    )

    st.session_state.analysis_level = selected_level

    # Custom styling for buttons
    toggle_style = f"""
        <style>
            div[data-testid="column"]:nth-child(1) button {{
                background-color: {'#ffffff' if st.session_state.analysis_level == 'SKU Level' else '#f0f0f5'};
                color: {'#000000' if st.session_state.analysis_level == 'SKU Level' else '#64748b'};
                font-weight: bold;
                border-radius: 6px;
                width: 100%;
            }}
            div[data-testid="column"]:nth-child(2) button {{
                background-color: {'#ffffff' if st.session_state.analysis_level == 'Parent Level' else '#f0f0f5'};
                color: {'#000000' if st.session_state.analysis_level == 'Parent Level' else '#64748b'};
                font-weight: bold;
                border-radius: 6px;
                width: 100%;
            }}
        </style>
    """
    st.markdown(toggle_style, unsafe_allow_html=True)

    # Upload files
    st.header("📁 Upload Transaction File")
    uploaded_file = st.file_uploader("", type=["csv"], key="main_csv")
    st.markdown("**Must include columns: item_name, quantity, sales_price, transaction_date, channel, est_unit_cost, launch_month**")

    st.header("📁 Upload Mandatory Items List")
    st.markdown("**Upload a CSV containing mandatory item_name values to exclude from assortment.**")
    mandatory_file = st.file_uploader("", type=["csv"], key="mandatory")

    st.subheader("➕ Upload Items for Classification (Optional)")
    st.markdown("Upload a CSV containing the `item_name` column to classify custom items manually.")
    item_file = st.file_uploader("Upload classification CSV", type=["csv"], key="item_file")
    has_uploaded_item_file = item_file is not None

    st.markdown("---")
    st.subheader("⚙️ Configure Classification Settings")
    if "config" not in st.session_state:
        st.session_state.config = {
            "threshold_months": 3,
            "data_divide_percent": 80,
            "avg_product_percent": 20,
            "below_avg_product_percent": 20,
            "business_unit": "",
            "min_sales_volume": 0
        }

    st.session_state.config["threshold_months"] = st.slider("Age for New Products (Months)", 0, 12, st.session_state.config["threshold_months"])
    st.session_state.config["data_divide_percent"] = st.slider("Revenue Threshold for High Performers (%)", 0, 100, st.session_state.config["data_divide_percent"])

    submit_disabled = uploaded_file is None

    if st.button("🚀 Set Threshold", disabled=submit_disabled):
        with st.spinner("Processing..."):
            try:
                nltk.download('punkt', quiet=True)
                start_time = time.time()
                t0 = time.time()

                print("🟡 Reading uploaded files...")
                config = st.session_state.config
                df = pd.read_csv(uploaded_file)
                print("✅ Loaded CSVs and config:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                # Handle optional mandatory file
                if mandatory_file:
                    mandatory_df = pd.read_csv(mandatory_file)
                    mandatory_list = mandatory_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                    config["mandatory_product"] = mandatory_list
                else:
                    config["mandatory_product"] = []
                print("🔵 Mandatory file reading time:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                if 'item_name' in df.columns:
                    df['item_name'] = df['item_name'].astype(str).str.strip()

                print("🔵 Column Correction:", round(time.time() - t0, 2), "s")
                t0 = time.time()
                transaction_df = process_csv_and_config(df, config)
                print("🔵 process_csv_and_config:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                file_path = "./classified_items_output.xlsx" # Assuming this file is in the same directory as the app; adjust if needed
                item_label_df = classify_products(transaction_df, config)
                print("🔵 classify_products:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                result_df = classify_items(file_path)
                print("🔵 classify_items:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                item_label_with_parent_df = item_with_parent(item_label_df, result_df)
                print("🔵 item_with_parent:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                product_with_label, product_without_label = identify_label(item_label_with_parent_df)
                print("🔵 identify_label:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                product_with_label_vec = generate_item_vectors(product_with_label)
                print("🔵 generate_item_vectors (with):", round(time.time() - t0, 2), "s")
                t0 = time.time()

                product_without_label_vec = generate_item_vectors(product_without_label)
                print("🔵 generate_item_vectors (without):", round(time.time() - t0, 2), "s")
                t0 = time.time()

                similar_df, remaining_df = find_similar_items(product_without_label_vec, product_with_label_vec)
                print("🔵 find_similar_items:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                final_labels_df = assign_max_priority_label(similar_df)
                print("🔵 assign_max_priority_label:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                Final_result = prepared_final_files(final_labels_df, product_with_label_vec, remaining_df)
                print("🔵 prepared_final_files:", round(time.time() - t0, 2), "s")

                Parent_result = summarize_labels_by_parent_class(Final_result)  # confidence df
                Ranked_result = prepare_ranked_output_by_margin(Final_result, transaction_df)
                invest_df, watch_df, sustain_df, sunset_df, summary_df = split_final_result_by_label(Final_result, transaction_df)
                summary_df["Label"] = summary_df["Label"].str.upper()

                margin_revenue_across_channel = movement_of_sku_across_channel(transaction_df)
                print("🔵 movement_of_sku_across_channel:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                channel_result = channel_wise_assortment(transaction_df, config)
                print("🔵 channel_wise_assortment:", round(time.time() - t0, 2), "s")
                t0 = time.time()

                similar_item_output = None
                message = ""

                Custom_item_result = None
                Unmatched_items = None
                if item_file:
                    print("📥 Received item_file for classification...")
                    item_df = pd.read_csv(item_file)
                    item_df['item_name'] = item_df['item_name'].astype(str).str.strip()

                print(f"✅ Total processing time: {round(time.time() - start_time, 2)}s")

                st.session_state.Final_result_df = Ranked_result[["parent_class", "item_name", "Label"]]
                st.session_state.Parent_result_df = Parent_result[["parent_class", "Invest", "Watch", "Sustain", "Sunset", "confidence"]]
                st.session_state.Ranked_result_df = Ranked_result
                st.session_state.Invest_df = invest_df
                st.session_state.Watch_df = watch_df
                st.session_state.Sustain_df = sustain_df
                st.session_state.Sunset_df = sunset_df
                st.session_state.Summary_df = summary_df[["Label", "Count", "Percentage"]]
                st.session_state.margin_revenue_across_channel_df = margin_revenue_across_channel[["item_name", "margin_Store", "margin_Vending", "margin_web", "margin_Builk order", "revenue_Store", "revenue_Vending", "revenue_web", "revenue_Builk order"]]
                st.session_state.channel_result_df = channel_result[["item_name", "Web Result", "Bulk Result", "Vending Result", "Store Result"]]
                st.session_state.data_ready = True
                st.success("✅ Processing complete!")
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Main content
if st.session_state.get("data_ready"):
    label_colors = {"INVEST": "#1f77b4", "SUSTAIN": "#2ca02c", "WATCH": "#ffbf00", "SUNSET": "#d62728"}
    label_desc = {
        "INVEST": "High growth potential",
        "SUSTAIN": "Maintain current performance",
        "WATCH": "Monitor closely",
        "SUNSET": "Consider discontinuation"
    }

    with st.container():
        st.markdown("""
            <style>
            .summary-card {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .label { font-size: 40px; font-weight: bold; }
            .count { font-size: 24px; margin-top: 10px; }
            .percentage { font-size: 18px; margin-top: 5px; }
            .desc { font-size: 14px; color: #666; }
            </style>
            <div class="row-gap">
        """, unsafe_allow_html=True)

        # 🔹 One row with 4 columns
        row = st.columns(4)
        for idx, label in enumerate(["INVEST", "SUSTAIN", "WATCH", "SUNSET"]):
            with row[idx]:
                count = st.session_state.Summary_df[st.session_state.Summary_df["Label"] == label]["Count"].values[0]
                percentage = st.session_state.Summary_df[st.session_state.Summary_df["Label"] == label]["Percentage"].values[0]
                percentage_str = f"{percentage:.2f}%"
                st.markdown(f"""
                    <div class="summary-card" style='color: {label_colors[label]};'>
                        <div class="label">{label}</div>
                        <div class="count">{count}</div>
                        <div class="percentage">{percentage_str}</div>
                        <div class="desc">{label_desc[label]}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.analysis_level == "SKU Level":
        st.markdown("#### 🔢 Strategic SKU Prioritization by Group")
        st.dataframe(st.session_state.Final_result_df.head(5))
        csv_data = st.session_state.Final_result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Prioritized SKU List", csv_data, "sku_output_list.csv", "text/csv")
    else:
        st.markdown("#### 🔢 SKU Label Distribution by Parent Category with Investment Confidence Marker")
        st.markdown("##### Invest = 1, Sustain = 0.8, Watch = 0.6, Sunset = 0.2")
        st.dataframe(st.session_state.Parent_result_df.head(5))
        csv_data = st.session_state.Parent_result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Prioritized Parent List", csv_data, "parent_output_list.csv", "text/csv")

    st.markdown("#### 🧰 Select Classification Label")
    col_order = ["item_name", "parent_class", "Label"]

    for label, df_key in zip(["📈 Invest", "👀 Watch", "🛠 Sustain", "📉 Sunset"], ["Invest_df", "Watch_df", "Sustain_df", "Sunset_df"]):
        st.subheader(label + " Items")
        st.dataframe(st.session_state[df_key][col_order].head(5))
        st.download_button(f"⬇️ Download {label.strip()} Items", st.session_state[df_key][col_order].to_csv(index=False).encode('utf-8'), f"{label.strip().lower()}_items.csv", "text/csv")

    st.markdown("#### 💰 Margin and Revenue Across Channels")
    st.dataframe(st.session_state.margin_revenue_across_channel_df.head(10))
    csv_margin_revenue = st.session_state.margin_revenue_across_channel_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Margin and Revenue Across Channels", csv_margin_revenue, "margin_revenue_across_channel_df.csv", "text/csv")

    st.markdown("#### 🤖 Channel Result")
    st.dataframe(st.session_state.channel_result_df.head(10))
    csv_channel = st.session_state.channel_result_df.to_csv(index=False).encode('utf-8')

    st.download_button("⬇️ Download Channel Result", csv_channel, "channel_result.csv", "text/csv")

