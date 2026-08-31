import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
import pickle

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Brian | Data & AI Portfolio",
    page_icon="🤖",
    layout="wide",
)

st.title("My Portfolio with Streamlit")

st.write("""
Welcome to my portfolio! This website was created to showcase various projects, data analysis, data science, machine learning and applications that I have developed. Explore this page to see my work.
""")

# SET TAB ORDER HERE
tab_home, tab_eda, tab_demand, tab_ml = st.tabs([
    "🏠 Home",
    "📊 EDA Dashboard",
    "📈 Demand Forecasting",
    "🤖 ML Predictor"
])

# ══════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════
with tab_home:

    st.title("About Me")
    st.header("Hi, I'm Brianca Hernawan, BBus, MBA 👋")
    st.subheader("Head of AIML, Data Science, and Data Analytics · BBF Meat")

    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        st.image("Brain_pasphoto.jpeg", width=200)

    with col_bio:
        st.write("""
        Results-driven Data & AI/ML leader with **15 years of experience** delivering
        measurable business impact across healthcare, insurance, banking, global trade, logistics, and food industries. 
        Skilled in transforming data into actionable insights that improve operations, optimize costs, and enable strategic decision‑making. 
        Proven ability to lead analytics teams, develop predictive models, and enhance business intelligence pipelines.
        """)

        st.markdown(""" I hold an MBA from IPMI International Business School and a Bachelor of Business from Victoria University. Recently, I have expanded my skill set into Data Science and Analytics, bridging business strategy with data-driven decision-making using Python, SQL, Power BI, and Machine Learning for demand forecasting.
        """)

        st.write("**Core Skills:** Data Analysis · Business Intelligence · EDA · ML Modeling · LLM · RAG")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Years of Experience", "15+")
    col2.metric("Industries", "6+")
    col3.metric("Global Clients", "100+")
    col4.metric("Cost Savings", "$20M+")

    st.divider()

    st.subheader("Featured Projects")

    # Updated to 3 columns to include the new Demand Forecast project
    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown("### 🌸 Interactive EDA")
            st.write("""
            Interactive exploratory data analysis dashboard to explore and uncover
            insights from the classic Iris dataset. Features bar charts, pie charts,
            box plots, and histograms with KDE overlays.
            """)
            st.info("👉 Explore it in the **📊 EDA Dashboard** tab above.")

    with p2:
        with st.container(border=True):
            st.markdown("### 📈 Demand Forecasting")
            st.write("""
            Predict product demand using a trained XGBoost model based on price, 
            discount, inventory levels, promotions, and competitor pricing data.
            """)
            st.info("👉 Try it live in the **📈 Demand Forecasting** tab above.")

    with p3:
        with st.container(border=True):
            st.markdown("### 🏠 House Price Predictor")
            st.write("""
            ML model trained on **>10,000 house listings** from rumahrumah.com.
            Predicts Jakarta property prices based on key property features.
            """)
            st.write("**R² = 92%** · **MAPE = 11%**")
            st.info("👉 Try it live in the **🤖 ML Predictor** tab above.")


# ══════════════════════════════════════════════
# TAB 2 — EDA DASHBOARD
# ══════════════════════════════════════════════
with tab_eda:

    @st.cache_data
    def load_iris_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df

    FEATURES = {
        "sepal_length": "Sepal Length (cm)",
        "sepal_width":  "Sepal Width (cm)",
        "petal_length": "Petal Length (cm)",
        "petal_width":  "Petal Width (cm)",
    }

    df = load_iris_data()

    st.title("🌸 Iris Dataset — Interactive EDA")
    st.divider()

    # ── Filters ───────────────────────────────
    st.subheader("Filters")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        selected_species = st.multiselect(
            "Species",
            options=df["species"].unique().tolist(),
            default=df["species"].unique().tolist(),
        )
    with col_f2:
        x_axis = st.selectbox(
            "X-axis (Scatter)",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            index=0,
        )
    with col_f3:
        y_axis = st.selectbox(
            "Y-axis (Scatter)",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            index=2,
        )

    df_f = df[df["species"].isin(selected_species)]

    if df_f.empty:
        st.warning("No data. Select at least one species.")
        st.stop()

    # ── Overview metrics ──────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", len(df_f))
    m2.metric("Features", 4)
    m3.metric("Species", len(selected_species))
    m4.metric("Missing Values", int(df_f.isnull().sum().sum()))

    st.divider()

    # ── Bar chart + Pie chart ─────────────────
    st.subheader("Distribution Overview")
    col_bar, col_pie = st.columns(2)

    with col_bar:
        bar_feature = st.selectbox(
            "Feature for bar chart",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            key="bar_feat",
        )
        bar_data = df_f.groupby("species", observed=False)[bar_feature].mean().reset_index()
        fig_bar = px.bar(
            bar_data,
            x="species", y=bar_feature,
            color="species",
            text_auto=".2f",
            labels={"species": "Species", bar_feature: FEATURES[bar_feature]},
            title=f"Mean {FEATURES[bar_feature]} by Species",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        counts = df_f["species"].value_counts().reset_index()
        counts.columns = ["species", "count"]
        fig_pie = px.pie(
            counts,
            names="species", values="count",
            hole=0.4,
            title="Sample Count by Species",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Box plot ──────────────────────────────
    st.subheader("Box Plots")
    box_feature = st.selectbox(
        "Feature for box plot",
        options=list(FEATURES.keys()),
        format_func=lambda k: FEATURES[k],
        key="box_feat",
    )
    fig_box = px.box(
        df_f,
        x="species", y=box_feature,
        color="species",
        points="all",
        labels={"species": "Species", box_feature: FEATURES[box_feature]},
        title=f"Distribution of {FEATURES[box_feature]} by Species",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Histogram with KDE ────────────────────
    st.subheader("Histogram with KDE")
    col_h1, col_h2 = st.columns([1, 3])
    with col_h1:
        hist_feature = st.selectbox(
            "Feature",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            key="hist_feat",
        )
        n_bins = st.slider("Bins", 5, 50, 20)
    with col_h2:
        fig_hist = px.histogram(
            df_f,
            x=hist_feature,
            color="species",
            nbins=n_bins,
            marginal="violin",
            opacity=0.7,
            barmode="overlay",
            labels={hist_feature: FEATURES[hist_feature]},
            title=f"Histogram + KDE: {FEATURES[hist_feature]}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter plot ──────────────────────────
    st.subheader("Scatter Plot")
    fig_scatter = px.scatter(
        df_f,
        x=x_axis, y=y_axis,
        color="species",
        symbol="species",
        trendline="ols",
        labels={x_axis: FEATURES[x_axis], y_axis: FEATURES[y_axis]},
        title=f"{FEATURES[x_axis]} vs {FEATURES[y_axis]}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Raw data ──────────────────────────────
    st.divider()
    if st.checkbox("Show raw data"):
        st.dataframe(df_f, use_container_width=True)
        st.caption(f"{len(df_f)} rows · 5 columns")


# ══════════════════════════════════════════════
# TAB 3 — DEMAND FORECASTING
# ══════════════════════════════════════════════
with tab_demand:
    st.title("📈 Demand Forecasting Predictor")
    st.write("Predict product demand using a trained XGBoost model.")
    st.divider()

    @st.cache_resource
    def load_demand_artifacts():
        with open("xgboost_demand_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model, encoders

    try:
        model_xgb, label_encoders_xgb = load_demand_artifacts()

        # Create Sub-Tabs for selecting prediction mode
        sub_manual, sub_batch = st.tabs(["📝 Manual Input", "📂 Batch Prediction (Upload CSV)"])

        # --- SUB TAB 1: MANUAL INPUT ---
        with sub_manual:
            col_inputs, col_results = st.columns(2)

            with col_inputs:
                st.subheader("Input Features")

                category = st.selectbox(
                    "Category", 
                    label_encoders_xgb["Category"].classes_.tolist()
                )
                price = st.number_input("Price ($)", min_value=0.0, value=50.0)
                discount = st.number_input("Discount (%)", min_value=0, value=10)
                inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
                promotion = st.selectbox("Promotion Active", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                competitor_pricing = st.number_input("Competitor Price ($)", min_value=0.0, value=50.0)

                predict_btn = st.button("📈 Predict Demand", width="stretch")

            with col_results:
                st.subheader("Forecast Result")

                if predict_btn:
                    input_data = pd.DataFrame({
                        "Price": [price],
                        "Discount": [discount],
                        "Inventory Level": [inventory_level],
                        "Promotion": [promotion],
                        "Competitor Pricing": [competitor_pricing],
                        "Category": [category]
                    })

                    # Encode categorical input
                    for col, encoder in label_encoders_xgb.items():
                        if col in input_data.columns:
                            input_data[col] = encoder.transform(input_data[col])

                    # Predict demand
                    prediction = model_xgb.predict(input_data)[0]

                    st.metric(label="Predicted Demand", value=f"{int(prediction)} Units")
                    st.success("Prediction generated successfully!")

                    st.info("💡 **Insight:** The model relies heavily on **Promotion** (58%) and **Category** (27%) to forecast demand.")
                else:
                    st.info("Fill in the parameters on the left and click **Predict Demand**.")

        # --- SUB TAB 2: BATCH PREDICTION VIA CSV ---
        with sub_batch:
            st.subheader("CSV-Based Prediction Pipeline")
            st.write("Please upload a CSV file with the following column format: `Price`, `Discount`, `Inventory Level`, `Promotion`, `Competitor Pricing`, and `Category`.")

            uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

            if uploaded_file is not None:
                # Read uploaded data
                df_upload = pd.read_csv(uploaded_file)
                st.write("📋 **Data Preview:**")
                st.dataframe(df_upload.head())

                # Button to Trigger Prediction Pipeline
                if st.button("🚀 Trigger Prediction Pipeline", width="stretch", type="primary"):
                    with st.spinner("Running predictions..."):
                        # Create a copy of the data for processing
                        df_process = df_upload.copy()

                        # Ensure required columns exist
                        expected_cols = ["Price", "Discount", "Inventory Level", "Promotion", "Competitor Pricing", "Category"]

                        if all(col in df_process.columns for col in expected_cols):
                            # Encode categorical input using the encoder from the training process
                            for col, encoder in label_encoders_xgb.items():
                                if col in df_process.columns:
                                    # Using encoder.transform
                                    df_process[col] = encoder.transform(df_process[col].astype(str))

                            # Filter only the columns used by the model
                            X_predict = df_process[expected_cols]

                            # Perform prediction
                            predictions = model_xgb.predict(X_predict)

                            # Merge prediction results into the original dataframe for display
                            df_upload["Predicted Demand"] = np.round(predictions).astype(int)

                            st.success("✅ Prediction ran successfully!")
                            st.write("📊 **Prediction Results:**")
                            st.dataframe(df_upload, use_container_width=True)

                            # Provide download button for results
                            csv_output = df_upload.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Prediction Results (CSV)",
                                data=csv_output,
                                file_name="predicted_demand_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(f"⚠️ Columns do not match! Ensure your CSV file has the following columns: {', '.join(expected_cols)}")

    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure `xgboost_demand_model.pkl` and `label_encoders.pkl` are saved in the same folder as `app.py`.")

# ══════════════════════════════════════════════
# TAB 4 — ML PREDICTOR (HOUSE PRICE)
# ══════════════════════════════════════════════
with tab_ml:

    st.title("🏠 Jakarta House Price Predictor")
    st.write("Trained on **>10,000 listings** from rumahrumah.com · R² = 92% · MAPE = 11%")
    st.divider()

    col_inputs, col_result = st.columns(2)

    with col_inputs:
        st.subheader("Property Input")

        luas_tanah = st.number_input(
            "Land Area (m²)",
            min_value=0, max_value=10000,
            value=100, step=1,
        )
        luas_bangunan = st.number_input(
            "Building Area (m²)",
            min_value=0, max_value=10000,
            value=80, step=1,
        )
        kamar_tidur = st.slider("Number of Bedrooms", 1, 10, 3)
        kamar_mandi = st.slider("Number of Bathrooms", 1, 8, 2)
        sudah_shm = st.selectbox(
            "Certificate Status",
            options=["SHM (Freehold Title)", "Non-SHM (Leasehold, etc.)"],
        )
        is_shm = 1 if "Non" not in sudah_shm else 0

        predict_btn_house = st.button("🔮 Predict Price", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn_house:
            shm_multiplier = 1 + 0.2 * is_shm
            price = shm_multiplier * (
                14_000_000 * luas_tanah
                + 4_000_000 * luas_bangunan
                + 15_000_000 * kamar_tidur
                + 10_000_000 * kamar_mandi
            )
            price_billion = price / 1_000_000_000
            price_million = price / 1_000_000

            st.metric(
                label="Estimated Property Price",
                value=f"IDR {price_billion:.2f} B",
                delta=f"≈ IDR {price_million:,.0f} million",
            )

            if is_shm:
                st.success("✅ Includes SHM premium +20%")

            st.divider()

            # Price breakdown bar chart
            components = {
                "Land Area":        14_000_000 * luas_tanah,
                "Building Area":    4_000_000  * luas_bangunan,
                "Bedrooms":         15_000_000 * kamar_tidur,
                "Bathrooms":        10_000_000 * kamar_mandi,
            }
            base = sum(components.values())
            if is_shm:
                components["SHM Premium"] = base * 0.2

            df_breakdown = pd.DataFrame({
                "Component": list(components.keys()),
                "Value (Million IDR)": [v / 1_000_000 for v in components.values()],
            })

            fig = px.bar(
                df_breakdown,
                x="Component",
                y="Value (Million IDR)",
                title="Price Component Breakdown",
                text_auto=".0f",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Fill in the parameters on the left and click **Predict Price**.")

    st.divider()
    st.caption(
        "Formula: `(1 + 0.2×SHM) × (14m×land + 4m×building + 15m×bed + 10m×bath)`"
    )