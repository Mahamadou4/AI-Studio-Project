import streamlit as st
import pandas as pd
import altair as alt
import json

from snowflake.snowpark.context import get_active_session
from ziyan_eda_arizona import render_exploratory_eda

def extract_final_answer(text):
    if "<final_answer>" in text:
        return text.split("<final_answer>")[1].split("</final_answer>")[0].strip()
    if "<think>" in text:
        return text.split("</think>")[-1].strip()
    return text

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {background-color: #f8fafc; font-family: 'Segoe UI', sans-serif;}
    h1, h2, h3, h4 {color: #0b3d91;}
    .sidebar .sidebar-content {background-color: #edf2f7;}
    footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0b3d91;
        color: white;
        text-align: center;
        padding: 6px 0;
        font-size: 13px;
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/Flag_of_Arizona.svg", width=100)
st.sidebar.title("📍 Navigation")

page = st.sidebar.radio(
    "Go to section:",
    [
        "Home",
        "Data Understanding",
        "County Analysis",
        "Exploratory Visual Analysis of Individual Variables",  
        "Modeling & Evaluation",
        "Insights",
        "AI Assistant",
    ]
)

session = get_active_session()

# ---------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------
if page == "Home":
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h1 style="color:#0b3d91; font-size:2.5rem;">🏜️ Arizona Social Vulnerability Dashboard</h1>
        <p style="font-size:1.1rem; color:#334155; max-width:800px; margin:auto;">
        Explore how <b>social vulnerability</b> and <b>demographic factors</b> interact across Arizona’s 15 counties —
        powered by <b>Snowflake</b> and <b>Streamlit</b>.
        </p>
        <p style="color:#475569; font-size:0.9rem;"><i>Developed by Team Snowflake 1C | Break Through Tech AI Studio</i></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    df = session.table("SVI.PUBLIC.ARIZONA_CLEAN").to_pandas()
    st.dataframe(df.head(), use_container_width=True)
    st.caption("Sample rows from the Arizona SVI dataset (loaded directly from Snowflake).")

# --------------------------------------------------
# DATA UNDERSTANDING (UPDATED)
# ---------------------------------------------------------------------
elif page == "Data Understanding":
    st.title("📊 Data Understanding: From Raw Sources to Final Dataset")
    st.markdown("""
        We begin by reviewing the **raw source data** (Social Vulnerability Index and Hospital Data) and then examine the **final merged and engineered dataset** used for our analysis.
    """)

    # --- 1. Load Raw DataTables ---
    # Fetch the data from the specified Snowflake tables
    try:
        # NOTE: SVI data is huge, we only grab the first 50 rows for display efficiency
        df_svi = session.table("SVI.PUBLIC.ARIZONA_2022_MAXTEMP").limit(50).to_pandas()
        df_hosp = session.table("SVI.PUBLIC.ARIZONA_LICENSED_HOSPITAL").limit(50).to_pandas()
        df_final = session.table("SVI.PUBLIC.SVI_HOSPITAL_MERGED").to_pandas()
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {e}")
        # Display placeholders if data fails to load to prevent app crash
        df_svi = pd.DataFrame() 
        df_hosp = pd.DataFrame()
        df_final = pd.DataFrame() 
        # return # You can uncomment this to stop the page if data fails

    # -------------------------------------
    # 📝 Raw Dataset Overview
    # -------------------------------------
    st.header("1. Raw Source Data Tables")

    # --- SVI/Max Temp Dataset ---
    st.markdown("### Arizona Social Vulnerability Index & Max Temp")
    
    # Safely get the actual column count in case the limited dataframe is empty
    try:
        svi_cols_count = len(session.table("SVI.PUBLIC.ARIZONA_SVI_CLEAN").columns)
    except:
        svi_cols_count = len(df_svi.columns)
        
    st.info(f"The original SVI table is a rich source, containing **{svi_cols_count:,} columns** before feature selection.")

    # Use st.expander for the large table
    with st.expander(f"Preview of Raw SVI/Max Temp Data (first {len(df_svi)} rows)"):
        st.dataframe(df_svi, use_container_width=True)
    
    st.caption("This data includes detailed demographic, social, and environmental indicators for US counties/tracts.")

    st.markdown("---")

    # --- Hospital Dataset ---
    st.markdown("### Arizona Licensed Hospital Data")
    st.write(f"**Rows:** {len(df_hosp):,} (sample) | **Columns:** {len(df_hosp.columns):,}")
    
    with st.expander(f"Preview of Raw Hospital Data (first {len(df_hosp)} rows)"):
        st.dataframe(df_hosp, use_container_width=True)

    st.caption("This source provides details like hospital type, location, and capacity.")

    st.markdown("---")
    
    # -------------------------------------
    # ⚙️ Final Merged & Engineered Dataset
    # -------------------------------------
    st.header("2. Final Processed Dataset: `SVI_HOSPITAL_MERGED`")
    
    st.write("""
        The raw tables were **merged, cleaned, and reduced** to create this final dataset. It combines only the most relevant features to predict medical desert status.
    """)
    st.write(f"**Rows:** {len(df_final):,} | **Columns:** **{len(df_final.columns):,}**")

    # Display the final dataset partially
    with st.expander("Preview of Final Dataset (first 10 rows)"):
        st.dataframe(df_final.head(10), use_container_width=True)

    # --- Final Data Visualization: Column Type Composition ---
    st.markdown("### Final Dataset ")
    
    if not df_final.empty:
        # Data Prep: Count the number of columns for each data type
        dtype_counts = df_final.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        # Create the Altair Bar Chart
        chart = (
            alt.Chart(dtype_counts)
            .mark_bar()
            .encode(
                x=alt.X('Data Type:N', title='Data Type'), 
                y=alt.Y('Count:Q', title='Number of Columns'), 
                tooltip=['Data Type', 'Count'],
                color=alt.Color('Data Type:N', scale=alt.Scale(range=['#0b3d91', '#85a7e6'])) # Using a custom color scheme
            )
            .properties(
                title="Distribution of Feature Data Types"
            )
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(f"A visualization showing the data types of the final **{len(df_final.columns)} columns** used for modeling.")
    else:
        st.warning("Cannot display feature composition as the final dataset is empty.")

    # --- Metric Reference Guide
    st.markdown("---")
    st.markdown("""
    ### Final Feature Reference Guide
    The 12 columns in the final dataset are:
    
    * **county** (Varchar) - The Arizona county.
    * **POPULATION_TOTAL** (Number) - Total population of the county.
    * **PCT_NO_VEHICLE** (Float) - Percentage of households with no vehicle (Social Vulnerability).
    * **PCT_UNINSURED** (Float) - Percentage of the population uninsured (Social Vulnerability).
    * **PCT_POVERTY** (Float) - Percentage of population below poverty line (Social Vulnerability).
    * **PCT_HEAT_RISK** (Float) - Estimated percentage of area under high maximum temperature (Environmental Risk).
    * **NUM_HOSPITALS** (Number) - Count of licensed hospitals in the county.
    * **TOTAL_CAPACITY** (Float) - Total licensed bed capacity of all hospitals.
    * **HOSPITALS_PER_10K** (Number) - Hospital count normalized by 10,000 population.
    * **BEDS_PER_10K** (Float) - Hospital bed count normalized by 10,000 population.
    * **CARRYING_CAPACITY_RATIO** (Float) - Custom metric: Ratio of hospital capacity to population.
    * **IS_MEDICAL_DESERT** (Number) - **Target Variable** (1 if medical desert, 0 otherwise).
    """)

    
# ---------------------------------------------------------------------
# COUNTY ANALYSIS
# ---------------------------------------------------------------------
elif page == "County Analysis":
    st.title("County-Level Analysis")

    df = session.table("SVI.PUBLIC.ARIZONA_CLEAN").to_pandas()

    metric_options = {
        "RPL_THEMES": "Social Vulnerability Index",
        "E_UNEMP": "Unemployment (Estimated Count)",
        "E_POV150": "Population Below 150% Poverty Line",
        "E_NOHSDP": "No High School Diploma"
    }

    metric = st.sidebar.selectbox(
        "Choose a metric to visualize:",
        list(metric_options.keys()),
        format_func=lambda x: metric_options[x]
    )

    counties = st.sidebar.multiselect(
        "Select counties:",
        options=df["COUNTY"].unique(),
        default=df["COUNTY"].unique()[:5]
    )

    pop_filter = st.slider(
        "Filter by minimum population:",
        min_value=int(df["E_TOTPOP"].min()),
        max_value=int(df["E_TOTPOP"].max()),
        value=int(df["E_TOTPOP"].median())
    )

    filtered_df = df[(df["E_TOTPOP"] >= pop_filter) & (df["COUNTY"].isin(counties))]

    st.subheader("📈 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Counties", len(df["COUNTY"].unique()))
    col2.metric("Avg Vulnerability", round(df["RPL_THEMES"].mean(), 2))
    col3.metric("Max Population", int(df["E_TOTPOP"].max()))

    # --- Visualization ---
    st.subheader("🌡️ Population vs Selected Metric")
    viz_df = filtered_df.rename(columns={
        "E_TOTPOP": "Estimated Total Population",
        metric: metric_options[metric]
    })
    st.scatter_chart(data=viz_df, x="Estimated Total Population", y=metric_options[metric], color="COUNTY")
    st.caption(f"Compares county population with **{metric_options[metric]}** across selected regions.")

# ---------------------------------------------------------------------
# >>> Exploratory EDA (Heat & Hospitals) 
# ---------------------------------------------------------------------
elif page == "Exploratory Visual Analysis of Individual Variables":
    # call eda_heat_hospitals.py 
    render_exploratory_eda()

# ---------------------------------------------------------------------
# MODELING & EVALUATION
# ---------------------------------------------------------------------

elif page == "Modeling & Evaluation":
# ---------------------------------------------------
# PERFORMANCE SUMMARY (TEXT)
# ---------------------------------------------------
    st.subheader("Performance Summary")
    
    st.markdown("""
    All three models achieved:
    
    - **Precision = 1.0**
    - **Recall = 1.0**
    - **Zero false positives**
    - **Correctly identified all medical desert counties**
    
    This identical performance strongly suggests the dataset has a 
    **clear, easy pattern** the models are learning.
    """)
    
    st.markdown("---")
    
    # ---------------------------------------------------
    # CHART UNDER THE TEXT (FULL WIDTH)
    # ---------------------------------------------------
    st.subheader("Model Precision & Recall")
    
    perf_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Precision": [1.0, 1.0, 1.0],
        "Recall": [1.0, 1.0, 1.0],
    })
    
    # reshape
    perf_long = perf_df.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall"],
        var_name="Metric",
        value_name="Score"
    )
    
    # create a combined category so each bar has a unique x position
    perf_long["X"] = perf_long["Model"] + " — " + perf_long["Metric"]
    
    chart = (
        alt.Chart(perf_long)
        .mark_bar(size=40)
        .encode(
            x=alt.X("X:N", title="", axis=alt.Axis(labelAngle=20)),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1.1])),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=["#ff69b4", "#ff8ccf"]),
                legend=alt.Legend(title="Metric")
            ),
            tooltip=["Model", "Metric", "Score"]
        )
        .properties(
            width=600,
            height=350,
            title="Model Precision & Recall"
        )
    )
    
    st.altair_chart(chart, use_container_width=True)




    st.markdown("---")

    # -----------------------------
    # RED FLAG SECTION
    # -----------------------------
    st.header("Why Perfect Scores Are a Red Flag")
    st.markdown("""
    Perfect accuracy in predictive modeling is **extremely rare**.
    These results likely indicate:

    - **Small dataset** → too few examples for robust learning  
    - **Possible data leakage** → models may be seeing the answer indirectly  
    - **Overfitting** → memorizing instead of generalizing  
    - **Limited generalization** → results may fail on new or larger datasets  
    """)

    st.markdown("---")

    # -----------------------------
    # NEXT STEPS
    # -----------------------------
    st.header("Recommended Next Steps")
    st.markdown("""
    To validate and strengthen the modeling approach:

    - Expand the dataset with more samples  
    - Use cross-validation  
    - Check carefully for feature leakage  
    - Evaluate generalization on held-out or external data  
    - Consider simplifying features or adding regularization  
    """)


# ---------------------------------------------------------------------
# INSIGHTS
# ---------------------------------------------------------------------
elif page == "Insights":
    st.title("Key Insights & Recommendations")
    st.markdown("""
    - High-vulnerability counties like **Gila**, **Graham**, and **Apache** show limited healthcare access.  
    - Education and poverty are key predictors of vulnerability.  
    - Future versions can integrate **heat risk**, **air quality**, and **hospital access layers**.
    """)

# ---------------------------------------------------------------------
# AI ASSISTANT (SNOWFLAKE CORTEX VERSION — NO OPENAI KEY REQUIRED)
# ---------------------------------------------------------------------
elif page == "AI Assistant":
    st.title("AI Assistant for Arizona SVI Dashboard")

    df = session.table("SVI.PUBLIC.ARIZONA_CLEAN").to_pandas()

    safe_df = df.describe().replace([float("inf"), float("-inf")], None).fillna(0)
    summary_json = safe_df.to_json()

    user_question = st.text_area("What would you like to know?", height=150)

    if st.button("Generate AI Answer"):
        if user_question:

            with st.spinner("AI is analyzing the data..."):

                prompt_text = (
                    "You are an expert data analyst for Arizona SVI.\n"
                    f"Dataset summary: {summary_json}\n\n"
                    f"User question: {user_question}\n"
                    "Provide a clear and concise answer.\n"
                    "Do not include chain-of-thought. Only provide the final answer."
                )

                response = session.sql("""
                    SELECT snowflake.cortex.complete(
                        'deepseek-r1',
                        ?
                    ) AS result
                """,
                params=[prompt_text]
                ).collect()[0]["RESULT"]

            clean = extract_final_answer(response)

            st.markdown("### AI Answer")
            st.write(clean)


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("""
<br><br><br>
<footer>
    © 2025 Arizona SVI Dashboard — Developed by <b>Team Snowflake 1C</b> | Break Through Tech AI Studio  
    Data Source: CDC SVI 2025 & Arizona Department of Health Services  
</footer>
""", unsafe_allow_html=True)

