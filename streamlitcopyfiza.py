import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from ziyan_eda_arizona import render_exploratory_eda
from openai import OpenAI



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

# ---------------------------------------------------------------------
# DATA UNDERSTANDING
# ---------------------------------------------------------------------

elif page == "Data Understanding":
    st.title("Data Understanding & Exploration")

    df = session.table("SVI.PUBLIC.ARIZONA_CLEAN").to_pandas()
    st.write(f"Each row represents a county in Arizona. **Rows:** {len(df):,} | **Columns:** {len(df.columns):,}")

    # --- Top 10 Most Populous Counties ---
    st.markdown("### 🌡️ Top 10 Most Populous Counties")
    top10 = df.sort_values("E_TOTPOP", ascending=False).head(10)
    top10 = top10.rename(columns={"E_TOTPOP": "Estimated Total Population"})
    st.bar_chart(data=top10, x="COUNTY", y="Estimated Total Population")
    st.caption("Displays the 10 most populous counties — population often correlates with vulnerability and resource strain.")

    # --- Population vs Social Vulnerability ---
    st.markdown("### Population vs Social Vulnerability Index")
    df_viz = df.rename(columns={
        "E_TOTPOP": "Estimated Total Population",
        "RPL_THEMES": "Social Vulnerability Index"
    })
    st.scatter_chart(
        data=df_viz,
        x="Estimated Total Population",
        y="Social Vulnerability Index",
        color="COUNTY"
    )
    st.caption("Shows how social vulnerability changes with county population levels.")

    # --- Reference Guide ---
    st.markdown("""
    ### 📘 Metric Reference Guide
    **RPL_THEMES** — Overall Social Vulnerability Index (Composite Score)  
    **E_UNEMP** — Estimated Unemployed Population  
    **E_POV150** — Population Below 150% of the Poverty Line  
    **E_NOHSDP** — Population Without a High School Diploma  
    **E_UNINSUR** — Uninsured Population  
    **E_MINRTY** — Minority Population  
    **E_DISABL** — Population with Disabilities  
    **E_TOTPOP** — Estimated Total Population  
    """)

    # -----------------------------
    # 🔥 FULL MN.PY LOGIC BELOW (embedded directly)
    # -----------------------------

    st.markdown("---")
    st.header("Additional Feature Exploration")

    # === Dataset Overview ===
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Column List")
    st.write(list(df.columns))

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum())

    # === Target Distribution ===
    st.subheader("Target Distribution (IS_MEDICAL_DESERT)")
    if "IS_MEDICAL_DESERT" in df.columns:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("IS_MEDICAL_DESERT:N", title="Medical Desert"),
                y="count()",
                color="IS_MEDICAL_DESERT:N"
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Column 'IS_MEDICAL_DESERT' not found in dataset.")

    # === Feature Distributions ===
    st.subheader("Numeric Feature Distributions")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col == "IS_MEDICAL_DESERT":
            continue

        st.markdown(f"### {col}")

        chart = (
            alt.Chart(df)
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
            )
        )
        st.altair_chart(chart, use_container_width=True)


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
# INSIGHTS
# ---------------------------------------------------------------------
elif page == "Insights":
    st.title("💡 Key Insights & Recommendations")
    st.markdown("""
    - High-vulnerability counties like **Gila**, **Graham**, and **Apache** show limited healthcare access.  
    - Education and poverty are key predictors of vulnerability.  
    - Future versions can integrate **heat risk**, **air quality**, and **hospital access layers**.
    """)

# ---------------------------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------------------------
elif page == "AI Assistant":
    st.title("🤖 AI Assistant for Arizona SVI Dashboard")
    st.write("Ask questions about counties, vulnerability, heat risk, hospitals, or any visualization.")

    # Load OpenAI key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Load dataset
    df = session.table("SVI.PUBLIC.ARIZONA_CLEAN").to_pandas()

    # User input
    user_question = st.text_area("What would you like to know?", height=150)

    # Ask AI
    if st.button("Generate AI Answer"):
        if user_question.strip():
            with st.spinner("AI is analyzing the data..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Arizona Social Vulnerability analyst. "
                                "Use clear reasoning, reference specific counties, and justify conclusions "
                                "using the dataset summary provided."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Here are the dataset's summary statistics: {df.describe().to_json()}\n\n"
                                f"Full column list: {df.columns.tolist()}\n\n"
                                f"Question: {user_question}"
                            )
                        }
                    ]
                )

            st.markdown("### 🔍 AI Answer")
            st.write(response.choices[0].message["content"])

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