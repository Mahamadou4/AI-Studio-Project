# ziyan_eda_arizona.py

# Exploratory EDA page: compare counties one variable at a time
# (heat, hospitals, social vulnerability).


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from snowflake.snowpark.context import get_active_session

# TABLE NAMES & COLUMNS

# import 
MERGED_TABLE_NAME = "SVI.PUBLIC.SVI_HOSPITAL_MERGED"  # merged county-level table
SVI_CLEAN_TABLE_NAME = "SVI.PUBLIC.SVICLEANED"        # for daytime population

# mapping
MERGED_COLUMN_MAP = {
    "county": "county",
    "HOSPITALS_PER_10K": "hospitals_per_10k",
    "BEDS_PER_10K": "beds_per_10k",
    "NUM_HOSPITALS": "num_hospitals",
    "POPULATION_TOTAL": "population_total",
    "TOTAL_CAPACITY": "total_capacity",
    "MAX_TEMP_2022": "max_temp_2022",
    "E_HEAT_RISK_MEAN": "e_heat_risk_mean",
    "MP_NOVEH_MEAN": "mp_noveh_mean",
    "MP_POV150_MEAN": "mp_pov150_mean",
    "MP_UNINSUR_MEAN": "mp_uninsur_mean",
    "IS_MEDICAL_DESERT": "is_medical_desert",
}


EDA_VARS = [
    "hospitals_per_10k",
    "beds_per_10k",
    "num_hospitals",
    "population_total",
    "total_capacity",
    "mp_pov150_mean",
    "mp_uninsur_mean",
    "mp_noveh_mean",
    "e_heat_risk_mean",
    "max_temp_2022",
    "e_daypop_mean",   
]

# make categories for user
THEME_VARS = {
    "Capacity and access": [
        "hospitals_per_10k",
        "beds_per_10k",
        "num_hospitals",
        "total_capacity",
    ],
    "Population and daytime presence": [
        "population_total",
        "e_daypop_mean",
    ],
    "Social barriers": [
        "mp_pov150_mean",
        "mp_uninsur_mean",
        "mp_noveh_mean",
    ],
    "Heat and climate": [
        "e_heat_risk_mean",
        "max_temp_2022",
    ],
}

# readable var names
VAR_LABELS = {
    "hospitals_per_10k": "Hospitals per 10k residents",
    "beds_per_10k": "Beds per 10k residents",
    "num_hospitals": "Number of hospitals",
    "population_total": "Total population",
    "total_capacity": "Total bed capacity",
    "mp_pov150_mean": "Poverty (relative to average)",
    "mp_uninsur_mean": "Uninsured (relative to average)",
    "mp_noveh_mean": "No-vehicle households (relative to average)",
    "e_heat_risk_mean": "Heat risk index",
    "max_temp_2022": "Maximum temperature (2022)",
    "e_daypop_mean": "Daytime population",
    "is_medical_desert": "Medical desert flag",
}


# DATA LOADING

@st.cache_data
def load_eda_data():
    """
    Load county-level data from Snowflake.
    1) Read SVI_HOSPITAL_MERGED.
    2) Rename columns with MERGED_COLUMN_MAP.
    3) Merge E_DAYPOP_MEAN from SVICLEANED (if available).
    """
    session = get_active_session()

    try:
        merged = session.table(MERGED_TABLE_NAME).to_pandas()
    except Exception as e:
        st.error(f"Error loading `{MERGED_TABLE_NAME}`: {e}")
        return None

    rename_dict = {
        snow_col: short_name
        for snow_col, short_name in MERGED_COLUMN_MAP.items()
        if snow_col in merged.columns
    }
    df = merged.rename(columns=rename_dict)

    if "county" not in df.columns:
        st.error("Merged table must include a `county` column.")
        return None

    try:
        svi_clean = session.table(SVI_CLEAN_TABLE_NAME).to_pandas()
        if "county" in svi_clean.columns and "E_DAYPOP_MEAN" in svi_clean.columns:
            svi_small = svi_clean[["county", "E_DAYPOP_MEAN"]].rename(
                columns={"E_DAYPOP_MEAN": "e_daypop_mean"}
            )
            df = df.merge(svi_small, on="county", how="left")
    except Exception as e:
        st.warning(f"Could not merge SVICLEANED (E_DAYPOP_MEAN): {e}")

    keep_cols = ["county"] + EDA_VARS + ["is_medical_desert"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def iqr_outliers(df, col):
    """
    Returns: cutoff value, and a DataFrame of outlier counties.
    """
    s = df[col].dropna()
    if s.empty:
        return np.nan, df.head(0)

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    cutoff = q3 + 1.5 * (q3 - q1)
    outliers = df[df[col] > cutoff]
    return cutoff, outliers


def theme_description(theme):
    """Short explanation for each theme."""
    if theme == "Capacity and access":
        return "Do counties have enough hospitals and beds for the people who live there?"
    if theme == "Population and daytime presence":
        return "Where are people physically present when heat events occur?"
    if theme == "Social barriers":
        return "How do poverty, insurance coverage and transport barriers differ across counties?"
    if theme == "Heat and climate":
        return "How intense is the heat that each county experiences?"
    return ""


# INTERPRETATION FOR A SINGLE VARIABLE

def interpretation_univariate(var, series, df):
    """
    Explain one variable in four parts:
    1) What it measures.
    2) How counties are distributed.
    3) Why it matters for heat & medical access.
    4) How it could be used in a model.
    """
    s = series.dropna()
    if s.empty:
        st.write("No data available for this variable.")
        return

    label = VAR_LABELS.get(var, var)
    median = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    vmin = s.min()
    vmax = s.max()

    ratio = None
    if vmin > 0:
        r = vmax / vmin
        if 1.0 < r <= 10.0:
            ratio = r

    sorted_df = df.loc[s.sort_values().index, ["county", var]]
    bottom3 = sorted_df.head(3)
    top3 = sorted_df.tail(3)

    # What this variable measures
    st.markdown("#### What this variable measures")

    if var == "hospitals_per_10k":
        var_explanation = (
            "Number of hospitals per 10,000 residents in each county. "
            "Higher values mean more facilities relative to population."
        )
    elif var == "beds_per_10k":
        var_explanation = (
            "Number of hospital beds per 10,000 residents. "
            "Higher values indicate greater inpatient capacity per person."
        )
    elif var == "num_hospitals":
        var_explanation = "Total count of hospitals in the county."
    elif var == "population_total":
        var_explanation = "Estimated resident population in the county."
    elif var == "total_capacity":
        var_explanation = "Total number of hospital beds across all hospitals in the county."
    elif var in ["mp_pov150_mean", "mp_uninsur_mean", "mp_noveh_mean"]:
        var_explanation = (
            "These variables are centred around zero and represent relative vulnerability. "
            "Values above 0 mean the county has more of this condition than the reference average; "
            "values below 0 mean less than average."
        )
    elif var == "e_heat_risk_mean":
        var_explanation = "Index summarising average heat risk; higher values mean greater heat exposure."
    elif var == "max_temp_2022":
        var_explanation = "Maximum recorded temperature in 2022 for the county."
    elif var == "e_daypop_mean":
        var_explanation = (
            "Estimated daytime population, including residents plus commuters and visitors present during the day."
        )
    else:
        var_explanation = "County-level summary for the metric shown on the chart."

    st.write(var_explanation)

    # Distribution and inequality
    st.markdown("#### How counties compare on this metric")

    bullets = [
        f"- A typical county is around **{median:,.1f} {label}**.",
        f"- Half of counties fall between **{q1:,.1f}** and **{q3:,.1f}**.",
        f"- The lowest observed value is **{vmin:,.1f}**, and the highest is **{vmax:,.1f}**.",
    ]
    if ratio is not None:
        bullets.append(
            f"- The highest county is about **{ratio:,.1f} times** the lowest on this metric."
        )
    st.write("\n".join(bullets))

    st.markdown(
        """
<div class="insight-callout">
The bar chart above is a ranking: counties on the left have the highest values on this metric,
and counties on the right have the lowest. This makes it easier to see which counties consistently
sit at the top or bottom.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Counties with the highest values (top 3)**")
        st.table(
            top3.rename(columns={"county": "County", var: label}).reset_index(drop=True)
        )
    with col2:
        st.markdown("**Counties with the lowest values (bottom 3)**")
        st.table(
            bottom3.rename(columns={"county": "County", var: label}).reset_index(drop=True)
        )

    # Why this matters for heat and access
    st.markdown("#### Why this is relevant for heat and medical access")

    if var == "hospitals_per_10k":
        text = (
            "Low values indicate potential gaps in facility access. "
            "During extreme heat, residents may face longer travel times and delays before reaching care."
        )
    elif var == "beds_per_10k":
        text = (
            "Low bed counts per resident suggest limited capacity to admit patients. "
            "Counties with both high heat risk and low beds per 10k may overload quickly."
        )
    elif var == "num_hospitals":
        text = (
            "Counties with very few hospitals have little redundancy. "
            "If one facility is unavailable, options for residents are limited."
        )
    elif var == "population_total":
        text = (
            "High-population counties require more infrastructure. "
            "If population is high but hospital access metrics are low, the system may be strained."
        )
    elif var == "total_capacity":
        text = (
            "Total bed capacity describes the size of the system. "
            "Low-capacity counties may struggle to manage larger numbers of heat-related admissions."
        )
    elif var == "mp_pov150_mean":
        text = (
            "Positive values indicate more residents near or below the poverty line than average. "
            "Financial constraints can limit access to cooling, transport and timely care."
        )
    elif var == "mp_uninsur_mean":
        text = (
            "Positive values indicate more uninsured residents than average. "
            "Uninsured individuals often delay care, which can worsen outcomes during heat events."
        )
    elif var == "mp_noveh_mean":
        text = (
            "Positive values indicate more households without vehicles than average. "
            "These residents may find it harder to reach hospitals when they need urgent care."
        )
    elif var == "e_heat_risk_mean":
        text = (
            "Higher scores indicate greater exposure to dangerous heat. "
            "Counties combining high heat risk with low capacity or high social barriers are of particular concern."
        )
    elif var == "max_temp_2022":
        text = (
            "High maximum temperatures increase physiological stress and can trigger more heat-related illness."
        )
    elif var == "e_daypop_mean":
        text = (
            "High daytime population means demand for emergency services may be higher than resident population alone suggests."
        )
    else:
        text = (
            "This metric captures one aspect of county-level risk or capacity that interacts with heat conditions."
        )

    st.write(text)

    # How this might be used in a model
    st.markdown("#### Possible use in a machine-learning model")

    if var in ["hospitals_per_10k", "beds_per_10k", "num_hospitals", "total_capacity"]:
        st.write(
            "- These variables summarise health system capacity.\n"
            "- In a model predicting heat-related hospitalisations by county, "
            "they help explain which counties are more likely to experience strain when demand increases."
        )
    elif var in ["mp_pov150_mean", "mp_uninsur_mean", "mp_noveh_mean"]:
        st.write(
            "- These variables represent social and economic barriers to care.\n"
            "- Using them as features lets the model capture how poverty, insurance and transport "
            "affect outcomes, beyond physical capacity alone."
        )
    elif var in ["e_heat_risk_mean", "max_temp_2022"]:
        st.write(
            "- These are exposure variables that quantify how extreme the heat is.\n"
            "- Combined with capacity and vulnerability variables, they help identify counties where high exposure "
            "and low access overlap."
        )
    elif var in ["population_total", "e_daypop_mean"]:
        st.write(
            "- These variables describe the size of the population at risk.\n"
            "- They can be used as predictors and also for constructing outcome rates "
            "(for example, hospitalisations per 10,000 residents)."
        )
    else:
        st.write(
            "- This variable can be included as a county-level feature after appropriate scaling or standardisation."
        )


# entry function to call from the main page.

def render_exploratory_eda():
    """
    Main function to draw the EDA page.
    This will be called from your main Streamlit app.
    """

    st.title("Exploratory EDA: "
    "Heat, Population, SVI, Access to Transportation and State-licensed Hospital.")
    st.write(
        "This section explores how Arizona counties differ in hospital capacity, social vulnerability, and exposure to heat. "
        "The exploratory view focuses on one variable at a time so it is easy to see which counties are consistently better or worse off."
    )

    df = load_eda_data()
    if df is None:
        return

    st.caption(f"Loaded data for {len(df)} counties in Arizona.")
    st.markdown("---")

    # Choose theme and metric
    st.subheader("1. Click to choose a theme and metric")

    available_vars = [v for v in EDA_VARS if v in df.columns]
    theme_var_map = {
        theme: [v for v in vars_list if v in available_vars]
        for theme, vars_list in THEME_VARS.items()
    }
    theme_var_map = {k: v for k, v in theme_var_map.items() if v}

    if not theme_var_map:
        st.error("No variables found. Check EDA_VARS and column mappings.")
        return

    theme = st.radio("Theme", list(theme_var_map.keys()), horizontal=True)
    st.caption(theme_description(theme))

    theme_vars = theme_var_map[theme]
    var = st.selectbox(
        "Metric (Click to select)",
        theme_vars,
        format_func=lambda x: VAR_LABELS.get(x, x),
    )
    label = VAR_LABELS.get(var, var)
    s = df[var].astype(float)

    # 5.2 County ranking chart
    st.markdown("---")
    st.subheader("2. County ranking on the selected metric")

    s_non = s.dropna()
    if s_non.empty:
        st.write("No data for this variable.")
        return

    q1 = s_non.quantile(0.25)
    q3 = s_non.quantile(0.75)

    df_bar = (
        df[["county", var]]
        .dropna()
        .sort_values(by=var, ascending=False)
        .reset_index(drop=True)
    )

    def bucket(x):
        if x >= q3:
            return "Top 25%"
        elif x <= q1:
            return "Bottom 25%"
        return "Middle"

    df_bar["Group"] = df_bar[var].apply(bucket)

    chart = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("county:N", sort="-y", title="County"),
            y=alt.Y(f"{var}:Q", title=label, scale=alt.Scale(nice=True)),
            color=alt.Color(
                "Group:N",
                title="Group",
                sort=["Top 25%", "Middle", "Bottom 25%"],
            ),
            tooltip=[
                "county",
                alt.Tooltip(var, title=label, format=",.2f"),
                "Group",
            ],
        )
        .properties(
            height=380,
            padding={"top": 30, "bottom": 10, "left": 5, "right": 5},
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12,
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # Interpretation and outliers
    st.markdown("---")
    st.subheader("3. Interpretation")

    interpretation_univariate(var, s, df)

    st.markdown("#### Counties that are unusually high (IQR rule)")
    cutoff, outliers = iqr_outliers(df, var)
    if outliers.empty or np.isnan(cutoff):
        st.write("No counties appear as strong high outliers on this metric.")
    else:
        st.write(
        f"""We use something called the interquartile range (IQR) to highlight counties that are much higher than a typical county. 
        This helps us see which places are not just high but unusually high on this metric.  
        Any county above approximately **{cutoff:,.2f} {label}** is higher than would be expected from a typical spread."""
        )
        st.dataframe(
            outliers[["county", var]]
            .sort_values(by=var, ascending=False)
            .rename(columns={"county": "County", var: label})
            .reset_index(drop=True)
        )
