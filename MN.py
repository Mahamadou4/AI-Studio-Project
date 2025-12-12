from snowflake.snowpark.context import get_active_session
session = get_active_session()

import pandas as pd
import altair as alt


def load_data():
    df = session.table("SVI.PUBLIC.SVI_HOSPITAL_MERGED").to_pandas()
    return df


def render_data_overview(df):
    import streamlit as st

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Column List")
    st.write(list(df.columns))

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum())


def render_target_distribution(df):
    import streamlit as st

    st.subheader("Target Distribution")

    if "IS_MEDICAL_DESERT" not in df.columns:
        st.error("Column IS_MEDICAL_DESERT not found")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("IS_MEDICAL_DESERT:N", title="Medical Desert"),
            y="count()",
            color="IS_MEDICAL_DESERT:N",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_feature_distributions(df):
    import streamlit as st

    st.subheader("Feature Distributions")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col == "IS_MEDICAL_DESERT":
            continue

        st.write(f"### {col}")

        chart = (
            alt.Chart(df)
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
            )
        )

        st.altair_chart(chart, use_container_width=True)