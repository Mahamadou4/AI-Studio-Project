# 🏜️ Arizona Social Vulnerability Dashboard (Streamlit on Snowflake)

An interactive **Streamlit application deployed inside Snowflake** that explores how **social vulnerability, heat risk, and healthcare access** intersect across Arizona counties.  
Built as part of **Break Through Tech AI Studio** by **Team Snowflake 1C**.

---

## 🚀 Overview

This dashboard connects **directly to Snowflake tables using Snowpark** and provides:

- Data understanding of raw and processed SVI + hospital datasets  
- County-level vulnerability and population analysis  
- Exploratory visual analysis of healthcare capacity, heat, and social barriers  
- Model performance interpretation  
- An AI assistant powered by **Snowflake Cortex** (no external API key required)

---

## 🧱 Tech Stack

- Snowflake Streamlit  
- Snowpark for Python  
- Streamlit  
- Pandas  
- Altair  
- Snowflake Cortex (LLM inference)

---

## Project Structure
├── app.py                     # Main Streamlit application
├── ziyan_eda_arizona.py        # Modular exploratory EDA logic
├── environment.yml             # Snowflake Python environment
└── README.md
---

## Data Sources (Snowflake Tables)

The app reads directly from Snowflake:

- `SVI.PUBLIC.ARIZONA_CLEAN`
- `SVI.PUBLIC.ARIZONA_2022_MAXTEMP`
- `SVI.PUBLIC.ARIZONA_LICENSED_HOSPITAL`
- `SVI.PUBLIC.SVI_HOSPITAL_MERGED`
- `SVI.PUBLIC.SVICLEANED`

No local CSVs are used.

---

## App Pages

### Home
- High-level introduction  
- Live preview of Arizona SVI data from Snowflake  

### Data Understanding
- Raw SVI and hospital data previews  
- Final engineered dataset (`SVI_HOSPITAL_MERGED`)  
- Feature reference guide  
- Data type composition visualization  

### County Analysis
- Population vs vulnerability metrics  
- Interactive county filtering and comparisons  

### Exploratory Visual Analysis
- Deep dive into:
  - Hospital capacity  
  - Social barriers  
  - Heat risk  
  - Population and daytime presence  
- County ranking charts  
- Outlier detection (IQR-based)  
- Plain-English interpretations for each variable  

### Modeling & Evaluation
- Summary of model performance  
- Interpretation of perfect scores  
- Discussion of overfitting and data leakage risks  
- Recommended next steps  

### AI Assistant
- Ask natural-language questions about the data  
- Uses **Snowflake Cortex (`deepseek-r1`)**  
- No OpenAI key required  
- Produces concise, policy-friendly insights  

---

## Environment Configuration

This app runs **inside Snowflake Streamlit**, using the following environment:

```yaml
name: app_environment
channels:
  - snowflake
dependencies:
  - python=3.10.*
  - snowflake-snowpark-python
  - streamlit
