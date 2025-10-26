import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("My Streamlit Dashboard")

# Example DataFrame
df = pd.DataFrame({
    "Category": ["A", "B", "C"],
    "Values": [10, 20, 30]
})

fig = px.bar(df, x="Category", y="Values", title="Example Bar Chart")
st.plotly_chart(fig)
