import pandas as pd
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import numpy as np




st.title("Heelo")
sns.set_theme(style="whitegrid")

# Make an example dataset with y ~ x
rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

# Plot the residuals after fitting a linear model
fig  = sns.residplot(x=x, y=y, lowess=True, color="g")
st.pyplot(fig)
