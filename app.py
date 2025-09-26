import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle
st.set_page_config(page_title="Simple Linear Regression Visualizer", layout="centered")

st.title("📊 Linear Regression Playground")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Preview of Data")
    st.write(df.head())

    target_col = st.selectbox("Select Target Column (y)", df.columns)
    feature_cols = st.multiselect("Select Feature Column(s) (X)", [col for col in df.columns if col != target_col])

    if target_col and feature_cols:
        X = df[feature_cols].values
        y = df[target_col].values

        if len(feature_cols) == 1:
            st.subheader("⚙️ Adjust Weight (w) and Bias (b) manually")

            w = st.slider("Weight (w)", -1000.0, 1000.0, 1.0, 0.1)
            b = st.slider("Bias (b)", -1000.0, 1000.0, 0.0, 0.1)

            y_pred_manual = w * X + b

            fig, ax = plt.subplots()
            ax.scatter(X, y, label="Data")
            ax.plot(X, y_pred_manual, color="red", label=f"y = {w:.2f}x + {b:.2f}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("📌 Manual weight/bias visualization is only available when you select ONE feature.")


        if st.button("Run Linear Regression (Auto Fit)"):
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)


            st.session_state["model"] = model
            st.session_state["feature_cols"] = feature_cols
            st.session_state["target_col"] = target_col
            st.session_state["df"] = df

            st.subheader("📈 Regression Results")
            eqn = " + ".join([f"{coef:.2f}*{name}" for coef, name in zip(model.coef_, feature_cols)])
            st.write(f"**Equation:** y = {eqn} + {model.intercept_:.2f}")
            st.write(f"**R² Score:** {r2_score(y, y_pred):.4f}")
            st.write(f"**MSE:** {mean_squared_error(y, y_pred):.4f}")
            model_bytes = pickle.dumps(model)
            st.download_button(
        label="📥 Download Trained Model",
        data=model_bytes,
        file_name="linear_regression_model.pkl",
        mime="application/octet-stream"
    )
           
          
            if len(feature_cols) == 1:
                fig, ax = plt.subplots()
                ax.scatter(X, y, label="Data")
                ax.plot(X, y_pred, color="green", label="Best Fit Line")
                ax.legend()
                st.pyplot(fig)


if "model" in st.session_state:
    st.subheader("🔮 Try Prediction with Your Own Inputs")

    model = st.session_state["model"]
    feature_cols = st.session_state["feature_cols"]
    df = st.session_state["df"]
    target_col = st.session_state["target_col"]

    input_values = []
    cols = st.columns(len(feature_cols))
    for i, feature in enumerate(feature_cols):
        val = cols[i].number_input(
            f"Enter value for {feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )
        input_values.append(val)

    if st.button("Predict"):
        pred = model.predict([input_values])[0]
        st.success(f"✅ Predicted {target_col}: {pred:.2f}")
