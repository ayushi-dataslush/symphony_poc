import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Sales Forecasting".upper())

    st.divider()
    st.sidebar.image("https://dataslush.com/wp-content/uploads/2023/03/Untitled-design-64-e1681395110215.png",
                     # use_column_width=True,
                     width=280)

    st.sidebar.info("""
        **Company Name:** DataSlush  
        **Industry:** Full Stack Data & AI Partner  
        **Website:** [www.dataslush.com](https://dataslush.com/)  

        Improve Business Decision Making with
        Your Full Stack Data & AI Partner
        Empowering Data-Driven Decisions: Scalable Data Engineering, Operationalized Machine Learning, and Actionable Analytics with DataSlush.
    """)
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Please Upload a CSV file:", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.sidebar.write("**Select Columns**:")
        date_column = st.sidebar.selectbox("Response Variable(X):", options=df.columns)
        target_column = st.sidebar.selectbox("Target Variable(y):", options=df.columns)
        periods = st.sidebar.number_input("Period:", min_value=30, max_value=365)

        if st.sidebar.button("FORCAST"):

            if date_column and target_column:
                df[date_column] = pd.to_datetime(df[date_column])

                df = df.rename(columns={date_column: 'ds', target_column: 'y'})

                if 'ds' in df.columns and 'y' in df.columns:
                    model = Prophet()
                    model.fit(df)

                    future = model.make_future_dataframe(periods=periods)
                    forecast = model.predict(future)

                    merged_df = pd.merge(forecast[['ds', 'yhat']], df[['ds', 'y']], on='ds', how='inner')

                    y_true = merged_df['y'].dropna()
                    y_pred = merged_df['yhat'].dropna()

                    if len(y_true) == len(y_pred):
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        mape = (abs(y_true - y_pred) / y_true).mean() * 100

                        st.subheader("Metrics:")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("MAE", round(mae, 1))
                        col2.metric("MSE", round(mse,1))
                        col3.metric("MAPE", round(mape,1))

                    else:
                        st.error("Error: The lengths of y_true and y_pred are inconsistent.")

                    st.subheader("Forecast Plot:")
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)

                    st.subheader("Forecast Components")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)
                else:
                    st.error("The DataFrame must have columns named 'ds' and 'y' after renaming.")
            else:
                st.error("Please select both date and target columns.")


if __name__ == "__main__":
    main()
