import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="CSV Visualization Tool", layout="wide")

# Title
st.title("CSV Data Visualization Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], help="Upload a CSV with at least one categorical and one numerical column.")

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        # Column selection
        st.subheader("Select Columns for Visualization")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(categorical_cols) == 0 or len(numerical_cols) == 0:
            st.warning("Your CSV must have at least one categorical and one numerical column.")
        else:
            categorical_col = st.selectbox("Categorical Column (for Pie Chart)", categorical_cols)
            numerical_col = st.selectbox("Numerical Column (for Pie Chart and Histogram)", numerical_cols)
            bins = st.slider("Number of Histogram Bins", 5, 50, 10)
            
            # Pie Chart
            st.subheader("Pie Chart")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.pie(df[numerical_col], labels=df[categorical_col], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            ax1.set_title(f'Distribution of {numerical_col} by {categorical_col}')
            ax1.axis('equal')
            st.pyplot(fig1)
            
            # Histogram
            st.subheader("Histogram")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(df[numerical_col], bins=bins, ax=ax2, color='skyblue')
            ax2.set_title(f'Distribution of {numerical_col}')
            ax2.set_xlabel(numerical_col)
            ax2.set_ylabel('Frequency')
            st.pyplot(fig2)
            
            # Save and download charts
            if st.button("Save Charts"):
                fig1.savefig('pie_chart.png', dpi=300)
                fig2.savefig('histogram.png', dpi=300)
                st.success("Charts saved as 'pie_chart.png' and 'histogram.png' in the server!")
            
            if st.button("Download Pie Chart"):
                with open('pie_chart.png', 'rb') as file:
                    st.download_button(label="Download Pie Chart", data=file, file_name="pie_chart.png", mime="image/png")
            if st.button("Download Histogram"):
                with open('histogram.png', 'rb') as file:
                    st.download_button(label="Download Histogram", data=file, file_name="histogram.png", mime="image/png")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
else:
    st.info("Upload a CSV file to visualize its data.")
