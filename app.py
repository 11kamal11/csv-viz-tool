import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import chardet

# Set page configuration
st.set_page_config(page_title="CSV Data Analysis Tool", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(ttl=3600)
def load_data(file):
    try:
        # Detect encoding
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        file.seek(0)
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        for delimiter in delimiters:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='warn', low_memory=False)
                if len(df.columns) > 1:
                    st.info(f"Loaded CSV with delimiter '{delimiter}' and encoding '{encoding}'")
                    return df
            except:
                continue
        # Fallback to Python engine
        file.seek(0)
        df = pd.read_csv(file, encoding=encoding, engine='python', low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}\n\nEnsure your CSV has headers and proper formatting.")
        return None

def main():
    st.title("CSV Data Analysis & Visualization")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Upload a CSV with at least one categorical and one numerical column.")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns!")
            
            # Data Analysis
            st.header("Data Analysis")
            st.subheader("Full Data Table")
            page_size = 10
            total_rows = len(df)
            max_pages = (total_rows + page_size - 1) // page_size
            col1, col2 = st.columns([1, 2])
            with col1:
                current_page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
            with col2:
                st.write(f"Showing page {current_page} of {max_pages} (Total rows: {total_rows})")
            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, hide_index=False)
            
            if st.checkbox("Show Full Dataset", help="Display all rows (may be slow for large datasets)"):
                st.dataframe(df, use_container_width=True, hide_index=False, height=400)
            
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes}), use_container_width=True, hide_index=True)
            
            # Visualizations
            st.header("Visualizations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols or not numeric_cols:
                st.warning("CSV must have at least one categorical and one numerical column.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    cat_col = st.selectbox("Categorical Column for Bar Chart", categorical_cols)
                with col2:
                    num_col = st.selectbox("Numerical Column for Bar Chart/Histogram", numeric_cols)
                bins = st.slider("Histogram Bins", 5, 50, 10, step=5)
                
                figs = []
                
                # Bar Chart
                st.subheader("Bar Chart")
                bar_data = df.groupby(cat_col)[num_col].mean().reset_index()
                fig1 = px.bar(bar_data, x=cat_col, y=num_col, title=f'Average {num_col} by {cat_col}',
                              color_discrete_sequence=['#ff7f0e'])
                fig1.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig1, use_container_width=True)
                figs.append((fig1, "bar_chart.png"))
                
                # Histogram
                st.subheader("Histogram")
                fig2 = px.histogram(df, x=num_col, nbins=bins, title=f'Distribution of {num_col}',
                                    color_discrete_sequence=['#1f77b4'])
                fig2.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig2, use_container_width=True)
                figs.append((fig2, "histogram.png"))
                
                # Download Charts
                st.header("Download Charts")
                chart_names = ["Bar Chart", "Histogram"]
                for i, (fig, fname) in enumerate(figs):
                    fig.write_image(fname, width=600, height=400)
                    with open(fname, 'rb') as file:
                        st.download_button(f"Download {chart_names[i]}", file, file_name=fname, mime="image/png")

if __name__ == "__main__":
    main()