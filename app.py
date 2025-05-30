import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import chardet
import re
from scipy import stats

# Configure page settings
st.set_page_config(
    page_title="CSV Data Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file):
    try:
        # First try to peek at the file content
        # Read first few lines for diagnosis
        preview = file.read(1024).decode('utf-8', errors='replace')
        st.write("File preview (first few bytes):")
        st.code(preview[:200])
        
        # Reset file pointer
        file.seek(0)
        
        # Try to detect encoding
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        st.write(f"Detected encoding: {encoding}")
        
        # Reset file pointer
        file.seek(0)
            
        # Try different delimiters with verbose output
        delimiters = [',', ';', '\t', '|']
        for delimiter in delimiters:
            try:
                st.write(f"Trying delimiter: '{delimiter}'")
                df = pd.read_csv(file, 
                               delimiter=delimiter,
                               encoding=encoding,
                               on_bad_lines='warn',
                               low_memory=False)
                
                if len(df.columns) > 1:
                    st.write(f"Successfully loaded with delimiter '{delimiter}'")
                    st.write(f"Found {len(df.columns)} columns: {list(df.columns)}")
                    return df
                else:
                    st.write(f"Only found {len(df.columns)} column(s) with delimiter '{delimiter}'")
            except Exception as e:
                st.write(f"Error with delimiter '{delimiter}': {str(e)}")
                file.seek(0)
                continue
        
        # If no delimiter worked, try pandas with Python engine
        st.write("Trying pandas with Python engine...")
        file.seek(0)
        df = pd.read_csv(file, encoding=encoding, engine='python', low_memory=False)
        return df
        
    except Exception as e:
        st.error(f"""Error reading file: {str(e)}
        
        Debugging information:
        1. File size: {file.size / 1024:.2f} KB
        2. Detected encoding: {encoding if 'encoding' in locals() else 'Unknown'}
        3. First few bytes shown above
        
        Common issues:
        1. Make sure your CSV file is properly formatted
        2. Check if the file has a header row
        3. The file might be empty or corrupted
        4. Try opening the file in a text editor to check its format
        5. The file might be using a different delimiter than expected
        """)
        return None

def check_csv_structure(file):
    """Check the CSV file structure and return header info"""
    try:
        # Read first few lines
        lines = []
        file.seek(0)  # Reset file pointer
        for _ in range(5):
            line = file.readline().decode('utf-8', errors='replace').strip()
            if not line:
                break
            lines.append(line)
    
        if not lines:
            st.error("File is empty")
            return None
    
        # Display preview
        st.write("First few lines of the file:")
        for i, line in enumerate(lines):
            st.code(f"Line {i+1}: {line}")
    
        # Try to detect if there's a header
        first_line_tokens = re.split(r'[,;\t|]', lines[0])
        second_line_tokens = re.split(r'[,;\t|]', lines[1]) if len(lines) > 1 else []
    
        # Check if first line might be a header
        has_header = False
        if second_line_tokens:
            # If first line has different data type pattern than second line,
            # it's likely a header
            first_numeric = all(token.replace('.','').isdigit() for token in first_line_tokens)
            second_numeric = all(token.replace('.','').isdigit() for token in second_line_tokens)
            has_header = first_numeric != second_numeric
    
        # Reset file pointer for subsequent reads
        file.seek(0)
        return has_header
    except Exception as e:
        st.error(f"Error checking file structure: {str(e)}")
        file.seek(0)  # Reset file pointer
        return None

def create_qq_plot(data, col):
    # Create Q-Q plot
    sorted_data = np.sort(data[col].dropna())
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    
    # Create the Q-Q plot using plotly
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers',
                               name='Data Points'))
    
    # Add the diagonal line
    slope = np.std(sorted_data)
    intercept = np.mean(sorted_data)
    line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
    line_y = slope * line_x + intercept
    
    qq_fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                               name='Normal Line', line=dict(dash='dash')))
    
    qq_fig.update_layout(title=f'Q-Q Plot for {col}',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles',
                        showlegend=True)
    
    return qq_fig

def create_visualizations(df):
    st.header("Data Visualizations")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Numeric Data Visualization
    if len(numeric_cols) > 0:
        st.subheader("Numeric Data Analysis")
        # Select columns for visualization
        selected_num_col = st.selectbox("Select a numeric column:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[selected_num_col], nbinsx=30))
            fig.update_layout(
                title=f'Histogram of {selected_num_col}',
                xaxis_title=selected_num_col,
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box Plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[selected_num_col], name=selected_num_col))
            fig.update_layout(
                title=f'Box Plot of {selected_num_col}',
                yaxis_title='Value'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Data Visualization
    if len(categorical_cols) > 0:
        st.subheader("Categorical Data Analysis")
        selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar Chart
            value_counts = df[selected_cat_col].value_counts()
            fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
            fig.update_layout(
                title=f'Bar Chart of {selected_cat_col}',
                xaxis_title=selected_cat_col,
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie Chart
            fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])
            fig.update_layout(title=f'Distribution of {selected_cat_col}')
            st.plotly_chart(fig, use_container_width=True)

def analyze_data(df):
    # Data Overview section
    st.header("Detailed Analysis")
    
    # Show basic info about the dataset
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Missing Values Analysis:")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.write("Data Type Summary:")
        dtype_df = pd.DataFrame({
            'Type': df.dtypes.value_counts().index,
            'Count': df.dtypes.value_counts().values
        })
        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Data Preview with pagination
    st.subheader("Paginated Data View")
    page_size = 5
    total_rows = len(df)
    max_pages = (total_rows + page_size - 1) // page_size
    
    col1, col2 = st.columns([2, 3])
    with col1:
        current_page = st.number_input("Page", min_value=1, max_value=max_pages, value=1)
    with col2:
        st.write(f"Showing page {current_page} of {max_pages} (Total rows: {total_rows})")
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Show paginated data in a proper table
    st.dataframe(
        df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=False
    )
    
    # Option to show full dataset with proper formatting
    if st.checkbox("Show full dataset"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=False,
            height=400  # Scrollable height
        )
    
    # Create visualizations with proper tables for data
    st.subheader("Data Visualizations")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("### Numeric Data Summary")
        numeric_summary = df[numeric_cols].describe()
        st.dataframe(
            numeric_summary,
            use_container_width=True,
            hide_index=False
        )
        
        create_visualizations(df)
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write("### Categorical Data Summary")
        for col in categorical_cols:
            st.write(f"#### {col} - Value Counts")
            value_counts_df = df[col].value_counts().reset_index()
            value_counts_df.columns = [col, 'Count']
            st.dataframe(
                value_counts_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Show QQ plots
    if len(numeric_cols) > 0:
        st.subheader("Distribution Analysis (Q-Q Plots)")
        for col in numeric_cols:
            qq_plot = create_qq_plot(df, col)
            st.write(f"### {col}")
            st.plotly_chart(qq_plot, use_container_width=True)

# Main app code
def main():
    st.title('Data Visualization Tool')    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Data Upload", "Stream Data"])

    with tab1:
        st.header("CSV File Upload")
        
        
        
        # File uploader with clear instructions
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to upload",
            type="csv",
            help="Upload a CSV file to analyze its contents and create visualizations"
        )
        
        if uploaded_file is not None:
            st.info(f"📂 File '{uploaded_file.name}' received - Size: {uploaded_file.size / 1024:.1f} KB")
            try:
                # Load the data with the enhanced load_data function
                df = load_data(uploaded_file)
                if df is not None and not df.empty:
                    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
                    create_visualizations(df)
                else:
                    st.error("Could not load the data. Please check if your file follows the format requirements above.")
                    st.info("Try opening your CSV file in a text editor to verify its contents and format.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please ensure your CSV file is properly formatted and try again.")

    with tab2:
        st.header("Stream Data")
        stream_url = st.text_input("Enter your streaming URL (e.g., YouTube, Twitch):")
        if stream_url:
            if st.button("View Stream"):
                try:
                    st.video(stream_url)
                except Exception as e:
                    st.error("Unable to load the stream. Please check the URL.")
                st.markdown(f"[Open in new window]({stream_url})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
