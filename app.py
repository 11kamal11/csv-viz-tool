import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import chardet

# Set page configuration
st.set_page_config(page_title="Data Analysis & Visualization Tool", layout="wide", initial_sidebar_state="expanded")

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

def create_qq_plot(data, col):
    sorted_data = np.sort(data[col].dropna())
    if len(sorted_data) == 0:
        return None
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    slope = np.std(sorted_data)
    intercept = np.mean(sorted_data)
    line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
    line_y = slope * line_x + intercept
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Data Points', marker=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Normal Line', line=dict(dash='dash', color='#ff7f0e')))
    fig.update_layout(
        title=f'Q-Q Plot for {col}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        template='plotly_white',
        font=dict(family="Arial", size=12),
        showlegend=True
    )
    return fig

def generate_conclusion(df, num_col, cat_col):
    summary_stats = df[num_col].describe()
    if '50%' not in summary_stats:
        return f"**Analysis Conclusion**: Insufficient data for {num_col} to compute detailed statistics."
    max_category = df.loc[df[num_col].idxmax(), cat_col] if cat_col in df.columns and not df[num_col].empty else "N/A"
    skew = abs(summary_stats['mean'] - summary_stats['50%']) > summary_stats['std'] / 2
    conclusion = f"""
    **Analysis Conclusion**:
    - **Average {num_col}**: {summary_stats['mean']:.2f} (Std: {summary_stats['std']:.2f})
    - **Range**: {summary_stats['min']:.2f} to {summary_stats['max']:.2f}
    - **Top Category**: '{max_category}' has the highest {num_col}
    - **Distribution**: {'Skewed' if skew else 'Relatively normal'} based on mean-median difference
    - **Insight**: {f'High variability in {num_col} suggests diverse data points.' if summary_stats['std'] > summary_stats['mean'] / 2 else f'Low variability in {num_col} indicates consistent data.'}
    """
    return conclusion

def main():
    st.title("Modern CSV Data Analysis & Visualization")
    
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
                    cat_col = st.selectbox("Categorical Column", categorical_cols)
                with col2:
                    num_col = st.selectbox("Numerical Column", numeric_cols)
                bins = st.slider("Histogram Bins", 5, 50, 10, step=5)
                
                figs = []
                
                # Pie Chart
                st.subheader("Pie Chart")
                value_counts = df.groupby(cat_col)[num_col].sum().reset_index()
                fig1 = px.pie(value_counts, names=cat_col, values=num_col, title=f'Distribution of {num_col} by {cat_col}',
                              color_discrete_sequence=px.colors.qualitative.Plotly)
                fig1.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig1, use_container_width=True)
                figs.append((fig1, "pie_chart.png"))
                
                # Histogram
                st.subheader("Histogram")
                fig2 = px.histogram(df, x=num_col, nbins=bins, title=f'Distribution of {num_col}',
                                   color_discrete_sequence=['#1f77b4'])
                fig2.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig2, use_container_width=True)
                figs.append((fig2, "histogram.png"))
                
                # Bar Chart
                st.subheader("Bar Chart")
                fig3 = px.bar(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}',
                             color_discrete_sequence=['#ff7f0e'])
                fig3.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig3, use_container_width=True)
                figs.append((fig3, "bar_chart.png"))
                
                # Scatter Plot
                st.subheader("Scatter Plot")
                if len(numeric_cols) >= 2:
                    num_col2 = st.selectbox("Second Numerical Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_num")
                    fig4 = px.scatter(df, x=num_col, y=num_col2, color=cat_col, title=f'{num_col} vs {num_col2}',
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    fig4.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                    st.plotly_chart(fig4, use_container_width=True)
                    figs.append((fig4, "scatter_plot.png"))
                else:
                    st.warning("Scatter plot requires at least two numerical columns.")
                    figs.append((None, "scatter_plot.png"))
                
                # Box Plot
                st.subheader("Box Plot")
                fig5 = px.box(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}',
                             color_discrete_sequence=['#2ca02c'])
                fig5.update_layout(template='plotly_white', font=dict(family="Arial", size=12))
                st.plotly_chart(fig5, use_container_width=True)
                figs.append((fig5, "box_plot.png"))
                
                # Q-Q Plot
                st.subheader("Q-Q Plot")
                fig6 = create_qq_plot(df, num_col)
                if fig6:
                    st.plotly_chart(fig6, use_container_width=True)
                    figs.append((fig6, "qq_plot.png"))
                else:
                    st.warning(f"No valid data for Q-Q plot of {num_col}.")
                    figs.append((None, "qq_plot.png"))
                
                # Conclusion
                st.header("Conclusion")
                conclusion = generate_conclusion(df, num_col, cat_col)
                st.markdown(conclusion)
                
                # Download Charts
                st.header("Download Charts")
                chart_names = ["Pie Chart", "Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Q-Q Plot"]
                for i, (fig, fname) in enumerate(figs):
                    if fig is None:
                        continue
                    fig.write_image(fname, width=600, height=400)
                    with open(fname, 'rb') as file:
                        st.download_button(f"Download {chart_names[i]}", file, file_name=fname, mime="image/png")

if __name__ == "__main__":
    main()