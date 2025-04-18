import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import logging
import numpy as np
from datetime import datetime
import re

# Configure logging
logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('ggplot')
sns.set_palette('Set2')

def generate_visualization(data, chart_type, entities, original_query):
    """
    Generate a visualization based on query results.
    
    Args:
        data (list): Query results as a list of dictionaries
        chart_type (str): Type of chart to generate (pie, bar, line, histogram)
        entities (list): List of entities detected in the query
        original_query (str): The original natural language query
        
    Returns:
        tuple: (image_bytes, insights)
            - image_bytes: PNG image of the chart as bytes
            - insights: Text insights about the data
    """
    logger.info(f"Generating {chart_type} visualization for {len(data)} records")
    
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        return _generate_empty_chart(), "No data available for this query."
    
    # Get chart title from query
    title = _generate_title_from_query(original_query)
    
    # Configure figure size
    plt.figure(figsize=(10, 6))
    
    # Create chart based on type
    if chart_type == 'pie':
        image_bytes, insights = _create_pie_chart(df, title, entities)
    elif chart_type == 'bar':
        image_bytes, insights = _create_bar_chart(df, title, entities)
    elif chart_type == 'line':
        image_bytes, insights = _create_line_chart(df, title, entities)
    elif chart_type == 'histogram':
        image_bytes, insights = _create_histogram(df, title, entities)
    else:
        # Default to bar chart
        image_bytes, insights = _create_bar_chart(df, title, entities)
    
    return image_bytes, insights

def _generate_title_from_query(query):
    """Generate a chart title from the original query."""
    # Remove question marks and make title case
    title = query.replace('?', '').strip().title()
    
    # Ensure title isn't too long
    if len(title) > 50:
        title = title[:47] + '...'
        
    return title

def _create_pie_chart(df, title, entities):
    """Create a pie chart without filtering out small values, and show a key/legend."""
    try:
        df = df.copy()

        # Attempt to convert numeric-looking columns
        df = df.apply(pd.to_numeric, errors='ignore')

        # Detect value and label columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Fetched data for PIE chart: {df}")

        if not numeric_cols or not text_cols:
            return _generate_empty_chart(), "Could not determine appropriate columns for pie chart."

        # Choose columns
        value_col = numeric_cols[0]
        label_col = text_cols[0]

        # Create pie chart and capture patch handles for legend
        wedges, texts, autotexts = plt.pie(
            df[value_col],
            labels=None,  # Labels in legend instead
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )

        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        plt.title(title)

        # Add a legend on the right
        plt.legend(
            wedges,
            df[label_col],
            title="Legend",
            loc="center left",
            bbox_to_anchor=(1, 0.5),  # Position it to the right of the pie
            fontsize='small'
        )

        # Generate insights
        top_item = df.loc[df[value_col].idxmax()]
        total_value = df[value_col].sum()

        insights = f"The largest segment is {top_item[label_col]} with {top_item[value_col]:,.2f} "
        insights += f"({(top_item[value_col] / total_value) * 100:.1f}% of total). "

        if len(df) > 1:
            second_top = df.nlargest(2, value_col).iloc[1]
            insights += f"Followed by {second_top[label_col]} with {second_top[value_col]:,.2f} "
            insights += f"({(second_top[value_col] / total_value) * 100:.1f}%)."

        # Save chart to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return buf.getvalue(), insights

    except Exception as e:
        logger.error(f"Error creating pie chart: {e}")
        return _generate_empty_chart(), f"Error creating pie chart: {e}"

    

def _create_bar_chart(df, title, entities):
    """Create a bar chart."""
    try:
        df = df.copy()
        df = df.apply(pd.to_numeric, errors='ignore')
        # Detect columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or not text_cols:
            return _generate_empty_chart(), "Could not determine appropriate columns for bar chart."
        
        # Use the first text column as x and first numeric column as y
        x_col = text_cols[0]
        y_col = numeric_cols[0]
        
        # Limit to top 10 items if more than 10
        if len(df) > 10:
            df = df.nlargest(10, y_col)
        
        # Sort by value descending
        df = df.sort_values(by=y_col, ascending=False)
        
        # Create bar chart
        bars = plt.bar(df[x_col], df[y_col])
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}', ha='center', va='bottom', rotation=0)
        
        # Generate insights
        top_item = df.iloc[0]
        
        insights = f"The highest value is {top_item[x_col]} with {top_item[y_col]:,.2f}. "
        
        if len(df) > 1:
            total_value = df[y_col].sum()
            insights += f"Top 3 items represent {(df.head(3)[y_col].sum() / total_value) * 100:.1f}% of the total. "
            if "employees" in entities or "employee" in entities:
                insights += f"There's a {(top_item[y_col] / df.iloc[-1][y_col]):.1f}x difference between the highest and lowest performing."
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf.getvalue(), insights
    
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        return _generate_empty_chart(), f"Error creating bar chart: {e}"

def _create_line_chart(df, title, entities):
    """Create a line chart, typically for time series data."""
    try:
        df = df.copy()
        logger.info(f"Fetched data for Line chart: {df}")

        # Step 1: Identify date and value columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = [col for col in df.columns if col not in date_cols]

        if not date_cols or not numeric_cols:
            return _generate_empty_chart(), "No valid date or numeric column found."

        x_col = date_cols[0]
        y_col = numeric_cols[0]

        # Step 2: Convert data types
        df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

        # Step 3: Drop invalid rows
        df = df.dropna(subset=[x_col, y_col])

        if df.empty:
            return _generate_empty_chart(), "No valid data to plot."

        # Step 4: Aggregate by date to handle duplicates
        df = df.groupby(x_col, as_index=False)[y_col].sum()

        # Step 5: Sort by date
        df = df.sort_values(by=x_col)

        # Step 6: Plot
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

        # Step 7: Insights
        max_point = df.loc[df[y_col].idxmax()]
        min_point = df.loc[df[y_col].idxmin()]

        insights = (
            f"Peak value of {max_point[y_col]:,.2f} occurred on {max_point[x_col].date()}. "
            f"Lowest value of {min_point[y_col]:,.2f} occurred on {min_point[x_col].date()}. "
        )

        if len(df) > 2:
            first_half_avg = df.iloc[:len(df)//2][y_col].mean()
            second_half_avg = df.iloc[len(df)//2:][y_col].mean()

            if second_half_avg > first_half_avg:
                trend = "an upward"
                pct_change = (second_half_avg / first_half_avg - 1) * 100
            else:
                trend = "a downward"
                pct_change = (1 - second_half_avg / first_half_avg) * 100

            insights += f"The data shows {trend} trend with a {pct_change:.1f}% change between first and second half."

        # Step 8: Save chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return buf.getvalue(), insights

    except Exception as e:
        logger.error(f"Error creating line chart: {e}")
        return _generate_empty_chart(), f"Error creating line chart: {e}"

def _create_histogram(df, title, entities):
    """Create a histogram."""
    try:
        # Find numeric columns for histogram
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Fetched data for Histogram: {df}")
        if not numeric_cols:
            return _generate_empty_chart(), "No numeric columns found for histogram."
        
        # Use the first numeric column
        value_col = numeric_cols[0]
        
        # Create histogram
        plt.hist(df[value_col], bins=min(10, len(df)), alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel(value_col)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        
        # Generate insights
        mean_val = df[value_col].mean()
        median_val = df[value_col].median()
        std_val = df[value_col].std()
        
        insights = f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, Standard Deviation: {std_val:.2f}. "
        
        # Distribution analysis
        if abs(mean_val - median_val) > std_val * 0.5:
            if mean_val > median_val:
                insights += "The distribution is right-skewed (has a tail extending toward higher values)."
            else:
                insights += "The distribution is left-skewed (has a tail extending toward lower values)."
        else:
            insights += "The distribution appears to be roughly symmetric."
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf.getvalue(), insights
    
    except Exception as e:
        logger.error(f"Error creating histogram: {e}")
        return _generate_empty_chart(), f"Error creating histogram: {e}"

def _generate_empty_chart():
    """Generate an empty chart when no data is available."""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "No data available", horizontalalignment='center', 
             verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()