import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from database import get_table_schema

# Configure logging
load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Condensed schema representation focused only on essential information
CONDENSED_SCHEMA = """
Key tables with essential fields:
- orders (id, merchant_id, customer_id, employee_id, table_id, order_date, order_type, order_status, gross_total, net_amount)
- order_items (id, merchant_id, order_id, menu_item_id, quantity)
- menu_items (id, merchant_id, category_id, name)
- menu_categories (id, merchant_id, name)
- menu_item_prices (id, merchant_id, menu_item_id, price)
- employees (id, merchant_id, name)
- customers (id, merchant_id, fullname)

Key relationships:
- order_items.order_id → orders.id
- order_items.menu_item_id → menu_items.id
- orders.employee_id → employees.id
- orders.customer_id → customers.id
- menu_items.category_id → menu_categories.id
"""

# Common query patterns to avoid regenerating SQL
QUERY_PATTERNS = {
    "best selling|popular|top|most sold item": {
        "sql": """
            SELECT mi.name as item_name, SUM(oi.quantity) as total_quantity
            FROM order_items oi
            JOIN menu_items mi ON oi.menu_item_id = mi.id
            WHERE oi.merchant_id = {merchant_id}
            GROUP BY oi.menu_item_id, mi.name
            ORDER BY total_quantity DESC
            LIMIT 10
        """,
        "chart_type": "bar",
        "entities": ["items", "sales"]
    },
    "best|top performing employee|staff": {
        "sql": """
            SELECT e.name as employee_name, COUNT(o.id) as order_count, SUM(o.net_amount) as total_sales
            FROM orders o
            JOIN employees e ON o.employee_id = e.id
            WHERE o.merchant_id = {merchant_id} AND o.paid_status = 1
            GROUP BY o.employee_id, e.name
            ORDER BY total_sales DESC
            LIMIT 10
        """,
        "chart_type": "bar",
        "entities": ["employees", "sales"]
    },
    "sales by category": {
        "sql": """
            SELECT mc.name as category_name, SUM(o.net_amount) as total_sales
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN menu_items mi ON oi.menu_item_id = mi.id
            JOIN menu_categories mc ON mi.category_id = mc.id
            WHERE o.merchant_id = {merchant_id} AND o.paid_status = 1
            GROUP BY mc.id, mc.name
            ORDER BY total_sales DESC
        """,
        "chart_type": "pie",
        "entities": ["categories", "sales"]
    },
    "sales by|over time|month|day|week|hour": {
        "sql": """
            SELECT DATE(o.order_date) as sale_date, SUM(o.net_amount) as daily_sales
            FROM orders o
            WHERE o.merchant_id = {merchant_id} AND o.paid_status = 1
            GROUP BY DATE(o.order_date)
            ORDER BY sale_date
            LIMIT 30
        """,
        "chart_type": "line",
        "entities": ["time", "sales"]
    }
}

def match_query_pattern(query):
    """
    Match the query against predefined patterns to avoid OpenAI API call.
    
    Args:
        query (str): Natural language query
        
    Returns:
        dict or None: Matched pattern or None if no match
    """
    query_lower = query.lower()
    
    for pattern, template in QUERY_PATTERNS.items():
        # Check if any pattern keyword is in the query
        keywords = pattern.split('|')
        if any(keyword in query_lower for keyword in keywords):
            return template
    
    return None

def process_natural_language_query(query, merchant_id=None):
    """
    Process a natural language query and convert it to SQL.
    Use pattern matching first, then fallback to OpenAI API.
    
    Args:
        query (str): Natural language query
        merchant_id (int, optional): Merchant ID to filter results
        
    Returns:
        tuple: (sql_query, chart_type, entities)
    """
    logger.info(f"Processing query: {query}")
    
    # Default merchant ID if not specified
    merchant_id = merchant_id or 1
    
    # Try pattern matching first to avoid API call
    matched_pattern = match_query_pattern(query)
    if matched_pattern:
        sql = matched_pattern["sql"].format(merchant_id=merchant_id)
        return sql.strip(), matched_pattern["chart_type"], matched_pattern["entities"]
    
    # If no pattern match, use OpenAI
    logger.info("No pattern match, using OpenAI")
    
    # Create a minimal prompt for the GPT model
    prompt = f"""
You are an SQL expert. Convert this question to SQL for a POS database:

"{query}"

Database schema:
{CONDENSED_SCHEMA}

* Always filter by merchant_id = {merchant_id}
* Limit results to 10 rows for rankings
* Return JSON only: {{"sql": "SQL_QUERY", "chart_type": "pie|bar|line|histogram", "entities": ["entity1", "entity2"]}}
"""
    
    try:
        # Call OpenAI API with shorter prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use 3.5-turbo instead of 4 to reduce tokens
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=500  # Limit response size
        )
        
        # Extract the response content
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(content)
            sql_query = result.get("sql", "").strip()
            chart_type = result.get("chart_type", "bar").lower()
            entities = result.get("entities", [])
            
            # Validate chart type
            valid_chart_types = ["pie", "bar", "line", "histogram"]
            if chart_type not in valid_chart_types:
                chart_type = "bar"  # Default to bar chart
            
            return sql_query, chart_type, entities
            
        except json.JSONDecodeError:
            # If JSON parsing fails, extract SQL with regex
            logger.warning("Failed to parse JSON response")
            import re
            
            # Try to extract SQL using regex pattern
            sql_pattern = r"```sql\s*(.*?)\s*```"
            sql_match = re.search(sql_pattern, content, re.DOTALL)
            
            if sql_match:
                sql_query = sql_match.group(1).strip()
            else:
                # Fallback to default query
                sql_query = """
                SELECT mi.name as item_name, SUM(oi.quantity) as total_quantity
                FROM order_items oi
                JOIN menu_items mi ON oi.menu_item_id = mi.id
                WHERE oi.merchant_id = {0}
                GROUP BY oi.menu_item_id, mi.name
                ORDER BY total_quantity DESC
                LIMIT 5
                """.format(merchant_id)
            
            return sql_query, "bar", ["fallback"]
    
    except Exception as e:
        logger.error(f"Error processing query with OpenAI: {e}")
        # Fallback to default query
        return """
        SELECT mi.name as item_name, SUM(oi.quantity) as total_quantity
        FROM order_items oi
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        WHERE oi.merchant_id = {0}
        GROUP BY oi.menu_item_id, mi.name
        ORDER BY total_quantity DESC
        LIMIT 5
        """.format(merchant_id), "bar", ["fallback"]