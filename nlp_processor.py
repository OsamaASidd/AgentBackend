import os
import json
import logging
# import openai
from dotenv import load_dotenv
from openai import OpenAI
from database import get_table_schema

# Configure logging
load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))# openai.api_key = os.getenv('OPENAI_API_KEY')

# DB schema as a string representation (for context to OpenAI)
SCHEMA_DESCRIPTION = """
Database schema for Synko POS:

- orders: Contains order information (id, merchant_id, customer_id, employee_id, table_id, order_date, order_type, order_status, gross_total, net_amount)
- order_items: Individual items in an order (id, merchant_id, order_id, menu_item_id, item_variation_id, quantity, status)
- menu_items: Available menu items (id, merchant_id, category_id, name, desc, status)
- menu_categories: Categories for menu items (id, merchant_id, name, status)
- menu_item_prices: Prices for menu items (id, merchant_id, menu_item_id, item_variation_id, price, status)
- employees: Employee information (id, user_id, merchant_id, job_id, name, email)
- employee_checkins: Track employee work hours (id, merchant_id, employee_id, checkin_time, checkout_time)
- customers: Customer information (id, merchant_id, fullname, email, phone)
- merchants: Business information (id, user_id, name, address)

Common ID relationships:
- orders.menu_item_id links to menu_items.id
- orders.employee_id links to employees.id
- orders.customer_id links to customers.id
- menu_items.category_id links to menu_categories.id
- order_items.order_id links to orders.id
- order_items.menu_item_id links to menu_items.id
"""

def process_natural_language_query(query, merchant_id=None):
    """
    Process a natural language query and convert it to SQL.
    
    Args:
        query (str): Natural language query
        merchant_id (int, optional): Merchant ID to filter results
        
    Returns:
        tuple: (sql_query, chart_type, entities)
            - sql_query: Generated SQL query
            - chart_type: Suggested visualization type (pie, bar, line, etc.)
            - entities: Key entities extracted from the query
    """
    logger.info(f"Processing natural language query: {query}")
    
    try:
        schema = get_table_schema()
        schema_json = json.dumps(schema, indent=2)
    except Exception as e:
        logger.warning(f"Could not get full schema, using simplified schema: {e}")
        schema_json = SCHEMA_DESCRIPTION
    
    # Create a prompt for the GPT model
    prompt = f"""
    You are an expert SQL query generator for a Point of Sale (POS) system. Given a natural language query, generate a valid SQL query that answers the question.
    
    DATABASE SCHEMA:
    {schema_json}
    
    RULES:
    1. Generate only the SQL query, no explanations or comments.
    2. Use proper table joins based on relationships described in the schema.
    3. Always include merchant_id filter if provided.
    4. For time-based queries (e.g., "this month"), use appropriate date functions.
    5. Format numbers appropriately (e.g., currency with 2 decimal places).
    6. Limit results to a reasonable number (e.g., top 10) for ranking queries.
    7. Along with the SQL query, suggest the most appropriate chart type (pie, bar, line, histogram) for visualizing the results.
    8. Identify the key entities being analyzed (e.g., "items", "employees", "sales", "time")
    
    USER QUERY: {query}
    MERCHANT_ID: {merchant_id if merchant_id else 'Not specified'}
    
    Output in the following JSON format:
    {{
        "sql": "your SQL query here",
        "chart_type": "one of: pie, bar, line, histogram",
        "entities": ["list", "of", "key", "entities"]
    }}
    """
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
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
                
            logger.info(f"Generated SQL: {sql_query}")
            logger.info(f"Suggested chart type: {chart_type}")
            
            return sql_query, chart_type, entities
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract just the SQL
            logger.warning("Failed to parse JSON response, attempting to extract SQL directly")
            lines = content.split('\n')
            sql_query = ""
            capture = False
            
            for line in lines:
                if line.strip().startswith("```sql"):
                    capture = True
                    continue
                elif line.strip() == "```" and capture:
                    break
                elif capture:
                    sql_query += line + "\n"
            
            sql_query = sql_query.strip()
            if not sql_query:
                # Fall back to using the entire response as SQL
                sql_query = content
                
            return sql_query, "bar", ["fallback"]
    
    except Exception as e:
        logger.error(f"Error processing query with OpenAI: {e}")
        raise Exception(f"Failed to process natural language query: {e}")

# Example query mappings for testing without API
TEST_QUERY_MAPPINGS = {
    "what is the most selling item": (
        """
        SELECT mi.name as item_name, SUM(oi.quantity) as total_quantity
        FROM order_items oi
        JOIN menu_items mi ON oi.menu_item_id = mi.id
        WHERE oi.merchant_id = 1
        GROUP BY oi.menu_item_id, mi.name
        ORDER BY total_quantity DESC
        LIMIT 10
        """, 
        "bar",
        ["items", "sales"]
    ),
    "who is the best performing employee": (
        """
        SELECT e.name as employee_name, COUNT(o.id) as order_count, SUM(o.net_amount) as total_sales
        FROM orders o
        JOIN employees e ON o.employee_id = e.id
        WHERE o.merchant_id = 1 AND o.paid_status = 1
        GROUP BY o.employee_id, e.name
        ORDER BY total_sales DESC
        LIMIT 10
        """,
        "bar",
        ["employees", "sales"]
    )
}

def test_query_processor(query, merchant_id=None):
    """Test function for local development without API calls"""
    query_lower = query.lower().strip()
    
    for key, value in TEST_QUERY_MAPPINGS.items():
        if key in query_lower:
            sql, chart_type, entities = value
            # Replace merchant_id if provided
            if merchant_id:
                sql = sql.replace("merchant_id = 1", f"merchant_id = {merchant_id}")
            return sql, chart_type, entities
    
    # Default fallback
    return """
    SELECT mi.name as item_name, SUM(oi.quantity) as total_quantity
    FROM order_items oi
    JOIN menu_items mi ON oi.menu_item_id = mi.id
    WHERE oi.merchant_id = 1
    GROUP BY oi.menu_item_id, mi.name
    ORDER BY total_quantity DESC
    LIMIT 5
    """, "bar", ["fallback"]