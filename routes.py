from flask import Blueprint, request, jsonify
import logging
import base64
import json

from database import execute_query
from nlp_processor import process_natural_language_query
from visualization import generate_visualization

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)

@api.route('/query', methods=['POST'])
def process_query():
    """
    Process a natural language query and return visualization.
    
    Request body should contain:
    {
        "query": "What is the most selling item?",
        "merchant_id": 123  # Optional, if user has multiple merchants
    }
    """
    try:
        data = request.json
        query = data.get('query')
        merchant_id = data.get('merchant_id')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Process natural language query to get SQL and chart type
        logger.info(f"Processing query: {query}")
        sql_query, chart_type, entities = process_natural_language_query(query, merchant_id)
        
        # Execute SQL query
        logger.info(f"Executing SQL: {sql_query}")
        results = execute_query(sql_query)
        
        if not results or len(results) == 0:
            return jsonify({
                "message": "No data found for your query",
                "sql": sql_query
            }), 200
        
        # Generate visualization
        logger.info(f"Generating {chart_type} visualization")
        chart_image, insights = generate_visualization(results, chart_type, entities, query)
        
        # Encode image as base64 for transmission
        encoded_image = base64.b64encode(chart_image).decode('utf-8')
        
        return jsonify({
            "chartImage": encoded_image,
            "chartType": chart_type,
            "insights": insights,
            "rawData": results[:10],  # Return first 10 rows for reference
            "sql": sql_query
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running."""
    return jsonify({"status": "healthy"}), 200

@api.route('/schema', methods=['GET'])
def get_schema():
    """Return the database schema for developer reference."""
    from database import get_table_schema
    
    try:
        schema = get_table_schema()
        return jsonify({"schema": schema})
    except Exception as e:
        logger.error(f"Error fetching schema: {str(e)}")
        return jsonify({"error": str(e)}), 500