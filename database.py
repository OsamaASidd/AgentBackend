import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Create and return a connection to the MySQL database.
    Connection details are read from environment variables.
    """
    try:
        host = os.getenv('DB_HOST', 'localhost')
        db = os.getenv('DB_NAME')
        user = "root"
        pwd = os.getenv('DB_PASSWORD')
        port = 3306
        
        logger.info(f"Connecting with host={host}, user={user}, db={db}, port={port}, pwd={'<blank>' if pwd == '' else '***'}")
        
        connection = mysql.connector.connect(
            host=host,
            database=db,
            user=user,
            port=port
        )
        
        if connection.is_connected():
            logger.info("Connected to MySQL database")
            return connection
            
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        raise Exception(f"Database connection failed: {e}")
    
    return None

def execute_query(query, params=None):
    """
    Execute SQL query and return results as a list of dictionaries.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        
    Returns:
        list: List of dictionaries containing query results
    """
    connection = None
    try:
        connection = get_db_connection()
        
        if connection is None:
            raise Exception("Failed to establish database connection")
        
        cursor = connection.cursor(dictionary=True)
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        query_type = query.strip().split()[0].lower()
        
        if query_type in ["insert", "update", "delete"]:
            connection.commit()
            if query_type == "insert":
                return [{"id": cursor.lastrowid}]
            
        results = cursor.fetchall()
        return results if results else []
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise Exception(f"Query execution failed: {e}")
        
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.debug("Database connection closed")

def get_table_schema():
    """
    Get schema information for all tables in the database.
    This is useful for the NLP processor to understand table relationships.
    
    Returns:
        dict: Dictionary with table names as keys and column information as values
    """
    schema_info = {}
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Get list of tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            
            schema_info[table_name] = []
            for column in columns:
                schema_info[table_name].append({
                    "name": column[0],
                    "type": column[1],
                    "nullable": column[2] == "YES",
                    "key": column[3],
                    "default": column[4],
                    "extra": column[5]
                })
        
        return schema_info
    
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return {}
    
    finally:
        if connection and connection.is_connected():
            connection.close()

def query_to_dataframe(query, params=None):
    """
    Execute SQL query and return results as a pandas DataFrame.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        
    Returns:
        DataFrame: Pandas DataFrame containing query results
    """
    try:
        results = execute_query(query, params)
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error converting query results to DataFrame: {e}")
        raise e