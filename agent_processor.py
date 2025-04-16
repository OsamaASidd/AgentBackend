import re
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from database import execute_query
import os
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class AgentProcessor:
    """
    Process agentic tasks like adding menu items, creating orders, etc.
    """
    
    def __init__(self, merchant_id=1):
        self.merchant_id = merchant_id
        
    def process_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Process user message to identify intent and extract relevant information.
        
        Args:
            user_message (str): User's natural language message
            
        Returns:
            dict: Response with action taken or follow-up questions
        """
        # Log the incoming message for debugging
        logger.info(f"Processing intent for message: {user_message}")
        
        intent, entities = self._identify_intent(user_message)
        
        # Log identified intent and entities
        logger.info(f"Identified intent: {intent}")
        logger.info(f"Extracted entities: {entities}")
        
        if intent == "add_menu_item":
            return self._handle_add_menu_item(entities, user_message)
        elif intent == "unknown":
            return {
                "success": False,
                "message": "I'm not sure what you'd like to do. Would you like to add a menu item, create an order, or manage employees?",
                "action_taken": None
            }
        else:
            # For future intents
            return {
                "success": False,
                "message": f"I understand you want to {intent}, but this feature is not yet implemented.",
                "action_taken": None
            }
    
    def _identify_intent(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Identify user intent and extract entities using pattern matching and NLP.
        
        Args:
            message (str): User's natural language message
            
        Returns:
            tuple: (intent_name, extracted_entities)
        """
        message_lower = message.lower()
        
        # Pattern matching for common intents
        add_item_patterns = [
            r"add (?:a |an )?(new )?(?:menu )?item",
            r"create (?:a |an )?(new )?(?:menu )?item",
            r"add (?:a |an )?(?:new )?dish",
            r"new menu item"
        ]
        
        # Check for add menu item intent
        for pattern in add_item_patterns:
            if re.search(pattern, message_lower):
                # Extract entities for menu item
                entities = self._extract_menu_item_entities(message)
                return "add_menu_item", entities
        
        # If message starts with "add" or has common phrases for adding items, default to add_menu_item
        if message_lower.startswith("add") or "add" in message_lower:
            entities = self._extract_menu_item_entities(message)
            # If we found a name, assume it's an add menu item intent
            if "name" in entities:
                return "add_menu_item", entities
        
        # If no patterns match, use OpenAI to identify intent (fallback)
        try:
            intent_data = self._get_intent_from_openai(message)
            return intent_data.get("intent", "unknown"), intent_data.get("entities", {})
        except Exception as e:
            logger.error(f"Error identifying intent with OpenAI: {e}")
            return "unknown", {}
    
    def _extract_menu_item_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities needed for adding a menu item using regex patterns.
        
        Args:
            message (str): User's natural language message
            
        Returns:
            dict: Extracted entities
        """
        entities = {}
        
        # Extract item name - look for patterns like "Add a [name]" or "[name] for €"
        name_patterns = [
            r"add (?:a |an )?(?:new )?(?:menu )?item called ([A-Za-z\s]+)",
            r"add (?:a |an )?([A-Za-z\s]+) for [€$£]",
            r"add (?:a |an )?([A-Za-z\s]+) (?:under|in|to) (?:the )?(?!menu)([A-Za-z\s]+)",
            r"add (?:a |an )?([A-Za-z\s]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities["name"] = match.group(1).strip()
                # If there's a second group and it might be a category, capture it
                if len(match.groups()) > 1 and match.group(2):
                    potential_category = match.group(2).strip()
                    # Don't use "menu item" as a category
                    if potential_category.lower() != "menu item" and potential_category.lower() != "menu":
                        entities["category"] = potential_category
                break
        
        # Extract price - look for currency symbols and numbers
        price_match = re.search(r"(?:for |price |costs |at |of )(?:[€$£])(\d+\.?\d*)", message, re.IGNORECASE)
        if price_match:
            entities["price"] = float(price_match.group(1))
        
        # If we don't have a category yet, try to extract it
        if "category" not in entities:
            category_match = re.search(r"(?:under|in|to) (?:the )?(?:category )?([A-Za-z\s]+)(?:category|section)?", message, re.IGNORECASE)
            if category_match:
                # Don't use "menu item" or "menu" as a category
                category = category_match.group(1).strip()
                if category.lower() != "menu item" and category.lower() != "menu":
                    entities["category"] = category
        
        # Set default category if none found
        if "category" not in entities or not entities["category"]:
            entities["category"] = "Uncategorized"
        
        # Extract availability options
        entities["available_for_takeaway"] = "takeaway" in message.lower() or "take away" in message.lower() or "take-away" in message.lower()
        entities["available_for_delivery"] = "delivery" in message.lower()
        
        return entities
    
    def _get_intent_from_openai(self, message: str) -> Dict[str, Any]:
        """
        Use OpenAI to identify intent and extract entities for complex messages.
        
        Args:
            message (str): User's natural language message
            
        Returns:
            dict: Intent and entities
        """
        prompt = f"""
        Identify the user intent and extract relevant entities from this message:
        
        "{message}"
        
        Possible intents:
        - add_menu_item: User wants to add a new menu item
        - create_order: User wants to create a new order
        - mark_attendance: User wants to mark employee attendance
        - unknown: Cannot determine intent
        
        For add_menu_item, extract:
        - name: Name of the menu item
        - price: Price of the item
        - category: Category of the item (default to "Uncategorized" if none specified)
        - available_for_takeaway: Boolean if available for takeaway
        - available_for_delivery: Boolean if available for delivery
        
        Return a JSON object with "intent" and "entities" keys.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, extract with regex
                intent_match = re.search(r'"intent":\s*"([^"]+)"', content)
                intent = intent_match.group(1) if intent_match else "unknown"
                
                return {"intent": intent, "entities": {}}
                
        except Exception as e:
            logger.error(f"Error processing with OpenAI: {e}")
            return {"intent": "unknown", "entities": {}}
    
    def _handle_add_menu_item(self, entities: Dict[str, Any], original_message: str) -> Dict[str, Any]:
        """
        Handle the intent to add a menu item.
        
        Args:
            entities (dict): Extracted entities
            original_message (str): Original user message
            
        Returns:
            dict: Response with action taken or follow-up questions
        """
        # Log the entities we're working with
        logger.info(f"Handling add_menu_item with entities: {entities}")
        
        # Check if we have all required information
        required_fields = ["name", "price", "category"]
        missing_fields = [field for field in required_fields if field not in entities or not entities[field]]
        
        # If missing information, ask follow-up questions
        if missing_fields:
            questions = []
            if "name" in missing_fields:
                questions.append("What would you like to name this menu item?")
            if "price" in missing_fields:
                questions.append("What is the price of this item?")
            if "category" in missing_fields:
                # Get available categories to suggest
                categories = self._get_available_categories()
                if categories:
                    category_list = ", ".join(categories[:5])
                    questions.append(f"Which category does this item belong to? Available categories include: {category_list}")
                else:
                    questions.append("Which category does this item belong to?")
            
            return {
                "success": False,
                "message": " ".join(questions),
                "missing_fields": missing_fields,
                "action_taken": None,
                "partial_entities": entities
            }
        
        # Try to add the menu item with error handling
        try:
            # Step 1: Check if menu_items table exists and get schema
            schema_info = self._get_table_schema("menu_items")
            if not schema_info:
                return {
                    "success": False,
                    "message": "Could not find menu_items table in the database. Please check your database setup.",
                    "action_taken": None
                }
            
            logger.info(f"Fetched schema for menu_items: {schema_info}")
            column_names = [col["name"] for col in schema_info]
            
            # Step 2: Check if category exists or create it
            try:
                category_id = self._get_or_create_category(entities["category"])
                logger.info(f"Using category ID: {category_id} for '{entities['category']}'")
                
                if not category_id or category_id <= 0:
                    return {
                        "success": False,
                        "message": f"Could not create or find category '{entities['category']}'. Please try a different category.",
                        "action_taken": None
                    }
            except Exception as e:
                logger.error(f"Category error: {e}")
                return {
                    "success": False,
                    "message": f"Error with category '{entities['category']}': {str(e)}. Please try a different category.",
                    "action_taken": None
                }
            
            # Step 3: Build dynamic insert query for menu item
            columns = ["merchant_id", "category_id", "name"]
            values = ["%s", "%s", "%s"]
            params = [self.merchant_id, category_id, entities["name"]]
            
            # Add optional fields based on schema
            if "available_for_takeaway" in column_names:
                columns.append("available_for_takeaway")
                values.append("%s")
                params.append(1 if entities.get("available_for_takeaway", False) else 0)
                
            if "available_for_delivery" in column_names:
                columns.append("available_for_delivery")
                values.append("%s")
                params.append(1 if entities.get("available_for_delivery", False) else 0)
            
            # Add status fields if they exist
            if "status" in column_names:
                columns.append("status")
                values.append("%s")
                params.append(1)  # Default to active
                
            if "delivery_status" in column_names:
                columns.append("delivery_status")
                values.append("%s")
                params.append(1)  # Default to active
                
            # Add timestamps if needed
            if "created_at" in column_names and "updated_at" in column_names:
                from datetime import datetime
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                columns.append("created_at")
                values.append("%s")
                params.append(now)
                
                columns.append("updated_at")
                values.append("%s")
                params.append(now)
            
            # Step 4: Execute menu item insert query
            menu_item_query = f"""
            INSERT INTO menu_items ({', '.join(columns)})
            VALUES ({', '.join(values)})
            """
            
            logger.info(f"Executing SQL: {menu_item_query} with params {params}")
            result = execute_query(menu_item_query, params)
            
            if not result or len(result) == 0:
                return {
                    "success": False,
                    "message": "Menu item was created but couldn't retrieve its ID.",
                    "action_taken": None
                }
                
            menu_item_id = result[0]["id"]
            logger.info(f"Created menu item with ID: {menu_item_id}")
            
            # Step 6: Check price table schema
            price_schema = self._get_table_schema("menu_item_prices")
            if not price_schema:
                return {
                    "success": False,
                    "message": f"Menu item '{entities['name']}' was created, but price table not found. Price couldn't be set.",
                    "action_taken": "add_menu_item",
                    "menu_item": {
                        "id": menu_item_id,
                        "name": entities["name"],
                        "category": entities["category"],
                        "price": entities["price"],
                        "available_for_takeaway": entities.get("available_for_takeaway", False),
                        "available_for_delivery": entities.get("available_for_delivery", False)
                    }
                }
            
            # Step 7: Insert price information
            price_query = """
            INSERT INTO menu_item_prices (merchant_id, menu_item_id, price)
            VALUES (%s, %s, %s)
            """
            price_params = (self.merchant_id, menu_item_id, entities["price"])
            
            logger.info(f"Executing SQL: {price_query} with params {price_params}")
            execute_query(price_query, price_params)
            
            # Step 8: Return success response
            return {
                "success": True,
                "message": f"Successfully added {entities['name']} to the menu under {entities['category']} category at €{entities['price']}.",
                "action_taken": "add_menu_item",
                "menu_item": {
                    "id": menu_item_id,
                    "name": entities["name"],
                    "category": entities["category"],
                    "price": entities["price"],
                    "available_for_takeaway": entities.get("available_for_takeaway", False),
                    "available_for_delivery": entities.get("available_for_delivery", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Error adding menu item: {e}")
            error_message = str(e)
            
            # Provide a user-friendly error message based on the exception
            if "foreign key constraint fails" in error_message.lower():
                return {
                    "success": False,
                    "message": f"Could not add the menu item because the category '{entities['category']}' doesn't exist in the database. Please try a different category.",
                    "action_taken": None,
                    "error": error_message
                }
            elif "duplicate entry" in error_message.lower():
                return {
                    "success": False,
                    "message": f"A menu item named '{entities['name']}' already exists. Please choose a different name.",
                    "action_taken": None,
                    "error": error_message
                }
            else:
                return {
                    "success": False,
                    "message": f"Error adding menu item: {str(e)}",
                    "action_taken": None,
                    "error": error_message
                }
                
    def _get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name (str): Name of the table to get schema for
            
        Returns:
            list: List of column information dictionaries
        """
        try:
            # Check if table exists first
            tables_query = "SHOW TABLES"
            tables_result = execute_query(tables_query)
            
            tables = [list(t.values())[0] for t in tables_result]
            logger.info(f"Available tables in database: {tables}")
            
            if table_name not in tables:
                raise Exception(f"Table '{table_name}' does not exist in the database")
            
            # Get table schema
            schema_query = f"DESCRIBE {table_name}"
            schema_result = execute_query(schema_query)
            
            columns = []
            for column in schema_result:
                columns.append({
                    "name": column.get("Field", ""),
                    "type": column.get("Type", ""),
                    "nullable": column.get("Null", "") == "YES",
                    "key": column.get("Key", ""),
                    "default": column.get("Default", ""),
                    "extra": column.get("Extra", "")
                })
            
            return columns
            
        except Exception as e:
            logger.error(f"Error fetching schema for table {table_name}: {e}")
            return []
    
    def _get_or_create_category(self, category_name: str) -> int:
        """
        Get category ID or create a new category if it doesn't exist.
        
        Args:
            category_name (str): Name of the category
            
        Returns:
            int: Category ID
        """
        try:
            # Log the category search
            logger.info(f"Looking up category: '{category_name}'")
            
            # Check if category exists
            query = """
            SELECT id FROM menu_categories
            WHERE name like %s
            """
            results = execute_query(query, (category_name,))
            logger.info(f"Category search results: {results}")
            
            if results and len(results) > 0:
                category_id = results[0]["id"]
                logger.info(f"Found existing category ID: {category_id}")
                return category_id
            
            # Category doesn't exist, check schema before creating it
            logger.info(f"Category '{category_name}' not found, checking schema before creating")
            
            # Get schema for menu_categories table
            schema = self._get_table_schema("menu_categories")
            logger.info(f"Category table schema: {schema}")
            
            # Get column names
            column_names = [col["name"] for col in schema]
            
            # Check for required columns
            if "id" not in column_names or "merchant_id" not in column_names or "name" not in column_names:
                raise Exception(f"menu_categories table is missing required columns. Found: {column_names}")
            
            # Prepare insert query with dynamic columns
            columns = ["merchant_id", "name"]
            values = ["%s", "%s"]
            params = [self.merchant_id, category_name]
            
            # Add timestamps if they exist
            if "created_at" in column_names and "updated_at" in column_names:
                from datetime import datetime
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                columns.append("created_at")
                values.append("%s")
                params.append(now)
                
                columns.append("updated_at")
                values.append("%s")
                params.append(now)
            
            # Add status if it exists
            if "status" in column_names:
                columns.append("status")
                values.append("%s")
                params.append(1)  # 1 = active
                
            # Create the insert query
            insert_query = f"""
            INSERT INTO menu_categories ({', '.join(columns)})
            VALUES ({', '.join(values)})
            """
            
            logger.info(f"Creating new category with query: {insert_query}")
            logger.info(f"Parameters: {params}")
            
            execute_query(insert_query, params)
            
            # Get the new category ID
            id_query = "SELECT LAST_INSERT_ID() as cat_id"
            result = execute_query(id_query)
            
            if result and len(result) > 0:
                category_id = result[0]["cat_id"]
                logger.info(f"Successfully created category '{category_name}' with ID: {category_id}")
                return category_id
            
            raise Exception(f"Failed to create new category '{category_name}'")
            
        except Exception as e:
            logger.error(f"Error in _get_or_create_category: {e}")
            # Don't mask the real error
            raise Exception(f"Category operation failed: {str(e)}")

# Class for managing conversation state and following up on incomplete tasks
class ConversationManager:
    """
    Manage conversation state and follow up on incomplete tasks.
    """
    
    def __init__(self):
        self.conversations = {}  # Store conversation state by session_id
    
    def process_message(self, session_id: str, message: str, merchant_id: int = 1) -> Dict[str, Any]:
        """
        Process a message in the context of an ongoing conversation.
        
        Args:
            session_id (str): Unique identifier for the conversation session
            message (str): User's message
            merchant_id (int): ID of the merchant
            
        Returns:
            dict: Response from the agent
        """
        # Log the incoming message
        logger.info(f"Processing message for session {session_id}: {message}")
        
        # Initialize conversation state if new session
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "current_intent": None,
                "partial_entities": {},
                "missing_fields": [],
                "merchant_id": merchant_id
            }
        
        conversation = self.conversations[session_id]
        
        # Check if we're in the middle of gathering info for an intent
        if conversation["current_intent"] and conversation["missing_fields"]:
            return self._continue_conversation(session_id, message)
        
        # Otherwise, process as a new intent
        agent = AgentProcessor(merchant_id=conversation["merchant_id"])
        result = agent.process_intent(message)
        
        # If we're missing fields, update conversation state
        if not result["success"] and "missing_fields" in result:
            conversation["current_intent"] = result.get("action_taken") or "add_menu_item"
            conversation["partial_entities"] = result["partial_entities"]
            conversation["missing_fields"] = result["missing_fields"]
        else:
            # Reset conversation if task completed or no follow-up needed
            self._reset_conversation(session_id)
        
        return result
    
    def _continue_conversation(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Continue an existing conversation to gather missing information.
        
        Args:
            session_id (str): Unique identifier for the conversation session
            message (str): User's follow-up message
            
        Returns:
            dict: Response with updated action or follow-up questions
        """
        conversation = self.conversations[session_id]
        
        # Log what we're continuing with
        logger.info(f"Continuing conversation for session {session_id}")
        logger.info(f"Current missing fields: {conversation['missing_fields']}")
        logger.info(f"Current partial entities: {conversation['partial_entities']}")
        
        # Update entities based on the response
        updated_entities = self._extract_follow_up_info(
            message, 
            conversation["missing_fields"][0], 
            conversation["partial_entities"]
        )
        
        # Merge updated entities with partial entities
        conversation["partial_entities"].update(updated_entities)
        
        # Remove fields that we've collected
        for field in updated_entities:
            if field in conversation["missing_fields"]:
                conversation["missing_fields"].remove(field)
        
        # If we still have missing fields, ask for the next one
        if conversation["missing_fields"]:
            next_field = conversation["missing_fields"][0]
            question = self._get_question_for_field(next_field, conversation["merchant_id"])
            
            return {
                "success": False,
                "message": question,
                "missing_fields": conversation["missing_fields"],
                "action_taken": None,
                "partial_entities": conversation["partial_entities"]
            }
        
        # All fields collected, proceed with the action
        agent = AgentProcessor(merchant_id=conversation["merchant_id"])
        
        if conversation["current_intent"] == "add_menu_item":
            result = agent._handle_add_menu_item(conversation["partial_entities"], "")
            
            # Reset conversation state
            self._reset_conversation(session_id)
            
            return result
        
        # Unknown intent (shouldn't happen)
        self._reset_conversation(session_id)
        return {
            "success": False,
            "message": "I'm not sure what you'd like to do next. Can I help you with something else?",
            "action_taken": None
        }
    
    def _extract_follow_up_info(self, message: str, field: str, partial_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entity information from a follow-up message.
        
        Args:
            message (str): User's follow-up message
            field (str): The field we're trying to extract
            partial_entities (dict): Existing partial entities
            
        Returns:
            dict: Extracted entity information
        """
        entity = {}
        
        if field == "name":
            # For name, just use the message as is (cleaned)
            entity["name"] = message.strip()
            
        elif field == "price":
            # For price, extract numeric value
            price_match = re.search(r"(?:[€$£])?(\d+\.?\d*)", message)
            if price_match:
                entity["price"] = float(price_match.group(1))
            else:
                # Try to convert whole message to float
                try:
                    entity["price"] = float(message.replace("€", "").replace("$", "").replace("£", "").strip())
                except ValueError:
                    # Default price if we can't extract
                    entity["price"] = 0.0
                    
        elif field == "category":
            # For category, just use the message as is (cleaned)
            entity["category"] = message.strip()
        
        return entity
    
    def _get_question_for_field(self, field: str, merchant_id: int) -> str:
        """
        Get a question to ask for a specific missing field.
        
        Args:
            field (str): The field to ask about
            merchant_id (int): ID of the merchant
            
        Returns:
            str: Question to ask the user
        """
        if field == "name":
            return "What would you like to name this menu item?"
            
        elif field == "price":
            return "What is the price of this item?"
            
        elif field == "category":
            # Get available categories
            agent = AgentProcessor(merchant_id=merchant_id)
            categories = agent._get_available_categories()
            
            if categories:
                category_list = ", ".join(categories[:5])
                return f"Which category does this item belong to? Available categories include: {category_list}"
            else:
                return "Which category does this item belong to?"
                
        return f"Could you provide the {field} for this menu item?"
    
    def _reset_conversation(self, session_id: str):
        """
        Reset the conversation state for a session.
        
        Args:
            session_id (str): Unique identifier for the conversation session
        """
        if session_id in self.conversations:
            merchant_id = self.conversations[session_id]["merchant_id"]
            self.conversations[session_id] = {
                "current_intent": None,
                "partial_entities": {},
                "missing_fields": [],
                "merchant_id": merchant_id
            }