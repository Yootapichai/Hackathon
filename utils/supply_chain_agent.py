import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import traceback
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from loguru import logger

load_dotenv()

class QueryCapturingSQLDatabase(SQLDatabase):
    """Custom SQLDatabase wrapper that captures executed queries"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_executed_query = None
        self.last_query_result = None
    
    def run(self, command: str, fetch: str = "all", **kwargs) -> str:
        """Override run method to capture queries"""
        try:
            # Store the query being executed
            self.last_executed_query = command.strip()
            
            # Execute the query using parent method with all parameters
            result = super().run(command, fetch, **kwargs)
            
            # Store the result
            self.last_query_result = result
            
            return result
        except Exception as e:
            # Reset on error
            self.last_executed_query = None
            self.last_query_result = None
            raise e
    
    def get_last_query_info(self):
        """Get the last executed query and its results"""
        return {
            "query": self.last_executed_query,
            "result": self.last_query_result
        }
    
    def clear_query_cache(self):
        """Clear stored query information"""
        self.last_executed_query = None
        self.last_query_result = None

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class SupplyChainAgent:
    def __init__(self):
        try:
            logger.info("Agent initialization started")
            
            self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY environment variable is required"
                logger.error(f"Agent initialization failed: {error_msg}")
                raise ValueError(error_msg)
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.google_api_key,
                temperature=0.0,
                verbose=True
            )
            
            # Initialize SQL database connection
            self.db = self._create_database_connection()
            
            # Log database connection info
            try:
                table_names = self.db.get_usable_table_names()
                logger.info(f"SQL connection initialized with {len(table_names)} tables")
                for table in table_names:
                    logger.debug(f"SQL table available: {table}")
            except Exception as e:
                logger.error(f"Database logging error: {e}")
            
            # Initialize LangChain memory for pandas agent
            self.memory = ConversationBufferWindowMemory(
                k=5,  # Keep last 5 conversation exchanges
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize LangGraph memory saver with SQLite persistence
            try:
                # Use proper SQLite initialization pattern
                import sqlite3
                conn = sqlite3.connect("conversations.db", check_same_thread=False)
                self.langgraph_memory = SqliteSaver(conn)
                logger.info("SQLite memory initialized")
            except Exception as e:
                # Fallback to in-memory storage
                logger.error(f"SQLite memory initialization failed, using in-memory storage: {e}")
                self.langgraph_memory = MemorySaver()
            
            # Database connection successful
            
            self._setup_tools()
            self._setup_agent()
            
            logger.info("Agent initialization complete")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _create_database_connection(self):
        """Create connection to Supabase PostgreSQL database"""
        try:
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_password = os.environ.get("SUPABASE_DB_PASSWORD")
            
            if not supabase_url or not supabase_password:
                raise ValueError("SUPABASE_URL and SUPABASE_DB_PASSWORD environment variables are required")
            
            # Extract project ID from Supabase URL
            project_id = supabase_url.split("//")[1].split(".")[0]
            
            # PostgreSQL connection string for Supabase
            db_uri = f"postgresql://postgres:{supabase_password}@db.{project_id}.supabase.co:5432/postgres"
            
            return QueryCapturingSQLDatabase.from_uri(db_uri)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _setup_tools(self):
        @tool
        def analyze_supply_chain_data(query: str) -> Dict[str, Any]:
            """Analyze supply chain data using SQL queries. Returns SQL, DataFrame, and natural language answer."""
            
            start_time = time.time()
            logger.info(f"Starting supply chain data analysis: {query[:100]}...")
            
            try:
                # Clear previous query cache
                self.db.clear_query_cache()
                
                # Enhanced system prompt for SQL agent with business context
                system_prefix = """
You are an expert supply chain data analyst with access to a PostgreSQL database containing:

Tables:
- material_master: Material information (material_name, polymer_type, shelf_life_in_month, downgrade_value_lost_percent)
- inventory: Current stock levels (balance_as_of_date, plant_name, material_name, batch_number, unrestricted_stock, stock_unit, stock_sell_value, currency)
- inbound: Incoming shipments (inbound_date, plant_name, material_name, net_quantity_mt)
- outbound: Outgoing shipments (outbound_date, plant_name, material_name, customer_number, mode_of_transport, net_quantity_mt)
- operation_costs: Storage and transfer costs (operation_category, cost_type, entity_name, entity_type, cost_amount, cost_unit, container_capacity_mt, currency)

Key Business Rules:
- Stock quantities are in KG for inventory, MT for inbound/outbound
- Different plants use different currencies (CNY for China, SGD for Singapore)
- Batch numbers track specific material lots
- Materials have shelf life and degradation rates

Always provide actionable insights and consider business context in your analysis.
Your answer must end with smile emoji
"""
                
                # Create SQL toolkit
                toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
                
                # Create SQL agent with proper memory integration
                try:
                    agent = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type="openai-tools",  # Changed from zero-shot-react-description
                        prefix=system_prefix,  # Use prefix parameter for openai-tools agent
                        handle_parsing_errors=True,
                        memory=self.memory  # Use proper LangChain memory
                    )
                except Exception as e:
                    logger.error(f"SQL agent creation with memory failed, trying without memory: {e}")
                    # Fallback without memory if it causes issues
                    agent = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type="openai-tools",  # Changed from zero-shot-react-description
                        prefix=system_prefix,  # Use prefix parameter for openai-tools agent
                        handle_parsing_errors=True
                    )
                
                # Use memory-aware invocation
                result = agent.invoke({"input": query})["output"]
                
                # Get the captured SQL query and create DataFrame
                query_info = self.db.get_last_query_info()
                sql_query = query_info.get("query", "No SQL query captured")
                
                # Create DataFrame from the executed query if available
                dataframe = None
                if sql_query and sql_query != "No SQL query captured":
                    try:
                        # Execute the same query to get DataFrame
                        dataframe = pd.read_sql_query(sql_query, self.db._engine)
                        logger.debug(f"SQL query result captured with shape: {dataframe.shape}")
                    except Exception as df_error:
                        logger.error(f"DataFrame creation failed for SQL: {sql_query[:100]}... Error: {df_error}")
                        # If DataFrame creation fails, still return the text result
                        pass
                
                execution_time = time.time() - start_time
                logger.info(f"Supply chain data analysis completed successfully in {execution_time:.2f}s")
                
                # Return structured response
                return {
                    "type": "text_with_sql_and_dataframe",
                    "text": result,
                    "sql_query": sql_query,
                    "dataframe": dataframe
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Supply chain data analysis failed in {execution_time:.2f}s: {e}")
                
                # Return error as text-only response
                return {
                    "type": "text",
                    "content": f"Error analyzing data: {str(e)}"
                }
        
        @tool 
        def create_visualization(query: str) -> Dict[str, Any]:
            """Create Plotly visualizations for supply chain data. Use this when users ask for charts, plots, or visual analysis."""
            
            start_time = time.time()
            logger.info(f"Starting visualization creation: {query[:100]}...")
            
            try:
                # Determine what type of visualization is needed
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['trend', 'time', 'monthly', 'daily', 'over time']):
                    result = self._create_time_series_chart(query)
                elif any(word in query_lower for word in ['top', 'highest', 'largest', 'ranking']):
                    result = self._create_ranking_chart(query)
                elif any(word in query_lower for word in ['inventory', 'stock', 'levels']):
                    result = self._create_inventory_chart(query)
                elif any(word in query_lower for word in ['cost', 'expense', 'storage', 'transfer']):
                    result = self._create_cost_chart(query)
                else:
                    result = self._create_general_chart(query)
                
                execution_time = time.time() - start_time
                
                if result["type"] == "plotly":
                    logger.info(f"Visualization created successfully: {result.get('description', 'unknown')}")
                else:
                    logger.error(f"Visualization creation failed: {result.get('message')}")
                
                logger.info(f"Visualization creation completed in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Visualization creation failed in {execution_time:.2f}s: {e}")
                return {"type": "error", "message": f"Error creating visualization: {str(e)}"}
        
        self.analyze_tool = analyze_supply_chain_data
        self.visualize_tool = create_visualization
    
    def _create_time_series_chart(self, query: str) -> Dict[str, Any]:
        """Create time-series visualizations"""
        try:
            # SQL query for monthly transaction trends (lowercase columns)
            sql_query = """
            WITH combined_transactions AS (
                SELECT 
                    inbound_date as transaction_date,
                    'INBOUND' as transaction_type,
                    net_quantity_mt as net_quantity_mt
                FROM inbound
                UNION ALL
                SELECT 
                    outbound_date as transaction_date,
                    'OUTBOUND' as transaction_type,
                    net_quantity_mt as net_quantity_mt
                FROM outbound
            ),
            monthly_data AS (
                SELECT 
                    DATE_TRUNC('month', transaction_date::DATE) as month,
                    transaction_type,
                    SUM(net_quantity_mt) as total_quantity
                FROM combined_transactions
                WHERE transaction_date IS NOT NULL
                GROUP BY DATE_TRUNC('month', transaction_date::DATE), transaction_type
                ORDER BY month, transaction_type
            )
            SELECT 
                TO_CHAR(month, 'YYYY-MM') as month_str,
                transaction_type,
                total_quantity
            FROM monthly_data;
            """
            
            # Execute query and get results
            df = pd.read_sql_query(sql_query, self.db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No transaction data found for time series"}
            
            fig = px.line(
                df, 
                x='month_str', 
                y='total_quantity', 
                color='transaction_type',
                title='Monthly Transaction Trends',
                labels={'total_quantity': 'Quantity (MT)', 'month_str': 'Month', 'transaction_type': 'Transaction Type'}
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Monthly transaction trends showing inbound vs outbound volumes"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating time series chart: {str(e)}"}
    
    def _create_ranking_chart(self, query: str) -> Dict[str, Any]:
        """Create ranking/top N visualizations"""
        try:
            if 'material' in query.lower():
                # Top materials by volume (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                material_totals AS (
                    SELECT 
                        material_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE material_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY material_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT material_name, total_volume FROM material_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(
                    df, 
                    x='total_volume', 
                    y='material_name',
                    orientation='h',
                    title='Top 10 Materials by Total Volume',
                    labels={'total_volume': 'Total Volume (MT)', 'material_name': 'Material'}
                )
                
            elif 'plant' in query.lower():
                # Top plants by volume (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        plant_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        plant_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                plant_totals AS (
                    SELECT 
                        plant_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE plant_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY plant_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT plant_name, total_volume FROM plant_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(
                    df, 
                    x='plant_name', 
                    y='total_volume',
                    title='Top 10 Plants by Total Volume',
                    labels={'total_volume': 'Total Volume (MT)', 'plant_name': 'Plant'}
                )
                
            else:
                # Default: top materials (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                material_totals AS (
                    SELECT 
                        material_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE material_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY material_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT material_name, total_volume FROM material_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(df, x='material_name', y='total_volume', title='Top 10 Materials by Volume')
            
            if df.empty:
                return {"type": "error", "message": "No data found for ranking chart"}
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Ranking chart based on your query"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating ranking chart: {str(e)}"}
    
    def _create_inventory_chart(self, query: str) -> Dict[str, Any]:
        """Create inventory-related visualizations"""
        try:
            # SQL query for inventory levels by plant (lowercase columns)
            sql_query = """
            SELECT 
                plant_name,
                SUM(unrestricted_stock) as total_inventory
            FROM inventory
            WHERE unrestricted_stock > 0
            GROUP BY plant_name
            ORDER BY total_inventory DESC;
            """
            
            df = pd.read_sql_query(sql_query, self.db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No inventory data found"}
            
            fig = px.pie(
                df, 
                values='total_inventory', 
                names='plant_name',
                title='Inventory Distribution by Plant'
            )
            
            return {"type": "plotly", "chart": fig, "description": "Current inventory distribution across plants"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating inventory chart: {str(e)}"}
    
    def _create_cost_chart(self, query: str) -> Dict[str, Any]:
        """Create cost-related visualizations"""
        try:
            if 'storage' in query.lower():
                # Storage costs by plant (lowercase columns)
                sql_query = """
                SELECT 
                    entity_name as plant_name,
                    cost_amount as storage_cost
                FROM operation_costs
                WHERE cost_type = 'Inventory Storage per MT per day'
                ORDER BY storage_cost DESC;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                if df.empty:
                    return {"type": "error", "message": "No storage cost data found"}
                
                fig = px.bar(
                    df, 
                    x='plant_name', 
                    y='storage_cost',
                    title='Storage Costs by Plant',
                    labels={'storage_cost': 'Cost per MT per Day', 'plant_name': 'Plant'}
                )
                
            elif 'transfer' in query.lower():
                # Transfer costs by transport mode (lowercase columns)
                sql_query = """
                SELECT 
                    entity_name as transport_mode,
                    cost_amount as transfer_cost
                FROM operation_costs
                WHERE cost_type = 'Transfer cost per container (24.75MT)'
                ORDER BY transfer_cost DESC;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                if df.empty:
                    return {"type": "error", "message": "No transfer cost data found"}
                
                fig = px.bar(
                    df, 
                    x='transport_mode', 
                    y='transfer_cost',
                    title='Transfer Costs by Transport Mode',
                    labels={'transfer_cost': 'Cost per Container', 'transport_mode': 'Transport Mode'}
                )
                
            else:
                # Default storage costs (lowercase columns)
                sql_query = """
                SELECT 
                    entity_name as plant_name,
                    cost_amount as storage_cost
                FROM operation_costs
                WHERE cost_type = 'Inventory Storage per MT per day'
                ORDER BY storage_cost DESC;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                if df.empty:
                    return {"type": "error", "message": "No cost data found"}
                
                fig = px.bar(df, x='plant_name', y='storage_cost', title='Storage Costs by Plant')
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Cost analysis visualization"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating cost chart: {str(e)}"}
    
    def _create_general_chart(self, query: str) -> Dict[str, Any]:
        """Create general visualizations"""
        try:
            # Default: transaction type distribution
            sql_query = """
            WITH transaction_counts AS (
                SELECT 'INBOUND' as transaction_type, COUNT(*) as count FROM inbound
                UNION ALL
                SELECT 'OUTBOUND' as transaction_type, COUNT(*) as count FROM outbound
            )
            SELECT transaction_type, count FROM transaction_counts;
            """
            
            df = pd.read_sql_query(sql_query, self.db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No transaction data found"}
            
            fig = px.pie(
                df, 
                values='count', 
                names='transaction_type',
                title='Transaction Type Distribution'
            )
            
            return {"type": "plotly", "chart": fig, "description": "General overview of transaction types"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating general chart: {str(e)}"}
    
    def _setup_agent(self):
        """Setup the LangGraph agent with proper memory management"""
        agent_node = create_react_agent(
            model=self.llm,
            tools=[self.analyze_tool, self.visualize_tool]
        )
        
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        
        # Compile with MemorySaver for persistence
        self.app = graph.compile(checkpointer=self.langgraph_memory)
    
    def process_query(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Process user query with proper memory management"""
        start_time = time.time()
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Create thread configuration for session persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Determine if this is a visualization request
            visualization_keywords = ['plot', 'chart', 'graph', 'show', 'visualize', 'display']
            is_viz_request = any(keyword in query.lower() for keyword in visualization_keywords)
            
            if is_viz_request:
                # Handle visualization requests directly
                viz_result = self.visualize_tool.invoke(query)
                if viz_result["type"] == "plotly":
                    response = {
                        "type": "text_with_chart",
                        "text": f"Here's a visualization based on your request: {viz_result['description']}",
                        "chart": viz_result["chart"]
                    }
                    processing_time = time.time() - start_time
                    logger.info(f"Chart response generated in {processing_time:.2f}s - text length: {len(response['text'])}")
                    
                    # Store in LangChain memory for pandas agent context
                    self.memory.save_context(
                        {"input": query},
                        {"output": response["text"]}
                    )
                    
                    return response
                else:
                    error_msg = viz_result.get("message", "Error creating visualization")
                    processing_time = time.time() - start_time
                    logger.error(f"Visualization error in {processing_time:.2f}s: {error_msg}")
                    return {"type": "error", "content": error_msg}
            
            # For data analysis questions and conversation memory
            else:
                # Use enhanced SQL tool that captures query and DataFrame
                analysis_result = self.analyze_tool.invoke(query)
                
                processing_time = time.time() - start_time
                
                # Handle the enhanced response with SQL and DataFrame
                if isinstance(analysis_result, dict) and analysis_result.get("type") == "text_with_sql_and_dataframe":
                    logger.info(f"SQL analysis response generated in {processing_time:.2f}s - text length: {len(analysis_result['text'])}")
                    
                    # Store in LangChain memory for context
                    self.memory.save_context(
                        {"input": query},
                        {"output": analysis_result["text"]}
                    )
                    
                    return analysis_result
                    
                # Handle error responses
                elif isinstance(analysis_result, dict) and analysis_result.get("type") == "text":
                    logger.error(f"Analysis error in {processing_time:.2f}s - content length: {len(analysis_result['content'])}")
                    return {"type": "error", "content": analysis_result["content"]}
                    
                # Fallback for unexpected response format
                else:
                    # If tool returns just text (old format), wrap it
                    response_text = str(analysis_result)
                    logger.info(f"Text response generated in {processing_time:.2f}s - length: {len(response_text)}")
                    
                    self.memory.save_context(
                        {"input": query},
                        {"output": response_text}
                    )
                    
                    return {"type": "text", "content": response_text}
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            logger.error(f"Query processing failed in {processing_time:.2f}s: {e}")
            return {"type": "error", "content": error_msg}
    
    def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Failed to get conversation history for thread {thread_id}: {e}")
            return []
    
    def clear_memory(self, thread_id: str = "default"):
        """Clear conversation memory for a specific thread"""
        try:
            # Clear LangChain memory
            self.memory.clear()
            
            # Clear LangGraph state (note: MemorySaver doesn't have direct clear method)
            # Memory will naturally expire or can be handled at the application level
            logger.info("Memory cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear memory for thread {thread_id}: {e}")