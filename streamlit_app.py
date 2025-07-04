import streamlit as st
import os
import glob
from datetime import datetime
from dotenv import load_dotenv
import sqlparse
from utils.supply_chain_agent import SupplyChainAgent
from agent.intent_agent import router_agent
from agent.agent_flow import handle_router
from agent.query_tools.monthly_throughput import analyst_thoughput
from loguru import logger

load_dotenv()

def format_sql_query(sql_query: str) -> str:
    """Format SQL query for better readability using sqlparse"""
    if not sql_query or sql_query == "No SQL query captured":
        return sql_query
    
    try:
        formatted = sqlparse.format(
            sql_query,
            reindent=True,
            keyword_case='upper',
            strip_comments=False,
            strip_whitespace=True,
            use_space_around_operators=True,
            indent_tabs=False,
            indent_width=2
        )
        return formatted.strip()
    except Exception:
        return sql_query

def main():
    st.set_page_config(
        page_title="Supply Chain Analytics Assistant", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏭 Supply Chain Analytics Assistant")
    st.markdown("Ask questions about inventory, transactions, costs, and get insights with visualizations")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Available Data")
        st.markdown("""
        **Datasets Available:**
        - 📦 Transaction History (Inbound/Outbound)
        - 📋 Inventory Snapshots
        - 🏪 Material Master Data
        - 💰 Storage & Transfer Costs
        """)
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "Show inventory levels for all materials",
            "What are the top 5 materials by outbound volume?",
            "Plot monthly transaction trends",
            "Which plants have the highest storage costs?",
            "Show me materials with low stock levels"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.user_input = question
        
        st.header("📊 Quick Analytics")
        if st.button("📈 Monthly Throughput Analysis"):
            st.session_state.call_throughput = True
        
        st.header("🔧 Settings")
        if st.button("🗑️ Clear Chat History"):
            logger.info("Chat history cleared by user")
            st.session_state.messages = []
            if "agent" in st.session_state:
                # Clear both LangChain and LangGraph memory
                st.session_state.agent.clear_memory(st.session_state.get("thread_id", "default"))
            st.rerun()
        
        # Debug Panel
        with st.expander("🔍 Debug & Logs"):
            st.subheader("Log Files")
            
            # Show available log files
            log_files = glob.glob("logs/*.log")
            if log_files:
                selected_log = st.selectbox("Select log file:", log_files)
                
                if st.button("📖 View Logs"):
                    try:
                        with open(selected_log, 'r') as f:
                            logs = f.read()
                        st.text_area("Log Contents:", logs, height=300)
                    except Exception as e:
                        st.error(f"Error reading log file: {e}")
            else:
                st.info("No log files found. Logs will appear after using the agent.")
            
            # Real-time agent status
            st.subheader("Agent Status")
            if "agent" in st.session_state:
                st.success("✅ Agent initialized")
                try:
                    table_names = st.session_state.agent.db.get_usable_table_names()
                    st.info(f"Database tables available: {len(table_names)}")
                    st.text(f"Tables: {', '.join(table_names)}")
                except:
                    st.info("SQL database connected")
                
                # Show memory status
                try:
                    history = st.session_state.agent.get_conversation_history(st.session_state.thread_id)
                    st.info(f"Conversation messages in memory: {len(history)}")
                except:
                    st.warning("Could not retrieve conversation history")
            else:
                st.warning("⚠️ Agent not initialized")
            
            # Session info
            st.subheader("Session Info")
            st.json({
                "messages_count": len(st.session_state.get("messages", [])),
                "thread_id": st.session_state.get("thread_id", "default"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Memory info
            st.subheader("Memory Status")
            if "agent" in st.session_state:
                st.info("✅ LangGraph SQLite persistence enabled")
                st.info("✅ LangChain window memory (5 exchanges)")
            else:
                st.warning("⚠️ Memory not initialized")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "agent" not in st.session_state:
        with st.spinner("🔄 Loading supply chain data and initializing agent..."):
            st.session_state.agent = SupplyChainAgent()
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "plotly":
                st.plotly_chart(message["content"], use_container_width=True, key=f"plotly_{idx}_{id(message)}")
            elif message["type"] == "dataframe":
                if message["content"] is not None and not message["content"].empty:
                    st.dataframe(message["content"], use_container_width=True)
            elif message["type"] == "sql_query":
                if message["content"] and message["content"] != "No SQL query captured":
                    with st.expander("🔍 SQL Query Used"):
                        st.code(format_sql_query(message["content"]), language="sql")
    
    # Chat input
    user_input = st.chat_input("Ask me about your supply chain data...")
    
    # Handle sample question clicks
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input
        st.session_state.user_input = ""
    
    # Handle throughput analysis button click
    if "call_throughput" in st.session_state and st.session_state.call_throughput:
        st.session_state.call_throughput = False
        
        # Add system message for throughput analysis
        st.session_state.messages.append({
            "role": "user", 
            "type": "text", 
            "content": "Monthly Throughput Analysis"
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown("Monthly Throughput Analysis")
        
        # Get throughput analysis
        with st.chat_message("assistant"):
            with st.spinner("🔄 Analyzing monthly throughput..."):
                try:
                    throughput_data = analyst_thoughput()
                    
                    # Convert to DataFrame for better display
                    import pandas as pd
                    df = pd.DataFrame(throughput_data)
                    
                    st.markdown("**Monthly Throughput Analysis Results:**")
                    st.dataframe(df, use_container_width=True)
                    
                    # Store in message history
                    st.session_state.messages.extend([
                        {"role": "assistant", "type": "text", "content": "**Monthly Throughput Analysis Results:**"},
                        {"role": "assistant", "type": "dataframe", "content": df}
                    ])
                    
                    logger.info(f"Monthly throughput analysis completed with {len(throughput_data)} records")
                    
                except Exception as e:
                    error_msg = f"❌ Error getting throughput analysis: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "type": "text", 
                        "content": error_msg
                    })
                    logger.error(f"Throughput analysis error: {str(e)}")
    
    if user_input:
        logger.info(f"User query: {user_input[:100]}")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    # Get intent classification and route to appropriate agent
                    intent = router_agent(user_input)
                    # print(f"Intent is: {intent}")
                    response_text = handle_router(
                        intent, 
                        user_input, 
                        chat_history=[], 
                        supply_chain_agent=st.session_state.agent, 
                        thread_id=st.session_state.thread_id
                    )
                    
                    # Handle if response is already a dict (from supply chain agent)
                    if isinstance(response_text, dict):
                        response = response_text
                    else:
                        response = {"type": "text", "content": response_text}
                    
                    # Handle different response types
                    if response["type"] == "text":
                        st.markdown(response["content"])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "text", 
                            "content": response["content"]
                        })
                        logger.info(f"Generated text response: {len(response['content'])} characters")
                    
                    elif response["type"] == "text_with_chart":
                        st.markdown(response["text"])
                        st.plotly_chart(response["chart"], use_container_width=True, key=f"chart_{st.session_state.thread_id}_{len(st.session_state.messages)}")
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "plotly", "content": response["chart"]}
                        ])
                        logger.info(f"Generated chart response with text: {len(response['text'])} characters")
                    
                    elif response["type"] == "text_with_dataframe":
                        st.markdown(response["text"])
                        st.dataframe(response["dataframe"], use_container_width=True)
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "dataframe", "content": response["dataframe"]}
                        ])
                        logger.info(f"Generated dataframe response with text: {len(response['text'])} characters")
                    
                    elif response["type"] == "text_with_sql_and_dataframe":
                        # Display natural language answer
                        st.markdown(response["text"])
                        
                        # Display DataFrame if available
                        if response["dataframe"] is not None and not response["dataframe"].empty:
                            st.subheader("📊 Raw Data Results")
                            st.dataframe(response["dataframe"], use_container_width=True)
                        
                        # Display SQL query in expandable section
                        if response["sql_query"] and response["sql_query"] != "No SQL query captured":
                            with st.expander("🔍 SQL Query Used"):
                                st.code(format_sql_query(response["sql_query"]), language="sql")
                        
                        # Store in message history
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "dataframe", "content": response["dataframe"]},
                            {"role": "assistant", "type": "sql_query", "content": response["sql_query"]}
                        ])
                        dataframe_shape = response["dataframe"].shape if response["dataframe"] is not None else None
                        sql_length = len(response["sql_query"]) if response["sql_query"] else 0
                        logger.info(f"Generated SQL and dataframe response: {len(response['text'])} chars, shape {dataframe_shape}, SQL {sql_length} chars")
                    
                    elif response["type"] == "error":
                        st.error(f"❌ {response['content']}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "text", 
                            "content": f"❌ {response['content']}"
                        })
                        logger.error(f"Response error: {response['content']}")
                        
                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "type": "text", 
                        "content": error_msg
                    })
                    logger.error(f"Streamlit error: {str(e)}")

if __name__ == "__main__":
    main()