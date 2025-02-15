import os
import uuid
import streamlit as st
import tempfile
from datetime import datetime
import glob
import openai
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
from langchain.schema.runnable import RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import sqlite3
import pandas as pd


openai.api_key = "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"

os.environ["OPENAI_API_KEY"] = openai.api_key
tmp_directory = tempfile.mkdtemp()

class HelpBot:

    def __init__(self):
        st.title("Help Bot")
        st.write("Ask anything about the Uploaded Document")
        self.last_db_history = None
        self.uploaded_files = []
        self.retriever = None
        self.llm = None
        self.setup_db()

    def setup_db(self):
        connect = sqlite3.connect('chat_memory.db')
        conn = connect.cursor()
        conn.execute("""CREATE TABLE IF NOT EXISTS chat_history
                     (timestamp TEXT, session_id TEXT, role TEXT, content TEXT)""")
        connect.commit()
        connect.close()

    def save_message_to_db(self, role, content, session_id=None):
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())

            connect = sqlite3.connect('chat_memory.db')
            conn = connect.cursor()
            timestamp = datetime.now().isoformat()
            conn.execute("""SELECT name FROM sqlite_master
                        WHERE type='table' AND name='chat_history'""")
            if not conn.fetchone():
                conn.execute('''CREATE TABLE IF NOT EXISTS chat_history
                            (timestamp TEXT, session_id TEXT, role TEXT, content TEXT)''')
                connect.commit()

            conn.execute("INSERT INTO chat_history VALUES (?, ?, ?, ?)",
                        (timestamp, session_id, role, content))
            connect.commit()
            connect.close()
        except Exception as e:
            st.error(f"Failed to save message to database: {str(e)}")
            self.setup_db()

    def get_chat_history(self, session_id=None):
        connect = sqlite3.connect('chat_memory.db')
        conn = connect.cursor()
        if session_id:
            conn.execute("SELECT * FROM chat_history WHERE session_id=?", (session_id,))
        else:
            conn.execute("SELECT * FROM chat_history")
        history = conn.fetchall()
        connect.close()
        return history

    def add_memory_controls(self):
        st.sidebar.markdown("### Manage memory")

        if st.sidebar.button("Clear Current Session"):
            st.session_state.messages = []
            st.experimental_rerun()

        if st.sidebar.button("Show Chat History"):
            st.sidebar.markdown("#### Chat History")
            for msg in st.session_state.messages:
                st.sidebar.write(f"**{msg['role']}**: {msg['content']}")

        if st.sidebar.button("Export Chat History"):
            try:
                history = self.get_chat_history()
                df = pd.DataFrame(history, columns=['timestamp', 'session_id', 'role', 'content'])
                st.sidebar.download_button(
                    label="Download History",
                    data=df.to_csv(index=False),
                    file_name="chat_history.csv",
                    mime="text/csv"
                )
                st.sidebar.success("Downloaded!")

            except Exception as e:
                st.sidebar.error(f"Failed to export: {str(e)}")

    def manage_sessions(self):
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

            st.sidebar.markdown("### Session Management")
            if st.sidebar.button("Start New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.experimental_rerun()

            return st.session_state.session_id

    def initialize_db_and_llm(self):  # This method sources the llm, uploads context and stores db
        with st.spinner("Loading model and processing documents..."):
            try:
                documents = self.load_docs()
                if not documents:
                    st.warning("No documents loaded. Please upload files first.")
                    return False

                embeddings=HuggingFaceEmbeddings()
                splits = self.split_docs(documents)
                vectorstore = Chroma.from_documents(
                    documents = splits,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                self.retriever = vectorstore.as_retriever()
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

                return True
            except Exception as e:
                st.error(f"An error occured while initializing: {str(e)}")
                return False

    def load_docs(self):
        # Load documents from the given directory.
        loader = DirectoryLoader(tmp_directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        return documents

    def split_docs(self, documents, chunk_size=500, chunk_overlap=20):
        # Split the documents into chunks.

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(documents)

        return splits

    def start_chat(self):
        self.setup_db()

        # Add memory controls to sidebar
        self.add_memory_controls()
        session_id = self.manage_sessions()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Ask me anything about the uploaded document"):
            # save to session state
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            # save to database
            self.save_message_to_db("user", user_prompt, session_id)
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                response = self.get_answer(user_prompt)
            # save assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            self.save_message_to_db("assistant", response, session_id)

    def get_answer(self, query: str):
        if self.retriever is None or self.llm is None:
            if not self.initialize_db_and_llm():
                return "No documents loaded or database connection issue. Please try uploading files again."

        memory = MemorySaver()
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})

        retriever_tool = create_retriever_tool(
            self.retriever,
            "search_documents",
            "Searches and returns documents relevant to the query."
        )

        tools = [retriever_tool]
        agent_executor = create_react_agent(self.llm, tools, checkpointer=memory)

        # Convert session messages to the format expected by the agent
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in st.session_state.messages
        ]

        # Add the current query to the chat history
        chat_history.append(HumanMessage(content=query))

        # Stream the agent's response
        response_placeholder = st.empty()
        full_response = ""

        for s in agent_executor.stream(
                {"messages": chat_history},
                config=config
        ):
            if isinstance(s, dict):
                if 'agent' in s and 'messages' in s['agent']:
                    for message in s['agent']['messages']:
                        if isinstance(message, AIMessage) and message.content:
                            full_response += message.content
                            response_placeholder.markdown(full_response)

                elif 'action' in s and 'messages' in s['action']:
                    for message in s['action']['messages']:
                        if isinstance(message, ToolMessage) and message.content:
                            # handle tool messages differently
                            tool_message = f"\n\nTool used: {message.name}\n{message.content}\n\n"
                            full_response += tool_message
                            response_placeholder.markdown(full_response)

        return full_response


# Initialize the HelpBot
helper = HelpBot()

context = st.sidebar.radio("Which knowledge base do you want to use?",
                           ["Already uploaded", "Upload new one"])

# File upload logic
if context == "Upload new one":
    uploaded_files = st.sidebar.file_uploader("choose a text file", accept_multiple_files=True)

    if uploaded_files:
        # Clear temp directory
        for file in os.listdir(tmp_directory):
            os.remove(os.path.join(tmp_directory, file))

        # Save uploaded files to the temporary directory
        for file in uploaded_files:
            with open(os.path.join(tmp_directory, file.name), "wb") as f:
                f.write(file.getvalue())

        helper.last_db_history = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

# Display current files in the knowledge base
current_files = os.listdir(tmp_directory)
if context == "Already uploaded":
    st.sidebar.write("Current Knowledge Base")
    if current_files:
        st.sidebar.write(current_files)
    else:
        st.sidebar.write("**No files uploaded**")

    if helper.last_db_history:
        st.sidebar.write(f"Last updated: {helper.last_db_history}")
else:
    if current_files:
        st.sidebar.write("Files ready to be processed:")
        st.sidebar.write(current_files)
    else:
        st.sidebar.write("No files selected for upload")

if context == "Upload new one" and uploaded_files is not None and len(uploaded_files):
    helper.last_db_history = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


helper.start_chat()