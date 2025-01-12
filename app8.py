import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "model" not in st.session_state:
    st.session_state.model = None

# Streamlit UI
st.set_page_config(page_title="Excel Vectorstore & RAG Chain", layout="wide")

st.title("📊 **Excel Vectorstore and RAG Chain Application**")
st.markdown(
    """
    This application allows you to upload an Excel file, create a vectorstore from its data, and then ask questions about the content using a **RAG chain** approach.
    """
)

st.sidebar.header("🔧 **Excel Upload and Sheet Selection**")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Display file preview when uploaded
if uploaded_file:
    st.sidebar.subheader("📄 **Preview of Uploaded Data**")
    df = pd.read_excel(uploaded_file)
    st.write(df)

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Load Excel and extract elements
    loader = UnstructuredExcelLoader(file_path=temp_file_path, mode="elements")
    docs = loader.load()
    st.session_state.docs = docs  # Store in session state
    st.success("📈 Table loaded successfully.")

    # Embedding model
    embedding = OllamaEmbeddings(model="gemma:2b")

    # Create FAISS vectorstore and store it in session state
    vectorstore = FAISS.from_documents(docs, embedding)
    st.session_state.vectorstore = vectorstore
    st.success("✨ Vectorstore created successfully!")

    # Convert vectorstore to retriever
    retriever = vectorstore.as_retriever()
    st.session_state.retriever = retriever

    # ChatGroq setup
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
    st.session_state.model = model
    st.success("🤖 ChatGroq model loaded successfully.")

# Page selection
page_selection = st.radio("Select Page", ("Query", "Plot"))

# Query Page
if page_selection == "Query":
    if st.session_state.vectorstore and st.session_state.retriever and st.session_state.model:
        # Prompt template
        message = """
        Answer this question using the provided table only:
        {question}

        table:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([("human", message)])

        # RAG chain setup
        rag_chain = {"context": st.session_state.retriever, "question": RunnablePassthrough()} | prompt | st.session_state.model

        # Input for query
        st.subheader("📝 **Ask a Question Based on the Sheet Data**")
        question = st.text_input("Enter your question below", "")

        if question:
            # Perform query and display response
            response = rag_chain.invoke(question)
            st.write(f"**Response**: {response.content}")

else:  # Plot Page
    if uploaded_file:
        # Plot options: dynamically select the plot type based on the query
        plot_type = st.selectbox("Select Plot Type", ["Scatter", "Line", "Bar", "Pie"])

        # Set the axis names (you can adjust these dynamically as well)
        if df.columns.size > 1:
            x_axis = st.selectbox("Select x-axis column", df.columns)
            y_axis = st.selectbox("Select y-axis column", df.columns)

            # Reduce the size of the plot
            fig, ax = plt.subplots(figsize=(8, 6))  # Smaller figure size

            if plot_type == "Scatter":
                # Scatter plot using matplotlib
                ax.scatter(df[x_axis], df[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{y_axis} vs {x_axis}")

            elif plot_type == "Line":
                # Line plot using matplotlib
                ax.plot(df[x_axis], df[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{y_axis} over time")

            elif plot_type == "Bar":
                # Bar plot using matplotlib
                ax.bar(df[x_axis], df[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{y_axis} distribution")

            elif plot_type == "Pie":
                # Pie chart using matplotlib
                ax.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%', startangle=90)
                ax.set_title(f"Proportions of {y_axis} by {x_axis}")

            # Display the plot
            st.pyplot(fig)
