import streamlit as st
import time
import tiktoken
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

MAX_LEN = 512
TEMPERATURE = 0.5
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5" #"hkunlp/instructor-xl"  # BAAI/bge-large-en-v1.5 #BAAI/bge-base-en-v1.5
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
LLM_MODEL = "tiiuae/falcon-7b"  # "hipnologo/falcon-7b-qlora-finetune-chatbot" #"tiiuae/falcon-7b-instruct" #'meta-llama/Llama-2-7b-chat-hf' #"google/flan-t5-xxl"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    start_time = time.time()

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    print("--- %s seconds ---" % (time.time() - start_time))
    return vectorstore


# Open AI solution
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs={"temperature": TEMPERATURE, "max_length": MAX_LEN})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with DOCs",
                       page_icon=":eyes:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with DOCs :eyes:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        ###########
        # raw_text = user_question
        # text_chunks = get_text_chunks(raw_text)
        # vectorstore = get_vectorstore(text_chunks)
        # st.session_state.conversation = get_conversation_chain(vectorstore)
        ##############
        handle_userinput(user_question)
    # st.text_input("Ask a question about your documents:")
    # st.write(user_template.replace("{{MSG}}", "Hello AI"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        # st.file_uploader("Upload your documents here and click on button", accept_multiple_files=True)
        # st.button("Process")
        pdf_docs = st.file_uploader(
            "Upload your documents here and click on button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Please wait. Loading..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text) # uncomment to see raw text before chunks
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks) # uncomment to see raw text with chunks
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
