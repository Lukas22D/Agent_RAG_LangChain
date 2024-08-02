import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa o modelo de linguagem da Cohere usando a chave da API
llm = ChatCohere(model='command-r-plus', cohere_api_key=os.getenv('COHERE_API_KEY'))

### Construir o recuperador ###
# Carrega os documentos da web a partir de uma URL específica
loader = WebBaseLoader(
    web_paths=("https://python.langchain.com/v0.2/docs/tutorials/agents/", "https://python.langchain.com/v0.2/docs/concepts/", "https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/")
)
docs = loader.load()

# Divide os documentos em partes menores para facilitar o processamento
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Cria uma loja de vetores a partir dos documentos divididos usando embeddings da Cohere
vectorstore = Chroma.from_documents(documents=splits, embedding=CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY')))
retriever = vectorstore.as_retriever()

### Contextualizar a pergunta ###
# Define o prompt do sistema para contextualizar perguntas
contextualize_q_system_prompt = (
   "Given a chat history and a user's question about Python code analysis, creation, correction, "
    "or related concepts (e.g., Langchain, libraries, frameworks), reformulate the question to be "
    "understandable without the chat history. Focus on technical aspects. Do NOT answer the question, "
    "only reformulate it if needed."
)
# Cria um template de prompt de chat com a mensagem do sistema e um espaço reservado para o histórico do chat
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Cria um recuperador que leva em conta o histórico do chat
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Responder a pergunta ###
# Define o prompt do sistema para responder perguntas
system_prompt = (
    "You are a Langchain expert specializing in Python code. Analyze, correct, and create high-quality "
    "Python code, explaining your reasoning clearly. Provide concise responses (3-5 sentences) with code "
    "examples when helpful. Identify potential errors or improvements and suggest best practices, "
    "referencing relevant Langchain features and documentation. Use the following context:\n\n"
    "{context}"
)
# Cria um template de prompt de chat com a mensagem do sistema e um espaço reservado para o histórico do chat
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Cria uma cadeia de documentos "stuff" para responder perguntas
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Cria uma cadeia de recuperação que combina o recuperador com consciência histórica e a cadeia de resposta de perguntas
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Gerenciar o histórico de chat com estado ###
# Dicionário para armazenar o histórico de mensagens do chat
store = {}

# Função para obter o histórico de uma sessão específica
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Cria uma cadeia de recuperação conversacional com histórico de mensagens
def conversational_rag_chain():
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

