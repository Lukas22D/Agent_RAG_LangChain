import streamlit as st 
import CohereAgent as CohereAgent
from streamlit_chat import message


def main():

    
    st.set_page_config(page_title = 'Agente LangChain', page_icon = ':books:')
    session_container = st.container()
    chat_history = st.container()

    if('session_id' in st.session_state):
     st.header('Agente LangChain')
     user_question = st.text_input('Digite sua pergunta aqui')
     if user_question:
        message(user_question, key= str(0) + '_user', is_user=True)
        with st.spinner('Bot está respondendo...'):
            response = st.session_state.conversation.invoke({'input': user_question}, config={ "configurable": {"session_id": st.session_state.session_id}})
            message(response['answer'], key= str(1) + '_bot', is_user=False)
            for i, text_message in enumerate(response['chat_history']):
                if i % 2 == 0:
                 message(text_message.content, key= str(i+2) + '_user', is_user=True)  
                else:
                 message(text_message.content, key= str(i+2) + '_bot', is_user=False) 
    
    if('session_id' not in st.session_state):
        with session_container:
            
            st.write('Iniciando nova sessão...')
            st.session_state.session_id = st.text_input('Digite o ID da sessão')
            st.session_state.conversation = CohereAgent.conversational_rag_chain() 
     
    
if __name__ == '__main__':
    main()