from llm import get_ai_response
import streamlit as st
import os 
from dotenv import load_dotenv



#pinecone vector 저장소에 tax.docx 파일을 임베딩벡터 형식으로 미리 저장해뒀으니 가능
   
    
    
st.set_page_config(page_title="소득세 챗봇", page_icon="★")


# 환경변수를 불러옴
load_dotenv()
#llm = ChatOpenAI(model='gpt-4o')



st.title("소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")




# Session State also supports attribute based syntax
if 'message_list' not in st.session_state:
    st.session_state.message_list=[]
    st.session_state.key = 'value'
    
#session_state에 message_list 에다가 사용자가 입력한게 계속 저장될것.
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    
if user_questsion := st.chat_input(placeholder="소득세데 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message(name="user"):
        st.write(user_questsion)
    #사용자가 입력한 질문 
    st.session_state.message_list.append({"role":"user","content":user_questsion})

    #user_questsion 는 사용자 메세지이다
    # 사용자 메세지로 ai llm에게 전달하는함수호출 
    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_response(user_questsion)
        with st.chat_message(name="ai"):
            ai_message = st.write_stream(ai_response)
            #사용자가 입력한 질문 
            st.session_state.message_list.append({"role":"ai","content":ai_message})
        


    

  

    

    
        