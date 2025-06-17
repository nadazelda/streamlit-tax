from langchain_upstage import UpstageEmbeddings #embedding
from langchain_pinecone import PineconeVectorStore #vector 
from langchain_upstage import ChatUpstage #ai 모델 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from config import answer_examples



store = {} #기본적으로 내부 저장소에 저장해서

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_history_retriever():
    retriever = get_retriever()
    llm = get_llm()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

##qa chain 관련
def get_ra_Chain():
    llm = get_llm()
    #few shot 을 넣어서 챗봇이 대답을 이런형식으로 유도하게합니다
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    #system_prompt는 기본이고
    #chat_history는 사용자와 응답할때까지 없으니가
    #few shot을 넣어서 마치  우린 이런식으로 대답을해왔어! 라고 착각하게 만든다.
    #
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # few shot 이라는걸 넣어주면서 대답형식을 유도합니다 
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    #전체 데이터를 다 보여주는게 아니라 pick으로 대답부분만 리턴한다 
    
    return conversational_rag_chain



def get_ai_response(user_message):
    qa_chain = get_ra_Chain()
    dictionary_chain = get_dictionary_chain()
    tax_chain = {"input":dictionary_chain}  |qa_chain
    #최종 연결된 tax_chain으로 사용자 메세지를 전달해서 ai 메세지를 가져온다 
    ai_response = tax_chain.stream( 
        {
            "question":user_message
         }, 
        config={
            "configurable":{"session_id":"abd123"}
        }
    )
    return ai_response


##llm 관련한 함수
def get_llm(model='solar-pro'):
    llm = ChatUpstage(api_key="up_Dmohp2sYgjEy5aW2sL9bvG6VTqgAn", model=model) # 키 환견ㄱ변수에서 못불러와서 임시로 이렇게 합니다
    return llm
##retriever 관련함수
def get_retriever():
    #embedding을로 문서 유사도 학습을 돌린다 
    embeddings = UpstageEmbeddings(
        api_key="up_Dmohp2sYgjEy5aW2sL9bvG6VTqgAn",
        model="embedding-query"
    )
    index_name = 'tax-index'
    # 여기서 from_documents 말고 from_existing_index를 사용해주셔야 합니다.
    database = PineconeVectorStore.from_existing_index(
                index_name=index_name, 
                embedding=embeddings
            )
    retriever=database.as_retriever()
    return retriever


#dictionary chain 관련해서 분리함수 
def get_dictionary_chain():    
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전 : {dictionary}
        질문 : {{question}}
    """)
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain