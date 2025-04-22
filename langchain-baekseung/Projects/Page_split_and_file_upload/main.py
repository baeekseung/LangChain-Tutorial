import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_teddynote.prompts import load_prompt
import glob

load_dotenv()

st.title("나만의 ChatGPT")

if "messages" not in st.session_state:
    # 대화기록 저장 목적
    st.session_state['messages'] = []

with st.sidebar:
    clear_button = st.button("대화 초기화")

    prompt_files = glob.glob("Prompt_temp/*.yaml")

    selected_prompt = st.selectbox(
    "프롬프트를 선택해 주세요.",
    prompt_files, index=0)

    task_input = st.text_input("task 입력","")


def print_messages():
    for chat_message in st.session_state.messages:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state.messages.append(ChatMessage(role=role, content=message))

def create_chain(prompt_type, task=""):

    prompt = load_prompt(prompt_type, encoding="utf-8")

    if task:
        prompt = prompt.partial(task=task)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain

if clear_button:
    st.session_state.messages = []
    st.rerun()

print_messages()

user_input = st.chat_input("Say something")

if user_input:
    # 사용자 입력 출력
    st.chat_message("user").write(user_input)

    chain = create_chain(selected_prompt, task=task_input)
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_response = ""
        for token in response:
            ai_response += token
            container.markdown(ai_response)

    # response = chain.invoke({"question": user_input})

    # st.chat_message("assistant").write(response)

    # 대화기록 저장 
    add_message("user", user_input)
    add_message("assistant", ai_response)



# 실행 방법 : streamlit run main.py