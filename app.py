import streamlit as st
from streamlit_lottie import st_lottie_spinner
from utils import handle_userinput, get_conversation_chain, get_pdf_text, get_text_chunks, get_vectorstore, page_bg_img
from utils import page_bg_img, css, load_lottieurl,hide_st_style
import time
import os


def main():
    st.set_page_config(page_title="AI PDF Query Master",
                       page_icon=":robot_face:")
    st.markdown(page_bg_img, unsafe_allow_html=True)    
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)

#   Take API key from user
    if 'text' not in st.session_state:
        st.session_state.text = ""
    def update():
        st.session_state.text += st.session_state.text_value
    with st.sidebar.form(key='my_form', clear_on_submit=True):
        st.sidebar.markdown(f"[Get your API KEY from here]({'https://platform.openai.com/account/api-keys'})")
        api_key = st.text_input('Enter Your OpenAI API KEY', value="", key='text_value')
        submit = st.form_submit_button(label='Submit', on_click=update)

    # Set API key as environment variable
    if submit:
        os.environ["OPENAI_API_KEY"] = api_key
        api_key = " "
        st.sidebar.success("API key has been set.")

    # Upload PDF documents
    st.subheader("Your documents")
    uploaded_docs = st.file_uploader(
        "", accept_multiple_files=True,type=(['pdf']))
    
    # Process uploaded documents
    if st.button("Process"):
        lotie_url= "https://lottie.host/d1d77544-9044-4dc0-8dff-83730fda924d/7vEeI2kFCd.json"
        lottie_progress = load_lottieurl(lotie_url)
        with st_lottie_spinner(lottie_progress, loop=True, key="Processing"):

            # Get text from PDF documents
            raw_text = get_pdf_text(uploaded_docs)
           
            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Create vector store from text chunks
            vectorstore = get_vectorstore(text_chunks)

            # Store conversation chain in session state
            st.session_state.conversation = get_conversation_chain(vectorstore)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Chat interface
    st.header("AI PDF Query Master :robot_face:")
    user_question = st.text_input("Input your prompt here")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
