import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_model():
    return ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.01,
        max_tokens=1024
    )

def getLLamaresponse(input_text, no_words, blog_style):
    try:
        llm = load_model()

        template = """
        Write a complete blog post for a {blog_style} audience on the topic: "{input_text}".
        The blog should be approximately {no_words} words with three sections:
        Introduction, Body, and Conclusion.
        Make sure to finish the Conclusion before stopping.
        """

        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "no_words"],
            template=template
        )
        formatted_prompt = prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            no_words=no_words
        )

        response = llm.invoke(formatted_prompt)
        return response.content

    except Exception as e:
        st.error("Something went wrong. Please try again.")
        st.exception(e)
        return None

st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

st.caption("Note: Word count is approximate — LLMs may not match it exactly.")

submit = st.button("Generate")

if submit:
    if not input_text.strip():
        st.error("Blog Topic cannot be empty.")
    elif not no_words.strip().isdigit() or int(no_words.strip()) <= 0:
        st.error("Number of words must be a positive integer.")
    else:
        with st.spinner("Generating blog..."):
            response = getLLamaresponse(input_text, int(no_words.strip()), blog_style)
            if response:
                st.write(response)