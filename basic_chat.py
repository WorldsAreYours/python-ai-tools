from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import gradio as gr

load_dotenv()

"""
Basic Agent to message the world
"""

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

groq_api_key = os.getenv("GROQ_API_KEY")

def setup_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional translator. Translate the following text from {input_language} to {output_language}. Only provide the translation, no explanations, quotes, or additional text."),
            ("human", "Translate this text: {input}")
        ]
    )
    return prompt | llm


def translate_text(input_lang, output_lang, input_text):
    chain = setup_chain()
    result = chain.invoke({
        "input_language": input_lang,
        "output_language": output_lang,
        "input": input_text
    })
    return result.content

iface = gr.Interface(
    fn = translate_text,
    inputs = [
        gr.Dropdown(choices=["English", "Spanish", "French"], label="From"),
        gr.Dropdown(choices=["English", "Spanish", "French"], label="To"),
        gr.Textbox(label="Text to translate")
    ],
    outputs = gr.Textbox(label="Translated Text"),
    title="Language Translator"
)


def main():
    """Main function to run the chat"""
    iface.launch(share=True)


if __name__ == "__main__":
    main()
    