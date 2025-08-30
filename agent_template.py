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

# Configuration
MODEL_NAME = "llama-3.1-8b-instant"
TITLE = "Agent Template"
SYSTEM_PROMPT = "You are a helpful AI assistant."
HUMAN_PROMPT = "User input: {input_text}"

llm = ChatGroq(model=MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"))

groq_api_key = os.getenv("GROQ_API_KEY")

prompt_template = "Prompt here"


def setup_chain(system_prompt=None, human_prompt=None):
    """
    Setup the LangChain chain with customizable prompts
    
    Args:
        system_prompt: System message prompt (optional)
        human_prompt: Human message prompt (optional)
        
    Returns:
        Configured LangChain chain
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    human_prompt = human_prompt or HUMAN_PROMPT
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
    )
    return prompt | llm


def agent_template(input_text):
    """
    Main agent function - customize this for your specific use case
    
    Args:
        input_text: User input to process
        
    Returns:
        Processed response from the agent
    """
    chain = setup_chain()
    result = chain.invoke({
        "input_text": input_text
    })
    return result.content

iface = gr.Interface(
    fn = agent_template,
    inputs = [
        gr.Textbox(label="Input text here:")
    ],
    outputs = gr.Textbox(label="Output here"),
    title=TITLE
)


def main():
    """Main function to run the chat"""
    iface.launch(share=True)


if __name__ == "__main__":
    main()
    