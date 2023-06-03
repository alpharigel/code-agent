"""
This is a simple streamlit app that uses langchain to create a simple agent 
that can create and edit python code. 

"""

import os
import re
import streamlit as st

from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from util.messages import message_style, message
from util.prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re


load_dotenv()


# Define which tools the agent can use to answer user queries
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

class CodeAgentOutputParser(AgentOutputParser):
    
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish

        ai_prefix = self.prompt.input_variables["ai_prefix"]
        human_prefix = self.prompt.input_variables["human_prefix"]

        # parse the first line from the output
        (first_line, rest) = llm_output.split("\n", maxsplit=1) 

        # check if the first line is a finish line  
        if f"From: {ai_prefix}, To: {human_prefix}" in first_line:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": rest.strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        selected_tool = first_line.split(":")[1].strip()
        action_input = rest.strip()

        # Return the action and action input
        return AgentAction(tool=selected_tool, tool_input=action_input, log=llm_output)

from langchain.tools import DuckDuckGoSearchRun
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
#prompt = PromptTemplate(
#            template= BAS  + format_prompt, 
#            input_variables=["current_file", "chat_history", "human_input"])
#chain = LLMChain(llm=self.llm, memory=self.memory, prompt=self.prompt, verbose=True)


    
class CodeAgent:
    """
    A class that represents a code agent. It takes in a base prompt and 
    format prompt and creates an LLMChain from AzureChatOpenAI, ConversationBufferMemory
    and PromptTemplate
    """

    def __init__(self, user_prompt, format_prompt):
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        self.prompt = PromptTemplate(
            template=user_prompt + format_prompt, 
            input_variables=["current_file", "chat_history", "human_input"])
        self.chain = LLMChain(llm=self.llm, memory=self.memory, prompt=self.prompt, verbose=True)

    def run(self, human_input, current_file):
        """
        Runs the chain with the human input and current file
        """
        result = self.chain.run(human_input=human_input, current_file=current_file, 
                                chat_history=self.memory.load_memory_variables({"chat_history"})
        )
        ret = parse_response(result)
        changed_file, ai_out = ret["code"], ret["text"]
        self.memory.chat_memory.add_user_message(human_input)
        self.memory.chat_memory.add_ai_message(ai_out)
        return changed_file, ai_out, result
    
    

def submit():
    st.session_state.submitted_input = st.session_state.user_input
    st.session_state.user_input = ""

def init_session(user_prompt=BASE_PROMPT, format_prompt=FORMAT_PROMPT):
    """Initializes the session state for submitted input, agent, and messages"""
    st.session_state.setdefault("submitted_input", "")
    st.session_state.setdefault("agent", CodeAgent(user_prompt, format_prompt))
    st.session_state.setdefault("user_messages", [])
    st.session_state.setdefault("ai_messages", [])
    st.session_state.setdefault("current_text", "")
    st.session_state.setdefault("last_input", "")
    

def main():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    script_name = os.path.basename(__file__)
    with open(os.path.join(script_directory, script_name), "r") as f:
        st.session_state.current_text = f.read()

    with st.sidebar:
        st.text_input("You: ", on_change=submit, key="user_input")
    user_input = st.session_state.submitted_input

    if user_input and user_input != st.session_state['last_input']:
        agent = st.session_state.agent
        changed_file, ai_out, result = agent.run(user_input, st.session_state.current_text)
        st.session_state.user_messages.append(user_input)
        st.session_state.ai_messages.append(ai_out)
        st.session_state.last_input = user_input
        if changed_file:
            st.session_state.current_text = changed_file

    st.write('Changed Result:')
    st.text(st.session_state.current_text)

    if st.button("Save"):
        with open(script_name, "w") as f:
            f.write(st.session_state.current_text)
        st.success("Saved!")

    message_style()
    with st.sidebar:
        # print the message from oldest to newest
        for user_message, ai_message in zip(st.session_state.user_messages[::-1], st.session_state.ai_messages[::-1]):
            message(user_message, is_user=True)
            message(ai_message, is_user=False)

if __name__ == "__main__":
    st.set_page_config(page_title="Shoemaker ", page_icon="👞")
    init_session()
    main()

