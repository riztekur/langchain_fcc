import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, load_tools, AgentType

api_key = st.secrets['OPENAI_API_KEY']
llm = OpenAI(temperature=0.3, openai_api_key=api_key)

def get_pet_name(pet_type, pet_color):
    prompt_template = PromptTemplate(
        input_variables=['pet_type'],
        template="Suggest me 5 cool full names for my pet {pet_type}. My pet color is {pet_color}"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="pet_name")
    response = name_chain(
        {
            'pet_type':pet_type,
            'pet_color':pet_color
        }
    )
    return response['pet_name']

def langchain_agent():
    tools = load_tools(['wikipedia','llm-math'],llm=llm)
    agent=initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(
        "What is the average age of a dog?"
    )

    return result

if __name__ == "__main__":
    col1, col2 = st.columns(2)
    with col1:
        pet_type = st.text_input('Insert pet type')
    with col2:
        pet_color = st.text_input('Insert pet color')
    if pet_type and pet_color:
        st.write(get_pet_name(pet_type, pet_color))
        st.write(langchain_agent())
    