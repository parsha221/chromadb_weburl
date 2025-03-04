import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel

# Define the Pydantic model
class AnswerModel(BaseModel):
    answer: str

def process_input(urls, question):
    model_local = ChatOllama(model="llama3.2")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | PydanticOutputParser(AnswerModel)
    )
    result = after_rag_chain.invoke(question)
    return result.answer

# Define Gradio interface
iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter URLs separated by new lines"), gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query with Ollama",
                     description="Enter URLs and a question to query the documents.")
iface.launch()


# from pydantic import BaseModel
# from langchain.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOllama

# # Define the Pydantic model
# class Joke(BaseModel):
#     setup: str
#     punchline: str

# # Create the prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     input_variables=["query"],
#     partial_variables={"format_instructions": PydanticOutputParser.get_format_instructions(Joke)}
# )

# # Use the PydanticOutputParser
# parser = PydanticOutputParser(Joke)
# model = ChatOllama(model="llama3.2")
# chain = prompt_template | model | parser

# # Example query
# query = "Tell me a joke."
# result = chain.invoke({"query": query})
# print(result)
