from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA
from langchain_classic import hub # LangChain v1 migration guide 참고
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_ai_message(user_message):

    # embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    embedding = UpstageEmbeddings(model="solar-embedding-1-large") # upstage에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding = embedding) # 이미 만들어진 vector db 사용

    llm = ChatUpstage()
    prompt = hub.pull('rlm/rag-prompt')
    retriever = database.as_retriever(search_kwargs = {'k':4})


    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever, #as_retriever 메서드는 langchain을 사용하면 다양한 vecter db에 사용가능
        chain_type_kwargs = {'prompt':prompt}
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문은 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전  : {dictionary}

        질문 : {{question}}

    """
    )

    dictionary_chain = prompt | llm | StrOutputParser()

    tax_chain = {"query" : dictionary_chain} | qa_chain
    ai_messge = tax_chain.invoke({'question' : user_message})
    return ai_messge['result']
