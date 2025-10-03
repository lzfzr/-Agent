import os
from dotenv import load_dotenv

# --- RAG 新增导入 ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# --- 原有导入 ---
from langchain.agents import AgentExecutor, create_react_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI  # 或者使用Ollama
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
#AI客服新增导入
from typing import Literal,TypedDict,Annotated
from langgraph.graph import StateGraph, END
import operator
from langchain_core.output_parsers import StrOutputParser

# --- 1. 环境与模型设置 ---
load_dotenv()
# 初始化一个LLM用于Agent的思考

llm1 = ChatOllama(model="gpt-oss:120b",temperature=0.3) # 请确保您已经下载了这个模型
llm=ChatGoogleGenerativeAI(temperature=0.3 , model='gemini-2.5-pro',)

# --- 2.  构建RAG知识库并创建工具 ---

knowledge_base_folder = r"F:\py.pytorch\llm\knowledge_chunk"
all_documents = []  # 创建一个空列表，用来存放所有文档的内容

# 2.2 遍历文件夹中的所有文件
print(f"正在从 '{knowledge_base_folder}' 文件夹中加载PDF文档...")
for filename in os.listdir(knowledge_base_folder):
    if filename.endswith(".pdf"):
        # 构造完整的文件路径
        pdf_path = os.path.join(knowledge_base_folder, filename)

        # 使用 PyMuPDFLoader 加载单个PDF文件
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()


        all_documents.extend(documents)
        print(f"- 已加载: {filename} (共 {len(documents)} 页)")

print(f"\n成功加载了 {len(os.listdir(knowledge_base_folder))} 个PDF文件。")

# 2.3 切分所有文档内容

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(all_documents)
print(f"所有文档被切分为 {len(docs)} 个小块。")

# 2.4 创建文本的向量化表示 (Embeddings)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("正在为所有文档创建向量数据库，这可能需要一些时间...")

# 2.5 创建并构建向量数据库 (Vector Store)

vector_store = FAISS.from_documents(docs, embeddings)
print("多文档向量数据库创建成功！")

# 2.6 创建检索器
retriever = vector_store.as_retriever()


@tool
def knowledge_base_search(query: str) -> str:
    """
    当需要回答关于产品手册、服务条款、技术文档等专业领域知识时，优先使用此工具。
    这是一个包含了多个专业文档的本地知识库。
    """
    relevant_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

# 2.6 【关键】将检索器包装成一个Agent可以使用的工具
@tool
def pdf_knowledge_base(query: str) -> str:
    """
    当需要回答关于“智能焊接系统”、“机器人焊接”、“路径规划”等技术问题时，优先使用此工具。
    这个工具可以从内部知识库中检索关于智能焊接技术的专业信息。
    输入应该是针对该文档内容的一个具体问题。
    """
    print(f"--- 正在知识库中检索: '{query}' ---")

    # 1. 【核心】执行真实的检索
    # retriever.invoke(query) 会在您的FAISS向量数据库中进行相似度搜索
    relevant_docs = retriever.invoke(query)

    # 2. 【核心】检查是否找到了相关内容
    if not relevant_docs:
        print("--- 在知识库中未找到相关信息 ---")
        return "知识库中没有找到与您问题直接相关的信息。"
    else:
        # 3. 【核心】将找到的多个文档块内容合并成一个字符串并返回
        # 这就是将真实检索结果返回给 Agent 的关键一步
        print(f"--- 找到了 {len(relevant_docs)} 个相关文档块 ---")
        return "\n\n".join([doc.page_content for doc in relevant_docs])


#订单查询工具
@tool
def search_orders(query: str) -> str:
    '''
    当用户咨询订单状态、物流、购买记录等问题时，使用此工具。输入应该时用户的ID或订单号。
    这个工具可以从《》中检索跟订单状态、物流、购买记录有关的问题。
    如果你没找到，就回复不知道
    '''
    print(f"--- 正在数据库中查询订单: {query} ---")
    # 此处为模拟返回结果
    if "12345" in query:
        return "找到订单号12345的状态是：已发货。"
    else:
        return "在数据库中没有找到相关订单信息。"

@tool
def product_suggestion_reply(query: str) -> str:
    """当用户的意图是提供产品建议时，使用此工具。"""
    return "非常感谢您宝贵的建议！我们已经将您的想法记录下来，并会转达给产品团队。"

@tool
def issue_resolved_reply(query: str) -> str:
    """当用户表示问题已解决或表示感谢时，使用此工具。"""
    return "太好了！很高兴能帮到您。如果未来还有其他问题，随时欢迎再次联系我们。"

@tool
def others_problme_reply(query: str) -> str:
    """当用户提出的问题与产品信息无关的时候，比如今天天气怎么样？我长得好看吗？"""
    return "请你提问与该产品有关的问题，谢谢"


#1定义工作流的状态
class GraphState(TypedDict):
    question: str
    web_search: str
    pdf_search: str
    final_answer: str
    classify_result: str
    response: str

#2.创建Agent来执行工具
# --- 准备完整的工具列表 ---
# 加载其他标准工具
search_tool = load_tools(["serpapi"], llm=llm)


other_tools = load_tools(['serpapi','llm-math','dalle-image-generator'],llm=llm)
# 将我们新创建的RAG工具加入列表
tools = other_tools + [pdf_knowledge_base] +[search_orders] +[product_suggestion_reply] + [issue_resolved_reply] + [others_problme_reply]

#Agent1:智能焊接相关任务咨询Agent
prompt_welding='''
你是一个智能焊接领域的专家，非常熟悉焊接智能焊接领域的知识
如果你碰到回答不了的问题，可以使用以下工具
{tools}

请严格遵循以下的思考与行动格式来回答问题:
Question: 这是用户提出的原始问题。
Thought: 在这里写下你为了解决问题，一步步的思考过程。
Action: 从 [{tool_names}] 中选择一个你要使用的工具。
Action Input: 你要提供给这个工具的具体输入内容。
Observation: 这是你执行工具后得到的结果。
... (这个 Thought/Action/Action Input/Observation 的循环可以重复)
Thought: 我现在已经收集到所有信息，可以给出最终答案了。
Final Answer: 这里写下针对原始问题的、最终的、完整的中文回答。

Question: {input}
Thought:{agent_scratchpad}
'''

prompt_WD=ChatPromptTemplate.from_template(prompt_welding)

brain_WD=create_react_agent(llm,tools,prompt_WD)

agent_WD=AgentExecutor(
    agent=brain_WD,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

#Agent2:订单咨询类Agent
prompt_orders='''
你是一个订单咨询客服专家，非常熟悉产品订单知识
如果你碰到回答不了的问题，可以使用以下工具
{tools}

请严格遵循以下的思考与行动格式来回答问题:
Question: 这是用户提出的原始问题。
Thought: 在这里写下你为了解决问题，一步步的思考过程。
Action: 从 [{tool_names}] 中选择一个你要使用的工具。
Action Input: 你要提供给这个工具的具体输入内容。
Observation: 这是你执行工具后得到的结果。
... (这个 Thought/Action/Action Input/Observation 的循环可以重复)
Thought: 我现在已经收集到所有信息，可以给出最终答案了。
Final Answer: 这里写下针对原始问题的、最终的、完整的中文回答。

Question: {input}
Thought:{agent_scratchpad}
'''

prompt_OD=ChatPromptTemplate.from_template(prompt_orders)

brain_OD=create_react_agent(llm,tools,prompt_OD)

agent_OD=AgentExecutor(
    agent=brain_OD,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

#Agent3:网上搜索类Agent
prompt_WEB='''
你是一个网上搜索专家，擅长从网上搜索用户想知道的知识，

如果你碰到回答不了的问题，可以使用以下工具
{tools}

请严格遵循以下的思考与行动格式来回答问题:
Question: 这是用户提出的原始问题。
Thought: 在这里写下你为了解决问题，一步步的思考过程。
Action: 从 [{tool_names}] 中选择一个你要使用的工具。
Action Input: 你要提供给这个工具的具体输入内容。
Observation: 这是你执行工具后得到的结果。
... (这个 Thought/Action/Action Input/Observation 的循环可以重复)
Thought: 我现在已经收集到所有信息，可以给出最终答案了。
Final Answer: 这里写下针对原始问题的、最终的、完整的中文回答。

Question: {input}
Thought:{agent_scratchpad}
'''

prompt_WEB=ChatPromptTemplate.from_template(prompt_WEB)

brain_WEB=create_react_agent(llm1, search_tool , prompt_WEB)

agent_WEB=AgentExecutor(
    agent=brain_WEB,
    tools=search_tool,
    verbose=True,
    handle_parsing_errors=True
)

#1定义工作流的状态
class GraphState(TypedDict):
    question: str
    classify_result: str
    #分支类
    web_search_response: str
    final_answer: str
    #单回答
    response: str

#3.定义节点工作器,构建工作流状态

# 工作分类节点函数
def problem_classifier_node(state: GraphState):
    '''
    这个节点负责对用户问题进行分类
    '''
    print("---NODE: 问题分类---")
    question = state["question"]

    prompt_template = '''
    #角色
    你是一个专业的AI产品客服，你的任务是将用户问出的问题进行分类

    # 核心职责 (Core Mission)
    你的核心职责是将问题分成6类

    #流程
    1.收集用户查询信息
    2.请将以下用户问题分类到最合适的类别中。
    类别选项：'智能焊接','订单咨询', '课程内容咨询', '产品建议', '问题已解决', '其他类'。
    3.如果你判断不出来用户的问题，就返回其他类

    #输出
    1.请只返回唯一的、最匹配的一个类别名称，不要添加任何其他解释或标点符号。


    用户问题: "{question}"
    '''
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 流程是：将输入填入prompt -> 调用llm -> 用parser提取出字符串结果
    classify_chain = prompt | llm | StrOutputParser()

    # 使用chain进行分类

    classify_result = classify_chain.invoke({'question': question})

    classify_result = classify_result.strip().replace("'", "")

    print(f"问题类别: {classify_result}")

    # 将分类结果存入状态，并返回分类结果用于路由
    return {"classify_result": classify_result}  # 返回classify_result用于条件判断


def web_search_node(state: GraphState):
    '''网络搜索节点'''
    print('''节点网络搜索''')
    response = agent_WEB.invoke({'input': state["question"]})
    return {"web_search_response": response['output']}

def welding_node(state: GraphState):
    '''智能焊接专家节点'''
    print('---node:智能焊接')
    response=agent_WD.invoke({'input': state["question"]})
    return {'response':response['output']}

def order_node(state: GraphState):
    '''订单专家节点'''
    print("---NODE: 订单专家---")
    response = agent_OD.invoke({"input": state["question"]})
    return {"response": response['output']}

def problem_solved_node(state: GraphState):
    '''问题解决节点'''
    print("---NODE: 问题解决节点---")
    return {'response':'好的，感谢你的咨询，您还有别的问题吗'}

def suggestion_apply_node(state: GraphState):
    '''建议提供节点'''
    print("---NODE: 建议提供节点---")
    return {'response':'好的，感谢你的建议'}

def others_node(state: GraphState):
    '''其他节点'''
    print("---NODE: 其他节点---")
    return {'response':'我不清楚您提问的问题，请提问相关问题'}

def synthesis_node(state: GraphState):
    '''答案聚合节点'''
    print("---NODE: 最终整合---")
    synthesis_prompt = ChatPromptTemplate.from_template("""
        你是一位资深分析师。请根据以下两份资料，综合、提炼出一个全面且有条理的最终答案来回答用户的问题。

        用户的原始问题是: {question}

        ---
        资料一 (来自内部知识库):
        {response}
        ---
        资料二 (来自网络搜索):
        {web_response}
        ---

        请生成你的最终分析报告:
        """)
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    final_answer = synthesis_chain.invoke({
        "question": state["question"],
        "welding_response": state["response"],
        "web_response": state["web_search_response"],
    })
    return {"final_answer": final_answer}



#4构件工作流
workflow=StateGraph(GraphState)

#添加各节点，'节点名称'，上方定义的节点函数，下面只需套用节点名称，即可使用相关函数内容
workflow.add_node('classifier',problem_classifier_node)
workflow.add_node("welding_agent", welding_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node('synthesis_agent', synthesis_node)
workflow.add_node("order_agent", order_node)
workflow.add_node("problem_solve", problem_solved_node)
workflow.add_node("suggestion_apply", suggestion_apply_node)
workflow.add_node("others", others_node)

#入口
workflow.set_entry_point("classifier")

#添加条件边，根据分类器**输出**决定下一节点

#分类器
def route_logic(state):
    if state["classify_result"] == '智能焊接':
        return 'welding_agent'
    elif state["classify_result"] in ['订单咨询', '课程内容咨询']:
        return 'order_agent'
    elif state["classify_result"] == '产品建议':
        return 'suggestion_apply'
    elif state["classify_result"] == '问题已解决':
        return 'problem_solve'
    elif state["classify_result"] == '其他类':
        return 'others'


#这里只设置分类后的问题流向
workflow.add_conditional_edges(
    "classifier",
    route_logic, # 使用分类器的输出来决定走向
    {
        "welding_agent": "welding_agent",
        "order_agent": "order_agent",
        "suggestion_apply": "suggestion_apply",
        "problem_solve": "problem_solve",
        "others": "others",
    }
)

# 问题流向具体分支后添加分支流向,到END即结尾
workflow.add_edge("welding_agent", 'web_search')
workflow.add_edge('web_search', 'synthesis_agent')
workflow.add_edge('synthesis_agent',END)

workflow.add_edge("order_agent", END)
workflow.add_edge("suggestion_apply", END)
workflow.add_edge("problem_solve", END)
workflow.add_edge("others", END)

# 编译成可运行的App
app = workflow.compile()

def run_workflow(question):
    print(f"\n\n您: {question}")
    final_state = app.invoke({"question": question})
    print(f"AI: {final_state['response']}")

#运行工作流
print("\nAI: 您好！我是AI智能客服，欢迎向我提问。")
while True:
    user_input = input("您: ")
    if user_input.lower() in ["退出", "exit"]:
        print("AI: 好的，期待下次再聊！")
        break

    response = run_workflow(user_input)

