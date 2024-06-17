import re
from ZhiPu_Model import zhipu_llm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def clean_query(query):
    query = query.replace("SQLResult:", "").replace("```", "").replace("sql", "").replace("SQLQuery:", "").replace("SQL", "")
    colon_index = query.find(':')
    if colon_index != -1:
        query = query[colon_index + 1:]
    colon_index_1 = query.find('：')
    if colon_index_1 != -1:
        query = query[colon_index_1 + 1:]

    # # 查找第一个大写的 SELECT，删除 SELECT 之前的所有内容
    # select_match = re.search(r'SELECT', query)
    # if select_match:
    #     query = query[select_match.start():]
    # # 查找 LIMIT 语句，确保是大写，并保留数字，删除其后的所有内容
    # limit_match = re.search(r'LIMIT \d+\s*', query)
    # if limit_match:
    #     # 从数字后的空格开始删除
    #     query = query[:limit_match.end()].strip()

    semicolon_index = query.find(';')
    if semicolon_index != -1:
        query = query[:semicolon_index]

    # 检查最末尾字符是否为英文句号或中文句号
    while query.endswith('.') or query.endswith('。'):
        # 向前找到最近的大写字母或中文字符
        match = re.search(r'([A-Z]|[^\x00-\x7F])[^A-Z]*[。\.]$', query)
        if match:
            # 删除从找到的字符到句号的所有内容
            query = query[:match.start()]

    return query.strip()  # 移除前后的空白字符以清洁结果


# 数据库连接信息
db_user = "postgres"
db_password = "postgres"  # 请更换成您的数据库密码
db_host = "localhost"
db_name = "postgres"

# 初始化 langchain 的 SQLDatabase 连接
db = SQLDatabase.from_uri(f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}")

# 输出数据库相关信息，以验证连接是否成功
print("本次数据库类型:", db.dialect)
print("可用的表格:", db.get_usable_table_names())

# 创建数据库查询链
generate_query = create_sql_query_chain(zhipu_llm, db)

execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


# 定义链式调用
chain = (
    RunnableLambda(
        lambda context: {
            'query': clean_query(generate_query.invoke(context)), 
            'question': context['question']
        }) # 生成并清理查询，保持问题文本
    |  
    RunnableLambda(
        lambda context: {
            'query': context['query'], 
            'question': context['question'], 
            'print': print("SQL查询语句:", context['query'])
        }) # 打印查询，继续传递'query'和'question'
    |  
    RunnableLambda(
        lambda context: {
            'result': execute_query.invoke(context['query']), 
            'query': context['query'], 
            'question': context['question']
        }) # 执行查询，维持'query', 'result'和'question'
    |  
    RunnableLambda(
        lambda context: {
            'question': context['question'], 
            'query': context['query'], 
            'result': context['result']
        }) # 保持完整的上下文
    |  
    answer_prompt   # 应用最终的重构答案处理
    | 
    zhipu_llm 
    | 
    StrOutputParser() 
)


# 调用链以生成最终答案
final_answer = chain.invoke({"question": "计算公司表前十个公司的项目数量"})
print("智谱回答:", final_answer)
