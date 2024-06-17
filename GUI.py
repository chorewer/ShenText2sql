import re
import tkinter as tk
from tkinter import scrolledtext
from ZhiPu_Model import zhipu_llm
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def clean_query(query):
    query = query.replace("SQLResult:", "").replace("```", "").replace("sql", "").replace("SQLQuery:", "").replace(
        "SQL", "")
    colon_index = query.find(':')
    if colon_index != -1:
        query = query[colon_index + 1:]
    colon_index_1 = query.find('：')
    if colon_index_1 != -1:
        query = query[colon_index_1 + 1:]

    # 查找第一个大写的 SELECT，删除 SELECT 之前的所有内容
    select_match = re.search(r'SELECT', query)
    if select_match:
        query = query[select_match.start():]
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

class SQLTranslatorApp:
    def __init__(self, master):
        self.master = master
        master.title("SQL_Translator")

        self.label = tk.Label(master, text="请输入自然语言查询:")
        self.label.pack()

        self.input_text = scrolledtext.ScrolledText(master, height=7, wrap=tk.WORD)
        self.input_text.pack()

        self.translate_button = tk.Button(master, text="查询", command=self.translate)
        self.translate_button.pack()

        self.output_label = tk.Label(master, text="生成的SQL查询语句:")
        self.output_label.pack()

        self.output_text = scrolledtext.ScrolledText(master, height=18, wrap=tk.WORD)
        self.output_text.pack()

        self.result_label = tk.Label(master, text="最终查询结果:")
        self.result_label.pack()

        self.result_text = scrolledtext.ScrolledText(master, height=18, wrap=tk.WORD)
        self.result_text.pack()

        # 数据库连接信息
        db_user = "postgres"
        db_password = "postgres"  # 请更换成您的数据库密码
        db_host = "localhost"
        db_name = "postgres"

        # 初始化 langchain 的 SQLDatabase 连接
        # db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
        self.db = SQLDatabase.from_uri(f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}")

        # 创建langchain工具和链
        self.generate_query = create_sql_query_chain(zhipu_llm, self.db)
        self.execute_query = QuerySQLDataBaseTool(db=self.db)
        self.answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )
        self.rephrase_answer = self.answer_prompt | zhipu_llm | StrOutputParser()

    def translate(self):
        user_question = self.input_text.get("1.0", tk.END).strip()
        sql_query = self.generate_query.invoke({"question": user_question})
        cleaned_query = clean_query(sql_query)
        sql_result = self.execute_query.invoke(cleaned_query)
        final_answer = self.rephrase_answer.invoke({"question": user_question, "query": cleaned_query, "result": sql_result})

        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, cleaned_query)

        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, final_answer)

root = tk.Tk()
app = SQLTranslatorApp(root)
root.mainloop()
