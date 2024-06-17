from ZhiPu_Model import zhipu_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    AIMessage(content="Hi."),
    # SystemMessage(content="Your role is a poet."),
    HumanMessage(content=" "),
    HumanMessage(content="only give me the result,no other words:the result of add 3 to 4"),
]
zhipu_llm.streaming = True
# print(zhipu_llm)

for chunk in zhipu_llm.stream("1+1"):
    # print(chunk.content, end="", flush=True)
    print(chunk.content)
