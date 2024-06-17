from langchain_openai import ChatOpenAI
import jwt
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


zhipuai_api_key = "e3d19e314a59df9e1c0fdbab5bf3aab9.DQyuj25vBVWhgIlT"

def generate_token(apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

zhipu_llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=generate_token(zhipuai_api_key,10000),  # 可以变得更长
        streaming=False,
        verbose=True
    )


# messages = [
#     # AIMessage(content="Hi."),
#     # SystemMessage(content="Your role is a poet."),
#     # HumanMessage(content="深圳2008年的GDP多少亿"),
#     HumanMessage(content="only give me the result,no other words:the result of add 3 to 4"),
# ]

# response = zhipu_llm.invoke(messages)
# print(response)
