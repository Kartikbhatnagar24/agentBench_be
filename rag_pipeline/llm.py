import dotenv
import os
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
dotenv.load_dotenv()
groq_api_key:str=os.environ.get("GROQ_API_KEY", "")

# def get_llm(streaming: bool = False):
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.3,
#         max_tokens=4096,
#         timeout=None,
#         max_retries=2,
#         api_key=api_key,
#         streaming=streaming,
#     )
#     return llm



from langchain_core.outputs import ChatResult

class ChatOpenAIGroqSafe(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", 1)
        if hasattr(self, "n") and self.n and self.n > 1:
            n = max(n, self.n)
        
        if n <= 1:
            kwargs["n"] = 1
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        original_n = getattr(self, "n", None)
        self.n = 1
        try:
            generations = []
            llm_output = None
            for _ in range(n):
                kwargs["n"] = 1
                res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
                llm_output = res.llm_output
            return ChatResult(generations=generations, llm_output=llm_output)
        finally:
            if original_n is not None:
                self.n = original_n
            else:
                delattr(self, "n")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", 1)
        if hasattr(self, "n") and self.n and self.n > 1:
            n = max(n, self.n)
        
        if n <= 1:
            kwargs["n"] = 1
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        original_n = getattr(self, "n", None)
        self.n = 1
        try:
            generations = []
            llm_output = None
            for _ in range(n):
                kwargs["n"] = 1
                res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
                llm_output = res.llm_output
            return ChatResult(generations=generations, llm_output=llm_output)
        finally:
            if original_n is not None:
                self.n = original_n
            else:
                delattr(self, "n")


def get_eval_llm():
    eval_llm = LangchainLLMWrapper(ChatOpenAIGroqSafe(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    callbacks=[]
))
    return eval_llm

def get_llm(streaming: bool = False):
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    streaming=streaming,
    api_key=groq_api_key,
    )
    return llm