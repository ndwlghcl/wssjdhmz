
from dotenv import load_dotenv
from os import getenv

from agent import Agent
from config import Config
from llm import LLM
from message import Message

load_dotenv()


RAG_TEST_PROMPT="""请仅根据下面提供的资料回答问题。如果资料中没有相关信息，就回答"资料中没有提到"，不要自己编造。

资料:
{context_text}

问题: {question}
"""

class RAGTest(Agent):
    def __init__(self, name: str, config: Config | None = None):

        if config is None:
            config = Config.from_env()

        self.llm = LLM()
        super().__init__(name=name, llm=self.llm, config=config)

        self.documents = ["“俄耳甫斯回头”不是‘我选择了看你一眼，哪怕因此失去你’，也不是‘我为了证明爱而回头’。更接近的是：**他想不回头。可是他无法把所爱的人变成一个抽象的、不可见的存在。爱具体的、鲜活的欧律狄刻，所以爱让他回头。**这也是故事最惆怅的地方：**他不是选择了一个错误的选项，而是连不犯这个错误的自由都没有。**",
             "一个人认为宇宙没有赋予生命意义当然也可以继续体验生命，从来都没有‘应该’，没有应该结束生命，只有想不想去做的区别。",
             "“品味”让人能够判断一个东西好不好、成熟不成熟、优秀还是平庸。随着阅读、观看和经验增加，一个人的判断标准会越来越丰富，也越来越能够迅速识别某种作品或经验属于什么类型。但这种能力同时存在一个反作用：当一个人已经拥有大量成熟的判断模板时，面对新的东西，可能会下意识地把它迅速归类，用过去已经形成的标准判断它，而不是暂时放下既有结论，真正观察它是什么。"]

        self.resp = self.llm.client.embeddings.create(model="text-embedding-v3", input=self.documents)

        self.store = [{"text": text, "vector": item.embedding}
                      for text, item in zip(self.documents, self.resp.data)]
    @staticmethod
    def cosine(a: list, b: list):
        dot_product=sum(x*y for x,y in zip(a,b))
        norm_a=sum(x**2 for x in a)**0.5
        norm_b=sum(x**2 for x in b)**0.5
        return dot_product/(norm_a*norm_b)

    def search(self,question:str, store:list, top_k:int=3):
        q=self.llm.client.embeddings.create(model="text-embedding-v3", input=[question])

        q_embedding = q.data[0].embedding

        score=[{"text":item["text"],"score":self.cosine(q_embedding, item["vector"])}
               for item in store]

        # 按 score 字段从大到小排序
        ranking = sorted(score, key=lambda item: item["score"], reverse=True)
        # 取前 top_k 名，只返回文字
        return [item["text"] for item in ranking[:top_k]]

    def run(self, question: str):
        # 检索到的多段资料要用空行拼成一段文字
        # （直接把列表塞进 format 会在提示词里留下 ['...', '...'] 的方括号和引号）
        context_text = "\n\n".join(self.search(question, self.store, top_k=2))
        prompt = RAG_TEST_PROMPT.format(context_text=context_text, question=question)
        message = Message(role="user", content=prompt)
        return self._stream_reply([message])


if __name__ == "__main__":
    rag = RAGTest("RAG助手")
    for question in ["品味的作用有哪些？", "俄耳甫斯为什么回头？", "认为生命没有意义就该结束生命吗？"]:
        print(f"\n问题: {question}")
        print("答案: ", end="")
        rag.run(question)



