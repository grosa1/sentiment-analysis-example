from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
import numpy as np
from langchain.prompts import ChatPromptTemplate


load_dotenv()

# setup langfuse handler
langfuse_handler = CallbackHandler()


# OpenAI rate limit params
MINUTES = 60
DEFAULT_MAX_REQUESTS_PER_MINUTE = 10000
DEFAULT_NUM_THREADS = 8


@sleep_and_retry
@limits(calls=DEFAULT_MAX_REQUESTS_PER_MINUTE, period=MINUTES)
def exec_chain(chain, chain_params, request_id):    
    return {"idx": request_id, "result": chain.invoke(chain_params, config={"callbacks": [langfuse_handler]})}


class SentimentClassification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text, either 'positive' or 'negative'")


def run_zeroshot_clf(docs: np.array, prompt: ChatPromptTemplate, model="gpt-4o-mini", temperature=0, output_schema=SentimentClassification):
    llm = ChatOpenAI(model=model, temperature=temperature)
    runnable_chain = prompt | llm.with_structured_output(schema=output_schema)

    # preprocess docs
    requests = [{"idx": i, "doc": docs[i]} for i in range(len(docs))]

    # Process the requests
    results = [None] * len(requests)
    with ThreadPoolExecutor(DEFAULT_NUM_THREADS) as p:
        for result in p.map(lambda request_data: exec_chain(chain=runnable_chain, chain_params={"input": request_data.get("doc")}, request_id=request_data.get("idx")), requests):
            results[result["idx"]] = result["result"]

    return results