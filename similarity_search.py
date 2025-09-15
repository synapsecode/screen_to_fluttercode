import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DISABLED"] = "1"

# Now import other libraries
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult
from dotenv import load_dotenv

import torch

# Optional: Print info to ensure MPS is disabled
print("Is MPS available?", torch.backends.mps.is_available())

load_dotenv()

def similarity_search(query):
    info_list = [
        {'id': '10001', 'text': 'name: arkTSBlank, description: ...'},
        {'id': '10002', 'text': 'name: arkTSButton, description: ...'}
    ]
    docs = [SearchResult(docid=info['id'], text=info['text'], score=None) for info in info_list]

    # Initialize the ranker with CPU forced
    ranker = SetwiseLlmRanker(
        model_name_or_path='google/flan-t5-large',
        tokenizer_name_or_path='google/flan-t5-large',
        device='mps',       # CPU execution
        num_child=10,
        scoring='generation',
        method='heapsort',
        k=10
    )

    # Perform reranking
    res = ranker.rerank(query, docs)[:2]
    doc_list = [sr.docid for sr in res]
    print(res)
    return doc_list

if __name__ == '__main__':
    similarity_search("Row")