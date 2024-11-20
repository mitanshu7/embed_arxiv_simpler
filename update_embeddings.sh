#!/bin/bash
source /home/milvus/miniforge3/bin/activate search_arxiv
python /home/milvus/embed_arxiv_simpler/update_embeddings.py >> /home/milvus/embed_arxiv_simpler/update_embeddings.log 2>&1