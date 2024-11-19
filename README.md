# Embed Arxiv

## Overview

This project involves downloading metadata from ArXiv and generating vector embeddings for the articles using an embedding model. The generated embeddings are compatible with [Milvus](https://milvus.io/).

Please find a copy of embeddings dataset on [HuggingFace](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus) generated using model [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).

## Features

- Metadata Download: Collect metadata for scientific articles from ArXiv.
- Embedding Generation: Use a pre-trained embedding model to generate vector representations for the articles.

## Installation

### Prerequisites

Satisfy prerequisites by issuing this command:
`pip install -r requirements.txt`

## Usage

Run the files in the following order:
1. Download Metadata:
- `prepare_metadata.py` to download the arXiv metadata from kaggle and split it year-wise.
  
2. Embed Abstract:
- `embed_abstract_all.py` to embed the abstract of the **all** papers in metadata.
- `embed_abstract_diff.py` to embed the abstract of the **new** papers in metadata.
   
## Extras:
1. `upload_hf.py` to upload files to [huggingface.co](https://huggingface.co/)
