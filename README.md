# Embed Arxiv (Backend for [PaperMatch](https://github.com/mitanshu7/PaperMatch))

## Overview

This project involves downloading metadata from ArXiv and generating vector embeddings for the articles using an embedding model. The generated embeddings are compatible with [Milvus](https://milvus.io/) vector database.

Please find a copy of embeddings dataset on [HuggingFace](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus) generated using model [mxbai-embed-large-v1](https://www.mixedbread.ai/docs/embeddings/mxbai-embed-large-v1).

## Features

- Metadata Download: Collect metadata for scientific articles from ArXiv.
- Embedding Generation: Use a pre-trained embedding model to generate vector representations for the articles.
- Updaing Embeddings: Update existing embeddings for new papers in metadata.

## Installation

### Prerequisites

Satisfy prerequisites by issuing this command:
`pip install -r requirements.txt`

### Environment Variables

Create a `.env` file in the root directory of the project with the following variables:

- HF_API_KEY="<SECRET>" # Required for uploading dataset to HuggingFace
- MXBAI_API_KEY = "<SECRET>" # Required for using mxbai-embed-large-v1 model via API 

See `.env.sample` for an example.  

## Usage

1. Create new embeddings:
- `create_embeddings.py` to embed the abstract of the **all** papers in metadata.

2. Update existing embeddings:
- `update_embeddings.py` to embed the abstract of the **new** papers in metadata.
   
## Keep embeddings updated:
1. Setup a crontab to run the script `update_embeddings.sh` every week, modify command accordingly.
```bash
crontab -e
0 0 * * 2 /bin/bash /home/milvus/embed_arxiv_simpler/update_embeddings.sh >> /home/milvus/embed_arxiv_simpler/update_embeddings_crontab.log 2>&1
crontab -l
```
This cron runs midnight every Tuesday.

## Contributing
Feel free to contribute to the project by submitting issues, pull requests, or suggestions. 
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Contact
For any questions or feedback, please contact [mitanshu.sukhwani@gmail.com](mailto:mitanshu.sukhwani@gmail.com).