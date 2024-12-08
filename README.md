# Embed Arxiv Simpler
## Frontend at [PaperMatch](https://github.com/mitanshu7/PaperMatch))

## Overview

This project involves downloading metadata from ArXiv and generating vector embeddings for the articles using an embedding model. The generated embeddings are compatible with [Milvus](https://milvus.io/) vector database.

Find a copy of embeddings dataset at [bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus) generated using model [mxbai-embed-large-v1](https://www.mixedbread.ai/docs/embeddings/mxbai-embed-large-v1).

Binary embeddings available at [bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary](https://huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary).

## Features

- Metadata Download: Collect metadata for scientific articles from ArXiv.
- Embedding Generation: Use a pre-trained embedding model to generate vector representations for the articles.
- Updaing Embeddings: Update existing embeddings for new papers in metadata.
- Milvus Prep: Setup Milvus vector database for efficient similarity search.

## Installation

### Prerequisites

Satisfy prerequisites by issuing this command:
`pip install -r requirements.txt`

### Environment Variables

Create a `.env` file in the root directory of the project with the following variables:

- HF_API_KEY="SECRET" # Required for uploading dataset to HuggingFace
- MXBAI_API_KEY = "SECRET" # Required for using mxbai-embed-large-v1 model via API 

See `.env.sample` for an example.  

## Usage

1. Create new embeddings:
- `create_embeddings.py` to embed the abstract of the **all** papers in metadata.
  - Read the configuration in the python script to suit your needs.

2. Binarise embeddings (Optional):
- `binarise_embeddings.py` to binarise the embeddings for Milvus.

3. Update existing embeddings (Weekly):
- `update_embeddings.py` to embed the abstract of the **new** papers in metadata.
  - Read the configuration in the python script to suit your needs.

4. Start Milvus (See setup [here](https://milvus.io/docs)):
- `bash start_milvus.sh start` to start Milvus. 
  - Remove `--restart always` and add `:Z` after each volume (`-v`) to make it compatible with podman + systemd. 

5. Prepare Milvus:
- `prepare_milvus.py` to load the embeddings into Milvus.
  - Read the configuration in the python script to suit your needs.
   
## Keep embeddings updated:
1. Setup a crontab to run the script `update_embeddings.sh` every week, modify command accordingly.
```bash
$ crontab -e

$ 0 0 * * 1 /bin/bash /home/[USERNAME]/embed_arxiv_simpler/update_embeddings.sh >> /home/[USERNAME]/embed_arxiv_simpler/update_embeddings.log 2>&1

$ crontab -l
```
This cron runs midnight every Monday.

## Keep vector database updated:
1. Setup a crontab to run the script `update_milvus.sh` every week, modify command accordingly.
```bash
$ crontab -e

$ 0 0 * * 2 /bin/bash /home/[USERNAME]/embed_arxiv_simpler/update_milvus.sh >> /home/[USERNAME]/embed_arxiv_simpler/update_milvus.log 2>&1

$ crontab -l
```
This cron runs midnight every Tuesday.

## Contributing
Feel free to contribute to the project by submitting issues, pull requests, or suggestions. 
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Contact
For any questions or feedback, please contact [mitanshu.sukhwani@gmail.com](mailto:mitanshu.sukhwani@gmail.com).