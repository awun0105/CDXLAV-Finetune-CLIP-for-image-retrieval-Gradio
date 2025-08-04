# CDXLAV-Finetune-CLIP-for-image-retrieval-Gradio

## How to run code locally and use the app ?
*example app deployed on hugging face space:* https://huggingface.co/spaces/anhquanlam/clip-image-search-app-deepfashion-multimodal

*Download the dataset at:* https://huggingface.co/datasets/anhquanlam/clip-deepfashion-multimodal

1. Go to .env file and adjust the Paths of the directory (dataset) on your computer (watch the example paths inside .env file).
  
2. create virtual environment.
3. open terminal in the app.py directory, activate the virtual env and 'pip install -r requirement.txt'
4. use 'python app.py' and the app will run, the console will give you a link. Go to that link and you will see the app running.

5. scan the image directory to create embeddings file and indexing for faiss, or just go on and search because i've already scanned and put the embed_data folder in the huggingface dataset.
