from db import SearchMechanism
from clip import CLIPSearcher
from clusterer import ImageIndexer
import os
import gradio as gr
from pathlib import Path
from PIL import Image
import torch
from dotenv import dotenv_values


# Cho phép chạy nếu lỗi OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
env = dotenv_values('.env')

# Gọi và nhận lại biến env từ file .env (có url, paths, v.v.)
# env = download_and_prepare_dataset()

# Đường dẫn thư mục ảnh và index
path = env['DEFAULT_IMAGES_PATH']
index_path = env['INDEX_PATH']
# Models cache được lưu ở đây
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']

# Khởi tạo các thành phần
clip_searcher = CLIPSearcher()
image_indexer = ImageIndexer(index_path)
search_mechanism = SearchMechanism(clip_searcher, image_indexer)


# Hàm xử lý tìm kiếm bằng văn bản
def search_by_text(text, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_text(text, top_k, use_cluster_search)

    if top_k_df is None or top_k_df.empty:
        print("No results found for input text.")
        return []

    images = []
    for _, row in top_k_df.iterrows():
        path = row['image_path']
        if os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception as e:
                print(f"Error opening image at {path}: {e}")
        else:
            print(f"Warning: Image not found → {path}")

    return images

# Hàm xử lý tìm kiếm bằng ảnh


def search_by_image(image, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_image(
        image, top_k, use_cluster_search)

    if top_k_df is None or top_k_df.empty:
        print("No similar images found.")
        return []

    images = []
    for _, row in top_k_df.iterrows():
        path = row["image_path"]
        if os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception as e:
                print(f"Error opening image at {path}: {e}")
        else:
            print(f"Warning: Image not found → {path}")

    return images
# Hàm tìm kiếm tổng hợp


def combined_search(search_type, text, image, top_k, use_cluster_search):
    if search_type == "Text":
        if not text:
            return gr.Warning("Please enter a text query.")
        return search_by_text(text, top_k, use_cluster_search)
    else:
        if not image:
            return gr.Warning("Please upload an image.")
        return search_by_image(image, top_k, use_cluster_search)

# Quét thư mục ảnh để tạo lại index


def scan_dir(path):
    if path is None or not os.path.exists(path):
        return gr.Info("Path does not exist")
    search_mechanism.scan_directory(Path(path))
    return path


# Tạo giao diện Gradio
with gr.Blocks() as webui:
    gr.Markdown("## CLIP Image Search App")
    path = gr.Textbox(label="Path", info="Path to scan images", value=path)
    scan_dir_btn = gr.Button("Scan Directory", variant="primary")
    with gr.Column():
        with gr.Row(equal_height=True):
            search_type = gr.Radio(
                choices=["Text", "Image"], label="Search by", value="Text")
            with gr.Column():
                top_k_slider = gr.Slider(
                    label="Top K", minimum=1, maximum=50, step=1, value=5)
                use_cluster_search = gr.Checkbox(
                    label="Use FAISS cluster search", value=False)
        with gr.Column(visible=True) as text_input:
            text = gr.Textbox(label="Text", placeholder="Enter text to search")

        with gr.Column(visible=False) as image_input:
            image = gr.Image(label="Image")

        def toggle_inputs(search_type):
            return gr.update(visible=search_type == "Text"), gr.update(visible=search_type == "Image")

        search_type.change(toggle_inputs, inputs=[search_type], outputs=[
                           text_input, image_input])

        search_btn = gr.Button("Search", variant="primary")

        gallery = gr.Gallery(label="Results", show_label=True,
                             columns=5, rows=2, height="auto", preview=False)
        image_info_score = gr.Textbox(
            label="Similarity Score", value="Select an image to view details")
        image_info_caption = gr.Textbox(label="Caption", value="")
    search_btn.click(
        fn=combined_search,
        inputs=[search_type, text, image, top_k_slider, use_cluster_search],
        outputs=gallery
    )
    scan_dir_btn.click(
        scan_dir,
        inputs=[path],
        outputs=path
    )


webui.queue()
webui.launch()
