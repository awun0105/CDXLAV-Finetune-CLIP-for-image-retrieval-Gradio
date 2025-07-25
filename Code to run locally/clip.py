
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


class CLIPSearcher:
    def __init__(self, model_id="anhquanlam/clip-finetuned-deepfashion", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(device)
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)

    def get_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inputs).cpu().detach().numpy()

    def get_image_features(self, image):
        inputs = self.processor(
            images=image, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs).cpu().detach().numpy()
