from ..font_tokenizer.base import FontRowGlAMTokenizer
from pagerlib.dtypes import ImageSegment
from pagerlib.extractors.page_extractor.font_emb_extractor import font_identifier
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

FONT_EMB_DIM = 512

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(18),
    transforms.CenterCrop((18, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model_and_device():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 71)
    model_path = Path(font_identifier.__file__).parent / 'font_identifier_model_lines.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.fc = nn.Identity()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device


def _row_to_vec_gpu(model, pil_image, device):
    image_tensor = _TRANSFORM(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model(image_tensor).squeeze()
    return vector.cpu().numpy()


class RowGLAMTokenizer(FontRowGlAMTokenizer):

    def __init__(self):
        self._model = None
        self._device = None
        super().__init__()

    @property
    def _model_and_device(self):
        if self._model is None:
            self._model, self._device = _build_model_and_device()
        return self._model, self._device

    def get_vec_font(self, row, pdf_img):
        model, device = self._model_and_device
        seg = ImageSegment(dict_p_size=row['segment'])
        row_img = seg.get_segment_from_img(pdf_img)
        row_cv2 = cv2.cvtColor(row_img, cv2.COLOR_RGB2GRAY)
        pil_image = Image.fromarray(row_cv2)
        return _row_to_vec_gpu(model, pil_image, device)

    def get_num_font_features(self) -> int:
        return FONT_EMB_DIM
