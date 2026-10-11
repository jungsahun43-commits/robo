"""CPU belt appearance inference with a pinned, explicitly selected checkpoint."""
import base64
import hashlib
import io
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.transforms.functional import to_tensor, normalize

from .belt_segmentation import ARCH, create_model

MAX_IMAGE_BYTES = 8*1024*1024


def letterbox(image):
    width,height = image.size
    scale = 640/max(width,height)
    w,h = max(1,round(width*scale)),max(1,round(height*scale))
    left,top = (640-w)//2,(640-h)//2
    padded = Image.new('RGB',(640,640),(123,116,103))
    padded.paste(image.resize((w,h),Image.Resampling.BILINEAR),(left,top))
    tensor = normalize(to_tensor(padded),[0.485,0.456,0.406],[0.229,0.224,0.225]).unsqueeze(0)
    return tensor,(left,top,w,h)


def decode_photo(payload):
    if not payload or len(payload)>MAX_IMAGE_BYTES:
        raise ValueError('Image bytes must be nonempty and at most8MB')
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width*source.height>12_000_000:
                raise ValueError('Image resolution exceeds12 megapixels')
            return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Readable JPEG/PNG photo required') from error


class BeltEngine:
    def __init__(self,checkpoint,expected_sha256):
        path = Path(checkpoint)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest!=expected_sha256:
            raise ValueError('Checkpoint SHA256 does not match the selected model')
        state = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if (state.get('architecture')!=ARCH or state.get('class')!='belt_crack'
                or state.get('input_size')!=640 or state.get('threshold')!=0.5):
            raise ValueError('Unsupported belt checkpoint contract')
        self.model = create_model(False).eval()
        self.model.load_state_dict(state['state_dict'],strict=True)
        self.model_sha256 = digest
        self.model_name = path.parent.name
        self.lock = RLock()

    def analyze(self,payload):
        rgb = decode_photo(payload)
        tensor,(left,top,w,h) = letterbox(rgb)
        with self.lock,torch.inference_mode():
            probability = self.model(tensor)['out'][0,0].sigmoid()[top:top+h,left:left+w]
        if not torch.isfinite(probability).all():
            raise RuntimeError('Nonfinite model output')
        heatmap = Image.fromarray(np.clip(probability.numpy()*255,0,255).astype('uint8'))
        native = heatmap.resize(rgb.size,Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        native.save(buffer,format='PNG')
        return {'schema':'safelog.equipment.belt-visual.v1','equipmentCategory':'conveyor_belt',
            'analysisType':'surface_crack_candidate','className':'belt_crack',
            'candidateDetected':bool((probability>0.5).any()),'maximumPixelScore':float(probability.max()),
            'pixelThreshold':0.5,'scoreIsMechanicalFailureProbability':False,
            'normalPhotoFprValidated':False,'mechanicalFailureState':'not_assessed',
            'humanConfirmationRequired':True,'modelName':self.model_name,'modelSha256':self.model_sha256,
            'inputSha256':hashlib.sha256(payload).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
            'heatmapPngBase64':base64.b64encode(buffer.getvalue()).decode('ascii'),
            'heatmapScale':'Raw sigmoid scores0..255, no per-photo normalization',
            'scope':'Belt surface crack/tear candidate only; no roller, motor, alignment or internal fault verdict'}
