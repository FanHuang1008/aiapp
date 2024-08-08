# pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
from .model import load_model
from .data import read_image
from .configs import cfg

# configs: .env .ini .cfg <-- pydantic
model = load_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)

def inference(image_path: str, topk: int = 5):
    # cli argument: $ aiapp ./test.png
    img = read_image(cfg.infer_model_name)

    # runtime interface: inputs -> (inference) -> output
    result = model.inference(img, topk=5)

    return result