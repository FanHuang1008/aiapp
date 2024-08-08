# pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
from .model import load_model
from .data import read_image

# cli argument: $ aiapp ./test.png
img = read_image("../test.png")

# configs: .env .ini .cfg <-- pydantic
model = load_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)

# runtime interface: inputs -> (inference) -> output
result = model.inference(img, topk=5)
print(result)