from src.model.clip import load
from src.utils import convert_models_to_fp32
import torch
from src.model.model import IM2TEXT
from app.params import INDEX_PATH, COLLECTION, CKPT_PATH
from app.utils import process_prompt, normalize
import faiss

from src.dataset import CIRRImageSplit

clip_model, preprocess_train, preprocess_val = load(name="ViT-L/14", jit=False)

convert_models_to_fp32(clip_model)


checkpoint = torch.load(CKPT_PATH)

clip_model = clip_model.cuda()

clip_sd = checkpoint["state_dict"]
clip_sd = {k[len("module.") :]: v for k, v in clip_sd.items()}  # strip the names
clip_model.load_state_dict(clip_sd)
print("Loaded pretrained CLIP model!")

img2text = IM2TEXT(
    embed_dim=clip_model.embed_dim,
    output_dim=clip_model.token_embedding.weight.shape[1],
)

img2text = img2text.cuda()

i2t_sd = checkpoint["state_dict_img2text"]
i2t_sd = {k[len("module.") :]: v for k, v in i2t_sd.items()}  # strip the names
img2text.load_state_dict(i2t_sd)
print("Loaded pretrained img2text model!")

index = faiss.read_index(INDEX_PATH)
print(f"Loaded FAISS index from {INDEX_PATH}!")

collection = CIRRImageSplit(
    transforms=None,
    root=r"C:\Users\Hoang\Code\IR project\New project\data",
    split=COLLECTION,
)

print(f"Loaded {COLLECTION} collection!")


def retrieve(ref_img, caption, top_k=5):
    img = preprocess_val(ref_img).unsqueeze(0)
    text = process_prompt([caption])

    img, text = img.cuda(), text.cuda()

    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        query_img_feat = img2text(image_features)
        composed_feature = clip_model.encode_text_img(text, query_img_feat)
        composed_feature = normalize(composed_feature)
        D, I = index.search(composed_feature.cpu().numpy(), top_k)

    return I[0], D[0]


def imgs_from_indices(indices):
    return [collection[i] for i in indices]
