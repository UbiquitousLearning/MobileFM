import logging
import torch
import data
import torchvision

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from datasets.flickr import flickr30k
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, force=True)

lora = False
linear_probing = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True

assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1

text_prompt = 'a photo of {}.'

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs=None,
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints/lora/100_flickr30k", postfix="_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints/lora/100_flickr30k", postfix="_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints/lora/100_flickr30k", postfix="_last")
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints/lora/100_flickr30k", postfix="_last")

model.eval()
model.to(device)

global_batch_size = 128
datadir = "./.datasets/flickr30k/flickr30k-images"
anne_dir = "./.datasets/flickr30k/results.csv"
test_ds = flickr30k(root_dir=datadir, anne_dir=anne_dir, split='test')
test_dl = DataLoader(dataset=test_ds, batch_size=global_batch_size, shuffle=False, drop_last=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

def get_vision_embeddings():
    img_dict = {}
    vision_embeddings = []
    with torch.no_grad():
        for batch_idx, (x, target, image_name) in enumerate(test_dl):
            x = x.to(device)
            inputs = {
                ModalityType.VISION: x,
            }
            embeddings = model(inputs)
            img_dict.update({image_name[i]: batch_idx * global_batch_size + i for i in range(len(image_name))})
            vision_embeddings.append(embeddings[ModalityType.VISION])
        vision_embeddings = torch.cat(vision_embeddings, dim=0)
    return img_dict, vision_embeddings.T

def run_inference(img_dict, vision_embeddings):
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (_, x, image_name) in enumerate(test_dl):
            text_x = [text_prompt.format(x[i]) for i in range(len(x))]
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_x, device),
            }
            
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.TEXT] @ vision_embeddings * (lora_factor if lora else 1)
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            target = torch.tensor([img_dict[name] for name in image_name]).to(device)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {test_correct / test_total * 100:.3f}%")
    
    return test_correct / test_total

img_dict, vision_embeddings = get_vision_embeddings()
print("Model Performance:", run_inference(img_dict, vision_embeddings))
