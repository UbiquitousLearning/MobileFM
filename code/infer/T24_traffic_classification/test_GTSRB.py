import logging
import torch
import data

import copy
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, force=True)

lora = False
linear_probing = False
device = "cuda" if torch.cuda.is_available() else "cpu"
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


classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]

template = 'a zoomed in photo of a "{}" traffic sign.'
text_list = [template.format(classname) for classname in classes]
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs=None,
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir="./.checkpoints", postfix="_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir="./.checkpoints", postfix="_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir="./.checkpoints", postfix="_last")
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints", postfix="_last")

model.eval()
model.to(device)

def run_inference():
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    datadir = "./.datasets"
    test_ds = GTSRB(datadir, split="test", download=False, transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            data_a, target = x.to(device), target.to(device)
            data_a = [data_a]
            text = copy.deepcopy(text_list)
            data_b = data.load_and_transform_text(text, device)
            data_b = [data_b]

            # class_a is always "vision" according to ImageBind
            feats_a = [model({ModalityType.VISION: data_a_i}) for data_a_i in data_a]
            feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
            # class_b could be any modality
            feats_b = [model({ModalityType.TEXT: data_b_i}) for data_b_i in data_b]
            feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

            match_value = feats_a_tensor @ feats_b_tensor.T

            result = torch.softmax(match_value, dim=-1)
            _, predicted = torch.max(result, -1)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(
            f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {test_correct / test_total * 100:.3f}%")

    return test_correct / test_total


Accuracy = run_inference()

print("Model Performance:", Accuracy)

