import logging
import torch

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
from torchvision import transforms
from datasets.veri import VERI
from torch.utils.data import DataLoader
logging.basicConfig(level=logging.INFO, force=True)

lora = False
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
load_head_post_proc_finetuned = True


if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1

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
data_dir = '.datasets/VeRi'

def get_embedding(model, device):
    dataset = VERI(data_dir, transform=data_transform, split='test')

    em_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    flag = 0
    with torch.no_grad():
        for batch_idx, (x, vehicle_id, camera_id) in enumerate(em_loader):
            x = x.to(device)
            inputs = {
                ModalityType.VISION: x,
            }
            embeddings = model(inputs)
            if not flag:
                all_embeddings = embeddings[ModalityType.VISION]
                flag = 1
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings[ModalityType.VISION]), dim=0)
            logging.info(f"batch_idx = {batch_idx}")
    return all_embeddings


model = imagebind_model.imagebind_huge(pretrained=True)

if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs=None,
                                        modality_names=[ModalityType.VISION]))
    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints", postfix="_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints", postfix="_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints", postfix="_last")

model.eval()
model.to(device)

veri_test_embeddings = torch.load('veri_test_embeddings',map_location=torch.device(device)) 
def run_inference():
    
    val_dataset = VERI('.datasets/VeRi',transform=data_transform,split='query')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (x, target, camera_id) in enumerate(val_loader):
            if batch_idx == 3:
                break
            x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.VISION: x,
            }
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.VISION] @ veri_test_embeddings.T * (
                lora_factor if lora else 1)

            result_1 = torch.softmax(match_value_1, dim=-1)
            top_k = torch.topk(result_1, k=15, dim=-1).indices 
            predicted = []
            for i in range(target.size(0)):
                for j in top_k[i]:
                    if camera_id[i] == val_dataset.image_to_cameraid(j):
                        if target[i] != val_dataset.image_to_veid(j):
                            predicted.append(val_dataset.image_to_veid(j))
                            break
                    else:
                        predicted.append(val_dataset.image_to_veid(j))
                        break
            predicted = torch.tensor(predicted).to(device)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(
                f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {test_correct / test_total * 100:.3f}%")
    return test_correct / test_total


Accuracy = run_inference()
print("Model Performance:", Accuracy)
