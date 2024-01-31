import logging
import torch
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from torchvision.datasets import Country211
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, force=True)

lora = True
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
    'Andorra',
    'United Arab Emirates',
    'Afghanistan',
    'Antigua and Barbuda',
    'Anguilla',
    'Albania',
    'Armenia',
    'Angola',
    'Antarctica',
    'Argentina',
    'Austria',
    'Australia',
    'Aruba',
    'Aland Islands',
    'Azerbaijan',
    'Bosnia and Herzegovina',
    'Barbados',
    'Bangladesh',
    'Belgium',
    'Burkina Faso',
    'Bulgaria',
    'Bahrain',
    'Benin',
    'Bermuda',
    'Brunei Darussalam',
    'Bolivia',
    'Bonaire, Saint Eustatius and Saba',
    'Brazil',
    'Bahamas',
    'Bhutan',
    'Botswana',
    'Belarus',
    'Belize',
    'Canada',
    'DR Congo',
    'Central African Republic',
    'Switzerland',
    "Cote d'Ivoire",
    'Cook Islands',
    'Chile',
    'Cameroon',
    'China',
    'Colombia',
    'Costa Rica',
    'Cuba',
    'Cabo Verde',
    'Curacao',
    'Cyprus',
    'Czech Republic',
    'Germany',
    'Denmark',
    'Dominica',
    'Dominican Republic',
    'Algeria',
    'Ecuador',
    'Estonia',
    'Egypt',
    'Spain',
    'Ethiopia',
    'Finland',
    'Fiji',
    'Falkland Islands',
    'Faeroe Islands',
    'France',
    'Gabon',
    'United Kingdom',
    'Grenada',
    'Georgia',
    'French Guiana',
    'Guernsey',
    'Ghana',
    'Gibraltar',
    'Greenland',
    'Gambia',
    'Guadeloupe',
    'Greece',
    'South Georgia and South Sandwich Is.',
    'Guatemala',
    'Guam',
    'Guyana',
    'Hong Kong',
    'Honduras',
    'Croatia',
    'Haiti',
    'Hungary',
    'Indonesia',
    'Ireland',
    'Israel',
    'Isle of Man',
    'India',
    'Iraq',
    'Iran',
    'Iceland',
    'Italy',
    'Jersey',
    'Jamaica',
    'Jordan',
    'Japan',
    'Kenya',
    'Kyrgyz Republic',
    'Cambodia',
    'St. Kitts and Nevis',
    'North Korea',
    'South Korea',
    'Kuwait',
    'Cayman Islands',
    'Kazakhstan',
    'Laos',
    'Lebanon',
    'St. Lucia',
    'Liechtenstein',
    'Sri Lanka',
    'Liberia',
    'Lithuania',
    'Luxembourg',
    'Latvia',
    'Libya',
    'Morocco',
    'Monaco',
    'Moldova',
    'Montenegro',
    'Saint-Martin',
    'Madagascar',
    'Macedonia',
    'Mali',
    'Myanmar',
    'Mongolia',
    'Macau',
    'Martinique',
    'Mauritania',
    'Malta',
    'Mauritius',
    'Maldives',
    'Malawi',
    'Mexico',
    'Malaysia',
    'Mozambique',
    'Namibia',
    'New Caledonia',
    'Nigeria',
    'Nicaragua',
    'Netherlands',
    'Norway',
    'Nepal',
    'New Zealand',
    'Oman',
    'Panama',
    'Peru',
    'French Polynesia',
    'Papua New Guinea',
    'Philippines',
    'Pakistan',
    'Poland',
    'Puerto Rico',
    'Palestine',
    'Portugal',
    'Palau',
    'Paraguay',
    'Qatar',
    'Reunion',
    'Romania',
    'Serbia',
    'Russia',
    'Rwanda',
    'Saudi Arabia',
    'Solomon Islands',
    'Seychelles',
    'Sudan',
    'Sweden',
    'Singapore',
    'St. Helena',
    'Slovenia',
    'Svalbard and Jan Mayen Islands',
    'Slovakia',
    'Sierra Leone',
    'San Marino',
    'Senegal',
    'Somalia',
    'South Sudan',
    'El Salvador',
    'Sint Maarten',
    'Syria',
    'Eswatini',
    'Togo',
    'Thailand',
    'Tajikistan',
    'Timor-Leste',
    'Turkmenistan',
    'Tunisia',
    'Tonga',
    'Turkey',
    'Trinidad and Tobago',
    'Taiwan',
    'Tanzania',
    'Ukraine',
    'Uganda',
    'United States',
    'Uruguay',
    'Uzbekistan',
    'Vatican',
    'Venezuela',
    'British Virgin Islands',
    'United States Virgin Islands',
    'Vietnam',
    'Vanuatu',
    'Samoa',
    'Kosovo',
    'Yemen',
    'South Africa',
    'Zambia',
    'Zimbabwe',
]

template = 'a photo showing the country of {}.'
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
    test_ds = Country211(datadir, split="test", download=True, transform=data_transform)

    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=True, drop_last=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            # print(x.shape)
            # print(target)
            x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            }

            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * (
                lora_factor if lora else 1)

            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(
                f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {test_correct / test_total * 100:.3f}%")

    return test_correct / test_total


Accuracy = run_inference()

print("Model Performance:", Accuracy)
