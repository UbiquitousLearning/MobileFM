import os
import logging
import torch
import data
import torchvision

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from datasets.kinetics import myKinetics400
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

text_list = ['abseiling',
            'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling', 'arranging flowers',
            'assembling computer', 'auctioning', 'baby waking up', 'baking cookies', 'balloon blowing', 'bandaging', 'barbequing', 'bartending',
            'beatboxing', 'bee keeping', 'belly dancing', 'bench pressing', 'bending back', 'bending metal', 'biking through snow', 'blasting sand',
            'blowing glass', 'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding', 'bookbinding', 'bouncing on trampoline',
            'bowling', 'braiding hair', 'breading or breadcrumbing', 'breakdancing', 'brush painting', 'brushing hair', 'brushing teeth',
            'building cabinet', 'building shed', 'bungee jumping', 'busking', 'canoeing or kayaking', 'capoeira', 'carrying baby', 'cartwheeling',
            'carving pumpkin', 'catching fish', 'catching or throwing baseball', 'catching or throwing frisbee', 'catching or throwing softball',
            'celebrating', 'changing oil', 'changing wheel', 'checking tires', 'cheerleading', 'chopping wood', 'clapping', 'clay pottery making',
            'clean and jerk', 'cleaning floor', 'cleaning gutters', 'cleaning pool', 'cleaning shoes', 'cleaning toilet', 'cleaning windows',
            'climbing a rope', 'climbing ladder', 'climbing tree', 'contact juggling', 'cooking chicken', 'cooking egg', 'cooking on campfire',
            'cooking sausages', 'counting money', 'country line dancing', 'cracking neck', 'crawling baby', 'crossing river', 'crying', 'curling hair',
            'cutting nails', 'cutting pineapple', 'cutting watermelon', 'dancing ballet', 'dancing charleston', 'dancing gangnam style', 'dancing macarena',
            'deadlifting', 'decorating the christmas tree', 'digging', 'dining', 'disc golfing', 'diving cliff', 'dodgeball', 'doing aerobics',
            'doing laundry', 'doing nails', 'drawing', 'dribbling basketball', 'drinking', 'drinking beer', 'drinking shots', 'driving car',
            'driving tractor', 'drop kicking', 'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger', 'eating cake', 'eating carrots',
            'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon', 'egg hunting', 'exercising arm',
            'exercising with an exercise ball', 'extinguishing fire', 'faceplanting', 'feeding birds', 'feeding fish', 'feeding goats', 'filling eyebrows',
            'finger snapping', 'fixing hair', 'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins', 'folding paper', 'front raises',
            'frying vegetables', 'garbage collecting', 'gargling', 'getting a haircut', 'getting a tattoo', 'giving or receiving award', 'golf chipping',
            'golf driving', 'golf putting', 'grinding meat', 'grooming dog', 'grooming horse', 'gymnastics tumbling', 'hammer throw', 'headbanging',
            'headbutting', 'high jump', 'high kick', 'hitting baseball', 'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging',
            'hula hooping', 'hurdling', 'hurling sport', 'ice climbing', 'ice fishing', 'ice skating', 'ironing', 'javelin throw', 'jetskiing', 'jogging',
            'juggling balls', 'juggling fire', 'juggling soccer ball', 'jumping into pool', 'jumpstyle dancing', 'kicking field goal', 'kicking soccer ball',
            'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing', 'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich',
            'making bed', 'making jewelry', 'making pizza', 'making snowman', 'making sushi', 'making tea', 'marching', 'massaging back', 'massaging feet',
            'massaging legs', 'massaging persons head', 'milking cow', 'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn', 'news anchoring',
            'opening bottle', 'opening present', 'paragliding', 'parasailing', 'parkour', 'passing American football in game',
            'passing American football not in game', 'peeling apples', 'peeling potatoes', 'petting animal not cat', 'petting cat', 'picking fruit',
            'planting trees', 'plastering', 'playing accordion', 'playing badminton', 'playing bagpipes', 'playing basketball', 'playing bass guitar',
            'playing cards', 'playing cello', 'playing chess', 'playing clarinet', 'playing controller', 'playing cricket', 'playing cymbals', 'playing didgeridoo',
            'playing drums', 'playing flute', 'playing guitar', 'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard',
            'playing kickball', 'playing monopoly', 'playing organ', 'playing paintball', 'playing piano', 'playing poker', 'playing recorder',
            'playing saxophone', 'playing squash or racquetball', 'playing tennis', 'playing trombone', 'playing trumpet', 'playing ukulele', 'playing violin',
            'playing volleyball', 'playing xylophone', 'pole vault', 'presenting weather forecast', 'pull ups', 'pumping fist', 'pumping gas', 'punching bag',
            'punching person boxing', 'push up', 'pushing car', 'pushing cart', 'pushing wheelchair', 'reading book', 'reading newspaper', 'recording music',
            'riding a bike', 'riding camel', 'riding elephant', 'riding mechanical bull', 'riding mountain bike', 'riding mule', 'riding or walking with horse',
            'riding scooter', 'riding unicycle', 'ripping paper', 'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating', 'running on treadmill',
            'sailing', 'salsa dancing', 'sanding floor', 'scrambling eggs', 'scuba diving', 'setting table', 'shaking hands', 'shaking head', 'sharpening knives',
            'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes', 'shooting basketball', 'shooting goal soccer', 'shot put', 'shoveling snow',
            'shredding paper', 'shuffling cards', 'side kick', 'sign language interpreting', 'singing', 'situp', 'skateboarding', 'ski jumping', 'skiing crosscountry',
            'skiing not slalom or crosscountry', 'skiing slalom', 'skipping rope', 'skydiving', 'slacklining', 'slapping', 'sled dog racing', 'smoking', 'smoking hookah',
            'snatch weight lifting', 'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spinning poi', 'spray painting',
            'spraying', 'springboard diving', 'squat', 'sticking tongue out', 'stomping grapes', 'stretching arm', 'stretching leg', 'strumming guitar', 'surfing crowd',
            'surfing water', 'sweeping floor', 'swimming backstroke', 'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs',
            'swinging on something', 'sword fighting', 'tai chi', 'taking a shower', 'tango dancing', 'tap dancing', 'tapping guitar', 'tapping pen', 'tasting beer',
            'tasting food', 'testifying', 'texting', 'throwing axe', 'throwing ball', 'throwing discus', 'tickling', 'tobogganing', 'tossing coin', 'tossing salad',
            'training dog', 'trapezing', 'trimming or shaving beard', 'trimming trees', 'triple jump', 'tying bow tie', 'tying knot not on a tie', 'tying tie', 'unboxing',
            'unloading truck', 'using computer', 'using remote controller not gaming', 'using segway', 'vault', 'waiting in line', 'walking the dog', 'washing dishes',
            'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding', 'watering plants', 'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs',
            'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba',
]
text_list = ['a video of {}.'.format(text.lower()) for text in text_list]

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs=None,
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints/lora/cifar100_new", postfix="_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints/lora/cifar100_new", postfix="_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints/lora/cifar100_new", postfix="_last")
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints/lora/cifar100_new", postfix="_last")

model.eval()
model.to(device)

def run_inference():
    datadir = './.datasets/kinetics'
    test_ds = myKinetics400(os.path.join(datadir, 'test'), split='test')
    test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            }
            
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.VISION]@embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1)

            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {test_correct / test_total * 100:.3f}%")
    
    return test_correct / test_total

print("Model Performance:", run_inference())
