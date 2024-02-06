# just transform the orignal ImageCaption dataset's format into the VisualInstruction dataset's format
import json
import csv
import os

if __name__ == '__main__':
    datasets = []
    base_path='/home/bingxing2/gpuuser590/yx/dataset/KeyWordSpotting'
    data = json.load(open('/home/bingxing2/gpuuser590/yx/dataset/KeyWordSpotting/train_orig.json','r'))
    for each in data:
        audio_name = os.path.join(base_path,each['file']) 
        if each['label'] not in ['yes','no','up','down','left','right','on','off','stop','go','_silence_']:
            each['label'] = '_unknown_'
        datasets.append({"image_name": audio_name,
                        "conversation": [
                            {
                                "from": "human",
                                "value": ""
                            },
                            {
                                "from": "gpt",
                                "value": each['label']
                            }
                        ]})


    with open('/home/bingxing2/gpuuser590/yx/dataset/KeyWordSpotting/train_select.json', 'w') as fp:
        json.dump(datasets, fp)