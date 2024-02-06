from speechbrain.utils.metric_stats import ErrorRateStats
import json


wer_metric = ErrorRateStats()
results = json.load(open('./results.json','r'))

num = 0
id = 0
for each in results:
    wer_metric.append([id], [each['text'].split(' ')], [each['target'].split(' ')])
    id += 1

wer_metric.write_stats(open('./wer.log','a+'))
WER = wer_metric.summarize("error_rate")
print(f"PandaGPT with Whisper-tiny-en as encoder on FSC-valid, WER: {WER}")