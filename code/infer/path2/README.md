## How to Run llama Lora Infer

### Currently included

- text classification on agnews

### Environment install

~~~shell
pip install -r requirements.txt
~~~

put your pretrained LLAMA checkpoint in ./lit-llama/model, and transform it into lit-llama form following the official guide [lit-llama/howto/download_weights.md at main · Lightning-AI/lit-llama (github.com)](https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md)

### Run Infer

~~~shell
chmod +x run.sh
./run.sh
~~~

