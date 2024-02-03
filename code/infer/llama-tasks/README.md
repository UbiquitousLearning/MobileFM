## How to Run llama Lora Infer

### Currently included

- text classification on agnews

### Environment install

~~~shell
pip install -r requirements.txt
~~~

put your pretrained LLAMA checkpoint in ./lit-llama/model, and transform it into lit-llama form following the official guide [lightning-AI](https://github.com/Lightning-AI/)

### Run Infer

~~~shell
chmod +x run.sh
./run.sh
~~~

