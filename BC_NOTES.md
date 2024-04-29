# Notes on Using LLM-Foundry for BeanCounter Training
## Custom Tokenizer
First, a custom tokenizer is trained on the BeanCounter dataset. I trained both a ~50K and 100K vocab tokenizers. In both cases, I included special tokens for 0-9 and whitespace up to 24 characters long. This can be done with HF. See notebook called `BeanCounter - Tokenizer.ipynb`.

## Setup
Mosaic recommends running LLM-Foundry from a Docker container. In order to expose the GPU(s) to a container we have to use the NVIDIA Container Toolkit, available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt). **NB** After installation be sure to restart the docker service, i.e., `sudo systemctl restart docker`

### NVIDIA Driver and CUDA Versions
The driver of the host machine must be able to support the CUDA version of the container (right now, the base Mosaic image uses 12.1). This sort of mismatched initially caused an issue. I had to update the driver in order to support CUDA 12.1 before it would work.

### Launch a Container
Once Docker and the Container Toolkit are installed, we can launch a container using Mosaic's images:
```bash
sudo docker run --runtime=nvidia \
    --gpus all \
    -v /mnt/spinning/:/mnt/spinning \
    -v /home/bl/llm-foundry/:/home/mosaicml/llm-foundry/ \
    --user mosaicml \
    -t \
    -d mosaicml/pytorch
```

Once the base image is running, we need to attach to the container and finish setup:
```bash
sudo docker exec -it {CONTAINER_ID} bash
```

Mosaic's setup is simple and available [here](https://github.com/mosaicml/llm-foundry?tab=readme-ov-file#with-docker-recommended).

## Conversion of HF Dataset to MDS Format
The MDS format provides a variety of performance enhancements including fast resumption of training. The dataset can be converted to MDS via the following command:
```bash
python data_prep/convert_dataset_hf.py \
    --dataset /mnt/spinning/beancounter_202402/beancounter/ \
    --data_subset full \
    --out_root /mnt/spinning/beancounter_202402/mml_beancounter \
    --splits train validation \
    --concat_tokens 2048 \
    --tokenizer /mnt/spinning/beancounter_202402/bc-tokenizer-50_432/ \
    --eos_text '<|endoftext|>'
```
**NB** A few things to keep in mind:
1. LLM-Foundry expects various dataset-specific constants to be defined in the `data_prep/convert_dataset_hf.py` script. If these are not defined then the script will fail.
2. The split name **must** equal the folder split name defined in the constants mentioned above. If not, then the split will be skipped.
3. Different subsets of the data can be defined, e.g., the smaller sample of BeanCounter for testing purposes.

## Model Training
```bash
# Train an MPT-125m model for 10 batches
composer train/train.py \
  train/yamls/pretrain/mpt-125m.yaml \
  data_local=/mnt/spinning/beancounter_202402/mml_beancounter_sample \
  train_loader.dataset.split=train \
  eval_loader.dataset.split=validation \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=/mnt/spinning/mpt-125m
```