# Would I Lie To You? Inference Time Alignment of Language Models using Direct Preference Heads

This repository is the official implementation of "Would I Lie To You? Inference Time Alignment of Language Models using Direct Preference Heads"

## Requirements

We recommend using the `nvcr.io/nvidia/pytorch:24.01-py3` docker container and installing all dependencies from `requirements.txt`.

An example dockerfile is included below:

```docker
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt install ninja-build
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
```

If you do not wish to use docker you may use `venv`, `conda` or any other environment to run all scripts.

### Envars
The following environment variables **must** be set:
- `WANDB_API_KEY` - Your WandB API key to track training/evaluation.
- `WANDB_PROJECT_NAME` - The WandB project destination for all runs.
- `HF_API_KEY` - Your Hugging Face API key to access protected datasets.
- `HF_CACHE_DIR` - The directory to store loaded models and datasets.

The following environment variables **must** be set, but can be set to any arbitrary value if you do not wish to pretrain from scratch:
- `PILE_PATH_PATTERN` - A python string format pattern pointing to the pile shards. Example: `.../the_pile/{:02d}.jsonl` or `placeholder`
- `PILE_SHARDS` - The number of shards available for use. Example: `24` or `0`

The following environment variables are **optional** but recommended:
- `TOKENIZERS_PARALLELISM=true` - Forces HF tokenizers to support parallelism.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Helps reduce fragmentation of the PyTorch allocator.

The following envars should be used for debugging `torch.compile` related issues:
- `TORCH_LOGS="+dynamo"` to enable dynamo logging
- `TORCHDYNAMO_VERBOSE=1` to force verbose dynamo logging
- `TORCH_LOGS=recompiles` to enable dynamo logging on recompiles

### Directory Structure
We recommend you use the following directory structure and for the built in functions to work all scripts must be run from the root directory.

```
.../direct-preference-heads/              <-- root directory
.../direct-preference-heads/cfg/          <-- config file directory
.../direct-preference-heads/src/          <-- source code directory
.../direct-preference-heads/checkpoints/  <-- where models are saved and loaded from
```

## Training

> NOTE: we recommend disabling WandB as otherwise it may attempt to link artifacts from runs that don't exist.

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

Using docker (or your preferred environemt) with the required envars set run the following command to obtain all evaluation results:    
`src/evaluation.py --dir=checkpoints/<model> --benchmark=all`

The results for GLUE will be saved in `checkpoints/<model>/benchmarks/glue_log/` and `checkpoints/<model>/benchmarks/glue_dph/`   
The results for GPT4All will be saved as `checkpoints/<model>/benchmarks/gpt4all.tsv`    
The results for RACE will be saved as `checkpoints/<model>/benchmarks/race.tsv`    

Note that you must manually zip and submit the GLUE results to the test server to obtain the benchmark scores.

## Pre-trained Models

All pretrained models can be downloaded from Hugging Face [here](https://huggingface.co/collections/Avelina/direct-preference-heads-preprint-6612d8a6fa3843352943fd43) and must be saved in the `/checkpoints/` directory to load correctly.

Note that when using `LSWTForDPH.generate(...)` the generation will stop when an `<|im_end|>` is predicted, however this token is NOT automatically added to the input context: there must be a manually added `<|im_end|>` token included at the end of the final assistant message for the `LSWTForDPH.compute_rewards(...)` method to calculate the reward correctly. If this token is not included the method will end up computing the reward for the final *user* message rather than the final *assistant* message which is undefined behaviour. 


>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

### GLUE
| System | MNLI<br>m/mm | QQP<br>F1/Acc | QNLI<br>Acc | SST-2<br>Acc | CoLA<br>M Corr | STS-B<br>P/S Corr | MRPC<br>F1/Acc | RTE<br>Acc | Score<br>(-WNLI) | WNLI<br>Acc | Score<br>(+WNLI) |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Ours (Vocab) | 34.1/34.7 | 28.2/42.9 | 50.2 | 58.0 |  0.9 | -0.9/99.2 | 69.4/57.4 | 50.9 | 42.8 | 34.9 | 41.9 |
| Ours (SFT)   | 73.6/75.0 | 59.1/82.8 | 81.4 | 90.8 | 22.7 | 80.6/92.4 | 80.6/75.2 | 71.4 | 72.0 | 38.4 | 68.2 |
| Ours (DPO)   | 78.8/80.2 | 65.6/85.6 | 87.0 | 93.3 | 36.5 | 83.7/94.4 | 83.9/79.1 | 73.9 | 77.0 | 37.7 | 72.7 |
| Ours (DPH)   | 80.0/80.6 | 65.8/85.3 | 87.5 | 94.0 | 43.8 | **85.3/93.0** | 85.5/80.2 | **75.3** | 78.6 | 46.6 | 75.0 |
| GPT-1        | 82.1/81.4 | 70.3/  -  | 87.4 | 91.3 | 45.4 | 82.0/80.0 | 82.3/  -  | 56.0 | -    | -    | 72.8 |
| BERT Base    | 84.6/83.4 | 71.2/  -  | 90.5 | 93.5 | 52.1 | -  /85.8  | 88.9/  -  | 66.4 | -    | -    | 78.3 |
| BERT Large   | **86.7/85.9** | **72.1/89.3** | **92.7** | **94.9** | **60.5** | 87.6/86.5 | **89.3/85.4** | 70.1 | **82.5** | **65.1** | **80.5** |

### GPT4All
| System | HellaSwag | OpenBookQA | WinoGrande | ARC-Challenge | ARC-Easy | BoolQ | PIQA | Average |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Ours (Vocab)   | 36.93 | 28.60 | 51.14 | 26.19 | 25.67 | 61.25 | 65.39 | 42.17 |
| Ours (SFT)     | 42.59 | 45.20 | 55.01 | 35.84 | 47.01 | 76.24 | 69.37 | 53.04 |
| Ours (DPO)     | 44.83 | 52.40 | 57.38 | 39.76 | 53.54 | **79.08** | 72.36 | 57.05 |
| Ours (DPH)     | **59.36** | **57.40** | **59.12** | **41.21** | **56.82** | 78.81 | 68.77 | **60.21** |
| Pythia-1.0B    | 47.16 | 31.40 | 53.43 | 27.05 | 48.99 | 60.83 | 69.21 | 48.30 |
| Pythia-1.4B    | 52.01 | 33.20 | 57.38 | 28.50 | 54.00 | 63.27 | 70.95 | 51.33 |
| TinyLlama (3T) | 59.20 | 36.00 | **59.12** | 30.12 | 55.25 | 57.83 | **73.29** | 52.99 |

### RACE

| System | RACE-middle | RACE-high | Average |
| --- | :-: | :-: | :-: |
| Ours (Vocab)   | 26.0 | 24.6 | 25.0 |
| Ours (SFT)     | 56.1 | 52.9 | 53.8 |
| Ours (DPO)     | 65.9 | 59.8 | 61.6 |
| Ours (DPH)     | **66.9** | **60.6** | **62.5** |
| GPT-1          | 62.9 | 57.4 | 59.0 |
| LLaMA 7B       | 61.1 | 46.9 | 51.0 |
| LLaMA 13B      | 61.6 | 47.2 | 51.4 |


## Contributing

If you would like to contribute we refer you to our primary repository here: https://github.com/Avelina9X/memory-transformer-pt4
