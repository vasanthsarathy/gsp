# gsp
Grounded Semantic Parsing (GSP): How do we get semantic parses of raw NL text that has visual and action grounding through a robotic architecture?


## Data Augmentation

### (1) Stylizer 

`$ gsp stylize -i data/tasks/dev/SpatialTrainingDataset2.json -n 2 -r 3`

This generates three variations for 2 utterances across six (default) styles. The input is the original file from diarc. 


### (2) Augmenter

`gsp augment -i gsp/results/SpatialTrainingDataset2_stylized.jsonl`


### (3) Prepare for finetuning


The command `$ gsp prepare` prepares the data for finetuning



## Finetuning config file:

[Config File](https://gist.github.com/vasanthsarathy/b6aedbcb15459e38a60a20c04a610bc5)
[Dataset on Huggingface hub](https://huggingface.co/datasets/vsarathy/DIARC-embodied-nlu-styled-4k)

### Overview of the steps for using Runpod

1. Prepare dataset. Make sure to include the fields "instruction", "input", "output", "text".
2. Upload the dataset to huggingface or something 
3. Create an axolotl config file 
4. Here are some things you should have ready-
    - Link to your *.yaml config file. 
    - Link to your dataset 
    - huggingface hub token
    - wandb token 
    - link to merge-script to merge adapter with larger model 
5. Create a runpod instance with axolotl on it. 
6. Run training
7. should automatically push the adapter to the hub
8. Merge adapter with model. 

### Overview of the steps for using Vast.ai 

5. Create vast.ai instance with docker to axolotl environment 

>> Need to check: can I just use the axolotl docker container within the instance or do I really need to pull the axolotl github repo and run the finetuning from there. My guess is that -- probably not. 

### What do you need 






