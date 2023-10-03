git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install -e .[flash-attn]
pip3 install -U git+https://github.com/huggingface/peft.git
huggingface-cli login
wandb login
