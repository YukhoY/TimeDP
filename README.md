
## Envirionment Setup
We recommand using conda as enviroment manager:
```bash
conda env create -f environment.yml
```

## Data Preparation
Pre-processed dataset can be found with this [link](https://huggingface.co/datasets/YukhoW/TimeDP-Data/blob/main/TimeDP-Data.zip).

## Examples
Training TimeDP:
```bash
python main_train.py --base configs/multi_domain_timedp.yaml --gpus 0, --logdir ./logs/ -sl 168 -up -nl 16 --batch_size 128 -lr 0.0001 -s 0
```

Training without PAM:
```bash
python main_train.py --base configs/multi_domain_timedp.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0
```

Training without domain prompts (unconditional generation model):
```bash
python main_train.py --base configs/multi_domain_timedp.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0 --uncond
```

Visualization of domain prompts:
```bash
python visualize.py --base configs/multi_domain_timedp.yaml --gpus 0, --logdir ./logs/ -sl 168 --batch_size 128 -lr 0.0001 -s 0 --uncond
```