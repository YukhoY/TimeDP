#%%
import os
import numpy as np
import torch
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from ldm.data.tsg_dataset import TSGDataset
from utils.init_utils import load_model_data
from utils.cli_utils import get_parser

data_root = os.environ['DATA_ROOT']

mix_dataset = [
        'electricity', 'solar', 'wind_4_seconds', 'traffic', 'taxi', 'pedestrian', 
        'kddcup', 'temp', 'rain', 'nn5', 'fred_md', 'exchange'
    ]

dataset_name_map = {
    'electricity': 'Electricity',
    'solar': 'Solar',
    'wind_4_seconds': 'Wind',
    'traffic': 'Traffic',
    'taxi': 'Taxi',
    'pedestrian': 'Pedestrian',
    'kddcup': 'Air Quality',
    'temp': 'Temperature',
    'rain': 'Rain',
    'nn5': 'NN5',
    'fred_md': 'Fred-MD',
    'exchange': 'Exchange'
}
dataset_color_map = {
    'electricity': 'tab:blue',
    'solar': 'tab:blue',
    'wind_4_seconds': 'tab:blue',
    'traffic': 'tab:green',
    'taxi': 'tab:green',
    'pedestrian': 'tab:green',
    'kddcup': 'tab:orange',
    'temp': 'tab:orange',
    'rain': 'tab:orange',
    'nn5': 'tab:purple',
    'fred_md': 'tab:purple',
    'exchange': 'tab:purple'
}
dataset_domain_map = {
    'electricity': 'Energy',
    'solar': 'Energy',
    'wind_4_seconds': 'Energy',
    'traffic': 'Transport',
    'taxi': 'Transport',
    'pedestrian': 'Transport',
    'kddcup': 'Nature',
    'temp': 'Nature',
    'rain': 'Nature',
    'nn5': 'Econ',
    'fred_md': 'Econ',
    'exchange': 'Econ'
}
    
def draw_domain_tsne_on_ax(ax1: plt.Axes, ax2: plt.Axes, all_repeat, num_dp=100):
    all_data = [x.cpu().numpy() for x in all_repeat]
    concat_data = np.concatenate(all_data, axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, perplexity=40, n_iter=500)
    transformed_data = tsne.fit_transform(concat_data)
    for i, data_name in enumerate(mix_dataset):
        tsne_results = transformed_data[i*num_dp:(i+1)*num_dp]
        ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.4, label=dataset_name_map[data_name])
        ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.4, color=dataset_color_map[data_name], label=dataset_domain_map[data_name])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    
    
#%%
if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    model, data, opt, logdir = load_model_data(parser)

    seq_len = opt.seq_len
    seed = opt.seed

    model = model.to('cuda')
    model.eval()
    nu = opt.num_latents

    num_dp = 100

    all_mask = []
    with torch.no_grad():
        for i, dataset in enumerate(mix_dataset):
            dataset_data = TSGDataset({dataset: data.norm_train_dict[dataset]})

            dataset_samples = []
            for idx in np.random.randint(dataset_data.__len__(), size=num_dp):
                dataset_samples.append(dataset_data.__getitem__(idx)['context'])
            dataset_samples = np.vstack(dataset_samples)
            
            x = torch.tensor(dataset_samples).to('cuda').float().unsqueeze(1)[:num_dp]
            z = model.get_first_stage_encoding(model.encode_first_stage(x))
            c, mask = model.get_learned_conditioning(x, return_mask=True)
            all_mask.append(mask)
    
    # tsne visulize
    sns.set_palette("Paired")
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

    draw_domain_tsne_on_ax(ax1, ax2, all_mask, num_dp=num_dp)
    fig1.tight_layout()
    fig1.savefig(logdir / "dataset_tsne_plot.pdf")
    fig2.tight_layout()
    fig2.savefig(logdir / "domain_tsne_plot.pdf")
    
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), dpi=200)
    for i, dataset_name in enumerate(mix_dataset):
        sns.heatmap(all_mask[i].cpu().numpy(), cmap='coolwarm',center=0,ax=axs[i//6, i%6])
        axs[i//6, i%6].set_title(dataset_name_map[dataset_name])
        axs[i//6, i%6].set_yticks([])
    plt.tight_layout()
    plt.savefig(logdir / "pam_heatmap.pdf")

    fig, axs = plt.subplots(4, 4, figsize=(8, 8), dpi=200)
    latents = model.cond_stage_model.latents.detach().repeat(10, 1, 1)
    for i_proto in range(16):
        mask = torch.zeros(latents.shape[0], latents.shape[1]).to('cuda') - 1  # 16 dims
        mask[:, i_proto] = 1
        with torch.no_grad():
            samples, z_denoise_row = model.sample_log(cond=latents, batch_size=latents.shape[0], ddim=False, cfg_scale=1, mask=mask)
        draw_samples = samples.detach().cpu().squeeze(1).numpy().mean(0)
        axs[i_proto // 4, i_proto % 4].plot(draw_samples)
        axs[i_proto // 4, i_proto % 4].set_title(f'Prototype No.{i_proto}')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(logdir / f"prototype_semantic.pdf")
    plt.show()

# %%
