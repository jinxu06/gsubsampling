# Group Equivariant Subsampling
This is a reference implementation for [Group Equivariant Subsampling](https://arxiv.org/abs/2106.05886) by Jin Xu, Hyunjik Kim, Tom Rainforth and Yee Whye Teh.

<p float="left">
  <img src="https://github.com/jinxu06/gsubsampling/blob/main/subsampling.gif" width="400">
  <img src="https://github.com/jinxu06/gsubsampling/blob/main/gae.gif" width="400">
</p>

## Dependencies

See `environment.yml`. For [Anaconda](https://docs.anaconda.com/anaconda/install/) users, please create conda environment with `conda env create -f environment.yml`

## Data

For [dSprites](https://github.com/deepmind/dsprites-dataset) and [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), data will be automatically downloaded and preprocessed before your first run.

For [multi-object datatsets](https://github.com/deepmind/multi_object_datasets) such as Multi-dSprites, please first run
```
python multi_object_datasets/load.py --dataset multi_dsprites --datadir "/tmp"
```

## Running the code

### Training autoencoders

Train `ConvAE`, `GConvAE-p4`, `GConvAE-p4m`, `GAE-p1`, `GAE-p4`, `GAE-p4m` on `dSprites`:
```
python main.py hydra.job.name=sample_complexity model=conv_ae run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
```
python main.py hydra.job.name=sample_complexity model=gconv_ae model.n_channels=21 model.fiber_group='rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
```
python main.py hydra.job.name=sample_complexity model=gconv_ae model.n_channels=15 model.fiber_group='flip_rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
```
python main.py hydra.job.name=sample_complexity model=eqv_ae run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
```
python main.py hydra.job.name=sample_complexity model=eqv_ae model.n_channels=26 model.fiber_group='rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
```
python main.py hydra.job.name=sample_complexity model=eqv_ae model.n_channels=18 model.fiber_group='flip_rot_2d' model.n_rot=4 model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
The numbers of channels are rescaled so that the above models have similar number of parameters. To train on `FashionMNIST`, one can simply set `data=fashion_mnist`. To show the progress bar during training, set `run.use_prog_bar=True`.


### Evaluate autoencoders

To visualise image reconstructions, set `run.mode=reconstruct`. For example, for `GAE-p1` on `dSprites`,
```
python main.py hydra.job.name=sample_complexity model=eqv_ae run.mode=reconstruct data=dsprites data.train_set_size=1600 run.random_seed=1
```
To evaluate the trained model, set set `run.mode=eval` and run:
```
python main.py hydra.job.name=sample_complexity model=eqv_ae run.mode=eval eval.which_set test data=dsprites data.train_set_size=1600 run.random_seed=1
```

### Out-of-distribution experiments

To train autoencoders on constrained data for out-of-distribution experiments, one can run (using `ConvAE` as an example):
```
python main.py hydra.job.name=ood model=conv_ae run.mode=train data=dsprites data.train_set_size=6400 data.constrained_transform="translation_rotation"
```

To regenerate the visualisation in the paper, use our python script at `py_scripts/out_of_distribution.py` (coming soon).

### Multi-object experiments

To train MONet baseline, run:
```
python main.py hydra.job.name=compare_to_monet model=monet run.mode=train data=multi_dsprites data.train_set_size=6400 data.batch_size=16 run.max_epochs=1000 run.random_seed=1
```
To train MONet-GAE-p1, run:
```
python main.py hydra.job.name=compare_to_monet model=eqv_monet run.mode=train data=multi_dsprites data.train_set_size=6400 data.batch_size=16 run.max_epochs=1000 run.random_seed=1
```

## About this repository

We use [Hydra](https://hydra.cc/) to specify configurations for experiments. " The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line." Our default hydra configurations can be found at `conf/`.

The directory `elm/` contains most of our research code. All the data loaders can be found at `elm/data_loader/`, and all the models can be found at `elm/model/`. Experimental results will be generated at `outputs/`, organised by dates and job names. By default, Logs and checkpoints are directed to `/tmp/log/`, but this can be reconfigured in `conf/config.yaml`.

## Contact 

To ask questions about code or report issues, please directly open an issue on github. 
To discuss research, please email jin.xu@stats.ox.ac.uk

## Acknowledgements

This repository includes code from two previous projects: [GENESIS](https://github.com/applied-ai-lab/genesis) and [Multi_Object_datasets](https://github.com/deepmind/multi_object_datasets). Their original licenses have been included.
