# gsubsampling
This is a reference implementation for "Group Equivariant Subsampling" by Jin Xu, Hyunjik Kim, Tom Rainforth and Yee Whye Teh.

Contact: jin.xu@stats.ox.ac.uk

When using this code, please cite the paper:
```
```

## Dependencies

See `environment.yml`. One can create identical conda environment with `conda env create -f environment.yml`

## Data

For [dSprites](https://github.com/deepmind/dsprites-dataset) and [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), data will be automatically downloaded and preprocessed before your first run.

For [multi-object datatsets](https://github.com/deepmind/multi_object_datasets) such as multi-dsprites and CLEVR6, sorry that we haven't got enough time to automate data downloading and preprocessing. But we are using `multi_object_datasets/read.py` to process the data.

## Running the code

Train `ConvAE` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=conv_ae run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
Train `GConvAE-p4` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=gconv_ae model.n_channels=21 model.fiber_group='rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
Train `GConvAE-p4m` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=gconv_ae model.n_channels=15 model.fiber_group='flip_rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
Train `GAE-p1` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=eqv_ae run.mode=train data=dsprites data.train_set_size=200,400,800 run.random_seed=1
```
Train `GAE-p4` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=eqv_ae model.n_channels=26 model.fiber_group='rot_2d' model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
Train `GAE-p4m` on `dSprites`:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=sample_complexity model=eqv_ae model.n_channels=18 model.fiber_group='flip_rot_2d' model.n_rot=4 model.n_rot=4 run.mode=train data=dsprites data.train_set_size=1600 run.random_seed=1
```
The channel numbers are rescaled so that all models have similar number of parameters.

To train on `FashionMNIST`, one can simply set `data=fashion_mnist`. To visualise image reconstructions, set `run.mode=reconstruct`. To evaluate the trained model, set set `run.mode=eval`.

To train models on constrained data for out-of-distribution experiments, 
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=ood model=conv_ae run.mode=train data=dsprites data.train_set_size=6400 data.constrained_transform="translation_rotation"
```

For multi-object experiments, one needs to download and preprocess the data using `multi_object_datasets/read.py`, and set `data.datadir` accordingly.
To train MONet baseline, run:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=compare_to_monet model=monet run.mode=train data=multi_dsprites data.train_set_size=6400 data.batch_size=16 run.max_epochs=1000 run.random_seed=1
```
To train MONet-GAE-p1, run:
```
CUDA_VISIBLE_DEVICES=0 python main.py hydra.job.name=compare_to_monet model=eqv_monet run.mode=train data=multi_dsprites data.train_set_size=6400 data.batch_size=16 run.max_epochs=1000 run.random_seed=1
```

## Acknowledgements

This repository includes code from two previous projects: [GENESIS](https://github.com/applied-ai-lab/genesis) and [Multi_Object_datasets](https://github.com/deepmind/multi_object_datasets). Their original licenses have been included.
