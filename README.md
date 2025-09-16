## Installation

This project's dependencies were managed using [uv][uv]. After installing uv and cloning the repository, the
dependencies can be installed by running `uv sync` in the repository.

## Description

This project reimplement a simple version of the [inVAE architecture][invae]. Instead of modelling a negative binomial
distribution with the decoder, I chose to model a gaussian distribution as it seemed to make more sense considering the
features are real values. I used the same hyperparameters as the paper recommends. I also tested different scheduler for
the learning rate (constant, reduce on plateau and cosine annealing) but opted for the reduce on plateau as it seemed to
yield the best results (according to the validation loss). As the reconstruction term of the validation loss seems to
improve over the first 50 epochs, I decided to linearly increase the beta term over the first 50 epochs of the training.

The training script is available and can be run using uv:

```
uv run train.py
```

The different losses (elbo, reconstruction and kullback leibler) are logged using mlflow. The mlflow tracking server can be started using the following command:

```
mlflow server --host 127.0.0.1 --port 8080
```

The weights of the best model are also available.

[uv]: https://docs.astral.sh/uv/
[invae]: https://www.biorxiv.org/content/10.1101/2024.12.06.627196v1.full.pdf
