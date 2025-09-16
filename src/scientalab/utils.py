import polars as pl
import torch
from sklearn.metrics import cluster
from torch import distributions


def reparametrize(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Sample from a normal distribution using the reparametrization trick

    Args:
        mean (torch.Tensor[float])
        variance (torch.Tensor[float])

    Returns:
        torch.Tensor[float]
    """
    std = torch.exp(0.5 * variance) + 1e-10
    epsilon = torch.randn_like(std)
    return mean + epsilon * std


def gaussian_likelihood(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Return the log of the probability density function of a gaussian distribution evaluated at a specific value

    Args:
        x (torch.Tensor[float]): Desired value
        mu (torch.Tensor[float])
        std (torch.Tensor[float])

    Returns:
        torch.Tensor[float]
    """
    gaussian_dist = distributions.Normal(loc=mu, scale=std)
    log_probs = gaussian_dist.log_prob(x)
    return log_probs


def negative_binomial_likelihood(x: torch.Tensor, total: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Return the log of the probability of a negative binomial distribution evaluated at a specific value

    Args:
        x (torch.Tensor[float]): Desired value
        total (torch.Tensor[int]): Number of successes
        probs (torch.Tensor[float]): probability of success

    Returns:
        torch.Tensor[float]
    """
    negative_binomial_dist = distributions.NegativeBinomial(total_count=total, probs=probs)
    log_probs = negative_binomial_dist.log_prob(x)
    return log_probs


def one_hot_encoding(x: str | float, vocabulary: list[str | float]) -> list[int]:
    """One hot encoding of a value based on the total vocabulary"""
    encoding = [0] * len(vocabulary)
    encoding[vocabulary.index(x)] = 1
    return encoding


def batch_ASW(data: pl.DataFrame, feature: str, label: str, batch: str) -> float:
    """Compute the batch average silhouette width for the specific features, label and batch data

    Args:
        data (pl.DataFrame)
        feature (str): column name of the features to consider
        label (str): column name of the label to consider
        batch (str): column name of the batch to consider

    Returns:
        float: silhouette score
    """
    gb = data.group_by(label)
    silhouette_score = 0
    n_labels = 0
    for _, group in gb:
        group_size = len(group)
        n_batches = group[batch].n_unique()

        if n_batches >= 2:
            silhouettes = cluster.silhouette_samples(group[feature].to_list(), group[batch].to_list())
            silhouettes = [1 - abs(silhouette) for silhouette in silhouettes]
            silhouette_score += sum(silhouettes) / group_size
            n_labels += 1

    return silhouette_score / n_labels
