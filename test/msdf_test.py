import torch

from utils.msdf import get_values_at_indices, get_grid, calculate_weights, find_closest_vertices_indices
from config import KERNEL_SIZE


def test_get_values_at_indices():
    M = 16
    N = 32
    Vi = torch.randn(M, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE)
    indices = torch.randint(0, KERNEL_SIZE, (N, M, 8, 3))

    naive_results = torch.zeros(indices.size(0), indices.size(1), indices.size(2)).to(Vi.device)
    for i in range(indices.size(0)):
        for j in range(indices.size(1)):
            for k in range(indices.size(2)):
                naive_results[i, j, k] = Vi[j, indices[i, j, k, 0], indices[i, j, k, 1], indices[i, j, k, 2]]

    values_at_indices = get_values_at_indices(Vi, indices)

    assert torch.all(naive_results == values_at_indices)


def test_get_grid():
    k_size = 3
    grid = get_grid(k_size)

    assert grid.size() == (k_size, k_size, k_size, 3)
    assert grid.sum() == 0
    assert grid.min() == -1
    assert grid.max() == 1
    assert grid.unique(sorted=True).tolist() == [-1, 0, 1]
    assert grid[0, 0, 0].tolist() == [-1, -1, -1]
    assert grid[1, 1, 1].tolist() == [0, 0, 0]
    assert grid[2, 2, 2].tolist() == [1, 1, 1]


def test_calculate_weights_all_negative():
    N = 16
    M = 32
    X = torch.ones(N, 3) - 10
    centers = torch.rand(M, 3)
    scales = torch.rand(M)

    weights = calculate_weights(X, centers, scales)

    assert weights.size() == (N, M)
    assert torch.all(weights == 0)


def test_calculate_weights_eye():
    N = 2
    M = 2
    X = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    centers = torch.tensor([[1, 1, 1], [-1, -1, -1]])
    scales = torch.tensor([1, 1])
    expected_weights = torch.eye(2).float()

    weights = calculate_weights(X, centers, scales).float()

    assert weights.size() == (N, M)
    assert torch.allclose(weights, expected_weights)


def test_calculate_weights_equal():
    N = 2
    M = 2
    X = torch.tensor([[1, 1, 1], [3, 3, 3]])
    centers = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    scales = torch.tensor([1, 1])
    expected_weights = torch.tensor([[0.5, 0.5], [0, 0]])

    weights = calculate_weights(X, centers, scales)

    assert weights.size() == (N, M)
    assert torch.allclose(weights, expected_weights)


def test_find_closest_vertices_indices_pt():
    x = torch.tensor([[-1, -1, -1], [1, 1, 1]])

    weights, indices = find_closest_vertices_indices(x)

    assert indices.size() == (2, 8, 3)
    assert weights.size() == (2, 8, 1)
    assert torch.allclose(weights.squeeze().sum(dim=-1), torch.ones(2))

    first_indices = indices[0]
    assert first_indices.unique(sorted=True).tolist() == [0, 1]
    assert weights[0, 0] == 1
    assert torch.all(weights[0, 1:] == 0)

    second_indices = indices[1]
    assert second_indices.unique(sorted=True).tolist() == [KERNEL_SIZE - 2 , KERNEL_SIZE - 1]
    assert weights[1, -1] == 1
    assert torch.all(weights[1, :-1] == 0)

