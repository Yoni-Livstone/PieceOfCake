import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def factor_pairs(x):
    min_pairs = 5

    def get_factor_pairs(n):
        pairs = []
        limit = int(abs(n) ** 0.5) + 1
        for i in range(1, limit):
            if n % i == 0 and i != 1:
                pairs.append((i, n // i))
        return pairs

    pairs = get_factor_pairs(x)

    offset = 1
    while len(pairs) < min_pairs:
        higher_pairs = get_factor_pairs(x + offset)
        for pair in higher_pairs:
            if (1 not in pair) and (pair not in pairs):
                pairs.append(pair)
        offset += 1

    return pairs


def calculate_piece_areas(x_cuts, y_cuts):
    x_coords = np.sort(np.concatenate(([0], x_cuts, [width])))
    y_coords = np.sort(np.concatenate(([0], y_cuts, [height])))

    piece_widths = np.diff(x_coords)
    piece_heights = np.diff(y_coords)

    areas = np.concatenate(np.outer(piece_widths, piece_heights))

    return areas


def loss_function(areas, requests):
    R = requests
    V = areas

    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return total_cost


def calculate_gradient(x_cuts, y_cuts, requests, curr_loss, epsilon=1e-3):
    grad_x_cuts = np.zeros_like(x_cuts, dtype=float)
    grad_y_cuts = np.zeros_like(y_cuts, dtype=float)

    for i in range(len(x_cuts)):
        x_cuts_eps = x_cuts.copy()
        x_cuts_eps[i] += epsilon
        areas_eps = calculate_piece_areas(x_cuts_eps, y_cuts)
        loss_eps = loss_function(areas_eps, requests)
        grad_x_cuts[i] = (loss_eps - curr_loss) / epsilon

    for i in range(len(y_cuts)):
        y_cuts_eps = y_cuts.copy()
        y_cuts_eps[i] += epsilon
        areas_eps = calculate_piece_areas(x_cuts, y_cuts_eps)
        loss_eps = loss_function(areas_eps, requests)
        grad_y_cuts[i] = (loss_eps - curr_loss) / epsilon

    return grad_x_cuts, grad_y_cuts


def gradient_descent(
    factors,
    requests,
    learning_rate=0.1,
    num_iterations=200,
    epsilon=1e-3,
    learning_rate_decay=1,
    num_restarts=10,
):
    best_loss = float("inf")
    best_x_cuts = None
    best_y_cuts = None
    all_losses = []

    for factor in factors:
        # print(f"Factor pair: {factor}")
        num_horizontal, num_vertical = factor

        restart_losses = []

        for i in range(num_restarts):
            x_cuts = np.array(np.random.randint(1, width, num_vertical), dtype=float)
            y_cuts = np.array(np.random.randint(1, height, num_horizontal), dtype=float)

            best_x_cuts = x_cuts.copy()
            best_y_cuts = y_cuts.copy()

            losses = []
            lr = learning_rate
            for i in range(num_iterations):
                lr = max(lr * learning_rate_decay, 1e-2)

                areas = calculate_piece_areas(x_cuts, y_cuts)
                loss = loss_function(areas, requests)
                losses.append(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_x_cuts = x_cuts.copy()
                    best_y_cuts = y_cuts.copy()

                grad_x_cuts, grad_y_cuts = calculate_gradient(
                    x_cuts, y_cuts, requests, loss, epsilon
                )

                x_cuts -= lr * grad_x_cuts
                y_cuts -= lr * grad_y_cuts
                # print(f'Iteration {i + 1}: Loss = {loss}, Best loss = {best_loss}')
            restart_losses.append(losses)
        all_losses.append(restart_losses)
    all_losses = np.array(all_losses)

    return best_x_cuts, best_y_cuts, all_losses


requests = {
    'g8': [
      79.69,
      46.99,
      85.05,
      47.85,
      99.57,
      18.06,
      45.45,
      82.25,
      85.96,
      10.4,
      97.74,
      33.82,
      92.54,
      22.59,
      83.18,
      25.34,
      45.36,
      93.45,
      59.57,
      94.85,
      46.51,
      62.71,
      15.13,
      35.17,
      44.58,
      99.34,
      66.05,
      72.79,
      88.21, 95.16, 63.82, 99.67, 64.55, 28.55, 18.45, 33.45, 28.11]
}

for request_name, request in requests.items():
    request = np.array(request, dtype=float)
    
    height = round(math.sqrt(1.05 * np.sum(request) / 1.6), 2)
    width = round(height * 1.6, 2)

    learning_rate = 1
    learning_rate_decay = 0.99
    num_iterations = 500
    epsilon = 1e-3
    num_restarts = 10

    factors = factor_pairs(len(request))
    best_x_cuts, best_y_cuts, all_losses = gradient_descent(
        factors, request, learning_rate, num_iterations, epsilon, learning_rate_decay, num_restarts
    )

    fig, axs = plt.subplots(len(factors), 1, figsize=(8, len(factors) * 5))
    for i, (num_horizontal, num_vertical) in enumerate(factors):
        ax = axs[i] if len(factors) > 1 else axs

        for j, losses in enumerate(all_losses[i]):
            ax.plot(losses, label=f"Restart {j}")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(f"{request_name}: {num_horizontal}x{num_horizontal}, best loss: {np.min(np.concatenate(all_losses[i]))}")
        ax.legend()

    print("Best x cuts:", best_x_cuts)
    print("Best y cuts:", best_y_cuts)

    fig.tight_layout()
    plt.savefig(f"{request_name}_{np.min(np.concatenate(all_losses))}.png")
    np.save(f"{request_name}_all_losses.npy", all_losses)
    np.save(f"{request_name}_best_x_cuts.npy", best_x_cuts)
    np.save(f"{request_name}_best_y_cuts.npy", best_y_cuts)
