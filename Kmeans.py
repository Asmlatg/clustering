import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import silhouette_score

# Function definitions (same as before)


def generate_gaussian_data(centers, num_points, dim):
    data = []
    for center in centers:
        cluster = np.random.normal(
            loc=center, scale=0.5, size=(num_points, dim))
        data.append(cluster)
    return np.vstack(data)


def generate_exponential_data(scales, num_points, dim):
    data = []
    for scale in scales:
        cluster = np.random.exponential(scale=scale, size=(num_points, dim))
        data.append(cluster)
    return np.vstack(data)


def plot_data(data, dim, title="Generated Data"):
    if dim == 1:
        plt.scatter(data, np.zeros_like(data), s=10)
        plt.title(f"{title} (1D)")
        plt.xlabel("Dimension 1")
        plt.show()
    elif dim == 2:
        plt.scatter(data[:, 0], data[:, 1], s=10)
        plt.title(f"{title} (2D)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10)
        ax.set_title(f"{title} (3D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.show()
    elif dim == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                        c=data[:, 3], cmap='viridis', s=10)
        ax.set_title(f"{title} (4D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        plt.colorbar(sc, label="Dimension 4")
        plt.show()


def kmeans(data, k, max_iters=100):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    sse_history = []
    for _ in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        sse = np.sum(np.min(distances, axis=0)**2)
        sse_history.append(sse)
        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels, sse_history


def init_kmeanspp(data, k):
    n_samples, n_features = data.shape
    centroids = [data[np.random.randint(n_samples)]]
    for _ in range(k - 1):
        distances = np.array([min([np.sum((x - c) ** 2)
                             for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        next_centroid = data[np.random.choice(n_samples, p=probabilities)]
        centroids.append(next_centroid)
    return np.array(centroids)


def kmeanspp(data, k, max_iters=100):
    centroids = init_kmeanspp(data, k)
    sse_history = []
    for _ in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        sse = np.sum(np.min(distances, axis=0)**2)
        sse_history.append(sse)
        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels, sse_history


def compare_kmeans_algorithms(data, k, n_runs=10):
    results = {
        'kmeans': {'times': [], 'iterations': [], 'final_sse': []},
        'kmeans++': {'times': [], 'iterations': [], 'final_sse': []}
    }
    for _ in range(n_runs):
        # K-means classique
        start_time = time()
        _, _, sse_history = kmeans(data, k)
        end_time = time()
        results['kmeans']['times'].append(end_time - start_time)
        results['kmeans']['iterations'].append(len(sse_history))
        results['kmeans']['final_sse'].append(sse_history[-1])

        # K-means++
        start_time = time()
        _, _, sse_history = kmeanspp(data, k)
        end_time = time()
        results['kmeans++']['times'].append(end_time - start_time)
        results['kmeans++']['iterations'].append(len(sse_history))
        results['kmeans++']['final_sse'].append(sse_history[-1])

    return results


def plot_cluster_comparison(data, dim, dist, k=3):
    centroids_kmeans, labels_kmeans, _ = kmeans(data, k)
    centroids_kmeanspp, labels_kmeanspp, _ = kmeanspp(data, k)

    if dim == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # K-means
        for i in range(k):
            mask = labels_kmeans == i
            ax1.scatter(data[mask], np.zeros_like(
                data[mask]), label=f'Cluster {i+1}')
        ax1.scatter(centroids_kmeans, np.zeros_like(centroids_kmeans),
                    c='red', marker='x', s=200, label='Centroids')
        ax1.set_title('K-means')
        ax1.legend()

        # K-means++
        for i in range(k):
            mask = labels_kmeanspp == i
            ax2.scatter(data[mask], np.zeros_like(
                data[mask]), label=f'Cluster {i+1}')
        ax2.scatter(centroids_kmeanspp, np.zeros_like(
            centroids_kmeanspp), c='red', marker='x', s=200, label='Centroids')
        ax2.set_title('K-means++')
        ax2.legend()

    elif dim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # K-means
        for i in range(k):
            mask = labels_kmeans == i
            ax1.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {i+1}')
        ax1.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1],
                    c='red', marker='x', s=200, label='Centroids')
        ax1.set_title('K-means')
        ax1.legend()

        # K-means++
        for i in range(k):
            mask = labels_kmeanspp == i
            ax2.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {i+1}')
        ax2.scatter(centroids_kmeanspp[:, 0], centroids_kmeanspp[:, 1],
                    c='red', marker='x', s=200, label='Centroids')
        ax2.set_title('K-means++')
        ax2.legend()

    elif dim == 3:
        fig = plt.figure(figsize=(15, 5))

        # K-means
        ax1 = fig.add_subplot(121, projection='3d')
        for i in range(k):
            mask = labels_kmeans == i
            ax1.scatter(data[mask, 0], data[mask, 1],
                        data[mask, 2], label=f'Cluster {i+1}')
        ax1.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1],
                    centroids_kmeans[:, 2], c='red', marker='x', s=200, label='Centroids')
        ax1.set_title('K-means')
        ax1.legend()

        # K-means++
        ax2 = fig.add_subplot(122, projection='3d')
        for i in range(k):
            mask = labels_kmeanspp == i
            ax2.scatter(data[mask, 0], data[mask, 1],
                        data[mask, 2], label=f'Cluster {i+1}')
        ax2.scatter(centroids_kmeanspp[:, 0], centroids_kmeanspp[:, 1],
                    centroids_kmeanspp[:, 2], c='red', marker='x', s=200, label='Centroids')
        ax2.set_title('K-means++')
        ax2.legend()

    plt.suptitle(f'Comparaison des clusters - {dist.capitalize()} Data {dim}D')
    plt.tight_layout()
    plt.show()

# Mini-Batch K-Means


def mini_batch_kmeans(data, k, batch_size=100, max_iters=100):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    sse_history = []
    for _ in range(max_iters):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        mini_batch = data[idx]
        distances = np.sqrt(
            ((mini_batch - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array(
            [mini_batch[labels == i].mean(axis=0) for i in range(k)])
        centroids = new_centroids
        sse = np.sum(np.min(distances, axis=0)**2)
        sse_history.append(sse)
    return centroids, labels, sse_history


def elbow_method(data, max_k=10):
    sse_values = []
    for k in range(1, max_k+1):
        centroids, labels, sse_history = kmeans(data, k)
        sse_values.append(sse_history[-1])  # Last SSE value
    plt.plot(range(1, max_k+1), sse_values, marker='o')
    plt.title("Elbow Method: SSE vs k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("SSE")
    plt.show()


def silhouette_method(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k+1):
        centroids, labels, _ = kmeans(data, k)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
    plt.title("Silhouette Method: Score vs k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="K-Means Clustering Comparison and Visualization")
    parser.add_argument('task', choices=['compare', 'visualize', 'visualize_algos', 'minibatch', 'find_best_k'],
                        help="Task to perform: 'compare', 'visualize', 'visualize algos', 'minibatch' or 'find_best_k'")

    parser.add_argument(
        '--dim', type=int, choices=[1, 2, 3, 4], default=2, help="Dimension of the data (1-4)")
    parser.add_argument(
        '--dist_type', choices=['gaussian', 'exponential'], help="Type of data distribution")
    parser.add_argument('--k', type=int, default=3,
                        help="Number of clusters for k-means")
    parser.add_argument('--n_runs', type=int, default=10,
                        help="Number of runs for k-means comparison")
    parser.add_argument('--num_points', type=int, default=100,
                        help="Number of points per cluster")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Batch size for Mini-Batch K-Means")
    parser.add_argument('--max_k', type=int, default=10,
                        help="Maximum value of k for finding the best k")

    args = parser.parse_args()

    if args.dist_type == 'gaussian':
        centers = [np.random.uniform(0, 10, args.dim) for _ in range(3)]
        data = generate_gaussian_data(centers, args.num_points, args.dim)
    elif args.dist_type == 'exponential':
        scales = [np.random.uniform(1, 3, args.dim) for _ in range(3)]
        data = generate_exponential_data(scales, args.num_points, args.dim)
    if args.task == "visualize":
        plot_data(data, args.dim,
                  title=f"Generated {args.dist_type.capitalize()} Data")

    if args.task == 'compare':
        results = compare_kmeans_algorithms(data, k=args.k, n_runs=args.n_runs)

        print("\nAverage Statistics:")
        for algo in ['kmeans', 'kmeans++']:
            print(f"\n{algo}:")
            print(
                f"Average Time: {np.mean(results[algo]['times']):.4f} seconds")
            print(
                f"Average Iterations: {np.mean(results[algo]['iterations']):.2f}")
            print(f"Average SSE: {np.mean(results[algo]['final_sse']):.2f}")

        # Plot SSE history for both K-Means and K-Means++
        plt.figure(figsize=(10, 6))

        # Plot SSE for K-Means
        plt.plot(results['kmeans']['final_sse'],
                 label='K-Means SSE', color='b', marker='o')

        # Plot SSE for K-Means++
        plt.plot(results['kmeans++']['final_sse'],
                 label='K-Means++ SSE', color='r', marker='x')

        plt.title(f"SSE History for K-Means and K-Means++ (k={args.k})")
        plt.xlabel('Run')
        plt.ylabel('SSE')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif args.task == 'visualize_algos':
        plot_cluster_comparison(data, args.dim, args.dist_type, k=args.k)

    elif args.task == 'minibatch':
        centroids_minibatch, labels_minibatch, sse_history_minibatch = mini_batch_kmeans(
            data, args.k, batch_size=args.batch_size)
        print("Mini-Batch K-Means SSE History:", sse_history_minibatch[-10:])
        plt.plot(sse_history_minibatch)
        plt.title("Mini-Batch K-Means SSE History")
        plt.show()

    elif args.task == 'find_best_k':
        elbow_method(data, max_k=args.max_k)
        silhouette_method(data, max_k=args.max_k)


if __name__ == '__main__':
    main()
