import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt('data_clustering.txt', delimiter=',')

# Візуалізація початкових даних
plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Data points')
plt.title("Input Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.savefig("input_data.png")
plt.show()

# Кількість кластерів
k = 5  # Задається кількість кластерів
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)

# Навчання моделі
kmeans.fit(data)

# Отримання результатів кластеризації
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# Візуалізація кластерів
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Clustered data')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.savefig("clustered_data.png")
plt.show()
# Визначення сітки
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Прогнозування на основі моделі
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Відображення сітки
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.5)
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Clustered data')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title("Cluster Boundaries with K-Means")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.savefig("cluster_boundaries.png")
plt.show()
