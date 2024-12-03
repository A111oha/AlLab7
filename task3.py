import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Завантаження даних із файлу
data = np.loadtxt('data_clustering.txt', delimiter=',')  # Завантажуємо дані, розділені комами
X = data  # Вхідні дані для кластеризації

# Оцінка ширини вікна для алгоритму зсуву середнього
# Параметр quantile впливає на ширину вікна: вищі значення зменшують кількість кластерів
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)  # Оцінка ширини вікна

# Навчання моделі кластеризації
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # Ініціалізація моделі
mean_shift.fit(X)  # Навчання моделі на даних

# Витягнення центрів кластерів
cluster_centers = mean_shift.cluster_centers_  # Координати центрів кластерів

# Витягнення міток кластерів для кожної точки
labels = mean_shift.labels_  # Мітки кластерів
n_clusters = len(np.unique(labels))  # Кількість кластерів

# Вивід інформації про кількість кластерів і їх центри
print(f"Кількість кластерів: {n_clusters}")
print(f"Центри кластерів:\n{cluster_centers}")

# Візуалізація результатів кластеризації
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, label='Data points')  # Точки даних із кластерними мітками
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='x', label='Cluster centers')  # Центри кластерів
plt.title("Mean Shift Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.savefig("mean_shift_clusters.png")
plt.show()
