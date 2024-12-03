import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()  # Завантажуємо набір даних Iris
X = iris['data']    # Витягуємо атрибути (довжина і ширина чашолистка і пелюстки)
y = iris['target']  # Витягуємо мітки класів (Setosa, Versicolour, Virginica)

# Масштабування даних для покращення збіжності алгоритму
scaler = StandardScaler()  # Ініціалізація стандартного масштабувальника
X_scaled = scaler.fit_transform(X)  # Масштабуємо дані (середнє = 0, стандартне відхилення = 1)

# Ініціалізація моделі K-середніх
kmeans = KMeans(n_clusters=3,  # Вказуємо кількість кластерів (Setosa, Versicolour, Virginica)
                init='k-means++',  # Використовуємо k-means++ для оптимальної ініціалізації центроїдів
                n_init=10,  # Кількість запусків алгоритму з різними початковими умовами
                max_iter=300,  # Максимальна кількість ітерацій для одного запуску
                random_state=42)  # Фіксуємо початковий стан для відтворюваності результатів

# Навчання моделі на даних
kmeans.fit(X_scaled)  # Навчаємо модель на масштабованих даних

# Прогнозування кластерів
y_kmeans = kmeans.predict(X_scaled)  # Отримуємо прогнозовані мітки кластерів

# Візуалізація результатів (за першими двома вимірами)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50, label='Clustered data')  # Дані з кластерами
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x', label='Centroids')  # Центроїди
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.grid()
plt.savefig("iris_clusters.png")
plt.show()
