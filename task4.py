import yfinance as yf
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import matplotlib.pyplot as plt

# Завантаження прив'язок символів до повних назв компаній
company_mapping = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}

# Завантаження даних для компаній
symbols = list(company_mapping.keys())
data = {}
variations = []

for symbol in symbols:
    try:
        stock_data = yf.download(symbol, start="2023-01-01", end="2023-12-31", progress=False)
        if not stock_data.empty:
            # Обчислення варіації (закриття - відкриття)
            variation = (stock_data['Close'] - stock_data['Open']).values
            variations.append(variation)
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")

# Перевірка: чи є дані
if not variations:
    raise ValueError("No valid stock data available for analysis.")

# Приведення розмірів рядів до мінімального спільного
min_length = min(len(v) for v in variations)
variations = np.array([v[:min_length] for v in variations])

# Нормалізація даних
# Нормалізація даних
scaler = StandardScaler()

# Перевірка розмірності variations
variations = np.array(variations)  # Переконаємося, що це NumPy масив
if variations.ndim == 3:
    variations = variations.squeeze(axis=-1)  # Видалення зайвого виміру, якщо він є

X = scaler.fit_transform(variations)
# Кластеризація через Affinity Propagation
model = AffinityPropagation(random_state=0)
model.fit(X)
labels = model.labels_

# Результати
for i, label in enumerate(labels):
    print(f"Company: {company_mapping[symbols[i]]}, Cluster: {label}")

# Візуалізація
plt.figure(figsize=(10, 6))
for i, variation in enumerate(variations):
    plt.plot(variation, label=f"{company_mapping[symbols[i]]} (Cluster {labels[i]})")
plt.legend()
plt.title("Stock Variations and Clustering")
plt.xlabel("Days")
plt.ylabel("Normalized Variation (Close - Open)")
plt.show()
