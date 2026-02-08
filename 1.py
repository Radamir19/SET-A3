import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('result.csv')

# Настройка графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# График 1: Сравнение оценки и истинного значения
ax1.plot(df['t'], df['Exact'], 'k-', label='Exact Cardinality (F0)', linewidth=2)
ax1.plot(df['t'], df['HLL_Mean'], 'r--', label='HyperLogLog Estimate (Mean)', linewidth=2)
ax1.set_title('График №1: Сравнение HLL и точного подсчета')
ax1.set_xlabel('Количество обработанных элементов')
ax1.set_ylabel('Количество уникальных элементов')
ax1.legend()
ax1.grid(True)

# График 2: Статистика и область неопределенности
ax2.plot(df['t'], df['HLL_Mean'], 'b-', label='E(Nt) - Mean Estimate')
# Область стандартного отклонения (Mean +/- Sigma)
ax2.fill_between(df['t'],
                 df['HLL_Mean'] - df['HLL_StdDev'],
                 df['HLL_Mean'] + df['HLL_StdDev'],
                 color='blue', alpha=0.2, label='Sigma (Std Dev)')

# Область теоретической ошибки (1.04 / sqrt(m))
ax2.plot(df['t'], df['HLL_Mean'] + df['Theoretical_Err'], 'g:', label='Theoretical Bound (+)')
ax2.plot(df['t'], df['HLL_Mean'] - df['Theoretical_Err'], 'g:', label='Theoretical Bound (-)')

ax2.set_title('График №2: Статистика оценки и неопределенность')
ax2.set_xlabel('Количество обработанных элементов')
ax2.set_ylabel('Оценка кардинальности')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()