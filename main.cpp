#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <set>
#include <map>
#include <cstdint>

//Инфраструктура

// Реализация Hash3 (32-bit)
// Это стандарт де-факто для вероятностных структур данных благодаря хорошему "лавинному эффекту"
uint32_t MurmurHash3(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    uint32_t c1 = 0xcc9e2d51;
    uint32_t c2 = 0x1b873593;

    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
            k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2; h1 ^= k1;
    };

    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
}

class HashFuncGen {
public:
    uint32_t operator()(const std::string& s) const {
        return MurmurHash3(s.c_str(), s.length(), 42);
    }
};

class RandomStreamGen {
private:
    std::mt19937 rng;
    std::uniform_int_distribution<int> char_dist;
    std::uniform_int_distribution<int> len_dist;
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-";

public:
    RandomStreamGen(unsigned int seed) : rng(seed), char_dist(0, 62), len_dist(5, 30) {}

    std::string next() {
        int len = len_dist(rng);
        std::string res;
        res.reserve(len);
        for (int i = 0; i < len; ++i) {
            res += chars[char_dist(rng)];
        }
        return res;
    }
};

//HyperLogLog

class HyperLogLog {
private:
    int b;          // количество бит для адресации регистра
    int m;          // количество регистров (2^b)
    double alphaMM; // константа alpha * m^2
    std::vector<uint8_t> registers;

    // Вспомогательная функция для подсчета ведущих нулей
    // Возвращает позицию первой единицы (1-based), т.е. количество нулей + 1
    uint8_t get_rank(uint32_t hash_val) {
        // Берем оставшиеся 32-b бит
        // Но проще считать нули во всем числе после сдвига b бит индекса в конец (или просто маскируя)
        // Для стандарта HLL берем w = (x << b) | (1 << (b-1))
        // Простая реализация:
        uint32_t w = hash_val << b; // Сдвигаем индекс влево, оставляя "тело" хеша
        // Если w = 0, значит все оставшиеся биты были нулями (маловероятно, но возможно)
        if (w == 0) return 32 - b + 1;

        // Считаем ведущие нули.
        // Используем __builtin_clz для GCC/Clang или цикл для переносимости
        uint8_t rank = 1;
        while ((w & 0x80000000) == 0) {
            rank++;
            w <<= 1;
        }
        return rank;
    }

public:
    HyperLogLog(int b_bits) : b(b_bits), m(1 << b_bits), registers(1 << b_bits, 0) {
        double alpha;
        if (m == 16) alpha = 0.673;
        else if (m == 32) alpha = 0.697;
        else if (m == 64) alpha = 0.709;
        else alpha = 0.7213 / (1.0 + 1.079 / m);

        alphaMM = alpha * m * m;
    }

    void add(const std::string& s) {
        uint32_t x = HashFuncGen()(s);
        uint32_t j = x >> (32 - b); // Первые b бит - индекс регистра
        uint8_t r = get_rank(x << b); // Оставшиеся биты для ранга

        if (r > registers[j]) {
            registers[j] = r;
        }
    }

    double count() const {
        double sum_inv = 0.0;
        int V = 0; // Количество пустых регистров (для коррекции малых значений)

        for (int val : registers) {
            sum_inv += std::pow(2.0, -val);
            if (val == 0) V++;
        }

        double E = alphaMM / sum_inv;

        // Коррекция малых диапазонов (Linear Counting)
        if (E <= 2.5 * m) {
            if (V > 0) {
                E = m * std::log((double)m / V);
            }
        }
        // Коррекция больших диапазонов (для 32-битного хеша)
        else if (E > (1.0 / 30.0) * 4294967296.0) {
            E = -4294967296.0 * std::log(1.0 - E / 4294967296.0);
        }

        return E;
    }
};

// Тестирование и сбор статистики

struct StatResult {
    double exact_cardinality;
    double avg_estimate;
    double std_dev;
};

int main() {
    int B_BITS = 12; // m = 2^12 = 4096 регистров. Стандартная ошибка ~ 1.04/sqrt(4096) = 1.62%
    int TOTAL_STREAM_SIZE = 100000;
    int STEP_SIZE = 5000; // Шаг измерения (5%)
    int NUM_EXPERIMENTS = 20; // Количество прогонов для усреднения

    // Ключ - шаг (момент времени t), Значение - вектор оценок разных прогонов
    std::map<int, std::vector<double>> estimates_at_t;
    std::map<int, double> exact_at_t; // Точное значение (оно будет примерно одинаковым, но усредним или возьмем из первого прогона)

    std::cout << "Running simulations..." << std::endl;

    for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
        RandomStreamGen streamGen(exp + 1000); // Разный seed для каждого эксперимента
        HyperLogLog hll(B_BITS);
        std::set<std::string> exact_counter; // Точный счетчик (множество)

        for (int t = 0; t <= TOTAL_STREAM_SIZE; ++t) {
            // Генерируем данные
            if (t > 0) {
                std::string s = streamGen.next();
                hll.add(s);
                exact_counter.insert(s);
            }

            // Фиксация результатов на контрольных точках
            if (t > 0 && t % STEP_SIZE == 0) {
                estimates_at_t[t].push_back(hll.count());
                // Для упрощения считаем, что точное количество +- одинаково для всех прогонов
                // (хотя из-за коллизий генератора строк оно может отличаться на единицы,
                //  но для графика возьмем точное значение текущего прогона)
                if (exp == 0) exact_at_t[t] = (double)exact_counter.size();
            }
        }
    }

    // Вывод CSV в консоль
    std::cout << "t,Exact,HLL_Mean,HLL_StdDev,Theoretical_Err" << std::endl;

    double theoretical_err = 1.04 / std::sqrt(1 << B_BITS); // 1.04 / sqrt(m)

    for (auto const& [t, values] : estimates_at_t) {
        double sum = 0.0;
        for (double v : values) sum += v;
        double mean = sum / values.size();

        double sq_sum = 0.0;
        for (double v : values) sq_sum += (v - mean) * (v - mean);
        double std_dev = std::sqrt(sq_sum / values.size());

        std::cout << t << ","
                  << exact_at_t[t] << ","
                  << mean << ","
                  << std_dev << ","
                  << (mean * theoretical_err) // Абсолютное значение теоретической ошибки
                  << std::endl;
    }

    return 0;
}