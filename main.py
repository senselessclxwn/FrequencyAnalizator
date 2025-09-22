import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math

# Настройка красивого отображения
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Алфавиты
RUS_ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
ENG_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

class SimpleTextAnalyzer:
    def clean_text(self, text):
        """Очистка текста - оставляем только буквы"""
        text = text.lower()
        # Определяем язык
        rus_count = sum(1 for c in text if c in RUS_ALPHABET)
        eng_count = sum(1 for c in text if c in ENG_ALPHABET)
        
        alphabet = RUS_ALPHABET if rus_count > eng_count else ENG_ALPHABET
        cleaned = ''.join(c for c in text if c in alphabet)
        return cleaned, alphabet
    
    def analyze(self, text):
        """Простой анализ текста"""
        text, alphabet = self.clean_text(text)
        n = len(alphabet)
        
        # Вероятности букв
        counts = Counter(text)
        total = len(text)
        probs = [counts.get(char, 0) / total for char in alphabet]
        
        # Биграммы
        bigram_counts = np.zeros((n, n))
        for i in range(len(text)-1):
            idx1 = alphabet.find(text[i])
            idx2 = alphabet.find(text[i+1])
            if idx1 != -1 and idx2 != -1:
                bigram_counts[idx1, idx2] += 1
        
        # Вероятности биграмм
        bigram_total = np.sum(bigram_counts)
        bigram_probs = bigram_counts / bigram_total if bigram_total > 0 else np.zeros((n, n))
        
        # Энтропия
        entropy = sum(-p * math.log2(p) for p in probs if p > 0)
        
        return {
            'alphabet': alphabet,
            'probs': probs,
            'bigram_probs': bigram_probs,
            'entropy': entropy,
            'length': len(text)
        }

def create_beautiful_plots(result, title):
    """Создает красивые графики для защиты"""
    
    # 1. Гистограмма частот букв
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(result['alphabet'])))
    bars = plt.bar(range(len(result['alphabet'])), result['probs'], color=colors)
    plt.title(f'Частоты букв: {title}\nЭнтропия: {result["entropy"]:.2f} бит/символ', fontsize=14)
    plt.xlabel('Буквы алфавита')
    plt.ylabel('Вероятность')
    
    # Подписываем каждую букву
    for i, (bar, char) in enumerate(zip(bars, result['alphabet'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{char}', ha='center', va='bottom', rotation=45)
    
    # 2. Тепловая карта биграмм
    plt.subplot(2, 2, 2)
    sns.heatmap(result['bigram_probs'], 
                xticklabels=list(result['alphabet']),
                yticklabels=list(result['alphabet']),
                cmap='YlOrRd', annot=False)
    plt.title('Тепловая карта биграмм', fontsize=14)
    plt.xlabel('Вторая буква')
    plt.ylabel('Первая буква')
    
    # 3. Круговая диаграмма самых частых букв
    plt.subplot(2, 2, 3)
    # Берем топ-10 букв
    indices = np.argsort(result['probs'])[-10:][::-1]
    top_letters = [result['alphabet'][i] for i in indices]
    top_probs = [result['probs'][i] for i in indices]
    
    plt.pie(top_probs, labels=top_letters, autopct='%1.1f%%', startangle=90)
    plt.title('Топ-10 самых частых букв', fontsize=14)
    
    # 4. Сравнение с равномерным распределением
    plt.subplot(2, 2, 4)
    uniform_probs = [1/len(result['alphabet'])] * len(result['alphabet'])
    
    x = range(len(result['alphabet']))
    width = 0.35
    plt.bar([i - width/2 for i in x], result['probs'], width, label='Фактические', alpha=0.7)
    plt.bar([i + width/2 for i in x], uniform_probs, width, label='Равномерные', alpha=0.7)
    plt.title('Сравнение с равномерным распределением', fontsize=14)
    plt.xlabel('Буквы')
    plt.ylabel('Вероятность')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def compare_texts(results, titles):
    """Сравнение нескольких текстов"""
    
    # Сравнение энтропий
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    entropies = [result['entropy'] for result in results]
    max_entropy = [math.log2(len(result['alphabet'])) for result in results]
    
    x = range(len(titles))
    plt.bar(x, entropies, alpha=0.7, label='Фактическая энтропия')
    plt.bar(x, max_entropy, alpha=0.3, label='Максимальная энтропия')
    plt.xticks(x, titles, rotation=45)
    plt.title('Сравнение энтропий текстов', fontsize=14)
    plt.ylabel('Биты на символ')
    plt.legend()
    
    # Эффективность (фактическая/максимальная)
    plt.subplot(2, 2, 2)
    efficiencies = [entropies[i] / max_entropy[i] for i in range(len(entropies))]
    plt.bar(titles, efficiencies, color='lightgreen')
    plt.xticks(rotation=45)
    plt.title('Эффективность использования алфавита', fontsize=14)
    plt.ylabel('Доля от максимальной энтропии')
    
    # Распределение частот букв для всех текстов
    plt.subplot(2, 1, 2)
    for i, result in enumerate(results):
        plt.plot(result['probs'], label=titles[i], linewidth=2)
    
    plt.title('Сравнение распределений частот букв', fontsize=14)
    plt.xlabel('Порядковый номер буквы в алфавите')
    plt.ylabel('Вероятность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Примеры текстов для демонстрации
texts = {
    'Русская литература': '''
    Война и мир - великий роман Толстого о судьбах людей во время войны с Наполеоном.
    Герои произведения проходят через испытания, любовь и потери, раскрывая глубину человеческой души.
    ''' * 10,
    
    'Русская наука': '''
    Квантовая механика изучает поведение частиц на атомном уровне. 
    Волновая функция описывает состояние системы согласно уравнению Шрёдингера.
    ''' * 10,
    
    'Английская литература': '''
    It was the best of times, it was the worst of times. The story of great expectations
    and dramatic turns of fate that shape the lives of characters in Victorian England.
    ''' * 10,
    
    'Английская наука': '''
    Machine learning algorithms improve through experience. Deep learning uses neural
    networks to extract patterns from large datasets for artificial intelligence applications.
    ''' * 10
}

def main():
    analyzer = SimpleTextAnalyzer()
    results = []
    titles = []
    
    print("🔍 АНАЛИЗАТОР ТЕКСТОВ - КРАСИВАЯ ВИЗУАЛИЗАЦИЯ")
    print("=" * 50)
    
    # Анализируем каждый текст
    for title, text in texts.items():
        print(f"📊 Анализируем: {title}")
        result = analyzer.analyze(text)
        results.append(result)
        titles.append(title)
        
        print(f"   Алфавит: {'русский' if len(result['alphabet']) == 32 else 'английский'}")
        print(f"   Длина текста: {result['length']} символов")
        print(f"   Энтропия: {result['entropy']:.3f} бит/символ")
        print(f"   Максимальная энтропия: {math.log2(len(result['alphabet'])):.3f} бит/символ")
        print()
        
        # Создаем красивые графики для каждого текста
        create_beautiful_plots(result, title)
    
    # Сравниваем все тексты
    print("📈 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВСЕХ ТЕКСТОВ")
    compare_texts(results, titles)
    
    # Простая таблица результатов
    print("\n" + "="*60)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*60)
    print(f"{'Текст':<20} {'Язык':<10} {'Энтропия':<10} {'Эффективность':<15}")
    print("-" * 60)
    
    for i, (title, result) in enumerate(zip(titles, results)):
        max_entropy = math.log2(len(result['alphabet']))
        efficiency = result['entropy'] / max_entropy
        language = 'русский' if len(result['alphabet']) == 32 else 'английский'
        
        print(f"{title:<20} {language:<10} {result['entropy']:<10.3f} {efficiency:<15.2%}")

if __name__ == "__main__":
    main()