import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import os
import sys
import time

# Важно для exe-файла
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный режим

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

RUS_ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
ENG_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

class SimpleTextAnalyzer:
    def clean_text(self, text):
        text = text.lower()
        rus_count = sum(1 for c in text if c in RUS_ALPHABET)
        eng_count = sum(1 for c in text if c in ENG_ALPHABET)
        alphabet = RUS_ALPHABET if rus_count > eng_count else ENG_ALPHABET
        cleaned = ''.join(c for c in text if c in alphabet)
        return cleaned, alphabet
    
    def analyze(self, text):
        text, alphabet = self.clean_text(text)
        n = len(alphabet)
        
        counts = Counter(text)
        total = len(text)
        probs = [counts.get(char, 0) / total for char in alphabet]
        
        bigram_counts = np.zeros((n, n))
        for i in range(len(text)-1):
            idx1 = alphabet.find(text[i])
            idx2 = alphabet.find(text[i+1])
            if idx1 != -1 and idx2 != -1:
                bigram_counts[idx1, idx2] += 1
        
        bigram_total = np.sum(bigram_counts)
        bigram_probs = bigram_counts / bigram_total if bigram_total > 0 else np.zeros((n, n))
        
        entropy = sum(-p * math.log2(p) for p in probs if p > 0)
        
        return {
            'alphabet': alphabet, 'probs': probs, 'bigram_probs': bigram_probs,
            'entropy': entropy, 'length': len(text)
        }

def save_plots(result, title, folder="results"):
    """Сохраняет графики в файлы вместо показа на экране"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 1. Гистограмма частот
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(result['alphabet'])))
    bars = plt.bar(range(len(result['alphabet'])), result['probs'], color=colors)
    
    for i, (bar, char) in enumerate(zip(bars, result['alphabet'])):
        height = bar.get_height()
        if height > 0.001:  # Подписываем только значимые столбцы
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{char}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.title(f'Частоты букв: {title}\nЭнтропия: {result["entropy"]:.2f} бит/символ')
    plt.xlabel('Буквы алфавита')
    plt.ylabel('Вероятность')
    plt.tight_layout()
    plt.savefig(f'{folder}/{title}_гистограмма.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Тепловая карта биграмм (только если алфавит не слишком большой)
    if len(result['alphabet']) <= 32:
        plt.figure(figsize=(10, 8))
        # Упрощаем подписи для больших алфавитов
        xticklabels = list(result['alphabet']) if len(result['alphabet']) <= 20 else False
        yticklabels = list(result['alphabet']) if len(result['alphabet']) <= 20 else False
        
        sns.heatmap(result['bigram_probs'], 
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    cmap='YlOrRd', annot=False)
        plt.title(f'Тепловая карта биграмм: {title}')
        plt.tight_layout()
        plt.savefig(f'{folder}/{title}_биграммы.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Круговая диаграмма топ-8 букв
    plt.figure(figsize=(8, 6))
    indices = np.argsort(result['probs'])[-8:][::-1]
    top_letters = [result['alphabet'][i] for i in indices]
    top_probs = [result['probs'][i] for i in indices]
    
    plt.pie(top_probs, labels=top_letters, autopct='%1.1f%%', startangle=90)
    plt.title(f'Топ-8 самых частых букв: {title}')
    plt.savefig(f'{folder}/{title}_топ8.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_report(results, titles, folder="results"):
    """Создает итоговый отчет"""
    plt.figure(figsize=(12, 8))
    
    # Сравнение энтропий
    entropies = [result['entropy'] for result in results]
    max_entropies = [math.log2(len(result['alphabet'])) for result in results]
    
    x = range(len(titles))
    plt.bar(x, entropies, alpha=0.7, label='Фактическая энтропия', color='skyblue')
    plt.bar(x, max_entropies, alpha=0.3, label='Максимальная энтропия', color='lightcoral')
    plt.xticks(x, titles, rotation=45, ha='right')
    plt.title('Сравнение энтропий текстов')
    plt.ylabel('Биты на символ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{folder}/сравнение_энтропий.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Создаем HTML отчет
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Анализ текстов</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
            .text-info {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .image {{ margin: 10px; text-align: center; background: white; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 400px; border: 1px solid #ddd; }}
            h2 {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Анализатор текстов</h1>
            <p>Частотный анализ и расчет энтропии</p>
        </div>
    """
    
    for i, (title, result) in enumerate(zip(titles, results)):
        efficiency = result['entropy'] / math.log2(len(result['alphabet']))
        html_content += f"""
        <div class='text-info'>
            <h2>📖 {title}</h2>
            <p><strong>Язык:</strong> {'русский' if len(result['alphabet']) == 32 else 'английский'}</p>
            <p><strong>Длина текста:</strong> {result['length']} символов</p>
            <p><strong>Энтропия H(A):</strong> {result['entropy']:.3f} бит/символ</p>
            <p><strong>Максимальная энтропия:</strong> {math.log2(len(result['alphabet'])):.3f} бит/символ</p>
            <p><strong>Эффективность:</strong> <span style="color: {'green' if efficiency > 0.8 else 'orange' if efficiency > 0.6 else 'red'}">{efficiency:.1%}</span></p>
        </div>
        <div class='images'>
            <div class='image'><img src='{title}_гистограмма.png'><br><strong>Гистограмма частот</strong></div>
            <div class='image'><img src='{title}_биграммы.png'><br><strong>Тепловая карта биграмм</strong></div>
            <div class='image'><img src='{title}_топ8.png'><br><strong>Топ-8 букв</strong></div>
        </div>
        <hr>
        """
    
    html_content += """
        <div class='text-info'>
            <h2>📈 Сравнение всех текстов</h2>
            <img src='сравнение_энтропий.png' style='max-width: 100%;'>
        </div>
    </body>
    </html>
    """
    
    with open(f'{folder}/отчет.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Создаем bat-файл для открытия отчета
    bat_content = f"""
@echo off
echo Открываю отчет в браузере...
start отчет.html
pause
"""
    with open(f'{folder}/открыть_отчет.bat', 'w', encoding='cp866') as f:
        f.write(bat_content)

def main():
    print("=" * 60)
    print("           🔍 АНАЛИЗАТОР ТЕКСТОВ")
    print("=" * 60)
    print("Запуск анализа...")
    
    # Примеры текстов
    texts = {
        'Русская литература': '''Война и мир - великий роман Толстого о судьбах людей во время войны с Наполеоном. Герои произведения проходят через испытания, любовь и потери.''' * 15,
        'Русская наука': '''Квантовая механика изучает поведение частиц на атомном уровне. Волновая функция описывает состояние системы согласно уравнению Шрёдингера.''' * 15,
        'Английская литература': '''It was the best of times, it was the worst of times. The story of great expectations and dramatic turns of fate in Victorian England.''' * 15,
        'Английская наука': '''Machine learning algorithms improve through experience. Deep learning uses neural networks to extract patterns from large datasets.''' * 15
    }
    
    analyzer = SimpleTextAnalyzer()
    results = []
    titles = []
    
    print("📊 Анализ текстов:")
    for title, text in texts.items():
        print(f"   📖 {title}...")
        result = analyzer.analyze(text)
        results.append(result)
        titles.append(title)
        save_plots(result, title)
    
    # Создаем итоговый отчет
    create_report(results, titles)
    
    print("✅ Анализ завершен!")
    print("📁 Результаты сохранены в папке 'results'")
    print("\nДля просмотра результатов:")
    print("1. Откройте папку 'results'")
    print("2. Запустите файл 'открыть_отчет.bat'")
    print("3. Или откройте 'отчет.html' в браузере")
    print("\nПрограмма завершит работу через 10 секунд...")
    
    # Автоматическое закрытие через 10 секунд
    time.sleep(10)

if __name__ == "__main__":
    main()