import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Inicialização do SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Nome base do ficheiro CSV
base_filename = 'tweets_sentiment_score_mentions'

if os.path.exists(base_filename + '.csv'):
    # Se o ficheiro já existir, adiciona um novo sufixo
    suffix_number = 1
    while os.path.exists(f'{base_filename}_{suffix_number}.csv'):
        suffix_number += 1
    new_filename = f'{base_filename}_{suffix_number}.csv'
else:
    # Se não existir, utiliza o ficheiro base
    new_filename = base_filename + '.csv'

# Lê o ficheiro CSV com os tweets
df = pd.read_csv('tweets.csv', encoding='utf-8')

# Inicializa listas para armazenar dados
sentiment_scores = []
sentiment_descriptions = []
mentions = []  # Lista para armazenar menções
mention_sentiments = []  # Lista para armazenar o sentimento das menções

# Loop pelos tweets
for tweet in df['Tweet']:
    # Obtem a polaridade dos scores utilizando Vader
    scores = sid.polarity_scores(tweet)

    # Verificar insultos e expressões negativas
    insult_words = ['idiota', 'burro', 'estúpido', 'fascista', 'prostituta política', 'merda']
    negative_expressions = ['vergonha', 'desilusão', 'preocupação', 'tristeza', 'revolta', 'desculpa', 'estranhamente',
                            'problemas', 'instabilidade', 'fraca', 'corrupção', 'descontentamento', 'insatisfação',
                            'revoltante', 'grave', 'desespero', 'lamentável', 'contrassenso', 'chocados', 'tragédia',
                            'desprezo', 'ameaça', 'falhança', 'medo', 'assustador', 'contrassenso', 'lamentável',
                            'revolta', 'abismal']

    # Verificar se o tweet contém insultos ou expressões negativas
    contains_insult = any(word in tweet.lower() for word in insult_words)
    contains_negative = any(exp in tweet.lower() for exp in negative_expressions)
    # Calcular a score com base na presença de insultos ou expressões negativas
    if contains_insult:
        sentiment_score = -1.0  # Score negativa para insultos
    elif contains_negative:
        sentiment_score = -0.5  # Score levemente negativa para expressões negativas
    else:
        sentiment_score = scores['compound']  # Score padrão do Vader

    # Atribui a descrição do sentimento de acordo com o score
    if sentiment_score > 0.6:
        sentiment_description = "5 - Muito positivo"
    elif sentiment_score > 0.2:
        sentiment_description = "4 - Positivo"
    elif sentiment_score > -0.1:
        sentiment_description = "3 - Neutro"
    elif sentiment_score > -0.4:
        sentiment_description = "2 - Negativo"
    else:
        sentiment_description = "1 - Muito negativo"

    # Extrai menções com expressão regular
    tweet_mentions = re.findall(r'@(\w+)', tweet)
    # Normaliza menções para minúsculas antes de adicionar
    normalized_mentions = [mention.lower() for mention in tweet_mentions]
    mentions.extend(normalized_mentions)
    mention_sentiments.extend([sentiment_description] * len(normalized_mentions))

    sentiment_scores.append(sentiment_score)
    sentiment_descriptions.append(sentiment_description)

# Adiciona as colunas ao DataFrame
df['Sentiment_Score'] = sentiment_scores
df['Sentiment_Description'] = sentiment_descriptions

# Contagem dos Scores de Sentimento e descrições
sentiment_score_count = df['Sentiment_Score'].value_counts().reset_index()
sentiment_score_count.columns = ['Sentiment_Score', 'Score_Frequency']
sentiment_desc_count = df['Sentiment_Description'].value_counts().reset_index()
sentiment_desc_count.columns = ['Sentiment_Description', 'Description_Frequency']

# Adiciona as colunas de frequência ao DataFrame
df = pd.merge(df, sentiment_score_count, on='Sentiment_Score', how='left')
df = pd.merge(df, sentiment_desc_count, on='Sentiment_Description', how='left')

# Guarda o DataFrame com scores e frequência dos scores e descrições no ficheiro CSV
df[['Sentiment_Score', 'Sentiment_Description', 'Score_Frequency', 'Description_Frequency']].to_csv(new_filename, index=False, encoding='utf-8')

# Conta a frequência de cada hashtag
hashtags_count = pd.Series(re.findall(r'#(\w+)', ' '.join(df['Tweet']))).value_counts().reset_index()
hashtags_count.columns = ['Hashtag', 'Frequency']

# Guarda a contagem de hashtags no ficheiro CSV 
hashtags_count.to_csv('hashtags_frequency.csv', index=False, encoding='utf-8')

# Conta a frequência de menções
mentions_count = pd.Series(mentions).value_counts().reset_index()
mentions_count.columns = ['Mention', 'Mention_Frequency']

# Guarda a contagem de menções no ficheiro CSV 
mentions_count.to_csv('mention_frequency.csv', index=False, encoding='utf-8')
# Cria um DataFrame para as menções e seus sentimentos
mentions_df = pd.DataFrame({
    'Mention': mentions,
    'Mention_Sentiment': mention_sentiments
})

# Guarda o DataFrame das menções e seus sentimentos no ficheiro CSV
mentions_df.to_csv('mention_sentiments.csv', index=False, encoding='utf-8')

# Exibe informações e o caminho dos ficheiros salvos
print(f'Dados de sentimento e frequência guardados em {new_filename}')
print(f'Contagem de hashtags guardada em hashtags_frequency.csv')
print(f'Contagem de menções guardada em mention_frequency.csv')
print(f'Sentimentos das menções guardados em mention_sentiments.csv')
print(df.head())
