import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from plotly.offline import init_notebook_mode

init_notebook_mode()

# Load datasets
avatar = pd.read_csv('avatar.csv', encoding='latin-1')

# Fill missing IMDb ratings with the average rating
avatar['imdb_rating'] = avatar['imdb_rating'].fillna(avatar['imdb_rating'].mean())

# Visualization of IMDb ratings across seasons
fig = px.bar(series, x='book', y='series_rating', template='simple_white', color_discrete_sequence=['#f18930'] * 3,
             opacity=0.6, text='series_rating', category_orders={'book': ['Water', 'Earth', 'Fire']},
             title='IMDb Rating Across Seasons')
fig.add_layout_image(
    dict(
        source="https://i.imgur.com/QWoqOZd.jpg",
        xref="x",
        yref="y",
        x=-0.5,
        y=10,
        sizex=3,
        sizey=10,
        opacity=0.7,
        sizing="stretch",
        layer="below")
)
fig.show()

# Insights on IMDb ratings
print("Average IMDb ratings for each season:")
print(series.groupby('book')['series_rating'].mean())

# Detailed episode ratings
fig = px.bar(data, x='Unnamed: 0', y='imdb_rating', color='book', hover_name='book_chapt', template='simple_white',
             color_discrete_map={'Fire': '#cd0000', 'Water': '#3399ff', 'Earth': '#663307'},
             labels={'imdb_rating': 'IMDb Rating', 'Unnamed: 0': 'Episode'})
fig.show()

# Analysis of director's impact on ratings
director_counts = pd.DataFrame(data['director'].value_counts()).reset_index()
director_counts.columns = ['Director Name', 'Number of Episodes']

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]], horizontal_spacing=0.2)

directorColors = ['#adbce6'] * 7
directorColors[5] = '#ba72d4'
director_rating = pd.DataFrame(data.groupby('director')['imdb_rating'].mean()).reset_index().sort_values(by='imdb_rating')
trace0 = go.Bar(y=director_rating['director'], x=director_rating['imdb_rating'], orientation='h',
                hovertext=director_rating['imdb_rating'], name='Director Average Ratings', marker_color=directorColors)
fig.add_trace(trace0, row=1, col=1)

trace1 = go.Pie(values=director_counts['Number of Episodes'], labels=director_counts['Director Name'],
                name='Director Number of Episodes')
fig.add_trace(trace1, row=1, col=2)

fig.update_layout(showlegend=False, title={'text': 'Directors and Their Average Rating', 'x': 0.5}, template='plotly_white')
fig.show()

# Character dialogues analysis
character_dialogues = pd.DataFrame({'Character': [], 'Number of Dialogues': [], 'Book': []})
for book in ['Water', 'Earth', 'Fire']:
    temp = pd.DataFrame(avatar[avatar['book'] == book]['character'].value_counts()).reset_index()
    temp.columns = ['Character', 'Number of Dialogues']
    temp['Book'] = book
    temp = temp.sort_values(by='Number of Dialogues', ascending=False)
    character_dialogues = pd.concat([character_dialogues, temp])

important_characters = ['Aang', 'Katara', 'Zuko', 'Sokka', 'Toph', 'Iroh', 'Azula']

# Dialogues Analysis by Book
bookColor = {'Fire': '#cd0000', 'Water': '#3399ff', 'Earth': '#663307'}
fig = make_subplots(rows=1, cols=3, subplot_titles=['Water', 'Earth', 'Fire'])
for i, book in enumerate(['Water', 'Earth', 'Fire']):
    temp = character_dialogues[(character_dialogues['Character'] != 'Scene Description') & (character_dialogues['Book'] == book)]
    trace = go.Bar(x=temp.iloc[:10][::-1]['Number of Dialogues'].values, y=temp.iloc[:10][::-1]['Character'].values,
                   orientation='h', marker_color=bookColor[book], name=book, opacity=0.8)
    fig.add_trace(trace, row=1, col=i + 1)
fig.update_layout(showlegend=False, template='plotly_white', title='Characters with Most Dialogues in Each Book')
fig.show()

# Sentiment analysis of character dialogues
sid = SentimentIntensityAnalyzer()
avatar['sentiment'] = avatar['character_words'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(avatar['sentiment'], kde=True, color='purple')
plt.title('Sentiment Distribution of Character Dialogues')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Important Characters' Dialogues Analysis
fig = px.bar(character_dialogues[character_dialogues['Character'].isin(important_characters)], template='gridon',
             title='Important Characters Number of Dialogues each season', x='Number of Dialogues', y='Character',
             orientation='h', color='Book', barmode='group',
             color_discrete_map={'Fire': '#cd0000', 'Water': '#3399ff', 'Earth': '#663307'})
fig.add_layout_image(
    dict(
        source="https://vignette.wikia.nocookie.net/avatar/images/1/12/Azula.png",
        x=0.25,
        y=0.9,
    ))
fig.add_layout_image(
    dict(
        source="https://vignette.wikia.nocookie.net/avatar/images/4/46/Toph_Beifong.png",
        x=0.42,
        y=0.77,
    ))
fig.add_layout_image(
    dict(
        source="https://vignette.wikia.nocookie.net/avatar/images/c/c1/Iroh_smiling.png",
        x=0.35,
        y=0.6,
    ))
fig.add_layout_image(
    dict(
        source="https://vignette.wikia.nocookie.net/avatar/images/4/4b/Zuko.png",
        x=0.62,
        y=0.47,
    ))
fig.add_layout_image(
    dict(
        source="https://vignette.wikia.nocookie.net/avatar/images/cc/Sokka.png",
        x=0.85,
        y=0.32,
    ))
fig.add_layout_image(
    dict(
        source="https://static.wikia.nocookie.net/loveinterest/images/c/cb/Avatar_Last_Airbender_Book_1_Screenshot_0047.jpg",
        x=0.85,
        y=0.18,
    ))
fig.add_layout_image(
    dict(
        source="https://comicvine1.cbsistatic.com/uploads/scale_small/11138/111385676/7212562-5667359844-41703.jpg",
        x=1.05,
        y=0.052,
    ))
fig.update_layout_images(dict(
    xref="paper",
    yref="paper",
    sizex=0.09,
    sizey=0.09,
    xanchor="right",
    yanchor="bottom"
))
fig.show()

# Top episodes with the most dialogues
chapter_dialogues = pd.DataFrame({'Chapter': [], 'Number of Dialogues': [], 'Book': []})
dialogue_df = avatar[avatar['character'] != 'Scene Description']
for book in ['Water', 'Earth', 'Fire']:
    temp = pd.DataFrame(dialogue_df[(dialogue_df['book'] == book)]['chapter'].value_counts()).reset_index()
    temp.columns = ['Chapter', 'Number of Dialogues']
    temp['Book'] = book
    chapter_dialogues = pd.concat([chapter_dialogues, temp])
chapter_dialogues = chapter_dialogues.sort_values(by='Number of Dialogues')
colors = ['#cd0000' if x == 'Fire' else '#3399ff' if x == 'Water' else '#663307' for x in chapter_dialogues['Book']]

trace = go.Bar(x=chapter_dialogues.iloc[:20]['Number of Dialogues'], y=chapter_dialogues.iloc[:20]['Chapter'],
               orientation='h', marker_color=colors)
fig = go.Figure([trace])
fig.update_layout(title={'text': 'Top 20 Episodes with the Most Number of Dialogues', 'x': 0.5},
                 xaxis_title="Number of Dialogues", yaxis_title="Chapter Name", template='plotly_white')
fig.show()

# Episodes with the least dialogues per IMDb rating
ratings = []
for i in range(len(chapter_dialogues)):
    chapter = chapter_dialogues.iloc[i]['Chapter']