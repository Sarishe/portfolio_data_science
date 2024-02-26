#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('fifa23_players.csv')


# In[3]:


df.head()


# In[4]:


#drop columns with urls
#age and dob give same info, dropping dob
#short_name and long_name give same info, dropping short_name
df.drop(['sofifa_id', 'player_url', 'player_face_url', 'club_logo_url','nation_logo_url','nation_flag_url', 'dob', 'short_name'], axis=1, inplace=True)


# In[5]:


df.info()


# In[6]:


#top100 players with the highest ranking
df[['long_name', 'overall']][0:100]


# In[7]:


#top 100 players with highest salaries compared to their overall ranking
top_salaries = df.sort_values('wage_eur',ascending=False)[0:100]
top_salaries[['long_name','wage_eur']]


# In[8]:


#compare highest ranking players vs highest salary players 
import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(12, 5))

x1 = df['overall'][0:100]
y1 = df['wage_eur'][0:100]
plt.subplot(1, 2, 1)
plt.title("Top100 players with the highest ranking")
plt.xlabel('Ranking')
plt.ylabel('Salary')
plt.scatter(x1, y1, color='red')

x = top_salaries['overall']
y = top_salaries['wage_eur']
plt.subplot(1, 2, 2)
plt.title("Top100 players with the highest salaries")
plt.xlabel('Ranking')
plt.ylabel('Salary')
plt.scatter(x, y, color='blue') 

plt.show()


# In[9]:


#top30 goalkeepers with the highest ranking
top_GK = df[df['club_position'] == 'GK'].sort_values('overall', ascending = False)[0:30]
top_GK[['long_name', 'club_position', 'overall']]


# In[10]:


#top30 teams with players of highest rankings
top_ranking_club = df.sort_values('overall',ascending=False)
top_ranking_club['club_name'].unique()[0:30]


# In[11]:


#top30 teams where players on average have the highest speed
speed = df.groupby(['club_name'])['movement_sprint_speed'].mean().reset_index(name="mean")
speed.sort_values('mean', ascending=False)[0:30]


# In[12]:


#I had some issues interpeting this task. Top of all leagues with the best dribbling players ['skill_dribbling']. 
#If we are looking at the best dribbler in each league
dribbling = df.groupby(['league_name'])['skill_dribbling'].max().reset_index(name="top_dribbling_skill")
dribbling.sort_values('top_dribbling_skill', ascending  = False)


# In[13]:


#Alternatively, we need to know what would be considered the "best" dribbling players, top100? 
#Then we can make a list of leagues that have the top 100 dribbling players
top_dribblers = df.sort_values('skill_dribbling',ascending=False)[0:100]
top_dribblers.groupby(['league_name']).count()


# In[14]:


#top30 teams with the highest ranking players with 1 GK, 4 CB/LB/RB/LWB/RWB, 4 CM/CAM/CDM/LM/RM, 2 ST/CF/LF/RF/LW/RW
df['club_team_id'].value_counts()


# In[15]:


df['club_name'].value_counts()


# In[16]:


#each club has 1 team id reference, so we can determine the 30 best teams rankings by checking mean score of all their players
#each club will have all the roles covered as well
teams = df.groupby(['club_name'])['overall'].mean().reset_index(name="mean_score")
teams.sort_values('mean_score', ascending = False)[0:30]


# In[17]:


import pandas as pd

gk_players = df[df['club_position'].str.upper() == 'GK'].groupby(['club_name', 'player_positions'])['overall'].max().reset_index()
def_players = df[df['club_position'].str.upper().isin(['CB', 'RB', 'LB', 'RWB', 'LWB'])].groupby(['club_name', 'player_positions'])['overall'].nlargest(4).reset_index()
midfielders = df[df['club_position'].str.upper().isin(['CM', 'CDM', 'CAM', 'RM', 'LM'])].groupby(['club_name', 'player_positions'])['overall'].nlargest(4).reset_index()
attackers = df[df['club_position'].str.upper().isin(['ST', 'CF', 'RW', 'LW'])].groupby(['club_name', 'player_positions'])['overall'].nlargest(2).reset_index()

best_players = pd.concat([gk_players, def_players, midfielders, attackers])

club_scores = best_players.groupby('club_name')['overall'].sum().nlargest(30).reset_index()

club_scores


# In[18]:


#distribution of players on different positions by age
grouped_data = df.groupby(['age', 'club_position']).size().unstack()
fig, ax = plt.subplots(figsize=(10, 8))
grouped_data.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Number of Players')
ax.set_title('Distribution of Players on Different Positions by Age')

plt.legend(title='Club Position', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


# In[19]:


#distribution of players on different positions by ranking
grouped_data1 = df.groupby(['overall', 'club_position']).size().unstack()
fig, ax = plt.subplots(figsize=(15, 8))
grouped_data1.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Ranking')
ax.set_ylabel('Number of Players')
ax.set_title('Distribution of Players on Different Positions by Ranking')

plt.legend(title='Club Position', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


# In[20]:


pip install pycountry-convert


# In[21]:


#distribution of different nationalities by ranking
import pycountry_convert as pc

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

df1=df.copy()
df1.replace(['Curacao', 'Korea DPR', 'Chinese Taipei', 'Guinea Bissau', 'Cape Verde Islands', 'Republic of Ireland', 'China PR', 'Congo DR', 'England', 'Scotland', 'Wales', 'Northern Ireland', 'Korea Republic'], ['Curaçao','North Korea', 'Taiwan','Guinea-Bissau','Cabo Verde', 'Ireland','China', 'Democratic Republic of the Congo', 'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom of Great Britain and Northern Ireland','United Kingdom of Great Britain and Northern Ireland', 'United Kingdom of Great Britain and Northern Ireland', 'South Korea'], inplace=True)
#dropped those countries that are not in ISO
df1.drop(df1[df1['nationality_name'] == 'Kosovo'].index, inplace = True)

continents=[]
for i in df1['nationality_name']:
    a = country_to_continent(i)
    continents.append(a)
df1['continents'] = continents


# In[22]:


grouped_data2 = df1.groupby(['overall', 'continents']).size().unstack()
fig, ax = plt.subplots(figsize = (15,8))
grouped_data2.plot(kind='bar', stacked = True, ax=ax )
ax.set_xlabel('Ranking')
ax.set_ylabel('Number of Players')
ax.set_title('Distribution of Different Nationalities by Ranking')

plt.legend(title='Continent', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()


# Metrics for best penalty taker should include the following data:
# 2 x attacking_heading_accuracy for precise shots
# 2 x skill_fk_accuracy for finding balance between placement and power of free kick
# 1.5 x mentality_penalties for keeping focus and composure
# 1 x skill_curve to be able to curve the shooting line if needed
# 
# best penalty taker = 2 * attacking_heading_accuracy + 2 * skill_fk_accuracy + 1.5 * mentality_penalties + 1 * skill_curve

# Metrics for best GK for penalty defence:
# 3 x goalkeeping_reflexes for quick reaction
# 2 x goalkeeping_positioning for choosing the right place for defence
# 1.5 x mentality_penalties same as for penalty taker, GK should be able to focus
# 1 x goalkeeping_handling for overall ball handling skill
# 
# best GK for penalties = 3 * goalkeeping_reflexes + 2 * goalkeeping_positioning + 1.5 * mentality_penalties + 1 * goalkeeping_handling

# In[23]:


#top10 teams with best penalty takers
def best_penalty(i):
    best_penalty = 2*i['attacking_heading_accuracy'] + 2*i['skill_fk_accuracy'] + 1.5*i['mentality_penalties'] + 1*i['skill_curve']
    return best_penalty

df1['penalty_score'] = df1.apply(best_penalty, axis = 1)
df1[['long_name', 'club_name', 'club_position', 'penalty_score']].sort_values('penalty_score', ascending=False)[0:10] 


# In[24]:


#top10 teams with best GKs for penalty
def best_penalty_GK(i):
    best_penalty = 3*i['goalkeeping_reflexes'] + 2*i['goalkeeping_positioning'] + 1.5*i['mentality_penalties'] + 1*i['goalkeeping_handling']
    return best_penalty

df1['penalty_GK'] = df1.apply(best_penalty_GK, axis = 1)
df1[['long_name', 'club_name', 'club_position', 'penalty_GK']].sort_values('penalty_GK', ascending=False)[0:10] 


# In[25]:


gks = df1[['long_name', 'club_name', 'club_position', 'penalty_GK']].sort_values('penalty_GK', ascending=False)[0:60] 
gks.groupby('club_name').size()


# In[26]:


pts = df1[['long_name', 'club_name', 'club_position', 'penalty_score']].sort_values('penalty_score', ascending=False)[0:60]
pts.groupby('club_name').size()


# If we select the top60 penalty takers and top60 GK penalty defenders according to the previously defined score, we can see that some teams have significantly higher number of strong players in only one category. For example, Manchester United has 5 penalty takers from the top60 players that we have analyzed, and only 1 GK from the top60 penalty Goalkeepers, as well as Real Madrid CF 3 to 1. Paris Saint-Germain has the balance of 3 to 2, while Tottenham Hotspur vice versa, 1 top60 penalty taker in comparison to 2 GK penalty defenders. 
# It may indicate the team's focus and hiring strategy in general. 

# What can the data be used for? 
# The most obvious usage of this data would be in hiring process for different foorball clubs. First of all, by analyzing current team comp and what positions are weaker (as it was visible in top60 comparison), it would be easier to identify which role would be a priority to hire first, as well as what skills of the potential candidate should be tested(for GK, for instance, goalkeeping_reflexes and can compare them to average of other players that are already in the club. Second of all, we can look for a specific fit for a position based on nationality\continent distribution. So hiring manager can go to the country that historically had good attakers, defenders, etc. Another use would be how to assign position to a new team player based on age and typical good players of the same age group (as it was in diagram for age/ranking comparison). Finally this data can help with team composition: by analyzing individual traits, speed, accuracy and other skills, it can advice which team position would be optimal for player. As well as indicate best situations for player, for example, who would have higher chances to score on penalty. If we had historical data of penalty results, we could have checked which players are best at taking penalty and what skills have contributed to that. 

# In[27]:


df.info()


# In[28]:


def EDA(df):
    total_na = df.isna().sum().sum()
    print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
    print("Total NA Values : %d " % (total_na))
    print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "Count Distinct", "NA Values"))
    col_name = df.columns
    dtypes = df.dtypes
    uniq = df.nunique()
    na_val = df.isna().sum()
    for i in range(len(df.columns)):
        print("%38s %10s   %10s %10s" % (col_name[i], dtypes[i], uniq[i], na_val[i]))

EDA(df)


# In[29]:


#there are 19239 rows but only 19219 distinct values in long_name. Checking for duplicates
duplicated = df[df.duplicated(['long_name'])] #found 20 duplicates, dropping the last ones


# In[30]:


df.drop_duplicates(['long_name'])


# In[31]:


df[df['club_position'].isnull()]


# In[32]:


#61 players have missing values in club_position, filling them in with the first value from player_position
df_split = df['player_positions'].str.split(', ')
df['player_positions'] = df_split
def extract_first_item(row):
    if pd.isna(row['club_position']):
        source_list = row['player_positions']
        if source_list:  
            return source_list[0]
    return row['club_position']

df['club_position'] = df.apply(extract_first_item, axis=1)
df['player_positions'] = df['player_positions'].apply(lambda x: ', '.join(x) if x else '')
df[df['club_position'].isnull()]


# In[33]:


'''Drop columns
-will not be used to predict possible player_positions: long_name, potential, club_team_id, 
club_name, league_name, club_loaned_from, club_joined, club_contract_valid_until, nationality_name
nation_team_id, nation_jersey_number, international_reputation

-a lot of missing values: nation_position(18480 missing values, can copy from player_positions, but then can 
risk multicollinearity), also dropping club_position

-may mislead model: player tags and player_traits are properly filled in only for top players'''

df.drop(['long_name', 'potential', 'club_team_id', 'club_name', 'league_name', 'club_loaned_from', 
         'club_joined', 'club_contract_valid_until', 'nationality_name', 'nation_team_id', 'nation_position', 'club_position', 'nation_jersey_number', 'international_reputation', 'player_tags', 'player_traits'], axis=1, inplace=True)


# In[34]:


df.info()


# In[35]:


#dividing variables to categorical and continuous
categorical = [variable for variable in df.columns if df[variable].dtype=='O']
continuous = [variable for variable in df.columns if df[variable].dtype!='O']


# In[36]:


df.describe()


# In[37]:


#League_level and club_jersey_number have missing values, filling in with mode

df['league_level'] = df['league_level'].fillna(df['league_level'].mode()[0])
df['club_jersey_number'] = df['club_jersey_number'].fillna(df['club_jersey_number'].mode()[0])


# In[38]:


#value_eur, wage_eur and release_clause_eur have missing values, applying log e to normalize and fillna

df['release_clause_eur'] = pd.Series(np.log(df['release_clause_eur'])).fillna(df['release_clause_eur'].mean())
df['value_eur'] = pd.Series(np.log(df['value_eur'])).fillna(df['value_eur'].mean())
df['wage_eur'] = pd.Series(np.log(df['wage_eur'])).fillna(df['wage_eur'].mean())


# In[39]:


#2132 missing values in columns: pace, shooting, passing, dribbling, defending, physic. All with GK
df['player_positions'][df['shooting'].isnull()].unique()


# In[40]:


df[df['player_positions'] == 'GK'].count()


# In[41]:


df['pace'] = df['pace'].fillna(0)
df['shooting'] = df['shooting'].fillna(0)
df['passing'] = df['passing'].fillna(0)
df['dribbling'] = df['dribbling'].fillna(0)
df['defending'] = df['defending'].fillna(0)
df['physic'] = df['physic'].fillna(0)

#goalkeeping_speed also has missing values for all other players that are not GK, filling them in with 0
df['goalkeeping_speed'] = df['goalkeeping_speed'].fillna(0)


# In[42]:


#checking age for outliers
df[df['age']>40]


# In[43]:


lower_boundries= []
upper_boundries= []
for i in ["age"]:
    IQR= df[i].quantile(0.75) - df[i].quantile(0.25)
    lower_bound = df[i].quantile(0.25) - (1.5*IQR)
    upper_bound = df[i].quantile(0.75) + (1.5*IQR)
    
    print(i, ":", lower_bound, ",",  upper_bound)
    
    lower_boundries.append(lower_bound)
    upper_boundries.append(upper_bound)


# In[44]:


df = df[df['age'] <= 41]


# In[45]:


#checking if data is well balanced in player positions
df['player_positions'].value_counts().head(20) 
#data is unbalanced and has different number of entries for each position


# In[46]:


for i in categorical:
    print(df[i].value_counts()/float(len(df)))


# Encoding work_rate, body_type, preferred_foot with label encoder

# In[47]:


df['work_rate'].unique()


# In[48]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['work_rate'] = le.fit_transform(df['work_rate'])

#saving encoding mapping 
work_rate_mapping = {index: label for index, label in enumerate(le.classes_)}
print(work_rate_mapping)


# In[49]:


df['body_type'].unique()


# In[50]:


df['body_type'] = le.fit_transform(df['body_type'])

#saving encoding mapping 
body_type_mapping = {index: label for index, label in enumerate(le.classes_)}
print(body_type_mapping)


# In[51]:


df['preferred_foot'].unique()


# In[52]:


df['preferred_foot'] = df['preferred_foot'].map({'Right': 0, 'Left': 1})


# In[53]:


df.info()


# Encoding player_positions with onehotencoder

# In[54]:


positions_encoded = pd.get_dummies(df['player_positions'].str.split(', ', expand=True).stack()).sum(level=0)
df= pd.concat([df, positions_encoded], axis=1)
df.drop(['player_positions'], axis=1, inplace=True)
df


# In[71]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import hamming_loss


# In[72]:


df.columns


# In[73]:


X = df.drop(columns=['CAM', 'CB', 'CDM', 'CF', 'CM', 'GK', 'LB', 'LM', 'LW', 'LWB', 'RB', 'RM', 'RW', 'RWB', 'ST'])

Y = df[['CAM', 'CB', 'CDM', 'CF', 'CM', 'GK', 'LB', 'LM', 'LW', 'LWB', 'RB', 'RM', 'RW', 'RWB', 'ST']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)


# In[74]:


classifier = MultiOutputClassifier(XGBClassifier(random_state=42))
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)


# In[75]:


for i in range(0, 15):
    classifier = XGBClassifier()
    score = cross_val_score(classifier, X, Y.iloc[:, [i]], cv=8, scoring = 'accuracy',
                       n_jobs = 4)
    
    print("Mean score for ", Y.columns[i], " ", np.mean(score))


# In[76]:


weights_per_feature = {key: [] for key in X.columns}

for i, target_column in enumerate(Y.columns):
    xgb_model = XGBClassifier()
    xgb_model.fit(X, Y[target_column])
    
    booster = xgb_model.get_booster()
    importance = booster.get_score(importance_type='weight')  

    for feature, score in importance.items():
        weights_per_feature[feature].append(score)

print(weights_per_feature)


# In[84]:


mean_weight_per_feature = pd.Series([np.mean(weights_per_feature[key]) for key in X.columns], index = X.columns)
mean_weight_per_feature = mean_weight_per_feature.sort_values(ascending=False)
mean_weight_per_feature.plot.bar(figsize = (16, 8))
plt.show()


# In[88]:


X[mean_weight_per_feature.iloc[:15].index.to_numpy()]


# In[89]:


classifier = MultiOutputClassifier(XGBClassifier(seed=0))
score = cross_val_score(classifier, X[mean_weight_per_feature.iloc[:20].index.to_numpy()], Y, cv=8, scoring = make_scorer(hamming_loss,greater_is_better=True),
                       n_jobs = 4)


# In[90]:


for i in range(0, 15):
    classifier = XGBClassifier(seed=0)
    score = cross_val_score(classifier, X[mean_weight_per_feature.iloc[:15].index.to_numpy()], Y.iloc[:, i], cv=8, scoring = 'accuracy',
                       n_jobs = 4)
    
    print("Mean score for ", Y.columns[i], " ", np.mean(score))


# In[ ]:




