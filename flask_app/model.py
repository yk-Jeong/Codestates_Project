import pandas as pd

df = pd.read_csv('all_drinks.csv')
df.rename(columns = lambda x: x.replace("str", ""), inplace=True)
df.rename(columns = {'Drink':'Name', 'idDrink':'ID'}, inplace = True) 

#DrinkThumb, Video, Instructions는 검증용 컬럼인 idDrink와 함께 별도의 df로 남겨둠 
df_imgdata = df[['ID', 'DrinkThumb', 'Video', 'Inuctions']]
df = df.drop(['dateModified', 'Unnamed: 0', 'DrinkThumb', 'Video', 'Inuctions'], axis=1)

df = df.copy()
df = df[['Name', 'ID', 'Alcoholic', 'Category', 'Glass', 'IBA', 'Ingredient1', 'Ingredient2', 'Ingredient3', 'Ingredient4', 'Ingredient5']]

#'Non alcoholic', 'Non Alcoholic' 통일
df['Alcoholic'] = df['Alcoholic'].replace('Non Alcoholic', 'Non alcoholic', regex=True) 
df['Alcoholic'] = df['Alcoholic'].fillna('Alcoholic')

#IBA 컬럼의 null값은 IBA 비공식이라는 뜻이므로 unofficial로 대체 
df['IBA'] = df['IBA'].fillna('Unofficial')

#대소문자를 통일하지 않아 Ingredients 등의 컬럼에 중복값이 생긴 것을 확인 -> 소문자로 통일하여 중복값을 제거하는 함수 적용
def engineering(df):
  df = df.copy()
  
  df['Ingredient1'] = df['Ingredient1'].str.lower()
  df['Ingredient2'] = df['Ingredient2'].str.lower()
  df['Ingredient3'] = df['Ingredient3'].str.lower()
  df['Ingredient4'] = df['Ingredient4'].str.lower()
  df['Ingredient5'] = df['Ingredient5'].str.lower()
  df['Glass'] = df['Glass'].str.lower()

  return df

df = engineering(df)

df_ingredient= df['Ingredient1'].astype(str) + ' '+ df['Ingredient2'].astype(str)+ ' '+ df['Ingredient3'].astype(str)+ ' '+ df['Ingredient4'].astype(str)+ ' '+ df['Ingredient5'].astype(str)
df_ingredient.replace('nan', '', regex=True, inplace=True) #n번째 성분이 없는(즉 원래는 결측치인) 경우 nan을 삭제
df_ingredient = df_ingredient.astype(str) #int가 없도록 str 처리 

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF Vectorizer 객체를 정의
tfidf = TfidfVectorizer()

#데이터를 적합하고 변환하여 필요한 TF-IDF 행렬을 구성
tfidf_matrix = tfidf.fit_transform(df_ingredient)

#546개 컬럼, 283개 단어 모음을 가진 TF-IDF 행렬로 추출
tfidf_matrix.shape 

# DataFrame의 인덱스를 재설정하고 역방향 매핑을 구성
df_ingredient = df_ingredient.reset_index()
indices = pd.Series(df_ingredient.index, index=df['Name']) #원본 데이터프레임의 Name column과 index 연결

#코사인 유사도 계산
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def model(name, cosine_sim=cosine_sim):
    idx = indices[name] #indices 데이터프레임에서 칵테일 이름과 일치하는 인덱스 번호를 받아옴
    sim_scores = list(enumerate(cosine_sim[idx])) #유사도 점수를 코사인 유사도 방식으로 계산

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4] #유사도 점수가 가장 높은 칵테일을 3개까지 출력

    cocktail_indices = [i[0] for i in sim_scores] #칵테일의 인덱스를 출력 

    # Return the top 3 most similar movies
    return df['Name'].iloc[cocktail_indices]

import pickle
with open('model.pkl', 'wb') as pickle_file:
  pickle.dump(model, pickle_file)