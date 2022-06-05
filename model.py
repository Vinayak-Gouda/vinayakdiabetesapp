import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('diabetes.csv')
#print(df)
#print(df.shape)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

model = LogisticRegression()
model.fit(x_train,y_train)
'''
prediction = model.predict([[6,148,72,35,0,33.6,0.627,50]])
print("The patient has diabetes:",prediction)
'''
file = open('model_plk','wb')
pickle.dump(model,file)




