import xgboost as xgb
import random
import numpy as np

a=2
b=-1

X = np.array([[random.random()] for _ in range(100)])
y = np.array([[a*X[i][0]+b] for i in range(100)])


xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3)
xgb_reg.fit(X,y)

res = xgb_reg.predict(np.array([[0.5]]))
print(res)