import numpy as np
from sklearn.linear_model import LinearRegression

X_test = np.random.randn(1000,10)
y_test = np.sum(X_test ,axis=1)
X_train = np.random.randn(10000,10)
y_train = np.sum(X_train ,axis=1)

reg = LinearRegression().fit(X_train, y_train)
print(f"Train Loss: {reg.score(X_train, y_train)}")

print(f"Test Loss: {reg.score(X_test, y_test)}")
