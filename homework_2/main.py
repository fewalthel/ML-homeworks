import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Загрузка данных
data = pd.read_csv("AmesHousing.csv")
data = data.drop(columns=['Order', 'PID'])

# Целевая переменная
target = 'SalePrice'
y = data[target]
X = data.drop(columns=[target])

# Выделим числовые и категориальные признаки
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Пайплайны предобработки
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Обработка признаков
X_preprocessed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

# DataFrame из признаков
X_df = pd.DataFrame(X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed,
                    columns=feature_names)

# Удаление скоррелированных признаков
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_df.drop(columns=to_drop, inplace=True)

# Визуализация данных (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_df)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.colorbar(label='SalePrice')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D визуализация данных (PCA)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Массив для хранения ошибок
alphas = np.logspace(-4, 1, 30)
rmse_values = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

# График зависимости ошибки от регуляризации
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_values, marker='o')
plt.xscale('log')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от регуляризации (Lasso)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Определение самого важного признака
best_alpha = alphas[np.argmin(rmse_values)]
best_model = Lasso(alpha=best_alpha, max_iter=10000)
best_model.fit(X_train, y_train)

coefs = pd.Series(best_model.coef_, index=X_df.columns)
top_feature = coefs.abs().sort_values(ascending=False).idxmax()
print("Наиболее важный признак:", top_feature)
