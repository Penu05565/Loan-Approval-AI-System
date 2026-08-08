import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_pipeline import LoanDataPreparationPipeline

raw_path = os.path.join(os.getcwd(), 'data', 'loan_dataset.csv')
pipeline = LoanDataPreparationPipeline(source_path=raw_path, top_k_features=999, test_size=0.2, random_state=42)
result = pipeline.run()

X_train = result.train_df.drop(columns=['Loan_ID', 'Loan_Status'])
y_train = result.train_df['Loan_Status']
X_test = result.test_df.drop(columns=['Loan_ID', 'Loan_Status'])
y_test = result.test_df['Loan_Status']

print('selected', len(pipeline.feature_selector.selected_columns), pipeline.feature_selector.selected_columns)
print('train', X_train.shape, 'test', X_test.shape)
print('columns', X_train.columns.tolist())

for cls_name, cls, params in [
    ('RF', RandomForestClassifier(random_state=42, class_weight='balanced'), {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
    }),
    ('GB', GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }),
]:
    grid = GridSearchCV(cls, params, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print('===', cls_name, 'best', grid.best_params_)
    print('acc', accuracy_score(y_test, y_pred), 'prec', precision_score(y_test, y_pred), 'rec', recall_score(y_test, y_pred), 'f1', f1_score(y_test, y_pred))
