import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import uniform, randint
from sklearn.model_selection import learning_curve

# Main function
def main():  
    algorithm = sys.argv[1]
    dataset_name = sys.argv[2]

    if algorithm not in ["logistic_regression", "mlp"]:
        print("Invalid algorithm")
        return
    
    if dataset_name not in ["AVC", "SalaryPrediction"]:
        print("Invalid dataset")
        return

    process_input(algorithm, dataset_name)

# Functie care citeste datele din fisierul de input si apeleaza algoritmul specificat
def process_input(algorithm, dataset_name): 
    full_input_path = "tema2_" + dataset_name + "/" + dataset_name + "_full.csv"
    train_input_path = "tema2_" + dataset_name + "/" + dataset_name + "_train.csv"
    test_input_path = "tema2_" + dataset_name + "/" + dataset_name + "_test.csv"

    full_dataset = pd.read_csv(full_input_path)
    train_dataset = pd.read_csv(train_input_path)
    test_dataset = pd.read_csv(test_input_path)

    if dataset_name == "AVC":
        full_dataset['tobacco_usage'] = full_dataset['tobacco_usage'].replace('not_defined', np.nan)
        train_dataset['tobacco_usage'] = train_dataset['tobacco_usage'].replace('not_defined', np.nan)
        test_dataset['tobacco_usage'] = test_dataset['tobacco_usage'].replace('not_defined', np.nan)
        full_dataset['biological_age_index'] = full_dataset['biological_age_index'].apply(lambda x: x if x >= 0 else np.nan)
        full_dataset['cardiovascular_issues'] = full_dataset['cardiovascular_issues'].replace(0, 'N')
        full_dataset['cardiovascular_issues'] = full_dataset['cardiovascular_issues'].replace(1, 'Y')
        full_dataset['high_blood_pressure'] = full_dataset['high_blood_pressure'].replace(0, 'N')
        full_dataset['high_blood_pressure'] = full_dataset['high_blood_pressure'].replace(1, 'Y')
        full_dataset['chaotic_sleep'] = full_dataset['chaotic_sleep'].replace(0, 'N')
        full_dataset['chaotic_sleep'] = full_dataset['chaotic_sleep'].replace(1, 'Y')
        full_dataset['cerebrovascular_accident'] = full_dataset['cerebrovascular_accident'].replace(0, 'N')
        full_dataset['cerebrovascular_accident'] = full_dataset['cerebrovascular_accident'].replace(1, 'Y')
        train_dataset['cardiovascular_issues'] = train_dataset['cardiovascular_issues'].replace(0, 'N')
        train_dataset['cardiovascular_issues'] = train_dataset['cardiovascular_issues'].replace(1, 'Y')
        train_dataset['high_blood_pressure'] = train_dataset['high_blood_pressure'].replace(0, 'N')
        train_dataset['high_blood_pressure'] = train_dataset['high_blood_pressure'].replace(1, 'Y')
        train_dataset['chaotic_sleep'] = train_dataset['chaotic_sleep'].replace(0, 'N')
        train_dataset['chaotic_sleep'] = train_dataset['chaotic_sleep'].replace(1, 'Y')
        train_dataset['cerebrovascular_accident'] = train_dataset['cerebrovascular_accident'].replace(0, 'N')
        train_dataset['cerebrovascular_accident'] = train_dataset['cerebrovascular_accident'].replace(1, 'Y')
        test_dataset['cardiovascular_issues'] = test_dataset['cardiovascular_issues'].replace(0, 'N')
        test_dataset['cardiovascular_issues'] = test_dataset['cardiovascular_issues'].replace(1, 'Y')
        test_dataset['high_blood_pressure'] = test_dataset['high_blood_pressure'].replace(0, 'N')
        test_dataset['high_blood_pressure'] = test_dataset['high_blood_pressure'].replace(1, 'Y')
        test_dataset['chaotic_sleep'] = test_dataset['chaotic_sleep'].replace(0, 'N')
        test_dataset['chaotic_sleep'] = test_dataset['chaotic_sleep'].replace(1, 'Y')
        test_dataset['cerebrovascular_accident'] = test_dataset['cerebrovascular_accident'].replace(0, 'N')
        test_dataset['cerebrovascular_accident'] = test_dataset['cerebrovascular_accident'].replace(1, 'Y')
        
    if dataset_name == "SalaryPrediction":
        full_dataset['country'] = full_dataset['country'].replace('?', np.nan)
        full_dataset['work_type'] = full_dataset['work_type'].replace('?', np.nan)
        full_dataset['job'] = full_dataset['job'].replace('?', np.nan)
        train_dataset['country'] = train_dataset['country'].replace('?', np.nan)
        train_dataset['work_type'] = train_dataset['work_type'].replace('?', np.nan)
        train_dataset['job'] = train_dataset['job'].replace('?', np.nan)
        test_dataset['country'] = test_dataset['country'].replace('?', np.nan)
        test_dataset['work_type'] = test_dataset['work_type'].replace('?', np.nan)
        test_dataset['job'] = test_dataset['job'].replace('?', np.nan)

    full_dataset, train_dataset, test_dataset, numerical_analysis, categorical_analysis, correlation_matrix= data_analysis(full_dataset, train_dataset, test_dataset)
    X_train, X_test, y_train, y_test = data_processing(train_dataset, test_dataset, numerical_analysis, categorical_analysis, correlation_matrix)


    if algorithm == "logistic_regression":
        logistic_regresion(X_train, X_test, y_train, y_test, dataset_name)
    else:
        mlp(X_train, X_test, y_train, y_test, dataset_name)
   
    
def data_analysis(dataset, train_dataset, test_dataset):
    
    numerical_analysis = dataset.describe()
    categorical_analysis = dataset.describe(include=['object'])
    

    print("Numerical analysis")
    print(numerical_analysis)
    plt.figure(figsize=(12, 8))
    numerical_analysis.drop(['count', 'mean', 'std']).boxplot()
    plt.xticks(rotation=45, ha='right') 
    plt.title('The ranges of values of continuous numerical attributes')
    plt.tight_layout()
    plt.show()

    print("Categorical analysis")
    print(categorical_analysis)
  
    for column in categorical_analysis.columns:
        plt.figure(figsize=(12, 8))
        dataset[column].value_counts().plot(kind='bar')
        plt.title(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 8))
    train_dataset.iloc[:, -1]
    train_dataset.iloc[:, -1].value_counts().plot(kind='bar')
    plt.title('Train dataset:' + train_dataset.columns[-1])
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    test_dataset.iloc[:, -1]
    test_dataset.iloc[:, -1].value_counts().plot(kind='bar')
    plt.title('Test dataset:' + test_dataset.columns[-1])
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show() 
    

    correlation_matrix_pearson = dataset.corr(method='pearson', numeric_only=True, min_periods=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.matshow(correlation_matrix_pearson, cmap='coolwarm')
    plt.colorbar(heatmap)
    ax.set_title('Correlation matrix for numerical attributes')
    ax.set_xticks(range(len(correlation_matrix_pearson.columns)))
    ax.set_xticklabels(correlation_matrix_pearson.columns, rotation=45, ha='left')
    ax.set_yticks(range(len(correlation_matrix_pearson.columns)))
    ax.set_yticklabels(correlation_matrix_pearson.columns)
    plt.tight_layout()
    plt.show()


    correlation_matrix_chi2 = pd.DataFrame(index=categorical_analysis.columns, columns=categorical_analysis.columns)
    # Calculate the Chi-square correlation matrix
    for attribute1 in categorical_analysis.columns:
        for attribute2 in categorical_analysis.columns:
            # Calculate the contingency table
            contingency_table = pd.crosstab(dataset[attribute1], dataset[attribute2])
            
            # Perform the Chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            # Store the p-value in the correlation matrix
            correlation_matrix_chi2.loc[attribute1, attribute2] = p_value


    correlation_matrix_chi2 = correlation_matrix_chi2.astype(float)
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.matshow(correlation_matrix_chi2, cmap='coolwarm')
    plt.colorbar(heatmap)
    ax.set_title('Correlation matrix for categorical attributes')
    ax.set_xticks(range(len(correlation_matrix_chi2.columns)))
    ax.set_xticklabels(correlation_matrix_chi2.columns, rotation=45, ha='left')
    ax.set_yticks(range(len(correlation_matrix_chi2.columns)))
    ax.set_yticklabels(correlation_matrix_chi2.columns)
    plt.tight_layout()
    plt.show()

    return dataset, train_dataset, test_dataset, numerical_analysis, categorical_analysis, correlation_matrix_pearson

def data_processing(train_dataset, test_dataset, numerical_analysis, categorical_analysis, correlation_matrix):

    #Daca avem corelare mai mare de 0.70 intre doua atribute numerice vom elimina unul din ele
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                print("Atribute puternic corelate: ", correlation_matrix.columns[i], correlation_matrix.columns[j])
                print ("Eliminam atributul: ", correlation_matrix.columns[j])
                train_dataset = train_dataset.drop(correlation_matrix.columns[j], axis=1)
                test_dataset = test_dataset.drop(correlation_matrix.columns[j], axis=1)
                numerical_analysis = numerical_analysis.drop(correlation_matrix.columns[j], axis=1)
                break

    #Identificam outlierii pentru atributele numerice si ii inlocuim cu nan
    for dataset in [train_dataset, test_dataset]:
        for column in numerical_analysis.columns:
            q1 = dataset[column].quantile(0.25)
            q3 = dataset[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            dataset[column] = dataset[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)


    # Daca avem valori lipsa in dataset, vom completa aceste valori cu cea mai frecventa pentru atributele categorice
    # si cu o metoda de imputare iterativa pentru atributele numerice
    numeric_imputer = IterativeImputer(random_state=0, missing_values=np.nan)
    categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Iterate through columns
    for column in train_dataset.columns:
        if column in categorical_analysis.columns:
            # For categorical columns, use SimpleImputer
            train_dataset[column] = categorical_imputer.fit_transform(train_dataset[[column]])[:, 0]
            test_dataset[column] = categorical_imputer.transform(test_dataset[[column]])[:, 0]
        else:
            # For numeric columns, use IterativeImputer
            train_dataset[column] = numeric_imputer.fit_transform(train_dataset[[column]])
            test_dataset[column] = numeric_imputer.transform(test_dataset[[column]])

    #Standardizarea datelor numerice
    scaler = StandardScaler()
    train_dataset[numerical_analysis.columns] = scaler.fit_transform(train_dataset[numerical_analysis.columns])
    test_dataset[numerical_analysis.columns] = scaler.transform(test_dataset[numerical_analysis.columns])
    
    #Ultima coloana din dataset este variabila țintă si trebuie sa o scoatem din dataset si sa o punem intr-o variabila separata
    y_train = train_dataset.iloc[:, -1]
    y_test = test_dataset.iloc[:, -1]
    train_dataset = train_dataset.drop(train_dataset.columns[-1], axis=1)
    test_dataset = test_dataset.drop(test_dataset.columns[-1], axis=1)
    categorical_analysis = categorical_analysis.drop(categorical_analysis.columns[-1], axis=1)

    #Pt variavila tinta vom folosi LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    #Transformam atributele categorice din train_dataset in atribute numerice folosind one-hot encoding
    # Vom folosi pandas.get_dummies
    X_train = pd.get_dummies(train_dataset, columns=categorical_analysis.columns, drop_first=True)
    X_test = pd.get_dummies(test_dataset, columns=categorical_analysis.columns, drop_first=True)

    return X_train, X_test, y_train, y_test
    
# Functie care implementeaza algoritmul de logistic regression
def logistic_regresion(X_train, X_test, y_train, y_test, dataset_name):
    

    # Ensure feature names match between training and testing datasets
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Add missing column with default value 0

    # Reorder columns to match order in training dataset
    X_test = X_test[X_train.columns]

    if dataset_name == 'AVC':
        class_weights = {0: 1, 1: 5}
    else:
        class_weights = {0: 1, 1: 1}
        
    clf = LogisticRegression(random_state=0, class_weight=class_weights, solver='liblinear', penalty='l1').fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Logistic Regression applied on ' + dataset_name + ' dataset', fontsize=13)   
    plt.show()

    print("Accuracy: ", clf.score(X_test, y_test))
    print("Precision: ", cm[1, 1] / (cm[1, 1] + cm[0, 1]))
    print("Recall: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    print("F1 score: ", 2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]))

    return

# Functie care implementeaza algoritmul de MLP
def mlp(X_train, X_test, y_train, y_test, dataset_name):

    # Ensure feature names match between training and testing datasets
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Add missing column with default value 0

    # Reorder columns to match the order in training dataset
    X_test = X_test[X_train.columns]

    clf = MLPClassifier(max_iter=1000, random_state=0, hidden_layer_sizes=(50, 50), activation='tanh', solver='lbfgs', alpha=0.01,early_stopping=False, validation_fraction=0.2, learning_rate='invscaling', learning_rate_init=0.001).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for MLP applied on ' + dataset_name + ' dataset', fontsize=13)
    plt.show()

    print("Accuracy: ", clf.score(X_test, y_test))
    print("Precision: ", cm[1, 1] / (cm[1, 1] + cm[0, 1]))
    print("Recall: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    print("F1 score: ", 2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]))   

    title = "Learning and Error Curves - MLP (Dataset: " + dataset_name + ")"
    plot_learning_and_error_curves(clf, title, X_train, y_train, cv=5)
    plt.show()

    return

def plot_learning_and_error_curves(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(14, 8))
    plt.suptitle(title, fontsize=14)
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    train_error_mean = 1 - train_scores_mean
    train_error_std = train_scores_std
    test_error_mean = 1 - test_scores_mean
    test_error_std = test_scores_std
    
    plt.xlabel("Training examples")
    plt.ylabel("Score / Error")
    
    plt.grid()
    
    # Plotting accuracy
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation accuracy")
    
    # Plotting error
    plt.fill_between(train_sizes, train_error_mean - train_error_std, train_error_mean + train_error_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_error_mean - test_error_std, test_error_mean + test_error_std, alpha=0.1, color="y")
    plt.plot(train_sizes, train_error_mean, 'o-', color="b", label="Training error")
    plt.plot(train_sizes, test_error_mean, 'o-', color="y", label="Validation error")
    
    plt.legend(loc="best")
    plt.title("Learning and Error Curves")
    
    plt.show()


if __name__ == "__main__":
    main()

