import warnings
warnings.filterwarnings('ignore')

# from sklearn import preprocessing

# Create an array of arguments to iteratively try out
sampling_strategy_arguments = np.arange(0.6, 1, 0.05)

average_accuracies = []
most_accurate_orders = []
most_stable_orders = []
standard_deviations = []

sampling_strategy_argument = 0.65

for i in range(3):
    print("1. split, SMOTE, scale")
    scores = []
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=3)
    
    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=8)
    X_train_SMOTE, y_train_SMOTE = smote.fit_sample(X_train, y_train.ravel())
    y_train_SMOTE = y_train_SMOTE.reshape(-1,1)
    
    # Create scaler for features and label
    X_train_SMOTE_scaler = StandardScaler().fit(X_train_SMOTE)
    X_test_scaler = StandardScaler().fit(X_test)
#     y_scaler = StandardScaler().fit(y_train_SMOTE)
    
    # Scale features and labels
    X_train_SMOTE_scaled = X_train_SMOTE_scaler.transform(X_train_SMOTE)
    X_test_scaled = X_test_scaler.transform(X_test)
#     y_train_SMOTE_scaled = y_scaler.transform(y_train_SMOTE)
    
    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_train_SMOTE_scaled, y=y_train_SMOTE)
    score = classifier.score(X_test_scaled, y_test)
    scores.append(score)
    
#     # Create, fit, and score the decision tree classifier
#     classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
#     classifier = classifier.fit(X=X_smote_train, y= y_smote_train)
#     score = classifier.score(X_smote_test, y_smote_test)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")

    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()
        
    ####################################################################################################

    print("2. split, scale, SMOTE")
    scores = []
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=3)
    
    # Create scaler for features and label
    X_train_scaler = StandardScaler().fit(X_train)
    X_test_scaler = StandardScaler().fit(X_test)
#     y_scaler = StandardScaler().fit(y_train)
    
    # Scale features and labels
    X_train_scaled = X_train_scaler.transform(X_train)
    X_test_scaled = X_test_scaler.transform(X_test)
#     y_train_scaled = y_scaler.transform(y_train)

    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=8)
    X_train_scaled_SMOTE, y_train_SMOTE = smote.fit_sample(X_train_scaled, y_train.ravel())
    y_train_SMOTE = y_train_SMOTE.reshape(-1,1)

    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_train_scaled_SMOTE, y=y_train_SMOTE)
    score = classifier.score(X_test_scaled, y_test)
    scores.append(score)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")
        
    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()
        
    ####################################################################################################

    print("3. SMOTE, split, scale")
    scores = []
        
    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=2)
    X_SMOTE, y_SMOTE = smote.fit_sample(X, y.ravel())
    y_SMOTE = y_SMOTE.reshape(-1,1)
    
    # Split the data into training and testing sets
    X_SMOTE_train, X_SMOTE_test, y_SMOTE_train, y_SMOTE_test = train_test_split(X_SMOTE, y_SMOTE, random_state=3)
    
    # Create scaler for features and label
    X_SMOTE_train_scaler = StandardScaler().fit(X_SMOTE_train)
    X_SMOTE_test_scaler = StandardScaler().fit(X_SMOTE_test)
#     y_scaler = StandardScaler().fit(y_SMOTE_train)
    
    # Scale features and labels
    X_SMOTE_train_scaled = X_SMOTE_train_scaler.transform(X_SMOTE_train)
    X_SMOTE_test_scaled = X_SMOTE_test_scaler.transform(X_SMOTE_test)
#     y_SMOTE_train_scaled = y_scaler.transform(y_train_SMOTE)
    
    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_SMOTE_train_scaled, y=y_SMOTE_train)
    score = classifier.score(X_SMOTE_test_scaled, y_SMOTE_test)
    scores.append(score)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")
        
    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()
        
    ####################################################################################################
        
        
    print("4. SMOTE, scale, split")
    scores = []
        
    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=2)
    X_SMOTE, y_SMOTE = smote.fit_sample(X, y.ravel())
    y_SMOTE = y_SMOTE.reshape(-1,1)
    
    # Create scaler for features and label
    X_scaler = StandardScaler().fit(X_SMOTE)
#     y_scaler = StandardScaler().fit(y_SMOTE)
    
    # Scale features and labels
    X_SMOTE_scaled = X_scaler.transform(X_SMOTE)
#     y_SMOTE_scaled = y_scaler.transform(y_SMOTE)
    
    # Split the data into training and testing sets
    X_SMOTE_scaled_train, X_SMOTE_scaled_test, y_SMOTE_train, y_SMOTE_test = train_test_split(X_SMOTE_scaled, y_SMOTE, random_state=3)
    
    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_SMOTE_scaled_train, y=y_SMOTE_train)
    score = classifier.score(X_test, y_test)
    scores.append(score)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")
        
    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()
        
    ####################################################################################################


    print("5. scale, split, SMOTE")
    scores = []
        
    # Create scaler for features and label
    X_scaler = StandardScaler().fit(X)
#     y_scaler = StandardScaler().fit(y_SMOTE)
    
    # Scale features and labels
    X_scaled = X_scaler.transform(X)
#     y_SMOTE_scaled = y_scaler.transform(y_SMOTE)

    # Split the data into training and testing sets
    X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, random_state=3)
    
    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=2)
    X_scaled_train_SMOTE, y_train_SMOTE = smote.fit_sample(X_scaled_train, y_train.ravel())
    y_train_SMOTE = y_train_SMOTE.reshape(-1,1)
    
    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_scaled_train_SMOTE, y=y_train_SMOTE)
    score = classifier.score(X_scaled_test, y_test)
    scores.append(score)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")
        
    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()
        
    ####################################################################################################

    print("6. scale, SMOTE, split")
    scores = []
        
    # Create scaler for features and label
    X_scaler = StandardScaler().fit(X)
#     y_scaler = StandardScaler().fit(y_SMOTE)
    
    # Scale features and labels
    X_scaled = X_scaler.transform(X)
#     y_SMOTE_scaled = y_scaler.transform(y_SMOTE)

    # Use SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy=sampling_strategy_argument, k_neighbors=2)
    X_scaled_SMOTE, y_SMOTE = smote.fit_sample(X_scaled, y.ravel())
    y_SMOTE = y_SMOTE.reshape(-1,1)

    # Split the data into training and testing sets
    X_scaled_SMOTE_train, X_scaled_SMOTE_test, y_SMOTE_train, y_SMOTE_test = train_test_split(X_scaled_SMOTE, y_SMOTE, random_state=3)
    
    # Create, fit, and score the decision tree classifier
    classifier = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=10)
    classifier = classifier.fit(X=X_scaled_SMOTE_train, y=y_SMOTE_train)
    score = classifier.score(X_scaled_SMOTE_test, y_SMOTE_test)
    scores.append(score)
    
    # Print a list of accuracies based on the current argument
    print(f"Setting the sampling_strategy parameter to {sampling_strategy_argument} yields an accuracy of {score}")  
        
    average_accuracy = sum(scores)/len(scores)
    average_accuracies.append(average_accuracy)
    standard_deviation = np.std(scores)
    print(f"Average accuracy: {average_accuracy}")
    print(f"Standard Deviation of accuracy: {standard_deviation}")
    print()

    ####################################################################################################

    print()

    if max(average_accuracies) == average_accuracies[0]:
        print("Most accurate order (on average) is 1: split, SMOTE, scale")
#         most_accurate_orders.append(1)
        most_accurate_order = 1
        
    elif max(average_accuracies) == average_accuracies[1]:
        print("Most accurate order (on average) is 2: split, scale, SMOTE")
#         most_accurate_orders.append(2)
        most_accurate_order = 2
        
    elif max(average_accuracies) == average_accuracies[2]:
        print("Most accurate order (on average) is 3: SMOTE, split, scale")
#         most_accurate_orders.append(3)
        most_accurate_order = 3
        
    elif max(average_accuracies) == average_accuracies[3]:
        print("Most accurate order (on average) is 4: SMOTE, scale, split")
#         most_accurate_orders.append(4)
        most_accurate_order = 4
        
    elif max(average_accuracies) == average_accuracies[4]:
        print("Most accurate order (on average) is 5: scale, split, SMOTE")
#         most_accurate_orders.append(5)
        most_accurate_order = 5
        
    elif max(average_accuracies) == average_accuracies[5]:
        print("Most accurate order (on average) is 6: scale, SMOTE, split")
#         most_accurate_orders.append(6)
        most_accurate_order = 6
        
print(f"Order number {most_accurate_order} was the most accurate the highest number of times.")
# print(f"Order number {} was the most stable (lowest standard deviation)")