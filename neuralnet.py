import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import config
from dataset import dataset
from models import model_decision_tree_classifier, model_logistic_regression, model_random_forest_classifier, \
    model_sequential, model_sequential_increase
from utils import structureCheck


APP_ROOT, DATASET_PATH, MODELS_PATH, MODEL_EXISTS, DATASET_FILE = structureCheck()
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X_ROW_TEST = dataset()


def training():
    print(X_TRAIN[0])

    # Escolher o model para treinar
    # Descomentar o modelo pretendido
    regModel = model_logistic_regression(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    decModel = model_decision_tree_classifier(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    classModel = model_random_forest_classifier(X_TRAIN, Y_TRAIN)
    print(
        '###############################################################################################################')
    # TODO: Save a class model test (ver se é utilizavel ou eliminar depois)

    print(
        '---------------------------------------------------------------------------------------------------------------')
    modelSequential, history_dict = model_sequential(X_TRAIN, Y_TRAIN)
    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=config.TRAINING_EVALUATE_VERBOSE_VALUE)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # # Guardar o modelo feito
    modelSequential.save(MODELS_PATH) if not MODEL_EXISTS else None

    print(
        '---------------------------------------------------------------------------------------------------------------')

    # Teste da avaliação e do predict do modelo sequencial
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = modelSequential.evaluate(X_TEST, Y_TEST, batch_size=config.TRAINING_EVALUATE_BATCH_SIZE_VALUE)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # Isto é para teste
    print('\n# Generate predictions for 3 samples')
    predictions = modelSequential.predict(X_TEST[:3])
    print('predictions shape:', predictions.shape)

    # The AUC score is simply the area under the curve which can be calculated with Simpson’s Rule. The bigger the AUC score the better our classifier is.
    # isto é a area a tracejado que aparece no grafico

    # AUC score of testing data

    y_test_pred = modelSequential.predict(X_TEST)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TEST, y_test_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Testing data AUC: ', auc_keras)

    # AUC score of training data

    y_train_pred = modelSequential.predict(X_TRAIN)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TRAIN, y_train_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Training data AUC: ', auc_keras)

    print('Take a batch of 10 examples from the training data and call model.predict on it.')
    example_batch = X_TRAIN[:10]
    example_result = modelSequential.predict(example_batch)
    print(example_result)


def acc_increase():

    if MODEL_EXISTS:
        modelGoal = tf.keras.models.load_model(MODELS_PATH)
        lossGoal, accGoal = modelGoal.evaluate(X_TEST, Y_TEST, verbose=config.TRAINING_EVALUATE_VERBOSE_VALUE)
        loss = lossGoal
        acc = accGoal
    else:
        print('There is no model to improve! Create one by using app.py first.')

    while config.INCREASE_ACC_ATTEMPTS <= config.INCREASE_ACC_MAX_ATTEMPTS:
        if acc <= accGoal:
            for layers in np.arange(config.MIN_LAYERS, config.MAX_LAYERS, config.RATE_LAYERS):
                for neurons in np.arange(config.MIN_NEURONS, config.MAX_NEURONS, config.RATE_NEURONS):
                    for dropout in np.arange(config.MIN_DROPOUT, config.MAX_DROPOUT, config.RATE_DROPOUT):
                        modelSequential, history_dict = model_sequential_increase(X_TRAIN, Y_TRAIN, layers, neurons, dropout)
                        loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=0)
                        print('Current Try: ', config.INCREASE_ACC_ATTEMPTS)
                        config.INCREASE_ACC_ATTEMPTS += 1
        else:
            break

    if acc > accGoal:
        modelSequential.save(MODELS_PATH)
        print("Test accuracy: ", acc)
        print("Model was improved by: ", (acc - accGoal))
    else:
        print('The model could not be improved!')


def predict(data):
    print('#' * 80)
    # Filtro para filtrar alguns caracteres que possa meter por engano
    raw_text = eval('"' + data.replace('"', '\\"') + '"')
    print(raw_text)
    print('-' * 80)
    print(data)
    print('#' * 80)

    result = [x.strip() for x in raw_text.split(',')]

    print('*' * 80)
    df = pd.DataFrame(result)

    # TODO: Precisa levar uma limpeza e meter isto bonito
    print(df)
    X = df.values
    print(X)
    XX = []
    X = np.insert(X, 30, XX)
    X = pd.DataFrame([X])
    X = X.values
    print(X)

    print(X_ROW_TEST)

    X = np.concatenate((X, X_ROW_TEST))

    sc = StandardScaler()
    X = sc.fit_transform(X)

    # X.columns = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
    #               'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 'radius_se',
    #               'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    #               'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worse', 'texture_worst',
    #               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    #               'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    print('--' * 80)
    print('')

    modelSequential = tf.keras.models.load_model(MODELS_PATH)

    # Avaliação do Modelo Sequencial
    print('')
    loss, acc = modelSequential.evaluate(X_TEST, Y_TEST, verbose=config.TRAINING_EVALUATE_BATCH_SIZE_VALUE)
    print("Test loss: ", loss)
    print("Test accuracy: ", acc)

    # AUC score of training data
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    y_train_pred = modelSequential.predict(X_TRAIN)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TRAIN, y_train_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('Training data AUC: ', auc_keras)

    print('')
    print('M = 1 e B = 0')
    pred = modelSequential.predict(X[:1])
    print(pred)
    print()

    print(pred[0].mean())

    print("Benigno" if pred.mean() <= 0.50 else "Maligno")
