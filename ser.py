import librosa
import pandas as pd
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_path = 'SERDataset/SERDataset/Train'
test_path = 'SERDataset/SERDataset/Test'

dataset_paths = {
    "Train": train_path,
    "Test": test_path
}

emotions = os.listdir(path=train_path)


def get_ids_df():
    ids = pd.DataFrame()
    for emotion in emotions:
        recordings_train = os.listdir(path=f"{train_path}/{emotion}")
        recordings_test = os.listdir(path=f"{test_path}/{emotion}")
        ids = pd.concat(
            [
                ids,
                pd.DataFrame({
                    "RecId": recordings_train,
                    "Emotion": [emotion] * len(recordings_train),
                    "Type": ['Train'] * len(recordings_train)
                }),
                pd.DataFrame({
                    "RecId": recordings_test,
                    "Emotion": [emotion] * len(recordings_test),
                    "Type": ['Test'] * len(recordings_test)
                })
            ], ignore_index=True
        )
    return ids


def extract_features(signal_sr):
    signal, sr = signal_sr[0], signal_sr[1]
    all_features = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    all_features = np.hstack((all_features, mfccs))

    spectral_centroids = np.mean(librosa.feature.spectral_centroid(signal, sr=sr).T, axis=0)
    all_features = np.hstack((all_features, spectral_centroids))
    return all_features


def load_file(file_path):
    return librosa.load(file_path, sr=16000)


def add_features(ids, type):
    files = ids.copy().loc[ids['Type'] == type]
    files['SignalSR'] = files.apply(
        lambda x: load_file(
            f"{dataset_paths[type]}/{x['Emotion']}/{x['RecId']}"
        ), axis=1
    )
    files['X'] = files['SignalSR'].apply(extract_features)
    files['y'] = files['Emotion'].apply(lambda x: emotions.index(x))
    return files


def train_model(x_train, y_train, model_name):
    cls = MLPClassifier(solver='adam', hidden_layer_sizes=(200,))
    cls.fit(x_train, y_train)
    pickle.dump(cls, open(model_name, "wb"))


def get_prediction(x_test, model_name):
    cls = pickle.load(open(model_name, "rb"))
    return cls.predict(x_test)


def execute():
    ids = get_ids_df()

    model_name = "ser_classifier_mlp.model"
    train = add_features(ids, type='Train')
    train_model(train['X'].tolist(), train['y'].tolist(), model_name)

    test = add_features(ids, type='Test')
    test['PredictedY'] = get_prediction(test['X'].tolist(), model_name)
    test['PredictedEmotion'] = test['PredictedY'].apply(lambda x: emotions[x])
    print("Accuracy:", accuracy_score(test['y'], test['PredictedY']))

    test.to_excel('Predictions.xlsx', index=False)


if __name__ == '__main__':
    execute()
