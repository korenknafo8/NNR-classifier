import time
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from typing import List


def prepare_data(data_trn, data_vld, data_tst):
    # this function prepare the data to be scaled and separate from the labels
    # extract data from the file
    trn_df = pd.read_csv(data_trn)
    vld_df = pd.read_csv(data_vld)
    tst_df = pd.read_csv(data_tst)
    # remove the labels
    trn_x = trn_df.drop(columns=['class'])
    vld_x = vld_df.drop(columns=['class'])
    tst_x = tst_df.drop(columns=['class'])
    # scale the the features
    scaler = StandardScaler()
    trn_scaled_features = pd.DataFrame(scaler.fit_transform(trn_x), columns=trn_x.columns)
    vld_scaled_features = pd.DataFrame(scaler.fit_transform(vld_x), columns=vld_x.columns)
    tst_scaled_features = pd.DataFrame(scaler.fit_transform(tst_x), columns=tst_x.columns)
    # only labels
    trn_labels = trn_df['class']
    vld_labels = vld_df['class']

    return trn_scaled_features, vld_scaled_features, tst_scaled_features, trn_labels, vld_labels


def tuple_to_ind_and_vec(tuple_to_change):
    # this function get a tuple from itertuples() and returns the ind and a vector
    ind = tuple_to_change[0]
    vec = list(tuple_to_change)
    vec.pop(0)

    return ind, vec


def prediction_with_nnr(data_x, data_labels, vector, radius):
    # this function predict vector label with nnr model according to known data
    labels_in_r = []
    closest_vec_r = 0

    for trn_tuple in data_x.itertuples():
        index, trn_vec = tuple_to_ind_and_vec(trn_tuple)
        distance_from_vec = euclidean(trn_vec, vector)
        if distance_from_vec < radius:
            labels_in_r.append(data_labels[index])
        if closest_vec_r > distance_from_vec or closest_vec_r == 0:
            closest_vec_r = distance_from_vec
            closest_label = data_labels[index]

    if len(labels_in_r) == 0:             # if there is no vectors in the wanted r we insert the closest one
        labels_in_r.append(closest_label)

    predicted_label = max(labels_in_r, key=labels_in_r.count)
    return predicted_label


def find_min_max_distance(trn_scaled_features, vld_scaled_features):
    # this function returns the minimum distance and the maximum distance between
    # two vectors in two different data in the first 1000 vectors in each
    min_dist = 0
    max_dist = 0

    for trn_tuple in trn_scaled_features.itertuples():
        trn_i, trn_vec = tuple_to_ind_and_vec(trn_tuple)
        for vld_tuple in vld_scaled_features.itertuples():
            vld_i, vld_vec = tuple_to_ind_and_vec(vld_tuple)
            curr_dist = euclidean(vld_vec, trn_vec)
            if curr_dist < min_dist or min_dist == 0:
                min_dist = curr_dist
            elif curr_dist > max_dist:
                max_dist = curr_dist
            if vld_i > 1000:
                break
        if trn_i > 1000:
            break
    return min_dist, max_dist


def radius_vld(trn_scaled_features, vld_scaled_features, trn_labels, vld_labels) -> float:
    # this function return the radius that is the most accurate during it search
    approx_min, approx_max = find_min_max_distance(trn_scaled_features, vld_scaled_features)
    curr_r = best_r = approx_min
    jumps = (approx_max - approx_min) / 8
    accurate = 0
    part_len = len(vld_labels) // 30
    correct_predictions = 0

    for vld_tuple in vld_scaled_features.itertuples():
        index, vld_vec = tuple_to_ind_and_vec(vld_tuple)
        predicted_label = prediction_with_nnr(trn_scaled_features, trn_labels, vld_vec, curr_r)

        if predicted_label == vld_labels[index]:
            correct_predictions += 1
        if (index + 1) % part_len == 0:
            curr_acc = correct_predictions/part_len
            if curr_acc < accurate:
                jumps = jumps / -2
                curr_r = best_r + jumps
                if curr_r < 0:
                    curr_r -= 2*curr_r
            else:
                best_r = curr_r
                curr_r = best_r + jumps
                accurate = curr_acc
            correct_predictions = 0
    return best_r


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    # this function classify the data tst and returns the predicted labels for the tst
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')

    trn_scaled_features, vld_scaled_features, tst_scaled_features, trn_y, vld_y = prepare_data(data_trn, data_vld, data_tst)
    best_rad = radius_vld(trn_scaled_features, vld_scaled_features, trn_y, vld_y)

    predictions = []
    for tuple_to_predict in tst_scaled_features.itertuples():
        index, vec_to_predict = tuple_to_ind_and_vec(tuple_to_predict)
        row_label = prediction_with_nnr(trn_scaled_features, trn_y, vec_to_predict, best_rad)
        predictions.append(row_label)

    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
