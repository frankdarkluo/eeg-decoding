import config
from feature_extraction import zuco_reader
from reldetect import reldetect_text_eeg_model, reldetect_eeg_gaze_model, reldetect_text_eeg4_model, reldetect_text_eeg_gaze_model
from ner import ner_text_model
from sentiment import sentiment_eeg_model, sentiment_eeg_gaze_model, sentiment_text_eeg_gaze_model, sentiment_text_eeg4_model, sentiment_text_random_model, sentiment_text_eeg_model
from data_helpers import save_results, sort_dict, load_matlab_files, load_data
import numpy as np

import json
import sys
import os
import tensorflow as tf
import random
from datetime import timedelta
import time

def sentiment_classification(config,feature_dict, label_dict, eeg_dict_theta, eeg_dict,gaze_dict,eeg_dict_alpha,
                             eeg_dict_beta, eeg_dict_gamma,parameter_dict, rand):

    if 'eeg4' in config.feature_set:
        fold_results = sentiment_text_eeg4_model.classifier(feature_dict, label_dict, eeg_dict_theta,
                                                            eeg_dict_alpha, eeg_dict_beta, eeg_dict_gamma,
                                                            config.embeddings, parameter_dict, rand)

    if 'eeg_raw' in config.feature_set:
        fold_results = sentiment_eeg_model.classifier(label_dict, eeg_dict, config.embeddings, parameter_dict, rand)
    elif 'text_eeg_eye_tracking' in config.feature_set:
        fold_results = sentiment_text_eeg_gaze_model.classifier(feature_dict, label_dict, eeg_dict, gaze_dict,
                                                                config.embeddings, parameter_dict, rand)
    elif 'eeg_eye_tracking' in config.feature_set:
        fold_results = sentiment_eeg_gaze_model.classifier(label_dict, eeg_dict, gaze_dict, config.embeddings,
                                                           parameter_dict, rand)

    elif 'random' in config.feature_set and 'eeg_theta' in config.feature_set:
        fold_results = sentiment_text_random_model.classifier(feature_dict, label_dict, eeg_dict,
                                                              config.embeddings, parameter_dict, rand)


    elif any(feature in config.feature_set for feature in
             ['combi_eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma']):
        fold_results = sentiment_text_eeg_model.classifier(feature_dict, label_dict,
                                                           eeg_dict, config.embeddings,
                                                           parameter_dict, rand)

    save_results(fold_results, config.class_task)

def main():
    eeg_dict, eeg_dir, feature_dict, label_dict=load_data()

    if config.run_eeg_extraction:
        # save EEG features
        with open(config.feature_dir + config.feature_set[0] + '_feats_file_' + config.class_task + '.json', 'w') as fp:
            json.dump(eeg_dict, fp)
        print("saved.")
        sys.exit()
    else:
        print("Reading EEG features from file!!")

        if 'eeg4' in config.feature_set:
            eeg_dict_theta = json.load(open(f"{eeg_dir}/eeg_theta_feats_file_{config.class_task}.json"))
            eeg_dict_beta = json.load(open(f"{eeg_dir}/eeg_beta_feats_file_{config.class_task}.json"))
            eeg_dict_alpha = json.load(open(f"{eeg_dir}/eeg_alpha_feats_file_{config.class_task}.json"))
            eeg_dict_gamma = json.load(open(f"{eeg_dir}/eeg_gamma_feats_file_{config.class_task}.json"))
            eeg_dict_alpha = sort_dict(eeg_dict_alpha)
            eeg_dict_beta = sort_dict(eeg_dict_beta)
            eeg_dict_theta = sort_dict(eeg_dict_theta)
            eeg_dict_gamma = sort_dict(eeg_dict_gamma)

        elif 'text_eeg_eye_tracking' in config.feature_set:
            eeg_dict = json.load(open(f"{eeg_dir}/combi_eeg_raw_feats_file_{config.class_task}.json"))
        else:
            eeg_dict = json.load(open(f"{eeg_dir}/{config.feature_set[0]}_feats_file_{config.class_task}.json"))

        print("done, ", len(eeg_dict), " sentences with EEG features.")

        print("Reading gaze features from file!!")
        gaze_dict = json.load(open("feature_extraction/features/gaze_feats_file_" + config.class_task + ".json"))

    feature_dict = sort_dict(feature_dict)
    label_dict = sort_dict(label_dict)
    eeg_dict = sort_dict(eeg_dict)
    gaze_dict = sort_dict(gaze_dict)

    print(len(feature_dict.keys()), len(label_dict))

    if 'eeg4' in config.feature_set:
        if len(set([len(feature_dict), len(label_dict), len(eeg_dict_alpha), len(eeg_dict_beta), len(eeg_dict_gamma),
                    len(eeg_dict_theta)])) > 1:
            print("WARNING: Not an equal number of sentences in features and labels!")
        print('len(feature_dict):\t', len(feature_dict))
        print('len(label_dict):\t', len(label_dict))
        print('len(eeg_dict_alpha):\t', len(eeg_dict_alpha))
        print('len(eeg_dict_beta):\t', len(eeg_dict_beta))
        print('len(eeg_dict_gamma):\t', len(eeg_dict_gamma))
        print('len(eeg_dict_theta):\t', len(eeg_dict_theta))
    else:
        eeg_dict_alpha, eeg_dict_beta, eeg_dict_gamma, eeg_dict_theta = None, None, None, None
        if len(feature_dict) != len(label_dict) or len(feature_dict) != len(eeg_dict) or len(label_dict) != len(
                eeg_dict):
            print("WARNING: Not an equal number of sentences in features and labels!")
        print('len(feature_dict): {}\nlen(label_dict): {}\nlen(eeg_dict): {}'.format(len(feature_dict), len(label_dict),
                                                                                     len(eeg_dict)))
        if "eye_tracking" in config.feature_set[0]:
            print('len(feature_dict): {}'.format(len(gaze_dict)))


    print('Starting Loop')
    start = time.time()
    count = 0

    def create_parameter_dict(rand, lstmDim, lstmLayers, denseDim, drop, bs, lr_val, e_val, inception_filters,
                              inception_kernel_sizes, inception_pool_size, inception_dense_dim):
        """Create a parameter dictionary from the given parameters."""
        return {
            "lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
            "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
            "epochs": e_val, "random_seed": rand, "inception_filters": inception_filters,
            "inception_dense_dim": inception_dense_dim, "inception_kernel_sizes": inception_kernel_sizes,
            "inception_pool_size": inception_pool_size
        }

    def run(parameter_dict, rand):
        """Return the appropriate model based on the parameter dictionary and task."""
        if config.class_task == 'reldetect':
            for threshold in config.rel_thresholds:
                if 'eeg4' in config.feature_set:
                    fold_results = reldetect_text_eeg4_model.classifier(feature_dict, label_dict, eeg_dict_theta,
                                                                        eeg_dict_alpha, eeg_dict_beta, eeg_dict_gamma,
                                                                        config.embeddings, parameter_dict, rand)
                elif any(feature in config.feature_set for feature in ['combi_eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma']):
                    fold_results = reldetect_text_eeg_model.classifier(feature_dict, label_dict, eeg_dict, config.embeddings,
                                                                       parameter_dict, rand, threshold)
                elif 'eeg_eye_tracking' in config.feature_set:
                    fold_results = reldetect_eeg_gaze_model.classifier(label_dict, eeg_dict, gaze_dict, config.embeddings,
                                                                       parameter_dict, rand, threshold)
                elif 'text_eeg_eye_tracking' in config.feature_set:
                    print(config.feature_set)
                    fold_results = reldetect_text_eeg_gaze_model.classifier(feature_dict, label_dict, eeg_dict,
                                                                            gaze_dict, config.embeddings,
                                                                            parameter_dict, rand, threshold)
                save_results(fold_results, config.class_task)

        elif config.class_task == 'sentiment-tri':
            sentiment_classification()

        elif config.class_task == 'sentiment-bin':
            print("dropping neutral sentences for binary sentiment classification")
            for s, label in list(label_dict.items()):
                # drop neutral sentences for binary sentiment classification
                if label == 2:
                    del label_dict[s]
                    del feature_dict[s]
                    if 'eeg4' in config.feature_set:
                        del eeg_dict_theta[s]
                        del eeg_dict_alpha[s]
                        del eeg_dict_beta[s]
                        del eeg_dict_gamma[s]
                    else:
                        del eeg_dict[s]

            sentiment_classification(config,feature_dict, label_dict, eeg_dict_theta, eeg_dict,gaze_dict,eeg_dict_alpha,
                             eeg_dict_beta, eeg_dict_gamma,parameter_dict, rand)

    for rand in config.random_seed_values:
        np.random.seed(rand)
        tf.random.set_seed(rand)
        os.environ['PYTHONHASHSEED'] = str(rand)
        random.seed(rand)
        parameter_dict = create_parameter_dict(rand, lstmDim=256,
                                               lstmLayers=1, denseDim=128, drop=0.3,
                                               bs=60, lr_val=0.001, e_val=1,
                                               inception_filters=14,
                                               inception_kernel_sizes=[1,4,7],
                                               inception_pool_size=3,
                                               inception_dense_dim=(128,16))

        run(parameter_dict, rand)

        elapsed = (time.time() - start)
        print('iteration {} done'.format(count))
        print('Time since starting the loop: {}'.format(timedelta(seconds=int(elapsed))))
        count += 1


if __name__ == '__main__':
    main()
