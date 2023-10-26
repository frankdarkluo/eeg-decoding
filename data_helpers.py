from datetime import date
import numpy as np
import config
import h5py
import collections
from feature_extraction import zuco_reader

def sort_dict(data_dict):
    data_dict=collections.OrderedDict(sorted(data_dict.items()))
    return data_dict

def load_data():

    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    eeg_dir = config.eeg_feature_dir
    print("TASK: ", config.class_task)
    print("Extracting", config.feature_set[0], "features....")

    for subject in config.subjects:
        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

    return eeg_dict, eeg_dir, feature_dict, label_dict

def load_matlab_files(task, subject):
    """loads Matlab files depending on which files are required for the chosen classification task"""

    if task.startswith("sentiment"):
        filename_sr = config.rootdir_zuco1 + "results" + subject + "_SR.mat"
        f_sr = h5py.File(filename_sr, 'r')
        sentence_data_sr = f_sr['sentenceData']

        return [(f_sr, sentence_data_sr)]

    elif task.startswith('reldetect'):
        loaded_matlab_data = []
        if subject.startswith('Z'):  # subjects from ZuCo 1
            # todo: add Sentiment data?
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        elif subject.startswith('Y'):  # subjects from ZuCo 1
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        else:
            print("UNDEFINED SUBJECT NAME")

        return [(f_nr, sentence_data_nr)]

    elif task.startswith('ner'):
        loaded_matlab_data = []
        if subject.startswith('Z'):  # subjects from ZuCo 1
            # load NR + sentiment task from ZuCo 1
            filename_nr = config.rootdir_zuco1 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
            filename_sr = config.rootdir_zuco1 + "results" + subject + "_SR.mat"
            f_sr = h5py.File(filename_sr, 'r')
            sentence_data_sr = f_sr['sentenceData']
            loaded_matlab_data.append((f_sr, sentence_data_sr))
        elif subject.startswith('Y'):  # subjects from ZuCo 1
            # load NR task from ZuCo 2
            filename_nr = config.rootdir_zuco2 + "results" + subject + "_NR.mat"
            f_nr = h5py.File(filename_nr, 'r')
            sentence_data_nr = f_nr['sentenceData']
            loaded_matlab_data.append((f_nr, sentence_data_nr))
        else:
            print("UNDEFINED SUBJECT NAME")

        return loaded_matlab_data


def save_results(fold_results_dict, task):
    """aggegates the results in fold_results_dict over all folds and
    saves hyper-parameters and results to a result file"""

    if config.class_task.startswith("sentiment"):
        base_path = 'sentiment/results'
    elif config.class_task == "ner":
        base_path = 'ner/results'
    elif config.class_task == "reldetect":
        base_path = 'reldetect/results'

    result_file_path = f"{base_path}/{date.today()}_results_{task}_{'-'.join(config.feature_set)}-{config.embeddings}.txt"
    result_file = open(result_file_path, 'w')

    # print header
    #if config.model is 'cnn':
     #   print("lstm_dim", "lstm_layers", "dense_dim", "dropout", "batch_size", "epochs", "lr", "embedding_type", "random_seed", "train_acc", "val_acc",
      #      "test_acc", "test_std", "avg_precision", "std_precision", "avg_recall", "std_recall", "avg_fscore", "std_fscore", "threshold", "folds",
       #     "training_time", 'best_ep', 'patience', 'min_delta', "model", "model_type", "inception_filters", "inception_kernel_sizes",
        #    "inception_pool_size", "inception_dense_dim", "data_percentage", file=result_file)
    #else:
     #   print("lstm_dim", "lstm_layers", "dense_dim", "dropout", "batch_size", "epochs", "lr", "embedding_type", "random_seed", "train_acc", "val_acc",
      #      "test_acc", "test_std", "avg_precision", "std_precision", "avg_recall", "std_recall", "avg_fscore", "std_fscore", "threshold", "folds",
       #     "training_time", 'best_ep', 'patience', 'min_delta', "model", "model_type", file=result_file)

    # training scores
    train_acc = np.mean([ep[-1] for ep in fold_results_dict['train-accuracy']])

    # validation scores
    val_acc = np.mean([ep[-1] for ep in fold_results_dict['val-accuracy']])

    # test scores
    avg_accuracy = np.mean(fold_results_dict['test-accuracy'])
    avg_precision = np.mean(fold_results_dict['precision'])
    avg_recall = np.mean(fold_results_dict['recall'])
    avg_fscore = np.mean(fold_results_dict['fscore'])
    std_accuracy = np.std(fold_results_dict['test-accuracy'])
    std_precision = np.std(fold_results_dict['precision'])
    std_recall = np.std(fold_results_dict['recall'])
    std_fscore = np.std(fold_results_dict['fscore'])
    threshold = fold_results_dict['threshold'] if 'threshold' in fold_results_dict else "-"
    best_eps = ",".join(map(str, fold_results_dict['best-e']))
    folds = config.folds
    model_type = config.model
    data_percentage = fold_results_dict['data_percentage']

    if config.model is 'cnn':
        inception_filters = fold_results_dict['inception_filters']
        inception_kernel_sizes = fold_results_dict['inception_kernel_sizes']
        inception_pool_size = fold_results_dict['inception_pool_size']
        inception_dense_dim = fold_results_dict['inception_dense_dim']


        inception_kernel_sizes = str(inception_kernel_sizes).replace(' ', '')
        inception_dense_dim = str(inception_dense_dim).replace(' ', '')

        print(" ".join(map(str, fold_results_dict['params'])),train_acc, val_acc, avg_accuracy, std_accuracy, avg_precision,
          std_precision, avg_recall, std_recall, avg_fscore, std_fscore, threshold, folds, fold_results_dict['training_time'], 
          best_eps, fold_results_dict['patience'], fold_results_dict['min_delta'], fold_results_dict['model'][-1], 
          model_type, inception_filters, inception_kernel_sizes, inception_pool_size, inception_dense_dim, data_percentage, file=result_file)
    
    else:
        print(" ".join(map(str, fold_results_dict['params'])),train_acc, val_acc, avg_accuracy, std_accuracy, avg_precision,
            std_precision, avg_recall, std_recall, avg_fscore, std_fscore, threshold, folds, fold_results_dict['training_time'], 
            best_eps, fold_results_dict['patience'], fold_results_dict['min_delta'], fold_results_dict['model'][-1], model_type, data_percentage, file=result_file)



