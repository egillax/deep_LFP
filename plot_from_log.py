import pickle
import numpy as np


def load_log_file(log_file):
    objects = []
    with (open(log_file, 'rb')) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    test_results = objects[0]['results'][0][len(objects[0]['results'][0])-1]
    log_data = objects[0]['results'][0][:len(objects[0]['results'][0])-1]
    train_results = log_data[::2]
    val_results = log_data[1::2]

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_map = []
    val_map = []
    train_confusion = np.empty((4, 4, len(train_results)))
    val_confusion = np.empty((4, 4, len(val_results)))
    for epoch in range(len(train_results)):
        train_loss.append(train_results[epoch]['Train']['loss'])
        val_loss.append(val_results[epoch]['Validation']['loss'])
        train_accuracy.append(train_results[epoch]['Train']['accuracy'])
        val_accuracy.append(val_results[epoch]['Validation']['accuracy'])
        train_map.append(train_results[epoch]['Train']['map'].numpy())
        val_map.append(val_results[epoch]['Validation']['map'].numpy())
        train_confusion[:, :, epoch] = train_results[epoch]['Train']['confusion']
        val_confusion[:, :, epoch] = val_results[epoch]['Validation']['confusion']

    train_data = {'loss': train_loss, 'accuracy': train_accuracy, 'map': train_map, 'confusion': train_confusion}
    val_data = {'loss': val_loss, 'accuracy': val_accuracy, 'map': val_map, 'confusion': val_confusion}
    test_data = {'accuracy': test_results['Test'].cpu.numpy()}

    return train_data, val_data, test_data

def main():
    log_file = '/data/eaxfjord/deep_LFP/logs/state_prediction_3sec_transfer/log'
    train, val, test = load_log_file(log_file)


if __name__ == '__main__':
    main()