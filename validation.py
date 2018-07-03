from utils import AverageMeter, calculate_accuracy


def val_epoch(data_loader, model, criterion, mlog):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, data in enumerate(data_loader):
        inputs, labels = data
        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy = calculate_accuracy(outputs, labels)

        # online ploter
        mlog.update_loss(loss, meter='loss')
        mlog.update_meter(outputs, labels, meters={'accuracy', 'map', 'confusion'})

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(accuracy, inputs.size(0))

    return losses.avg

