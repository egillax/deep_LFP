from utils import AverageMeter, calculate_accuracy


def train_epoch(data_loader, model, criterion, optimizer, mlog
                ):
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, data in enumerate(data_loader):
        inputs, labels = data

        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        outputs = model(inputs).cuda()
        loss = criterion(outputs, labels)

        accuracy = calculate_accuracy(outputs, labels)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(accuracy, inputs.size(0))

        # online plotter
        mlog.update_loss(loss, meter='loss')
        mlog.update_meter(outputs, labels, meters={'accuracy', 'map', 'confusion'})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

