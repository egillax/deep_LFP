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
        mlog.update_meter(outputs, labels, meters={'accuracy', 'confusion'})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

        # online plotter
        mlog.update_loss(loss, meter='loss')
        mlog.update_meter(outputs, labels, meters={'accuracy', 'confusion'})

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(accuracy, inputs.size(0))

    return losses.avg


def test_epoch(data_loader, model, criterion, mlog):
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

    return losses.avg, accuracies.avg