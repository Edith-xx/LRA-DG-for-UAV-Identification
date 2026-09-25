import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from models.Proposed import Extractor
from Classifier.Proposed import Classifier
from utils import *
import numpy as np
from dataloader import *
from contrastive_loss import *
import random
#from confusion_matrix import confusion
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置pytorch内置hash函数种子，保证不同运行环境下字典等数据结构的哈希结果一致
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False禁用
class Config:
    def __init__(
            self,
            batch_size: int = 32,
            test_batch_size: int = 8,
            epochs: int = 100,
            lr: float = 0.001,
            save_path: str = 'model_weight/SDG_Extractor.pth',
            save_path_fc: str = 'model_weight/SDG_Classifier.pth',
            device_num: int = 0,
            rand_num: int = 30,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.save_path_fc = save_path_fc
        self.device_num = device_num
        self.rand_num = rand_num
def coral(features_s, features_e, labels_s, labels_e):
    unique_labels = torch.unique(torch.cat([labels_s, labels_e]))
    total_loss = 0.0
    count = 0
    for label in unique_labels:
        mask_s = (labels_s == label)
        mask_e = (labels_e == label)
        feat_s = features_s[mask_s]
        feat_e = features_e[mask_e]
        if len(feat_s) < 2 or len(feat_e) < 2:
            continue
        mean_s = feat_s.mean(0, keepdim=True)
        mean_e = feat_e.mean(0, keepdim=True)
        cent_s = feat_s - mean_s
        cent_e = feat_e - mean_e
        cov_s = (cent_s.t() @ cent_s) / (len(feat_s) - 1)
        cov_e = (cent_e.t() @ cent_e) / (len(feat_e) - 1)
        mean_diff = (mean_s - mean_e).pow(2).mean()
        cov_diff = (cov_s - cov_e).pow(2).mean()
        total_loss += mean_diff + cov_diff
        count += 1
    return total_loss / (count + 1e-8)
'''
def coral(x, y):
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()
    return mean_diff + cova_diff
'''
def train(extractor, classifier, criterion, train_dataloader, optimizer1, optimizer2, epoch, writer, device_num):
    extractor.train()
    classifier.train()
    device = torch.device("cuda:" + str(device_num))
    correct1 = 0
    correct = 0
    running_loss = 0.0
    omega1 = 0.001
    omega2 = 0.001
    con_criterion = SupConLoss()
    for data_nnl in train_dataloader:
        data, target, target1 = data_nnl
        target = target.squeeze().long()
        target1 = target1.squeeze().long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
            target1 = target1.to(device)
        features1, features2, SD_features, SD1_features = extractor(data)
        zsrc = torch.cat([SD_features.unsqueeze(1), SD1_features.unsqueeze(1)], dim=1)
        conloss1 = con_criterion(zsrc, target)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output, output1 = classifier(features1)#先输出的是个体分类
        output = F.log_softmax(output, dim=1)
        output1 = F.log_softmax(output1, dim=1)
        loss1 = criterion(output, target)
        loss2 = criterion(output1, target1)
        cls_loss = loss1 + loss2
        coral_loss = coral(features1, features2, target, target)#类别和个体都试试
        total_loss = cls_loss + omega1*coral_loss + omega2*conloss1
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
        running_loss += total_loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        pred1 = output1.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
    mean_loss = running_loss / len(train_dataloader)  # len(t1)表示t1数据集的批次数量
    mean_accuracy = 100 * correct / len(train_dataloader.dataset)  # 所有样本中预测正确的数量除以总的样本个数
    mean_accuracy1 = 100 * correct1 / len(train_dataloader.dataset) # 所有样本中预测正确的数量除以总的样本个数
    print(
        'Train Epoch: {} \tLoss: {:.6f}, Accuracy_devices: {}/{} ({:0f}%), Accuracy_classes: {}/{} ({:0f}%)\n'.format(
            epoch,
            mean_loss,
            correct,
            len(train_dataloader.dataset),
            mean_accuracy,
            correct1,
            len(train_dataloader.dataset),
            mean_accuracy1,
        )
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Accuracy1/train', 100.0 * correct1 / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', mean_loss, epoch)  # 用于训练准确率和分类器损失的可视化
    return mean_loss, mean_accuracy, mean_accuracy1

def evaluate(extractor, classifier, criterion, val_dataloader, epoch, writer, device_num):
    extractor.eval()
    classifier.eval()
    val_loss = 0
    val_loss1 = 0
    correct = 0
    correct1 = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target, target1 in val_dataloader:
            target = target.squeeze().long()
            target1 = target1.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                target1 = target1.to(device)
            features,_,_,_ = extractor(data)
            output, output1 = classifier(features)
            output = F.log_softmax(output, dim=1)
            output1 = F.log_softmax(output1, dim=1)
            val_loss += criterion(output, target).item()
            val_loss1 += criterion(output1, target1).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred1 = output1.argmax(dim=1, keepdim=True)
            correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
        val_loss /= len(val_dataloader)
        val_loss1 /= len(val_dataloader)
        mean_accuracy = 100 * correct / len(val_dataloader.dataset)
        mean_accuracy1 = 100 * correct1 / len(val_dataloader.dataset)
        fmt = '\nValidation set: loss_devices: {:.4f}, loss_classes: {:.4f}, Accuracy_devices: {}/{} ({:0f}%), Accuracy_classes: {}/{} ({:0f}%)\n'
        print(
            fmt.format(
                val_loss,
                val_loss1,
                correct,
                len(val_dataloader.dataset),
                mean_accuracy,
                correct1,
                len(val_dataloader.dataset),
                mean_accuracy1,
            )
        )
        accuracy = 100.0 * correct / len(val_dataloader.dataset)
        accuracy1 = 100.0 * correct1 / len(val_dataloader.dataset)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        writer.add_scalar('Accuracy1/validation', accuracy1, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        return val_loss, val_loss1, accuracy, accuracy1

def test(extractor, classifier, test_dataloader):
    extractor.eval()
    classifier.eval()
    correct = 0
    correct1 = 0
    pred_classes = []
    pred_devices = []
    real_classes = []
    real_devices = []
    with torch.no_grad():
        for data, target, target1 in test_dataloader:
            target = target.long()
            target1 = target1.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target1 = target1.cuda()
            features,_,_,_ = extractor(data)
            output, output1 = classifier(features)
            output = F.log_softmax(output, dim=1)
            output1 = F.log_softmax(output1, dim=1)
            pred = output.argmax(dim=1, keepdim=True)  # 预测结果
            pred1 = output1.argmax(dim=1, keepdim=True)
            correct_mask = pred.eq(target.view_as(pred))  # 创建一个正确分类的掩码
            correct1_mask = pred1.eq(target1.view_as(pred1))
            correct += correct_mask.sum().item()
            correct1 += correct1_mask.sum().item()
            pred_devices[len(pred_devices):len(target) - 1] = pred.tolist()  # 切片赋值的方式允许我们替换现有列表中的特定范围，以及将其他列表的元素添加到已有列表中的特定位置
            real_devices[len(real_devices):len(target) - 1] = target.tolist()  # target是真实标签的列表，target_pred是预测结果的列表
            pred_classes[len(pred_classes):len(target1) - 1] = pred1.tolist()  # 切片赋值的方式允许我们替换现有列表中的特定范围，以及将其他列表的元素添加到已有列表中的特定位置
            real_classes[len(real_classes):len(target1) - 1] = target1.tolist()  # target是真实标签的列表，target_pred是预测结果的列表
        pred_devices = np.array(pred_devices)
        pred_classes = np.array(pred_classes)
        real_devices = np.array(real_devices)
        real_classes = np.array(real_classes)
    accuracy = correct / len(test_dataloader.dataset)
    accuracy1 = correct1 / len(test_dataloader.dataset)
    print("Accuracy_devices:", accuracy)
    print("Accuracy_classes:", accuracy1)
    return pred_classes, real_classes, pred_devices, real_devices

def train_and_evaluate(extractor, classifier, loss_function, train_dataloader, val_dataloader, optimizer1, optimizer2, epochs, writer, save_path_extractor, save_path_classifier, device_num):
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []
    current_max_val_accuracy = 1  # 初始化当前最小的测试损失为一个较大的值，判断模型是否有改进
    current_loss = 50
    time_start1 = time.time()
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        train_loss, train_acc1, train_acc2 = train(extractor, classifier, loss_function, train_dataloader, optimizer1, optimizer2, epoch, writer, device_num)
        train_losses.append(train_loss)
        train_accies.append(train_acc1)
        val_loss, val_loss1, val_accuracy, val_accuracy1 = evaluate(extractor, classifier, loss_function, val_dataloader, epoch, writer, device_num)
        val_loss = val_loss + val_loss1
        val_losses.append(val_loss)
        val_accies.append(val_accuracy)
        val_accuracy = val_accuracy + val_accuracy1
        if val_accuracy > current_max_val_accuracy or val_loss < current_loss:
            print("Model improved: ", end="")
            if val_accuracy > current_max_val_accuracy:
                print("Accuracy increased from {:.4f} to {:.4f}".format(current_max_val_accuracy, val_accuracy),
                      end="; ")
                current_max_val_accuracy = val_accuracy
            if val_loss < current_loss:
                print("Loss decreased from {:.6f} to {:.6f}".format(current_loss, val_loss), end="; ")
                current_loss = val_loss
            print("\nNew model weight is saved.")
            torch.save(extractor, save_path_extractor)
            torch.save(classifier, save_path_classifier)
        else:
            print("No improvement: Accuracy = {:.4f}, Loss = {:.6f}".format(val_accuracy, val_loss))
        time_end = time.time()
        time_sum = time_end - time_start
        print("time for each epoch is: %s" % time_sum)
        print("------------------------------------------------")
        torch.cuda.empty_cache()  # 每轮训练结束清空未使用的显存
    time_end1 = time.time()
    Ave_epoch_time = (time_end1 - time_start1) / epochs
    print("Avgtime for each epoch is: %s" % Ave_epoch_time)
    return train_losses, train_accies, val_losses, val_accies

if __name__ == '__main__':
    conf = Config()
    writer = SummaryWriter("logs")
    device = torch.device("cuda:" + str(conf.device_num))
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    run_for = 'Train'
    if run_for == 'Train':
        X_train, X_val, Y_train, Y_val, Class_train, Class_val = read_train_data()
        X_test, _, Y_test, _, Class_test, _ = read_test_data()
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(Class_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val), torch.Tensor(Class_val))
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test), torch.Tensor(Class_test))
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, drop_last=True)#drop_last为True则丢失最后一个不完整的批次
        val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset)
        extractor = Extractor().to(device)
        classifier = Classifier().to(device)
        print('-----------------------')
        print(extractor)
        loss = nn.NLLLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)
        optim_extractor = torch.optim.Adam(extractor.parameters(), lr=conf.lr, weight_decay=0)
        optim_classifier = torch.optim.Adam(classifier.parameters(), lr=conf.lr, weight_decay=0)
        time_start = time.time()
        train_losses, train_accies, val_losses, val_accies = train_and_evaluate(extractor,
                                                                                classifier,
                                                                                loss_function=loss,
                                                                                train_dataloader=train_dataloader,
                                                                                val_dataloader=val_dataloader,
                                                                                optimizer1=optim_extractor, optimizer2=optim_classifier,
                                                                                epochs=conf.epochs,
                                                                                writer=writer, save_path_extractor=conf.save_path,
                                                                                save_path_classifier=conf.save_path_fc,
                                                                                device_num=conf.device_num,)
        time_end = time.time()
        time_sum = time_end - time_start
        print("total training time is: %s" % time_sum)
        extractor = torch.load('model_weight/SDG_Extractor.pth')
        classifier = torch.load('model_weight/SDG_Classifier.pth')
        pred_classes, real_classes, pred_devices, real_devices = test(extractor, classifier, test_dataloader)



