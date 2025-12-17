import json
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

##############################
########## CLASSES ###########
##############################

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive Pooling을 통해 Feature map 크기를 (4, 4)로 변환
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
class classification():
    """
    Defines a machine learning classification pipeline using the CNN network.
    """
    def __init__(self, n_channels, n_classes, task, learning_rate, name):
        self.model = CNN(in_channels=n_channels, num_classes=n_classes).cuda()
        self.best_model = None
        self.data_flag = name
        if task == "multi-label, binary-class": 
            self.criterion = nn.BCEWithLogitsLoss()
        else: 
            self.criterion = nn.CrossEntropyLoss()
        self.task = task
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.training_done = False
        self.train_accuracy_per_epoch = []
        self.val_accuracy_per_epoch = []
        self.val_AUC_per_epoch = []
        self.best_val_accuracy = float("-inf")
        self.test_accuracy = None
        self.test_AUC = None

    def train(self, train_loader, val_loader, epochs=3):
        for epoch in range(epochs):
            print(f"===================\nEpoch {epoch}\n")
            train_correct = 0
            train_total = 0
            self.model.train()
            for inputs, targets in tqdm(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs.cuda())
                if self.task == "multi-label, binary-class":
                    targets = targets.to(torch.float32).cuda()
                else:
                    targets = targets.squeeze().long().cuda()
                loss = self.criterion(outputs, targets)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_correct += sum(targets == torch.argmax(outputs, 1))
                train_total += len(outputs)
            acc = train_correct.item() / train_total
            self.train_accuracy_per_epoch.append(acc)
            print(f"train -- accuracy: {round(acc,4)}")
            val_acc, val_AUC = self.test(val_loader, split="val")
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model = self.model.state_dict()
            self.val_accuracy_per_epoch.append(val_acc)
            self.val_AUC_per_epoch.append(val_AUC)
        self.training_done = True
        print("===================")

    def test(self, test_loader, label_names=None, split="test",
             display_confusion_matrix=False, use_best_model=False):
        """
        Runs the test phase for the trained model.
        이때, test 데이터는 "/home/prml/YY/dataset/test/metadata.csv"에 있는 라벨 정보를 사용하는 것으로 가정합니다.
        (test_loader는 해당 경로를 기반으로 생성되어야 합니다.)
        """
        if use_best_model: 
            self.model.load_state_dict(self.best_model)
        self.model.eval()
        y_true = torch.tensor([]).cuda()
        y_score = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs.cuda())
                if self.task == "multi-label, binary-class":
                    targets = targets.to(torch.float32).cuda()
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long().cuda()
                    outputs = outputs.softmax(dim=-1).cuda()
                    targets = targets.float().resize_(len(targets), 1)
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            y_true = y_true.detach().cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            y_preds = np.argmax(y_score, axis=1)
        
        accuracy = accuracy_score(y_true, y_preds)
        try:
            AUC = roc_auc_score(y_true, y_score, multi_class="ovr")
        except Exception as e:
            print("Warning: AUC calculation error:", e)
            AUC = 0.0

        if display_confusion_matrix:
            print(f"{split} -- accuracy: {round(accuracy,4)}, AUC: {round(AUC,4)}")
            cm = confusion_matrix(y_true.tolist(), y_preds.tolist())
            if label_names is None:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            else:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
            plt.figure(figsize=(10,10))
            disp.plot()
            plt.xticks(rotation=90)
            plt.show()
        else:
            print(f"{split} -- accuracy: {round(accuracy,4)}, AUC: {round(AUC,4)}")
        return accuracy, AUC

##############################
######### FUNCTIONS ##########
##############################

def print_accuracy_convergence(training_accs, validation_accs, validation_AUC, test_acc, test_AUC):
    plt.figure(figsize=(10, 5))
    plt.plot(training_accs)
    plt.plot(validation_accs)
    plt.plot(validation_AUC)
    plt.plot(len(training_accs)-1, test_acc, 'ro')
    plt.plot(len(training_accs)-1, test_AUC, '^')
    plt.title("Training & Validation Accuracy per Epoch")
    plt.legend(["Training Acc.", "Validation Acc.", "Validation AUC", 
                "Test Acc. (model with best val. acc.)", "Test AUC (model with best val. acc.)"])
    plt.show()

def run_classifier_pipeline(name, info_flags, imported_data,
                            learning_rate=0.001, epochs=20,
                            DA_technique=""):
    """
    Runs the training and testing process for the classifier declared above.
    Note: Ensure that the test_loader (imported_data[4]) is built using images from
    "/home/prml/YY/dataset/test/images" and labels from "/home/prml/YY/dataset/test/metadata.csv".
    """
    info = info_flags[name][0]
    print(json.dumps(info, sort_keys=False, indent=2))
    clf = classification(
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        task=len(info["task"]),  # 필요에 따라 task 설정을 조정하세요.
        learning_rate=learning_rate, 
        name=name
    )
    print(summary(clf.model, input_size=(info["n_channels"], 224, 224)))
    clf.train(
        train_loader=imported_data[3], 
        val_loader=imported_data[5], 
        epochs=epochs
    )
    clf.test_accuracy, clf.test_AUC = clf.test(
        test_loader=imported_data[4], 
        label_names=info["label"].values(), 
        display_confusion_matrix=True,
        use_best_model=True
    )
    print_accuracy_convergence(clf.train_accuracy_per_epoch, clf.val_accuracy_per_epoch,
                               clf.val_AUC_per_epoch, clf.test_accuracy, clf.test_AUC) 
    if not os.path.exists("trained_models/classifier_wo_data_augmentation/"):
        os.makedirs("trained_models/classifier_wo_data_augmentation/")
    if name not in os.listdir("trained_models/classifier_wo_data_augmentation/"): 
        os.makedirs(f"trained_models/classifier_wo_data_augmentation/{name}")
    torch.save(clf.model.state_dict(),
               f"trained_models/classifier_wo_data_augmentation/{name}/{name}{DA_technique}_epochs{epochs}.pth")
    return clf
