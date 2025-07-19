import torch
from load_from_clip import load_model, encode_img
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

#four different classes
BREAST  = 0
OTHER = 1
ABDOMEN = 2
TYROID = 3

#anatomical classifier 
class multiClassifier(torch.nn.Module):

    def __init__(self, init_mode):
    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        INPUT_DIM = 512
        HIDDEN_NUM = 50
        OUTPUT_NUM = 4


        super().__init__()
        self.lin1 = torch.nn.Linear(INPUT_DIM, HIDDEN_NUM)
        self.lin2 = torch.nn.Linear(HIDDEN_NUM, 30)
        self.lin3 = torch.nn.Linear(30, OUTPUT_NUM)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        if init_mode == "xavier":
            torch.nn.init.xavier_uniform(self.lin1.weight)
            torch.nn.init.xavier_uniform(self.lin2.weight)
            torch.nn.init.xavier_uniform(self.lin3.weight)
        elif init_mode == "kaiming":
            torch.nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='sigmoid')
            torch.nn.init.kaiming_normal_(self.lin2.weight, mode='fan_in', nonlinearity='sigmoid')
            torch.nn.init.kaiming_normal_(self.lin3.weight, mode='fan_in', nonlinearity='sigmoid')
        elif init_mode == "orthogonal":
            torch.nn.init.orthogonal_(self.lin1.weight)
            torch.nn.init.orthogonal_(self.lin2.weight)
            torch.nn.init.orthogonal_(self.lin3.weight)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.log_softmax(x)
        return x

def load_dataset(path : str, label: int, image_path: str, model):
    X_train = []
    Y_train = []
    with open(path, "r", encoding="utf-8") as file:
        for f in file:
            Y_train.append(label)

            correct_line = json.loads(f) 
            new_path = image_path + correct_line["modalities"][0]['value']

            X_train.append(encode_img(model, new_path))
            
    return (torch.stack(X_train), torch.tensor(Y_train))

class BodyPartsDataset(Dataset):
    
    def __init__(self, model, model_name, load, path):
        if not load:
            BUSI = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-breast-radiopedia-2.jsonl", BREAST, "", model))
            CAMUS = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-heart-radiopedia-2.jsonl", OTHER, "", model))
            COVIDUS = load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-lungs-radiopedia-2.jsonl", OTHER, "", model)
            CT2 = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-abdomen-radiopedia-2.jsonl", ABDOMEN, "", model))
            DDTI = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-thyroid-radiopedia-2.jsonl", TYROID, "", model))

            self.data = torch.cat([BUSI[0], CAMUS[0], COVIDUS[0], CT2[0], DDTI[0]], dim=0)
            self.labels = torch.cat([BUSI[1], CAMUS[1], COVIDUS[1], CT2[1], DDTI[1]], dim=0)

            torch.save(self.data, "data_emb_" + model_name + ".pt")
            torch.save(self.labels, "data_lab_" + model_name + ".pt")
        else:
            self.data = torch.load(path + "/data_emb_"+ model_name +".pt")
            self.labels = torch.load(path + "/data_emb_"+ model_name + ".pt")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class BodyPartsDatasetTEST(Dataset):
    
    def __init__(self, model, model_name, load, save_path):
        if not load:
            BUSI = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-breast-radiopedia-2_test.jsonl", BREAST, "", model))
            CAMUS = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-heart-radiopedia-2_test.jsonl", OTHER, "", model))
            COVIDUS = load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-lungs-radiopedia-2_test.jsonl", OTHER, "", model)
            CT2 = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-abdomen-radiopedia-2_test.jsonl", ABDOMEN, "", model))
            DDTI = (load_dataset("/mloscratch/users/deschryv/clipFineTune/ultrasound_evaluation/classifier-thyroid-radiopedia-2_test.jsonl", TYROID, "", model))

            self.data = torch.cat([BUSI[0], CAMUS[0], COVIDUS[0], CT2[0], DDTI[0]], dim=0)
            self.labels = torch.cat([BUSI[1], CAMUS[1], COVIDUS[1], CT2[1], DDTI[1]], dim=0)

            torch.save(self.data, save_path + "/data_emb_test_"+ model_name + ".pt")
            torch.save(self.labels, save_path + "/data_lab_test_" + model_name + ".pt")
        else:
            self.data = torch.load(save_path + "/data_emb_test_"+ model_name +".pt")
            self.labels = torch.load(save_path + "/data_lab_test_"+ model_name +".pt")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_1(seed, init_method, data_loader, train_dataset): 

    torch.manual_seed(seed)
    model = multiClassifier(init_method)
    labelsWEIGHT = np.array(train_dataset.labels)
    labelsWEIGHT = np.unique(labelsWEIGHT.astype(int))
    print(labelsWEIGHT)

    class_weights = compute_class_weight(class_weight='balanced', classes=labelsWEIGHT, y=np.array(train_dataset.labels))
    #class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    weights = torch.tensor(class_weights, dtype=torch.float)

    loss = torch.nn.CrossEntropyLoss(weight=weights)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N_EPOCH = 1000
    losses = []
    model.train()
    for epoch in range(N_EPOCH):

        epoch_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, lab = data
            optimizer.zero_grad()
            y = model(inputs)

            l = loss(y, lab)
            l.backward()
        
            optimizer.step()
            epoch_loss += l.item()

        average_loss = epoch_loss / len(data_loader)
        losses.append(average_loss)

    return average_loss, model

#trains the neural classifier with different initialization and choose the one with the lowest train loss
def multiple_train(data_loader, train_dataset):
    models = [train_1(41, "xavier",data_loader, train_dataset), train_1(14, "kaiming", data_loader, train_dataset), train_1(5, "orthogonal", data_loader, train_dataset)]
    average_loss_l = [models[0][0], models[1][0], models[2][0]]
    ind = average_loss_l.index(min(average_loss_l))
    print("the best model has a train loss of : " + str(average_loss_l[ind]))
    best_model_1 = models[ind][1]
    return best_model_1

#evaluates the neural classifier @model on the test dataset @test_loader
def evaluate(model, test_loader):
    model.eval()
    total = 0
    correct = 0

    BATCH_SIZE = 64

    breast_count = 0
    breast_good = 0
    other_count = 0
    other_good = 0
    abdomen_count = 0
    abdomen_good = 0
    tyroid_count = 0
    tyroid_good = 0


    for input, label in test_loader:
        output = model(input)
        y_test_hat = torch.tensor(torch.max(output.data, 1).indices.numpy())
        
        total += len(y_test_hat)
        for i in range(len(y_test_hat)):
            if label[i] == BREAST:
                breast_count += 1
                if y_test_hat[i] == BREAST:
                    breast_good += 1
                    
            if label[i] == OTHER:
                other_count += 1
                if y_test_hat[i] == OTHER:
                    other_good += 1
            if label[i] == ABDOMEN:
                abdomen_count += 1
                if y_test_hat[i] == ABDOMEN:
                    abdomen_good += 1
            if label[i] == TYROID:
                tyroid_count += 1
                if y_test_hat[i] == TYROID:
                    tyroid_good += 1
            
            if y_test_hat[i] == label[i]:
                correct += 1
        
        
    print("total : " + str(total) + " and correct : " + str(correct) + " Accuracy : " + f"{(correct/total)*100:0,.2f}"+"%")
    print("BREAST number of samples: " + str(breast_count) + " Accuracy : " + f"{(breast_good/breast_count)*100:0,.2f}"+"%")
    print("OTHER number of samples: " + str(other_count) +" Accuracy : " + f"{(other_good/other_count)*100:0,.2f}"+"%")
    print("ABDOMEN number of samples:  "+ str(abdomen_count) +" Accuracy : " + f"{(abdomen_good/abdomen_count)*100:0,.2f}"+"%")
    print("TYROID number of samples: "+ str(tyroid_count) +" Accuracy : " + f"{(tyroid_good/tyroid_count)*100:0,.2f}"+"%")

#evaluates tho @model_name image encoder stored in @model_path
def evaluate_pipeline(model_path, model_name):
    evaluated_clip = load_model(model_path)
    #put LOAD to True if the embeddings have already been computed
    LOAD = False
    SAVE_PATH = ""
    train_dataset = BodyPartsDataset(evaluated_clip, model_name, LOAD, SAVE_PATH)
    classifier = multiple_train(DataLoader(dataset=train_dataset, batch_size=64), train_dataset)
    test_dataset = BodyPartsDatasetTEST(evaluated_clip, model_name, False, SAVE_PATH)
    evaluate(classifier, DataLoader(dataset=test_dataset, batch_size=64))

def main():

    #evaluated image encoders
    clips = [(model, "/mloscratch/users/deschryv/models/" + model) for model in os.listdir("/mloscratch/users/deschryv/models") if model.startswith("us-2")]

    model_names = [clips[i][0] for i in range(len(clips))]
    model_paths = [clips[i][1] for i in range(len(clips))]

    for encoder in range(len(model_paths)):
        print("evaluation of the model : " + model_names[encoder])
        evaluate_pipeline(model_paths[encoder], model_names[encoder])

if __name__ == "__main__":
    main()