import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchmetrics import PearsonCorrCoef


def create_hierarchical_penalty_matrix(num_classes, thegma):
    """
    Create a matrix of size num_classes x num_classes where each entry (i, j)
    contains the hierarchical penalty for predicting class j when the true class is i.
    """
    # Generate a grid of label indices
    indices = torch.arange(num_classes).unsqueeze(0)
    # Calculate the absolute difference between indices and transpose
    absolute_differences = torch.abs(indices - indices.T)
    # Calculate the hierarchical penalty matrix
    penalty_matrix = torch.exp(-thegma * absolute_differences)
    return penalty_matrix


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, thegma):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta  # 新增参数beta用于调节两个loss的权重
        self.thegma = thegma 
        # 预计算罚分矩阵
        self.penalty_matrix = create_hierarchical_penalty_matrix(num_classes, thegma)

    def forward(self, logits, targets):
        # 确保罚分矩阵与logits在同一个设备上
        self.penalty_matrix = self.penalty_matrix.to(logits.device)
        
        # 计算标准的交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')

        # 根据真实类别收集每个预测对应的罚分
        penalties = self.penalty_matrix[targets, :]
        
        # 应用罚分到log概率上
        log_probs = F.log_softmax(logits, dim=1)
        weighted_log_probs = penalties * log_probs
        
        # 计算最终的加权log概率损失
        structured_contrastive_loss = -torch.sum(weighted_log_probs, dim=1).mean()

        # 计算Pearson相关损失
        logits_flat = logits.view(-1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_flat = targets_one_hot.view(-1)
        
        logits_mean = logits_flat.mean()
        targets_mean = targets_flat.mean()
        
        logits_centered = logits_flat - logits_mean
        targets_centered = targets_flat - targets_mean
        
        correlation = torch.sum(logits_centered * targets_centered) / (torch.sqrt(torch.sum(logits_centered ** 2)) * torch.sqrt(torch.sum(targets_centered ** 2)))
        pearson_loss = -correlation

        # 结合两种损失
        combined_loss = self.alpha*ce_loss + self.thegma * structured_contrastive_loss + self.beta * pearson_loss

        return combined_loss

df = pd.read_csv("/root/trac2_CONVT_train_p.csv",encoding='ISO-8859-1')
df_dev = pd.read_csv("/root/trac2_CONVT_dev.csv",encoding='ISO-8859-1')

# EmotionClass:
emotion_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75,3.5,4.5,5.5]
emotion_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# EmotionalPolarityClass:
emotionalPolarity_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 3]
emotionalPolarity_groups = [0, 1, 2, 3,4]

# EmpathyClass:
empathy_bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75,3.25,5.5]
empathy_groups = [0, 1, 2,3,4,5, 6,7]

# turn discrete values into classes
def value_to_class(bins, groups, df_name, column_name):
    class_col = column_name + "Class"
    df_name[class_col] = pd.cut(df_name[column_name], bins, labels=groups)


# train, get class labels
value_to_class(emotion_bins, emotion_groups, df, 'Emotion')
value_to_class(emotionalPolarity_bins, emotionalPolarity_groups, df, 'EmotionalPolarity')
value_to_class(empathy_bins, empathy_groups, df, 'Empathy')
# dev, get class labels
value_to_class(emotion_bins, emotion_groups, df_dev, 'Emotion')
value_to_class(emotionalPolarity_bins, emotionalPolarity_groups, df_dev, 'EmotionalPolarity')
value_to_class(empathy_bins, empathy_groups, df_dev, 'Empathy')


# online loading:
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# local loading:
# random the order of samples
random.seed(42)
df = df.sample(frac=1).reset_index(drop=True)
df_dev = df_dev.sample(frac=1).reset_index(drop=True)


# # 数据集中1为正面，0为反面
class Task2Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['EmotionClass']
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


#处理数据不平衡
from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np
import pandas as pd
from collections import Counter 

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Assuming 'input_ids' and 'attention_mask' are keys in your encodings
        # print("Length of input_ids:", len(self.encodings['input_ids']))
        # print("Length of attention_mask:", len(self.encodings['attention_mask']))
        # print("Length of labels:", len(self.labels))
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        label = self.labels[idx]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)


def data_balanced(df,tokenizer):

    text = df['text'].tolist()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    x_train_bert = encoding['input_ids']
    original_attention_mask = encoding['attention_mask']
    # x_train_bert = x_train_bert.reshape(-1, 1)

    y_train = df['EmpathyClass']
    print(f"原始数据类别分布：{Counter(y_train)}")
    if y_train.dtype == float:
        y_train = y_train.astype(int)

    smote = SMOTE(random_state=42)
    ros = RandomOverSampler(random_state=42)

    try:
        smote = SMOTE(k_neighbors=1)
        X_train_res, y_train_res = smote.fit_resample(x_train_bert, y_train)
        # attention_mask_res = original_attention_mask[smote.sample_indices_]
        attention_mask_res = np.ones_like(X_train_res)

        print('SMOTE后的train数据类别分布:', Counter(y_train_res))

    except ValueError as e:
        print("SMOTE 错误:", e)
        X_train_res, y_train_res = ros.fit_resample(x_train_bert, y_train)
        attention_mask_res = original_attention_mask[ros.sample_indices_]
        print('RandomOverSampler后的train数据类别分布:', Counter(y_train_res))
        
    dataset = TextDataset(encodings={'input_ids': X_train_res, 'attention_mask': attention_mask_res}, labels= y_train_res)
    # Check if __len__ is callable and returns the expected length
    return dataset

def evaluate(model,dev_loader):
    model.eval()
    total_eval_accuracy = 0
    y_pred = torch.tensor([0, 0]).to(device)
    y_truth = torch.tensor([0, 0]).to(device)
    for batch in tqdm(dev_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        y_pred = torch.cat((y_pred, preds), dim=0)
        y_truth = torch.cat((y_truth, labels), dim=0)

        accuracy = (preds == labels).float().mean()
        total_eval_accuracy += accuracy.item()

    pearson_corr = pearson(y_pred.to(torch.float), y_truth.to(torch.float))
    average_eval_accuracy = total_eval_accuracy / len(dev_loader)
    result = pearson_corr.item()
    if  result > 0.63:
        preds_np = y_truth.detach().cpu().numpy()
        # 创建 DataFrame
        preds_df = pd.DataFrame(preds_np)
        # 保存为 CSV 文件
        preds_df.to_csv('/root/Emotion_y_truth.csv', index=False)
        print("y_truth have been saved to 'Emotion_y_truth.csv'")
        
        # 如果 logits 在 GPU 上，确保转移到 CPU 并转换为 numpy 数组
        logits_np = y_pred.detach().cpu().numpy()
        # 创建 DataFrame
        logits_df = pd.DataFrame(logits_np)
        # 保存为 TSV 文件
        logits_df.to_csv('Emotion'+str(result)+'y_pred.csv', sep='\t', index=False)
        print("y_pred have been saved to 'Emotion'"+str(result)+'y_pred.csv')

    return average_eval_accuracy, pearson_corr.item()

epochs = 2

def train_boosted_model(model,train_loader,dev_loader,name):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch %1 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                total_preds = torch.tensor([0, 0]).to(device)
                y_truth = torch.tensor([0, 0]).to(device)
                for batch in tqdm(dev_loader, desc="Evaluating"):
                    batch_inputs = batch['input_ids'].to(device)
                    batch_masks = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(batch_inputs, attention_mask=batch_masks)
                    predicted = torch.argmax(outputs.logits, dim=-1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_preds = torch.cat((total_preds, predicted), dim=0)

                    y_truth = torch.cat((y_truth, labels), dim=0)
            accuracy = correct / total
            pearson_corr = pearson(total_preds.to(torch.float), y_truth.to(torch.float))
            result = pearson_corr.item()

            print(f"Model :{name}, epoch {epoch+1},total_loss is :{total_loss}, Accuracy: {accuracy},Pearson is:{result}")
    return model, accuracy

        # Evaluate the model and update weights

# Function to predict with a boosted ensemble
def predict_with_boosting(model,dev_loader):
    model.eval()
    model_predictions = torch.tensor([0, 0]).to(device)
    y_truth = torch.tensor([0, 0]).to(device)
    for batch in tqdm(dev_loader, desc="Evaluating"):
        batch_inputs = batch['input_ids'].to(device)
        batch_masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(batch_inputs, attention_mask=batch_masks)
            preds = torch.argmax(outputs.logits, dim=-1)
        y_truth = torch.cat((y_truth, labels), dim=0)
        model_predictions = torch.cat((model_predictions, preds), dim=0)
    return  model_predictions,y_truth # Weighted average

from transformers import AutoModelForSequenceClassification, AutoTokenizer

modelsname =['/root/autodl-tmp/model/deberta_v3_base','autodl-tmp/model/roberta-base','/root/bert-base-uncased']
NUM_MODELS = len(modelsname)
models = []
model_weights = [1.0] * NUM_MODELS
model_index =0
for name in modelsname:
    if name == 'autodl-tmp/model/roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(name)
        model = RobertaForSequenceClassification.from_pretrained(name, num_labels=len(emotion_groups))
    elif name == '/root/bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(name)
        model = BertForSequenceClassification.from_pretrained(name, num_labels=len(emotion_groups))
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(emotion_groups))
    
    dev_dataset = Task2Dataset(df[:], tokenizer)
    #df_train_resampled = data_balanced(df,tokenizer)
    train_loader = DataLoader(dev_dataset, batch_size=128, shuffle=True)
    dev_dataset = Task2Dataset(df_dev[:], tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
    # params
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # 使用多个GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pearson = PearsonCorrCoef().to(device)
    num_classes = [9,5,8]
    loss_func = CombinedLoss(num_classes=num_classes[0], alpha=1, beta=1,thegma =1)
    model, accuracy = train_boosted_model(model,train_loader,dev_loader,name)
    models.append(model)
    model_weights[model_index] = accuracy
    model_index =model_index+1

model_i =0 #model index
total_predictions = None
for name in modelsname:
    if name == 'autodl-tmp/model/roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(name)
    elif name == '/root/bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)

    dev_dataset = Task2Dataset(df_dev[:], tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
    
    model_predictions,y_truth = predict_with_boosting(models[model_i], dev_loader)

    if total_predictions is None:
        total_predictions = model_predictions * model_weights[model_i]
    else:
        total_predictions += model_predictions * model_weights[model_i]
    
    model_i =model_i+1

pearson_corr = pearson(total_predictions.to(torch.float), y_truth.to(torch.float))
result = pearson_corr.item()
print(f"Final Pearson is:{result}")

if  result > 0.63:
    # 如果 logits 在 GPU 上，确保转移到 CPU 并转换为 numpy 数组
    logits_np = total_predictions.detach().cpu().numpy()
    # 创建 DataFrame
    logits_df = pd.DataFrame(logits_np)
    # 保存为 TSV 文件
    logits_df.to_csv('boosting_bert_Emotion'+str(result)+'total_preds.csv', sep='\t', index=False)
    print("total_preds have been saved to 'Emotion'"+str(result)+'total_preds.csv')

    torch.save(model, '/root/models/boosting_bert_Emotion' + str(result) + '.pth')

    
    print(f"Model is model:{model_i},Pearson is:{result}")

final_predictions = total_predictions / sum(model_weights) 

print("Final Boosted Predictions:", final_predictions)
