import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import ASVspoof2021
from models import TeacherModel, StudentModel
from losses import oc_softmax_loss, am_softmax_loss, doc_softmax_loss

# Knowledge Distillation Loss
def knowledge_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
    student_probs = torch.log_softmax(student_logits / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs) * (temperature ** 2)
    return loss

def train_one_class_model(model, dataloader, optimizer, loss_fn, epoch, device):
    model.train()
    total_loss = 0

    # Expecting 4 values from the dataloader
    for batch_idx, (lfcc_feat, mfcc_feat, filenames, labels) in enumerate(dataloader):
        if labels.item() == 1:  # Only train on bonafide samples
            lfcc_feat = lfcc_feat.to(device)
            mfcc_feat = mfcc_feat.to(device)
            labels = labels.to(device)

            logits = model(torch.cat([lfcc_feat, mfcc_feat], dim=1))  # Concatenate LFCC and MFCC
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}], One-Class Loss: {avg_loss:.4f}")
    return avg_loss


def distillation_train(student_model, teacher_model, dataloader, optimizer, loss_fn, temperature, device):
    student_model.train()
    teacher_model.eval()  # Teacher model is fixed

    total_loss = 0

    for batch_idx, (lfcc_feat, mfcc_feat, filenames, tags, labels) in enumerate(dataloader):
        lfcc_feat = lfcc_feat.to(device)
        mfcc_feat = mfcc_feat.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(torch.cat([lfcc_feat, mfcc_feat], dim=1))

        student_logits = student_model(torch.cat([lfcc_feat, mfcc_feat], dim=1))

        kd_loss = knowledge_distillation_loss(student_logits, teacher_logits, temperature)
        task_loss = loss_fn(student_logits, labels)
        total_loss_val = kd_loss + task_loss

        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        total_loss += total_loss_val.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Distillation Training, Loss: {avg_loss:.4f}")
    return avg_loss

def plot_loss(epochs, loss_values, title):
    """
    Plots the loss curve.
    
    Args:
        epochs: List of epoch numbers.
        loss_values: List of loss values.
        title: Title of the plot (e.g., One-Class Loss or Distillation Loss).
    """
    plt.plot(epochs, loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    temperature = 2.0  # For knowledge distillation
    loss_type = "oc_softmax"  # For one-class learning and distillation

    # Paths
    path_to_features = r'C:\Users\Serilda\Desktop\VPD\features_extracted'
    path_to_protocol = r'C:\Users\Serilda\Desktop\VPD\protocols'

    # Load dataset and dataloader
    dataset = ASVspoof2021(path_to_features=path_to_features, path_to_protocol=path_to_protocol, part='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # List to store loss values
    one_class_loss_values = []
    distillation_loss_values = []
    epochs_list = list(range(epochs))

    # One-Class Training (Teacher Model)
    for epoch in epochs_list:
        avg_loss = train_one_class_model(teacher_model, dataloader, optimizer, oc_softmax_loss, epoch, device)
        one_class_loss_values.append(avg_loss)  # Store loss for plotting

    # Plot One-Class Loss
    plot_loss(epochs_list, one_class_loss_values, title="One-Class Training Loss")

    # Train the student model using distillation
    for epoch in epochs_list:
        avg_loss = distillation_train(student_model, teacher_model, dataloader, optimizer, oc_softmax_loss, temperature, device)
        distillation_loss_values.append(avg_loss)  # Store loss for plotting

    # Plot Distillation Loss
    plot_loss(epochs_list, distillation_loss_values, title="Distillation Training Loss")
