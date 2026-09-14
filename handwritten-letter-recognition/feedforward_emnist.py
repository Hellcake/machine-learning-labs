import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Определяем устройство: CUDA (если доступно) или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# --------------------------
# Гиперпараметры и конфигурация
# --------------------------
batch_size = 64
num_epochs = 5
learning_rate = 0.001
hidden_size = 128       # Изменяемое число нейронов в скрытом слое
input_size = 28 * 28    # Размер входного изображения: 28x28 пикселей
num_classes = 26        # Для EMNIST Letters (буквы)

# Выбор передаточной функции: 'relu', 'sigmoid' или 'tanh'
activation_choice = 'sigmoid'
if activation_choice == 'relu':
    activation_fn = nn.ReLU()
elif activation_choice == 'sigmoid':
    activation_fn = nn.Sigmoid()
elif activation_choice == 'tanh':
    activation_fn = nn.Tanh()
elif activation_choice == 'leaky_relu':
    activation_fn = nn.LeakyReLU()
else:
    activation_fn = nn.ReLU()  # Значение по умолчанию
print(f"Используем передаточную функцию: {activation_choice}, количество нейронов: {hidden_size}")
# --------------------------
# Определение модели FFNN (Feedforward Neural Network)
# --------------------------
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Преобразуем изображение [batch_size, 1, 28, 28] в вектор [batch_size, 784]
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

# --------------------------
# Загрузка и предобработка данных (EMNIST Letters)
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Стандартные значения для MNIST/EMNIST
])

train_dataset = datasets.EMNIST(root='data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='data', split='letters', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda")
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda")
)

# --------------------------
# Инициализация модели, функции потерь, оптимизатора и scaler для AMP
# --------------------------
model = FFNN(input_size, hidden_size, num_classes, activation_fn).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

train_losses = []
test_accuracies = []

# --------------------------
# Цикл обучения с поддержкой CUDA и AMP
# --------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = (labels - 1).to(device, non_blocking=True)  # Корректировка меток: вычитаем 1
        
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Оценка модели на тестовой выборке во время обучения
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = (labels - 1).to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# Сохраняем финальное состояние модели
model_save_path = "ffnn_emnist_letters.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Модель сохранена по пути: {model_save_path}")

# --------------------------
# Отдельное тестирование сохранённой модели
# --------------------------
# Создаем новую модель того же типа и загружаем сохраненные веса
loaded_model = FFNN(input_size, hidden_size, num_classes, activation_fn).to(device)
loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
loaded_model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = (labels - 1).to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = loaded_model(images)
        else:
            outputs = loaded_model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

final_accuracy = test_correct / test_total
print(f"Финальная точность на тестовой выборке: {final_accuracy:.4f}")

# --------------------------
# Визуализация результатов обучения
# --------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker='o')
plt.title('Test Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
