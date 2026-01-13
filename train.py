import warnings
import json
from skimage import io
import skimage
from skimage.transform import resize
from visualize import *
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model3 import *
import os
from tqdm import tqdm
from torchmetrics.classification import JaccardIndex
import torch
import torch.nn as nn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#用model3
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
class BalloonDataset(Dataset):
    def __init__(self, annotations, dataset_dir, img_size=(128, 128), transform=None):
        self.annotations = annotations
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        tid = list(self.annotations.keys())[idx]
        a = self.annotations[tid]
        mask, image, _, _, _, _ = get_mask(a, self.dataset_dir)
        # cv2.imwrite("image.jpg", image)
        # cv2.imwrite("mask.jpg", mask*255)
        # print(mask.shape)
        # print(image.shape)
        # cv2.imshow("image", image)
        # print("mask:",np.max(mask))
        # Resize and normalize
        mask = resize(mask, self.img_size, mode='constant', preserve_range=True).astype(np.float32)
        image = resize(image, self.img_size, mode='constant', preserve_range=True).astype(np.float32) / 255.0
        # print(mask.shape)
        #
        # print(image.shape)
        # print("\n")
        # Convert to tensor and add channels
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Single channel mask
        #image = torch.tensor(image, dtype=torch.float32)  # Channels-first for PyTorch
        image = image.clone().detach().float()  # Channels-first for PyTorch

        # print(image.shape)
        return image, mask

# def get_mask(a, dataset_dir):
#     image_path = os.path.join(dataset_dir, a['filename'])
#     image = io.imread(image_path)
#     height, width = image.shape[:2]
#     polygons = [r['shape_attributes'] for r in a['regions'].values()]
#     mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
#
#     for i, p in enumerate(polygons):
#         rr, cc = polygon(p['all_points_y'], p['all_points_x'])
#         rr = list(map(lambda x: height - 1 if x > height - 1 else x, rr))
#         cc = list(map(lambda x: width - 1 if x > width - 1 else x, cc))
#         mask[rr, cc, i] = 1
#
#     mask = mask.astype(bool)
#     return mask, image, height, width, None, None

def get_mask(a, dataset_dir):
    image_path = os.path.join(dataset_dir, a['filename'])
    image = io.imread(image_path)
    height, width = image.shape[:2]
    polygons = [r['shape_attributes'] for r in a['regions'].values()]
    mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        # print(max(cc))
        rr = list(map(lambda x: height-1 if x > height-1 else x, rr))
        cc = list(map(lambda x: width-1 if x > width-1 else x, cc))
        # print("i:",i)
        mask[rr, cc, i] = 1

    mask, class_ids = mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    # boxes = extract_bboxes(mask)
    boxes = extract_bboxes(resize(mask, (128, 128), mode='constant', preserve_range=True))

    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]

    class_id = top_ids[0]
    # Pull masks of instances belonging to the same class.
    m = mask[:, :, np.where(class_ids == class_id)[0]]
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)

    return m, image, height, width, class_ids, boxes

### 加载数据集
annotations_path = "dataset/balloon/train_fake/via_region_data.json"
dataset_dir = 'dataset/balloon/train_fake'
annotations = json.load(open(annotations_path))
train_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = BalloonDataset(annotations, dataset_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# 验证数据集
annotations_test_path = "dataset/balloon/val/via_region_data.json"
testset_dir = 'dataset/balloon/val'
annotations_test = json.load(open(annotations_test_path))
test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)


# Training and validation functions remain the same as provided in your code
#model = UNet(128, 128, 3).to(device)
#model = UNet(n_channels=3, n_classes=1).to(device)
model = UNet(in_channels=3, out_channels=1).to(device)
#model = UNetL(in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=300, resume_path=None):
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path))

    best_model_wts = model.state_dict()
    best_iou = 0.0

    # 设置 Jaccard Index（IoU）指标，适用于二分类任务
    iou_metric = JaccardIndex(task='binary', num_classes=2).to(device)

    train_losses = []
    val_losses = []
    val_ious = []

    try:
        for epoch in tqdm(range(num_epochs)):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.bool().float()  # 将 mask 转换为 float 类型

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.bool().float()  # 确保 mask 的类型是 bool -> float

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

                    # 计算 IoU
                    preds = (outputs > 0.5).float()  # 使用阈值 0.5 进行二值化预测
                    val_iou += iou_metric(preds, masks).item() * images.size(0)  # 按批量大小加权求和

            # 记录损失和 IoU
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_iou /= len(val_loader.dataset)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ious.append(val_iou)

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val IoU: {val_iou:.4f}')

            # 每5轮检查并保存模型
            if (epoch + 1) % 5 == 0:
                if val_iou > best_iou:  # 如果当前模型是最优的
                    best_iou = val_iou
                    best_model_wts = model.state_dict()
                    print(f"New best model found at epoch {epoch + 1} with IoU: {best_iou:.4f}")
                    torch.save(best_model_wts, f'saved_models/model_epoch_{epoch + 1}.pth')

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")

    # 保存当前模型权重
    torch.save(model.state_dict(), f'saved_models/current_model.pth')

    # 绘制中断时的训练/验证曲线
    plot_metrics(train_losses, val_losses, val_ious)

    # 恢复最佳模型权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'saved_models/model_{num_epochs}.pth')

    return model



def plot_metrics(train_losses, val_losses, val_ious):
    """绘制训练和验证的损失下降曲线及 IoU 曲线"""
    epochs = len(train_losses)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # 绘制 IoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), val_ious, label="Validation IoU")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()
    plt.title("IoU Curve")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 权重文件路径（如果需要继续训练）
    #resume_path = "best_model_epoch_170.pth"  # 替换为你的权重文件路径

    # 训练模型并保存结果
    model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=300)
    #model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=300, resume_path=resume_path)




