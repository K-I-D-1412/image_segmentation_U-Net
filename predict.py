from train_new import *
# 补全 predict 函数

def plot_comparison(image, mask, pred, iou, save_path=None):
    """展示原始图像、真实掩码和预测掩码的对比，并显示 IoU 值"""
    # 确保 mask 和 pred 是 2D 的
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.astype(np.uint8))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title(f"Predicted Mask\nIoU: {iou:.4f}")
    axs[2].axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def predict(model, loader, device):
    """模型预测函数，返回所有预测结果"""
    model.eval()
    preds_all = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # 使用 0.5 阈值进行二值化
            preds_all.append(preds.cpu().numpy())
    return np.concatenate(preds_all, axis=0)

# 补全 plot_predictions 函数
def plot_predictions(images, masks, preds, ious, save_dir="test_predictions"):
    """
    展示和保存预测结果。
    :param images: 原始图像
    :param masks: 真实掩模
    :param preds: 预测掩模
    :param ious: 每幅图的 IoU 值
    :param save_dir: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(images.shape[0]):
        image = (images[i].transpose(1, 2, 0) * 255).astype(np.uint8)  # 反归一化
        mask = (masks[i] * 255).astype(np.uint8)
        pred = (preds[i] * 255).astype(np.uint8)

        # 保存对比图
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plot_comparison(image, mask, pred, ious[i], save_path)


# 计算平均 Loss 和 IoU，并保存预测结果
def evaluate_and_save_predictions(model, test_loader, criterion, device, save_dir="test_results"):
    """
    计算测试集的平均 Loss 和 IoU，并保存预测结果图片。
    :param model: 已训练好的模型
    :param test_loader: 测试集 DataLoader
    :param criterion: 损失函数
    :param device: 设备
    :param save_dir: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 设置 Jaccard Index（IoU）指标，适用于二分类任务
    iou_metric = JaccardIndex(task="binary", num_classes=2).to(device)

    test_loss = 0.0
    test_iou = 0.0
    total_samples = 0  # 记录总样本数，用于加权求平均
    ious = []  # 存储每幅图的 IoU 值

    # 遍历测试集
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            # 移动数据到设备
            images, masks = images.to(device), masks.to(device)

            # 确保标签格式与验证阶段一致
            masks = masks.bool().float()

            # 模型预测
            outputs = model(images)

            # 计算 Loss
            loss = criterion(outputs, masks)
            batch_loss = loss.item() * images.size(0)

            # 计算 IoU
            preds = (outputs > 0.5).float()  # 阈值为 0.5 的二值化
            batch_iou = iou_metric(preds, masks).item() * images.size(0)

            # 累计损失和 IoU
            test_loss += batch_loss
            test_iou += batch_iou
            total_samples += images.size(0)

            # 计算每幅图的 IoU 并存储
            for j in range(images.size(0)):
                iou = iou_metric(preds[j].unsqueeze(0), masks[j].unsqueeze(0)).item()
                ious.append(iou)

                # 保存图片
                # 提取单个样本的原图、标签和预测结果
                image = images[j].cpu().permute(1, 2, 0).numpy() * 255.0  # 转换到 [0, 255] 范围
                mask = masks[j].cpu().squeeze().numpy() * 255.0  # 标签
                pred = preds[j].cpu().squeeze().numpy() * 255.0  # 预测结果

                # 构建保存路径
                save_path = os.path.join(save_dir, f"sample_{i * images.size(0) + j}.png")
                plot_comparison(image, mask, pred, iou, save_path)

            # 打印每个批次的损失和 IoU
            print(f"Batch {i + 1}/{len(test_loader)}: "
                  f"Loss = {batch_loss / images.size(0):.4f}, "
                  f"IoU = {batch_iou / images.size(0):.4f}")

    # 计算测试集平均损失和 IoU
    test_loss /= total_samples
    test_iou /= total_samples

    print(f"\nTest Loss (Average): {test_loss:.4f}, Test IoU (Average): {test_iou:.4f}")
    return test_loss, test_iou, ious


# 示例：评估模型并保存预测结果
if __name__ == "__main__":
    # 定义测试集和 DataLoader
    test_dataset = BalloonDataset(annotations_test, testset_dir, transform=train_transform)
    test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

    # 加载训练好的模型
    model = UNet(3, 1).to(device)
    model.load_state_dict(torch.load("current_model.pth"))
    model.eval()

    # 定义损失函数
    criterion = torch.nn.BCEWithLogitsLoss()

    # 模型预测、保存和评估
    preds_test = predict(model, test_loader, device)
    test_loss, test_iou, ious = evaluate_and_save_predictions(model, test_loader, criterion, device)

    # 保存预测结果
    test_images, test_masks = next(iter(test_loader))
    plot_predictions(
        test_images.cpu().numpy(),
        test_masks.cpu().numpy(),
        preds_test[: len(test_images)],
        ious[: len(test_images)],
        show=False
    )