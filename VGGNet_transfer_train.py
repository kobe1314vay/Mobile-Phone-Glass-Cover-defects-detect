from datautils.create_datasets import MPGC_Single_DET, Mini_ImageNet, Mini_MPGC_Single_DET
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils import transforms as mytransform
from torch.utils.data import DataLoader
from model.VGGNet import vgg
import os
from result_visualize.loss_acc import save_data, plot, create_confuison_matrix
import torchvision.models as models
# from result_visualize.log_out import Logger
# import sys

#打包数据
def collate(batch):
    return tuple(zip(*batch))

# 数据变换
image_mean = [0.485, 0.456, 0.406]  # imagenet数据集的均值
image_std = [0.229, 0.224, 0.225]  # imagenet数据集的方差

image_mean_own = [0.5,0.5,0.5] #不使用迁移学习时使用此标准差
image_std_own = [0.5,0.5,0.5] #不使用迁移学习时使用此标准差

# 自定义数据变换
my_data_transform = {
    "train": mytransform.Compose([mytransform.ToTensor(),
                                 mytransform.RandomHorizontalFlip(0.5),
                                 mytransform.Normalize(image_mean_own, image_std_own)]),
    "val": mytransform.Compose([mytransform.ToTensor(),
                               mytransform.Normalize(image_mean, image_std)])
}

#官方数据变换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

def select_datasets(args):
    '''
    选择数据集类型
    datasets_type = 'MPGC_Single_DET' -- 自定义数据集
                  = 'Mini_MPGC_Single_DET' -- 自定义Mini版数据集
                  = 'MiniImagenet' -- Mini-ImageNet数据集
    :return: train_loader, val_loader
    '''
    datasets_type = args['datasets_type']
    datasets_root = args['datasets_root']
    batch_size = args['batch_size']
    num_workers = args['num_workers']
    train_loader = None
    val_loader = None
    val_dataset = None

    if datasets_type == 'MPGC_Single_DET':
        train_dataset = MPGC_Single_DET(datasets_root, my_data_transform["train"], True)

        val_dataset = MPGC_Single_DET(datasets_root, my_data_transform["val"], False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)
    elif datasets_type == 'MiniImagenet':
        json_path = os.path.join(datasets_root, 'classes_name.json')

        train_dataset = Mini_ImageNet(datasets_root, csv_name="new_train.csv", json_path=json_path,
                                           transform=data_transform["train"])

        val_dataset = Mini_ImageNet(datasets_root, csv_name="new_val.csv", json_path=json_path,
                                           transform=data_transform["val"])

        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle = True,
                                 num_workers=num_workers)

        val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle = True,
                                 num_workers=num_workers)
    else:
        train_dataset = Mini_MPGC_Single_DET(datasets_root, my_data_transform["train"], True)

        val_dataset = Mini_MPGC_Single_DET(datasets_root, my_data_transform["val"], False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)
    if train_loader != None and val_loader != None:
        return train_loader, val_loader, val_dataset
    else:
        print('no found datasets!')

def train_model(epoch, model, device, save_weights_path, save_full_model_path, save_visualization_result, model_name, train_loader,
                val_loader, val_dataset, print_train_loss=False):
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器
    best_acc = 0.0
    epochs = epoch  # 迭代次数
    loss_txt = []  # 记录损失函数值
    acc_txt = []   # 记录准确率

    print('loss_type:{} optimizer:{}'.format(loss_function, optimizer))
    print("start training......")
    for epoch in range(epochs):
        # 模型训练
        model.train()  # 模型的训练模式
        running_loss = 0.0
        running_loss_copy = 0.0  # 每组数据打印损失函数
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()  # 梯度清零 防止累加
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()   # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 记录每批次的损失函数和
            running_loss_copy += loss.item()
            if step % 20 == 19 and print_train_loss:
                print("[epoch:%d %2d] loss:%.3f" % (epoch+1, step, running_loss_copy/19)) #每20组数据打印一次损失函数
                running_loss_copy = 0.0
        loss_txt.append(running_loss / len(train_loader))  # 每epoch记录损失函数值


        #模型测试
        print("starting valitate......")
        model.eval()
        total_correct = 0.0 #预测正确的样本个数
        total_number = len(val_dataset) #总样本数
        with torch.no_grad():
            for val_data in val_loader:
                val_images,val_labels = val_data
                optimizer.zero_grad()
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                total_correct+=(predict_y == val_labels.to(device)).sum().item()
            acc = total_correct/total_number
            acc_txt.append(acc) #每epoch记录准确率
            if acc>best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_weights_path)   #保存准确率最好时的模型
                torch.save(model, save_full_model_path)

            print("[epoch:%d] train_loss: %.3f  test_acc: %.3f" % (epoch+1,running_loss/len(train_loader), acc))

    #保存可视化结果
    save_data(loss_txt, acc_txt, model_name)
    plot(loss_txt, acc_txt, save_visualization_result, model_name)
    print('Finished Training')

def main(args):

    #将输出信息存到文件中
    # sys.stdout = Logger('./save_visualization_result/vgg16.out', sys.stdout)
    # sys.stderr = Logger('./save_visualization_result/vgg16.err', sys.stderr)

    #选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据保存路径
    save_visualize_dir = args['save_visualize_dir']
    save_weights_dir = args['save_weights_dir']
    save_weights_path = os.path.join(save_weights_dir, 'vgg16Net_transfer.pth')
    save_full_model_path = os.path.join(save_weights_dir, 'Fullvgg16Net_transfer.pth')  # 保存整个模型

    # 训练集和验证集
    train_loader, val_loader, val_dataset = select_datasets(args)

    # 创建模型
    num_classes = args['num_classes']  # 类别个数
    model = models.vgg16()
    model_weight_path = './pre_weights/vgg16_Imagenet.pth'
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
    model.classifier[6].out_features = num_classes
    model.to(device)

    # 可视化结果的保存名称 默认为模型名
    model_name = "vgg16Net_transfer"

    #是否每20个epoch打印训练集损失函数
    print_train_loss = args['print_train_loss']

    # 迭代轮数
    epochs = args['epochs']
    #训练模型
    # train_model(epochs, model, device, save_weights_path, save_full_model_path, save_visualize_dir, model_name, train_loader, val_loader,
    #             val_dataset, print_train_loss)

    #打印混淆矩阵 验证模型 保存实验结果
    model.load_state_dict(torch.load(save_weights_path))
    model.to(device)
    create_confuison_matrix(model, val_loader, save_visualize_dir, model_name, device)


if __name__ == "__main__":
    import argparse  # 导入命令行参数解析器包

    parser = argparse.ArgumentParser(
        description=__doc__)   # 创建命令行解析器

    parser.add_argument('--datasets_type', default='MPGC_Single_DET', help='选择哪个数据集训练:\n'
                                                                                'Mini_MPGC_Single_DET -- mini版的自定义数据集\n'
                                                                                'MPGC_Single_DET -- 自定义的主数据集\n'
                                                                                'MiniImagenet -- mini版的ImageNet\n')
    parser.add_argument('--datasets_root', default='F:/datasets/super_MPGCCLASS', help='数据集的根目录\n'
                                                                                 'Mini_MPGC_Single_DET and '
                                                                                 'MPGC_Single_DET -- F:/datasets/MPGCCLASS\n'
                                                                                 'MiniImagenet -- F:/datasets/MiniImagenet')
    parser.add_argument('--num_classes', default=2, type=int, help='类别个数\n'
                                                                   '自定义数据集：2'
                                                                   'Mini_ImageNet：100')
    parser.add_argument('--save_weights_dir', default='./save_weights', help='权重保存目录')
    parser.add_argument('--save_visualize_dir', default='./save_visualization_result', help='可视化结果保存目录')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch大小')
    parser.add_argument('--num_workers', default=4, type=int, help='cpu线程数')
    parser.add_argument('--epochs', default=5, type=int, help='迭代轮数')
    parser.add_argument('--print_train_loss', default=False, help='是否每20个epoch打印训练集损失函数')

    args = parser.parse_args()  # 解析命令
    args = vars(args)   # 为了方便使用，转化为字典形式


    print(args)  # 打印参数
    main(args)






