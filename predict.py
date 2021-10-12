import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.ResNet import resnet34
from model.AlexNet import AlexNet
from model.MDNet2 import MDNet2

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("F:/datasets/MPGCCLASS/IMAGES/defects_10003.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_indict = ["no_defects", "defects"]
# create model

# model = resnet34(num_classes=4)
# load model weights
# model.load_state_dict(torch.load(model_weight_path))
# model_weight_path = "./save_weights/FullAlexNet.pth"  # 直接加载整个模型和参数 不需要重新定义模型
# model = torch.load(model_weight_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class_indict = ["no_defects", "defects"]
model = MDNet2(num_classes=2)
model_weight_path = "F:/超算实验结果/分类网络/MPGC_Single_DET/2021_3_9_实验1/MDNet_transfer/weights/MDNet2_transfer.pth"
model.load_state_dict(torch.load(model_weight_path))
model.to(device)




print(model)
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img.to(device)))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).cpu().numpy()
print(output)
print(predict_cla)
print(class_indict[predict_cla])
plt.show()