import torch
from torch.autograd import Variable
import struct
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert


cls_names = ['乱堆物料', '人骑车_行驶状态', '出店经营', '卖衣服游商', '合规广告类', '合规的早餐车', '垃圾桶未满溢', '垃圾桶满溢', '平摊', '广告误检',
'店内(橱窗内)晾晒', '店内经营', '废弃摊位_不在经营的摊位', '打包垃圾', '撑伞', '无照经营游商', '晒玉米粮食等', '暴露垃圾', '气模拱门',
'沿街晾晒', '渣土堆积', '盆栽花卉', '矩摊', '石头_假山_雕塑_墙体等', '砖块堆积', '篮筐', '经营性物资', '误检', '车上堆积', '车辆无遮挡',
'车辆有遮挡', '运动中游商', '违规不上报广告类', '违规广告类', '遮盖布']

cls_nums = len(cls_names)

model_type = "RepVGG-B0"

if model_type in ["mobilenet_v3_small", "mobilenet_v3_large"]:
    model = models.__dict__[model_type](pretrained=True)
    fc_input_nums = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features=fc_input_nums, out_features=cls_nums, bias=True)
    cpkt = torch.load(f"./ckpt/ckpt_{model_type}_linear_ce/checkpoint_best.pth.tar", map_location=torch.device('cpu'))[
        'state_dict']
    new_weights = {k[7:]: v for k, v in cpkt.items()}
    model.load_state_dict(new_weights)
elif model_type == "mv3_small_zt":
    model = mobilenetv3_small()
    model.classifier = torch.nn.Linear(in_features=576, out_features=cls_nums, bias=True)
    cpkt = torch.load(f"./ckpt_linear_ce/cls_model_20230111_clsnums35.pth.tar", map_location=torch.device('cpu'))['state_dict']
    new_weights = {k[7:]: v for k, v in cpkt.items()}
    model.load_state_dict(new_weights)
elif model_type in ["RepVGG-A0", "RepVGG-A1", "RepVGG-A2", "RepVGG-B0", "RepVGG-B1", "RepVGG-B2"]:
    train_model = get_RepVGG_func_by_name(model_type)(deploy=False)
    fc_input_nums = train_model.linear.in_features
    train_model.linear = torch.nn.Linear(in_features=fc_input_nums, out_features=cls_nums, bias=True)

    cpkt = torch.load(f"./ckpt/ckpt_{model_type}_linear_ce/checkpoint_best.pth.tar", map_location=torch.device('cpu'))[
        'state_dict']
    new_weights = {k[7:]: v for k, v in cpkt.items()}
    train_model.load_state_dict(new_weights)

    model = repvgg_model_convert(train_model)

image = torch.ones(1, 3, 224, 224)
if torch.cuda.is_available():
    model.cuda()
    image = image.cuda()

model.eval()
print(model)
print('image shape ', image.shape)
preds = model(image)

f = open(f"./wts/{model_type}.wts", 'w')
f.write("{}\n".format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")

