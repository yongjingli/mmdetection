import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from functools import partial
from types import MethodType
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.cnn import ConvModule


def export_onnx_dcn_op():
    debug_path = "/userdata/liyj/data/test_data/depth/debug"
    torch_op_name = "mmdet_dcn"
    in_channel = 3
    out_channel = 10

    conv_module_dcn = ConvModule(in_channel, out_channel, 3, padding=1,
                             conv_cfg=dict(type='DCNv2'), norm_cfg=dict(type='BN'),
                                 bias=True)

    # conv_module_dcn = ConvModule(in_channel, out_channel, 3, padding=1,
    #                          conv_cfg=None, norm_cfg=dict(type='BN'),
    #                              bias=True)

    conv_module_dcn.eval()

    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    with torch.no_grad():
        outputs = conv_module_dcn(x)

    input = x.detach().cpu().numpy()
    save_path = debug_path + "/" + torch_op_name + "_pt_input_" + str(0) + ".npy"
    np.save(save_path, input)
    save_path = debug_path + "/" + torch_op_name + "_pt_input_" + str(0) + ".bin"
    input.tofile(save_path)

    input_len = 1
    for s in input.shape:
        input_len = input_len * s
    print("input_len:", input_len)

    for i, output in enumerate(outputs):
        output = output.detach().cpu().numpy()
        save_path = debug_path + "/" + torch_op_name + "_pt_output_" + str(i) + ".npy"
        np.save(save_path, output)

        input_len = 1
        for s in output.shape:
            input_len = input_len * s
        print("output_len {}:{}".format(i, input_len))

    out_path = debug_path + "/" + torch_op_name + ".onnx"
    with torch.no_grad():
        torch.onnx.export(
            conv_module_dcn,
            args=x,
            f=out_path,
            input_names=['input'],
            output_names=['output'],
            #operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            #opset_version=11,
            enable_onnx_checker=False,)


def forward_dummy(self, img):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    outs = torch.cat([outs[0][0], outs[1][0], outs[2][0]], dim=1)
    return outs


def onnx_export(self, img, img_metas, with_nms=True):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)

    outs = torch.cat([outs[0][0], outs[1][0], outs[2][0]], dim=1)

    return outs


def export_onnx_centernet():
    # CTResNetNeck, add bias for DCN
    debug_path = "/userdata/liyj/data/test_data/depth/debug"
    torch_op_name = "mmdet_centernet"

    # centernet
    config_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco.py'
    checkpoint_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)

    origin_forward = model.forward
    model.onnx_export = MethodType(onnx_export, model)
    model.forward = MethodType(forward_dummy, model)
    # model.forward = model.forward_dummy

    x = torch.randn(1, 3, 512, 512, requires_grad=True)
    x = x.to(device)

    input = x.detach().cpu().numpy()
    save_path = debug_path + "/" + torch_op_name + "_pt_input_" + str(0) + ".npy"
    np.save(save_path, input)
    save_path = debug_path + "/" + torch_op_name + "_pt_input_" + str(0) + ".bin"
    input.tofile(save_path)

    input_len = 1
    for s in input.shape:
        input_len = input_len * s
    print("input_len:", input_len)

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        output = outputs.detach().cpu().numpy()
        save_path = debug_path + "/" + torch_op_name + "_pt_output_" + str(0) + ".npy"
        np.save(save_path, output)

        input_len = 1
        for s in output.shape:
            input_len = input_len * s
        print("output_len {}:{}".format(0, input_len))

    out_path = debug_path + "/" + torch_op_name + ".onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=x,
            f=out_path,
            input_names=['input'],
            output_names=['output'],
            # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # opset_version=11,
            enable_onnx_checker=False, )

    print("export_onnx_centernet Done")


def debug_dcn_trt_different():
    pt_out_path = "/mnt/data10/liyj/programs/tensorrt_deploy_test/debug_result/mmdet_dcn2/mmdet_centernet_pt_output_0.npy"

    pt_out_0 = np.load(pt_out_path).reshape(-1)
    trt_out_path = "/mnt/data10/liyj/programs/tensorrt_deploy_test/debug_result/mmdet_dcn2/trt_dcn_out.bin"
    trt_out = np.fromfile(trt_out_path, dtype=np.float32)

    trt_out_0 = trt_out

    # diff1 = abs(pt_out_0 - trt_out_0)
    diff0 = abs(pt_out_0 - trt_out_0)
    print("max_0:", np.max(diff0))
    # print("min:", np.min(diff1))
    print("mean_0:", np.mean(diff0))

    pt_out_path = "/mnt/data10/liyj/programs/tensorrt_deploy_test/debug_result/mmdet_dcn2/mmdet_centernet_pt_output_0.npy"
    pt_out = np.load(pt_out_path)
    print(pt_out.shape)

    diff_ori = diff0.reshape(pt_out.shape)
    print(np.max(diff_ori[:, 82:84, :, :]))


if __name__ == "__main__":
    print("Start Proc...")
    export_onnx_dcn_op()
    # export_onnx_centernet()
    debug_dcn_trt_different()
    print("eNd Proc...")