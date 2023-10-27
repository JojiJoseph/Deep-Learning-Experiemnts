import cv2
import numpy as np
import torch

def generate_cam(model, image_tensor, class_idx):
    model.eval()
    target = None
    inp = None
    def activation_hook_func(module, inp_, out):
        nonlocal target, inp
        inp = inp_[0].detach()
        target = out.detach()
    
    model.avgpool.register_forward_hook(activation_hook_func)

    scores = model(image_tensor)

    weights = model.fc.weight

    weights = weights[class_idx][None   , :, None, None]


    cam = weights * inp


    cam = cam.detach().numpy()

    cam = np.sum(cam[0],0 )

    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))  # Resize to the input image size
    cam = cam - np.min(cam)  # Normalize
    cam = cam / np.max(cam)
    cam = np.uint8(cam*255)
    return cam
    

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    image = cv2.imread('./kitten_and_puppy.webp')
    # image = cv2.imread('./cat_and_dog.webp')
    # image = cv2.imread('./Girl_and_cat.jpg')
    # print(image)
    image = cv2.resize(image, (1024, 1024))
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)/255.
    image_tensor = 2 * (image_tensor - 0.5)
    cam = generate_cam(model, image_tensor, 207)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
    img_out = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
    cv2.imshow("Cam", img_out)
    cv2.waitKey(0)

"""
ImageNet classes

cat - 281, 282, 283, 284, 285, 286, 287, 288, 289, 290
dog - 151, 152, 153, 154, 155, 156, 157, 158, 159, 160

german shepherd - 235

"""