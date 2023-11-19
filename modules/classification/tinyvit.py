from modules.classification.base import ClassifyBase
import numpy as np
from PIL import Image
import onnxruntime

from torchvision import transforms
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
IMAGENET_DEFAULT_MEAN= (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD= (0.229, 0.224, 0.225)



class Classifier(ClassifyBase):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self._cfg = cfg
        providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(cfg["model_path"], providers=providers)

    def _preprocess(self, img):
        size = int((256 / 224) * 224)
        tfms = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        #convert image cv2 to Image PIL
        img=np.asarray(img)
        img=Image.fromarray(img)
        img = tfms(img).unsqueeze(0)
        return img

    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    
    def classify(self, img_src):
        img = self._preprocess(img_src)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        scores = self.session.run([output_name], {input_name: self.to_numpy(img)})[0]
        class_predict = np.argsort(-scores, axis=1)[:, :2].squeeze()[0]
        prob = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, np.newaxis]
        prob = prob[:, :2].squeeze()  
        if prob[class_predict]<0.6:
            class_predict=1-class_predict    
        return class_predict

        
    
    
   