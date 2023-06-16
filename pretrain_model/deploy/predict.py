from lib import *
from config import *
from utils import *
from imagetransform import ImageTransform

class_index = ['cats', 'dogs']

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        predict_label = self.clas_index[max_id]
        return predict_label


predictor = Predictor(class_index)


def predict(img):

    # prepare net work
    net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    net.classifier[6] = nn.Linear(in_features= 4096, out_features= 2)
    net.eval()


    # prepare model
    model = load_model(net, save_path)

    # prepare input img

    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase= 'test')
    img = img.unsqueeze_(0) #(c, h,w) ->(1,c,h,w)

    output = model(img)
    response = predictor.predict_max(output)
    return response

