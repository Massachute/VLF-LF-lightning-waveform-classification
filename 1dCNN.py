import argparse
from fastai.basics import *
import numpy as np
from fastai.basics import *
from timeseries.all import *


def i2o(x):
    return lbl_dict.__getitem__(x.data.item())


def getCAM(csvRow,cls,index,model,i2o):
    batch = (Tensor(csvRow), TensorCategory(int(cls)).to(DEVICE))
    [fig, weights] = show_cam(batch, model, layer=5, i2o=i2o, func_cam=cam_acts, reduction='mean', force_scale=False, cmap='jet', show=False)
    fig.savefig(camPicPath+'\\cam{}.png'.format(i))
    print('saving pic{}'.format(index))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--i',default="test.csv", type=str,help='input files')
    parser.add_argument('--m',default="1dinception_v3.pkl", type=str,help ='model path')
    parser.add_argument('--co',default="picturetest", type=str,help='cam pic dir.')
    parser.add_argument('--ro',default="predictRessult.txt", type=str,help='result dir.')
    args = parser.parse_args()

    csvpath = args.i
    modelpath = args.m
    campath = args.co
    resultpath = args.ro
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn = load_learner(modelpath)
    model = learn.model.to(DEVICE).eval()
    camPicPath = campath
    predictResult = []
    lbl_dict = dict([(0, 'rs'), (1, 'pb'), (2, 'nb'), (3, 'ic')])
    i=1
    with open(csvpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            target = np.array(list(map(float, row[0:2500]))).reshape(1, 2500)
            res = learn.predict(target)
            predictResult.append(res[0])
            getCAM(target,res[0],i,model,i2o)
            print('{},result = {}'.format(i, lbl_dict[res[0]]))
            i = i+1
    np.savetxt(resultpath, predictResult, fmt='%s')






