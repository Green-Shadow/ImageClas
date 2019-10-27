from fastai import *
from fastai.vision import *

path = untar_data("https://s3.amazonaws.com/fast-ai-imageclas/food-101")

tfms = get_transforms()
data = ImageDataBunch.from_folder(path, train='images', valid_pct = 0.15, ds_tfms = tfms, size = 224)
#data.show_batch()

learn = cnn_learner(data, models.resnet50, metrics=accuracy)

learn.fit_one_cycle(10)
learn.unfreeze()
learn.fit_one_cycle(10)