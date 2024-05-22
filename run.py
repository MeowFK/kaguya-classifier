from fastai.vision.all import *

model = load_learner('kaguya_model1.pkl')
res = model.predict('images/kaguya.jpg')
print(res[0])

print(model.dls.vocab)