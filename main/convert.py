import pickle
import sys
sys.path.append('..')
from core.cnn01 import WordCNNbp





with open('checkpoints/mr_wordcnn01_bp_relu_2.pkl', 'rb') as f:
    model_bp = pickle.load(f)

model = WordCNNbp(num_classes=2, act='sign', sigmoid=False, softmax=True,
                 ndim=200, drop_p=0, bias=True)
model.load_state_dict(model_bp.model.state_dict())
model_bp.model = model.cuda()

with open('checkpoints/mr_wordcnn01_bp_relu_2.pkl', 'wb') as f:
    pickle.dump(model_bp, f)