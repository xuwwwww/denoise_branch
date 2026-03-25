from .basic_template import TrainTask
from models.DUGAN.DUGAN import DUGAN
from models.DUGAN.DUGAN_MoE import DUGAN_MoE
from models.REDCNN.REDCNN import REDCNN

model_dict = {
    'DUGAN': DUGAN,
    'DUGAN_MoE': DUGAN_MoE,
    'REDCNN': REDCNN
}
