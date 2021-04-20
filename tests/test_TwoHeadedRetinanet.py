#test two class headed output
import torch
from TwoHeadedRetinanet import create
from deepforest import main
import pytest

@pytest.fixture()
def trained_baseline():
    m = main.deepforest()
    m.use_release()
    
    return m.model
    
def test_create(trained_baseline):
    model = create(trained_model=trained_baseline)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    prediction = model(x)    
    
    
    
    

