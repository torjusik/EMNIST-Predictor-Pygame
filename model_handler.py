import torch
from neural_network import Neural_network
class Model_handler:
    def save(model, PATH):
        torch.save(model.state_dict(), PATH)
            
    def load(model, PATH):
        model.load_state_dict(torch.load(PATH))
        model.eval()