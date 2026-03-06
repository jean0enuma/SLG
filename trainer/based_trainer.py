#MAE(Masked Autoencoder) trainer is a subclass of BaseTrainer.
from abc import ABC, ABCMeta, abstractmethod
class BaseTrainer(ABC):
    def compute_loss(self, batch, model, criterion):
        pass
    def train(self, model, optimizer, criterion, train_loader, device):
       pass
    def eval(self, model, criterion, test_loader, device):
        pass
    def fit(self,model,optimizer,scheduler,criterion,train_loader,eval_loader,test_loader,device,early_stopping=None):
        pass
