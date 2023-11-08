#imports
import torch

#classifier class
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, accelerator):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.accelerator = accelerator
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid(),
        )
        self.to("cuda")                             #use GPU
        
    def reset_weights(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        return self.mlp(x)
    
    def loss_fn(self):
        return torch.nn.BCELoss()                   #binary cross entropy loss

    #function for training on the GPU and returning the loss on validation set
    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, L2_alpha, prints = False):
        self.reset_weights()                                                            
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=L2_alpha) #L2 regularization
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(X_train.float().cuda())
            loss = self.loss_fn()(y_pred, y_train.float().cuda())
            self.accelerator.backward(loss)
            optimizer.step()
            if epoch % (epochs//10) == 0:
                if prints:
                    print("Epoch %d: train loss %f" % (epoch, loss.item()))
        y_pred = self.forward(X_val.float().cuda())
        loss = self.loss_fn()(y_pred, y_val.float().cuda())
        if prints:
            print("Validation loss %f" % loss.item())
        return(loss.item())

    #function for finetuning the hyperparameters
    def finetuning(self, learning_rate_list, epochs_list, L2_alphas, X_train, y_train, X_val, y_val, prints = False):
        best_loss = 1
        best_state = None
        best_hyperparams = (0, 0, 0)
        for learning_rate in learning_rate_list:
            for epochs in epochs_list:
                for L2_alpha in L2_alphas:
                    self.reset_weights()
                    loss = self.train(X_train, y_train, X_val, y_val, epochs, learning_rate, L2_alpha, prints)
                    if loss < best_loss:
                        best_loss = loss
                        best_state = self.state_dict()
                        best_hyperparams = (learning_rate, epochs, L2_alpha)
        return(best_loss, best_state, best_hyperparams)