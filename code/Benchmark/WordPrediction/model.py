import torch
from torch import nn
from params import *


# Model
class NNLM(nn.Module):
    def __init__(self, n_class):
        super(NNLM, self).__init__()
        self.n_step = n_step
        self.emb_size = emb_size

        self.C = nn.Embedding(n_class, emb_size)
        self.w1 = nn.Linear(n_step * emb_size, n_hidden, bias=False)
        self.b1 = nn.Parameter(torch.ones(n_hidden))
        self.w2 = nn.Linear(n_hidden, n_class, bias=False)
        self.w3 = nn.Linear(n_step * emb_size, n_class, bias=False)

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.emb_size)
        Y1 = torch.tanh(self.b1 + self.w1(X))
        b2 = self.w3(X)
        Y2 = b2 + self.w2(Y1)
        return Y2


# Model
class TextRNN(nn.Module):
    def __init__(self, n_class):
        super(TextRNN, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model


# RNNLM based on Attention
class TextRNN_attention(nn.Module):
    def __init__(self, n_class):
        super(TextRNN_attention, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(2 * n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output = outputs[-1]
        attention = []
        for it in outputs[:-1]:
            attention.append(torch.mul(it, output).sum(dim=1).tolist())
        attention = torch.tensor(attention).to(outputs.device)
        attention = attention.transpose(0, 1)
        attention = nn.functional.softmax(attention, dim=1).transpose(0, 1)

        # get soft attention
        attention_output = torch.zeros(outputs.size()[1], n_hidden).to(outputs.device)
        for i in range(outputs.size()[0] - 1):
            attention_output += torch.mul(attention[i], outputs[i].transpose(0, 1)).transpose(0, 1)
        output = torch.cat((attention_output, output), 1)
        # joint ouput output:[batch_size, 2*n_hidden]
        model = torch.mm(output,self.W.weight.t()) + self.b  # model : [batch_size, n_class]
        # 这里返回值可以只保留model，返回attention的目的是为了将attention矩阵保存成热力图
        return model
