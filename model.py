# codeing: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import sys


class DANTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        dense_units=[100, 100]
    ):
        super(DANTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        modulelist = []
        for i in dense_units:
            modulelist.append(nn.Sequential(
            nn.Dropout(dr),
            nn.LazyLinear(i), # input size is inferred
            nn.ReLU()
        ))
        self.feedforwards = nn.ModuleList(modulelist)
        self.projection = nn.LazyLinear(num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]

        #seperate the embedding of the word (to do NER on) from that of the whole sentence
        embedded_split = torch.split(embedded, split_size_or_sections=1, dim=1)
        embedded_word = embedded_split[0] # [batch_size, 1, embedded_size]
        embedded_sentence = torch.cat(embedded_split[1:], dim=1) # [batch_size, seq_len-1, embedded_size]

        #concat the embedding of the word (to do NER on) with the averaged sentence embedding
        avg_embedded = embedded_sentence.mean(dim=1) # [batch_size, embedded_size]
        embedded_word = torch.squeeze(embedded_word) # [batch_size, embedded_size]
        x = torch.cat((embedded_word, avg_embedded), dim=1) # [batch_size, 2*embedded_size]

        for feedforward in self.feedforwards:
            x = feedforward(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x


class DenseTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        dense_units=[100, 100]
    ):
        super(DenseTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        modulelist = []
        for i in dense_units:
            modulelist.append(nn.Sequential(
            nn.Dropout(dr),
            nn.LazyLinear(i), # input size is inferred
            nn.ReLU()
        ))
        self.feedforwards = nn.ModuleList(modulelist)
        self.projection = nn.LazyLinear(num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x = torch.flatten(embedded, start_dim=1, end_dim=2) # [batch_size, seq_len*embedded_size] 

        for feedforward in self.feedforwards:
            x = feedforward(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x

class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0,
                 num_classes=2,
                 dr=0.2,
                 filter_widths=[3,4],
                 num_filters=100,
                 num_conv_layers=3,
                 intermediate_pool_size=3, **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        # create the list "convblocks" consisting of only first convblock
        convblocks = [nn.ModuleList([
            nn.ModuleList([nn.Conv1d(in_channels=emb_output_dim, out_channels=num_filters,
                                     kernel_size=filter_width) for filter_width in filter_widths]),
            nn.ReLU(),
            nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
        ])]
        # append subsequent convblocks
        for i in range(num_conv_layers - 1):
            convblocks.append(nn.ModuleList([
                nn.ModuleList([nn.Conv1d(in_channels=(num_filters*len(filter_widths)), out_channels=num_filters,
                                         kernel_size=filter_width) for filter_width in filter_widths]),
                nn.ReLU(),
                nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
            ]))
        self.convblocks = nn.ModuleList(convblocks)
        self.globalpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.projection = nn.Linear(in_features=(num_filters*len(filter_widths)), out_features=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def forward(self, data, mask):
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x = embedded.transpose(1, 2).contiguous() # [batch_size, embedded_size, seq_len]
        for i in range(len(self.convblocks)):
            convblock = self.convblocks[i] #first block's input size: [batch_size, embedded_size, seq_len]
            xs = [conv(x) for conv in convblock[0]] # [batch_size, num_filters, (seq_len - filter_width + 1)]
            xs = [convblock[1](x) for x in xs] #relu
            xs = [convblock[2](x) for x in xs] #maxpool: [batch_size, num_filters, (seq_len - filter_width + 1)/pool_size]
            # ensure that all tensors have the same size in the 2nd dim
            min_output_length = min([x.shape[2] for x in xs])
            xs = [torch.narrow(input=x, dim=2, start=0, length=min_output_length) for x in xs]
            # concat along the num_filter dimension
            x = torch.cat(xs, dim=1) # [batch_size, len(filter_width) * num_filters, (seq_len - filter_width + 1)/pool_size]
            # with each subsequent iteration, the 2nd dim would be divided down by pool_size
            
        x = self.globalpool(x) # [batch_size, len(filter_width) * num_filters, 1]
        x = torch.squeeze(x)   # [batch_size, len(filter_width) * num_filters]
        x = self.projection(x) # [batch_size, num_classes]
        return x
    

class AttnTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        dense_units=[100, 100],
    ):
        super(AttnTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim

        """
        The Deep Averaging Network is very simple. It performs "average
        pooling" over word embeddings, i.e. it averages the individual
        embedding vectors.

        Following the averaging, there should be one or more feedforward
        layers.  You can construct a feedforward layer by combining nn.Linear
        and an activation function. Before each feedforward layer there should
        also be a Dropout layer that randomly sets values of the input to 0
        with a probability `dr`.

        Main structure
        --------------
        1.) Embedding layer
        2.) An ordered collection (nn.Sequential) of feedforward blocks
            a.) Dropout layer
            b.) Linear layer
            c.) Activation layer
        3.) Final projection layer to go from hidden layers to output class
        """
        ## .. TODO ..
        ## initialize model
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.multihead_attn = nn.MultiheadAttention(emb_output_dim, num_heads=10)
        modulelist = []
        for i in dense_units:
            modulelist.append(nn.Sequential(
            nn.Dropout(dr),
            nn.LazyLinear(i), # input size is inferred
            nn.ReLU()
        ))
        self.feedforwards = nn.ModuleList(modulelist)
        self.projection = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        While PyTorch will automatically initialize weights for you this can be suboptimal.
        This method should use "Xavier initialization" which draws values from a Uniform distribution.
        The relevant PyTorch function will be torch.nn.init.xavier_uniform_.

        To get the weights of a given layer/module, use `module.weight.data`.
        For feedforward layers we will also need a bias which you can set to 0 using `module.bias.data.zero_()`
        """
        # torch.nn.init.xavier_uniform_
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        """
        for layer in self.feedforwards:
            torch.nn.init.xavier_uniform_(layer[1].weight)
        """
        #torch.nn.init.xavier_uniform_(self.projection.weight)
        pass

    def from_embedding(self, embedding, mask):
        """
        This is where the 'magic' happens! Use this function to:
        1.) Mask out embeddings that belong to padding tokens
        2.) Pass the embeddings through the 'DAN encoder'
            a.) Average pool the embeddings
            b.) Pass through feedforward layers
            c.) Project so that output dim equals number of classes
        """
        ## .. TODO ..
        ## forward pass (from the outputs of the embedding)

        embedded = embedding # [batch_size, seq_len, embedded_size]
        # 1.) Mask out embeddings that belong to padding tokens
        for i in range(embedded.shape[0]): #32
            for j in range(embedded.shape[1]): #128
                if(mask[i][j]==0):
                    embedded[i][j] = 0
        #print(embedded.shape)
        x, _ = self.multihead_attn(query=embedded, key=embedded, value=embedded) # [batch_size, seq_len, embedded_size]
        #print(x.shape)
        x = torch.flatten(embedded, start_dim=1, end_dim=2) # [batch_size, seq_len*embedded_size]
        #print(x.shape)
        #sys.exit()
        #x = torch.flatten(embedded, start_dim=1, end_dim=2) # [batch_size, seq_len*embedded_size] 
        #print(x.shape)
        #sys.exit()
        #embedded_split = torch.split(embedded, split_size_or_sections=1, dim=1)
        #embedded_word = embedded_split[0] # [batch_size, 1, embedded_size]
        #embedded_sentence = torch.cat(embedded_split[1:], dim=1) # [batch_size, seq_len-1, embedded_size]
        #x = torch.split(embedded, split_size_or_sections=1, dim=1)
        # 2a.) Average pool the embeddings
        #avg_embedded = embedded_sentence.mean(dim=1) # [batch_size, embedded_size]
        #embedded_word = torch.squeeze(embedded_word) # [batch_size, embedded_size]
        #x = torch.cat((embedded_word, avg_embedded), dim=1) # [batch_size, 2*embedded_size]
        #print(x.shape)
        #sys.exit()
        # 2b.) Pass through feedforward layers
        #x = avg_embedded
        for feedforward in self.feedforwards:
            x = feedforward(x)
        # 2c.) Project so that output dim equals number of classes
        x = self.projection(x)
        #x = self.softmax(x) # torch.Size([32, 4])
        return x
        #pass

    def forward(self, data, mask):
        # since I use (-1) to represents padding in the data, let's mask it out first
        data = (data * mask)
        #data = F.one_hot(data,num_classes=self.emb_input_dim)
        embedded = self.embedding(data) # torch.Size([32, 128, 50])
        
        return self.from_embedding(embedded, mask)


class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    hidden_size : int
        Dimension size for hidden states within the LSTM
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    """

    def __init__(
        self, emb_input_dim=0, emb_output_dim=0, hidden_size=50, num_classes=2, dr=0.2
    ):
        super(LSTMTextClassifier, self).__init__()
        ## .. TODO ..
        ## initialize model

        # 1.) Set up embeddings (nn.Embedding)
        # 2.) Set up LSTM encoder (nn.LSTM)
        #   * The assignment doesn't specify how many layers (sol'n: 2)
        # 3.) Set up global MAX pooling (nn.AdaptiveMaxPool1d)
        # 4.) Set up final projection layer (hidden -> output)
        #   * Don't forget dropout!
        #   * These can be combined using nn.Sequential

        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim) # torch.Size([32, 128, 50])
        self.lstm = nn.LSTM(input_size=emb_output_dim, hidden_size=hidden_size, num_layers=1, batch_first=True) # torch.Size([32, 128, 100])
        # then I'll transpose it to torch.Size([32, 100, 128]), tensor = tensor.transpose(1, 2)
        self.maxpool = nn.AdaptiveMaxPool1d(1) # [32, 100, 1] tensor = torch.squeeze(tensor)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.LazyLinear(num_classes)

        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)
        """elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)"""

    def from_embedding(self, embedded, mask):
        """
        This function should take care of the rest of .forward() after the
        tokens have been embedded. That is, it should take the embedded tokens,
        pass them through the LSTM encoder and apply global max pooling and a
        feedforward layer to obtain a vector of size `num_classes`.
        """
        ## .. TODO ..
        ## forward pass (from the outputs of the embedding)
        # torch.Size([32, 128, 50])
        """
        # 1.) Mask out embeddings that belong to padding tokens
        for i in range(embedded.shape[0]): #32
            for j in range(embedded.shape[1]): #128
                if(mask[i][j]==0):
                    embedded[i][j] = 0
        """
        
        """
        x = self.lstm(embedded) # torch.Size([32, 16, 100])
        #print(x[0].shape)
        x = self.projection(x[0]) #torch.Size([32, 16, 8])
        x = x.transpose(1,2) #torch.Size([32, 8, 16])
        #x = self.maxpool(x[0])
        #x = torch.squeeze(x)
        #x = self.dropout(x)
        #print(x.shape)
        #x = self.projection(x)
        #print(x.shape)
        #sys.exit()
        """
        x = self.lstm(embedded) # torch.Size([32, 128, 100])
        #print(x.shape)
        x = x[0].transpose(1,2) # torch.Size([32, 100, 128])
        x = self.maxpool(x) # [32, 100, 1]
        x = torch.squeeze(x) # [32, 100]
        x = self.dropout(x)
        x = self.projection(x) # [32, 4]
        #sys.exit()
        return x

    def forward(self, data, mask):
        # since I use (-1) to represents padding in the data, let's mask it out first
        #print(data.shape)
        data = (data * mask)
        #print(data.shape)
        #data = F.one_hot(data,num_classes=self.emb_input_dim)
        embedded = self.embedding(data) # torch.Size([32, 128, 50])
        #print(data.shape)
        return self.from_embedding(embedded, mask)


class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0,
                 num_classes=2,
                 dr=0.2,
                 filter_widths=[3,4],
                 num_filters=100,
                 num_conv_layers=3,
                 intermediate_pool_size=3, **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)
        ## .. TODO ..
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        # create the list "convblocks" consisting of only first convblock
        convblocks = [nn.ModuleList([
            nn.ModuleList([nn.Conv1d(in_channels=emb_output_dim, out_channels=num_filters,
                                     kernel_size=filter_width) for filter_width in filter_widths]),
            nn.ReLU(),
            nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
        ])]
        # append subsequent convblocks
        for i in range(num_conv_layers - 1):
            convblocks.append(nn.ModuleList([
                nn.ModuleList([nn.Conv1d(in_channels=(num_filters*len(filter_widths)), out_channels=num_filters,
                                         kernel_size=filter_width) for filter_width in filter_widths]),
                nn.ReLU(),
                nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
            ]))
        self.convblocks = nn.ModuleList(convblocks)
        self.globalpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.projection = nn.Linear(in_features=(num_filters*len(filter_widths)), out_features=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded, mask):
        ## .. TODO ..
        x = embedded.transpose(1, 2).contiguous() #[32, 128, 50] and then becomes [32, 50, 128]
        for i in range(len(self.convblocks)):
            convblock = self.convblocks[i] #first block's input size: [32, 50, 128]
            xs = [conv(x) for conv in convblock[0]] # convolution: output sizes [32, 100, 126] and [32, 100, 125]
            xs = [convblock[1](x) for x in xs] #relu
            xs = [convblock[2](x) for x in xs] #maxpool: [32, 100, 42] and [32, 100, 42]
            # I am going to concat along the kernel dimension (100 in this case)
            min_output_length = min([x.shape[2] for x in xs])
            xs = [torch.narrow(input=x, dim=2, start=0, length=min_output_length) for x in xs] #ensure that both tensors are [32, 100, 42]
            x = torch.cat(xs, dim=1) #[32, 200, 42]
            """ Not sure if this is the "correct" method, but I find this to be quite efficient.
            Since each kernal has similar length, few information is lost, especially with ceil_mode=True in MaxPool """
            #2nd round: [32, 200, 13], #3rd round: [32, 200, 4]
        x = torch.squeeze(self.globalpool(x)) # [32, 200, 1] and then becomes [32, 200]
        x = self.projection(x) # [32, 4]
        return x

    def forward(self, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(embedded, mask)


