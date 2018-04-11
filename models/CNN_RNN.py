from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(CNN_RNN,self).__init__()
        self.model_name = 'CNN_RNN'
        self.args = args
        
        Ks = args.kernel_sizes
        Ci = args.embed_dim
        Co = args.kernel_num
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.convs = nn.ModuleList([ nn.Sequential(
                                            nn.Conv1d(Ci,Co,K),
                                            nn.BatchNorm1d(Co),
                                            nn.LeakyReLU(inplace=True),

                                            nn.Conv1d(Co,Co,K),
                                            nn.BatchNorm1d(Co),
                                            nn.LeakyReLU(inplace=True)
                                     )
                                    for K in Ks])
        self.sent_RNN = nn.GRU(
                        input_size = Co * len(Ks),
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.fc = nn.Sequential(
                nn.Linear(2*H,2*H),
                nn.BatchNorm1d(2*H),
                nn.Tanh()
                )
        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out
    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out
    def forward(self,x,doc_lens):
        sent_lens = torch.sum(torch.sign(x),dim=1).data 
        H = self.args.hidden_size
        x = self.embed(x)                                                       # (N,L,D)
        # word level GRU
        x = [conv(x.permute(0,2,1)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        # make sent features(pad with zeros)
        x = self.pad_doc(x,doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)
        docs = self.fc(docs)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = docs[index].unsqueeze(0)
            s = Variable(torch.zeros(1,2*H)).cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]])).cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]])).cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
