import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################################################################################
class BasicDependencyParserModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, word_emb_dim=100, pos_emb_dim=25,
                 hidden_dim=125, mlp_dim_out=100, lstm_layers=2):

        super(BasicDependencyParserModel, self).__init__()

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)

        self.lstm = nn.LSTM(input_size=(word_emb_dim+pos_emb_dim), hidden_size=hidden_dim, num_layers=lstm_layers,
                            bidirectional=True)
        self.mlp_h = nn.Linear(hidden_dim*2, mlp_dim_out)
        self.mlp_m = nn.Linear(hidden_dim*2, mlp_dim_out)
        self.activation = nn.Tanh()
        self.mlp = nn.Linear(mlp_dim_out, 1)

    def forward(self, sentence):
        word_embed_idx, pos_embed_idx, headers, _ = sentence
        word_embeds = self.word_embedding(word_embed_idx.to(device))  # [batch_size, seq_length, emb_dim]
        pos_embeds = self.pos_embedding(pos_embed_idx.to(device))  # [batch_size, seq_length, emb_dim]
        embeds = torch.cat((word_embeds, pos_embeds), dim=2)  # [batch_size, seq_length, 2*emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        h_out = self.mlp_h(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]
        m_out = self.mlp_m(lstm_out).view(1, lstm_out.shape[0], -1)   # [batch_size, seq_length, mlp_size]

        # calculate scores
        scores = torch.unsqueeze(h_out, 2) + torch.unsqueeze(m_out, 1)  # [batch_size, seq_length, seq_length, mlp_size]
        scores = self.mlp(self.activation(scores))  # [batch_size, seq_length, seq_length, 1]
        scores = scores.view(1, scores.shape[1], scores.shape[2])
        return scores

    ##################################################################################################################
    def evaluate(model, test_dataloader):
        acc = 0
        with torch.no_grad():
            for batch_idx, input_data in enumerate(test_dataloader):
                words_idx_tensor, pos_idx_tensor, sentence_length = input_data
                tag_scores = model(words_idx_tensor)
                tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)

                _, indices = torch.max(tag_scores, 1)
                acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
            acc = acc / len(test_dataloader)
        return acc

    ##################################################################################################################
    def save(self, path):
        torch.save(self, path)

    ##################################################################################################################
    @staticmethod
    def load(path):
        net = torch.load(path)
        net.eval()
        return net


