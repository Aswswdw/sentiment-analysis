import torch
import torch.nn as nn
from utils import load_SA_data


class SentimentRNN(nn.Module):
    """基于 RNN 的情感分析模型。

    前向流程：
        索引 -> 词向量(Embedding) -> RNN/LSTM/GRU 编码 -> 时间步平均池化 -> 全连接 -> 二分类

    对应课件里「RNN 做句子分类」一节：
        - 句子编码用「对所有隐状态求平均」的方法（比只用最后一步隐状态更好）
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 rnn_type='lstm', bidirectional=False, num_classes=2, dropout=0.0):
        super().__init__()

        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type.lower()]
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)

        
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        
        emb = self.embedding(x)              
        out, _ = self.rnn(emb)               
        sent = out.mean(dim=1)               
        return self.fc(sent)                 


def evaluate(model, data_iter, device):
    """在给定数据集上计算分类准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            pred = logits.argmax(dim=1)
            
            correct = correct + (pred == labels).sum()
            total += labels.size(0)
    return correct.item() / total   


def train(model, train_iter, test_iter, epochs, lr, device, save_path=None):
    """训练一个模型。

    - 每个 epoch 打印 loss / train_acc / test_acc
    - 当测试准确率创新高时，把模型保存到 save_path（标准做法：存「最佳」而非「最后」的模型）
    - 返回测试集上的最佳准确率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for features, labels in train_iter:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            
            total_loss = total_loss + loss.detach()
            n_batches += 1

        train_acc = evaluate(model, train_iter, device)
        test_acc = evaluate(model, test_iter, device)
        avg_loss = total_loss.item() / n_batches
        print(f'epoch {epoch:3d}/{epochs} | loss {avg_loss:6.4f} '
              f'| train_acc {train_acc:.4f} | test_acc {test_acc:.4f}', flush=True)

        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f'  -> 已保存最佳模型到 {save_path} (test_acc = {best_test_acc:.4f})')

    return best_test_acc


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    data_dir = 'data/'          
    batch_size, max_len = 16, 100
    epochs, lr = 10, 1e-3

    train_iter, test_iter, vocab = load_SA_data(
        data_dir + 'pos.csv', data_dir + 'neg.csv', batch_size, max_len)

    device = get_device()
    print(f'vocab size = {len(vocab)} | device = {device}')

    
    rnn_types = ['rnn', 'lstm', 'gru']

    results = {}
    for rnn_type in rnn_types:
        print(f'\n===== 训练 {rnn_type.upper()} =====')
        model = SentimentRNN(vocab_size=len(vocab),
                             embed_dim=128,
                             hidden_dim=128,
                             num_layers=1,
                             rnn_type=rnn_type,
                             bidirectional=False)
        model.to(device)

        results[rnn_type] = train(model, train_iter, test_iter,
                                  epochs, lr, device,
                                  save_path=f'model_{rnn_type}.pth')


    print('\n========== 模型对比表 ==========')
    print(f'{"模型":<8}{"测试准确率":<12}')
    print('-' * 22)
    for name, acc in results.items():
        print(f'{name.upper():<8}{acc:.4f}')

    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write('模型\t测试准确率\n')
        for name, acc in results.items():
            f.write(f'{name.upper()}\t{acc:.4f}\n')
    print('\n对比表已写入 results.txt')


if __name__ == '__main__':
    main()
