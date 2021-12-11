import datetime
import sys

sys.path.append("..")
from ADGAT.Model import *
from Baseline.Model import LSTM
from ADGAT.utils import *
import pickle
import torch
from torch import optim
from tqdm import tqdm
import gc

# args:
device = "0"
max_epoch = 1000
wait_epoch_threshold = 1000
save = True
DEVICE = "cuda:" + device
# DEVICE = "cpu"

criterion = torch.nn.BCELoss()


def load_dataset(DEVICE, relation):
    with open('../Data/relations_author_source/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('../Data/relations_author_source/y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open("../Data/relations_author_source/x_short_pe_ps_pb.pkl", 'rb') as handle:
        alternatives = pickle.load(handle)
    # with open("../Data/relations_author_source/y_.pkl", 'rb') as handle:
    #     alternatives = alternatives.reshape(list(alternatives.shape) + [1])

    markets = markets.astype(np.float64)
    x_market = torch.tensor(markets, device=DEVICE)
    x_market.to(torch.double)

    x_alternative = torch.tensor(alternatives, device=DEVICE)
    x_alternative.to(torch.double)

    if relation != "None":
        with open('../Data/relations_author_source/' + relation + '_relation.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        relation_static = torch.tensor(relation_static, device=DEVICE)
        relation_static.to(torch.double)
    else:
        relation_static = None
    y = torch.tensor(y_load, device=DEVICE)
    y = (y.T > y.median(dim=1)[0]).T.to(torch.long)

    return x_market, y, x_alternative, relation_static


def train(model, x_train, x_alt_train, y_train, relation_static=None, optimizer=None, rnn_length=None, clip=None):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 15
    for i in tqdm(train_seq):
        output = model(x_train[i - rnn_length + 1: i + 1], x_alt_train[i - rnn_length + 1: i + 1],
                       relation_static=relation_static)
        loss = criterion(output, y_train[i].double())
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return total_loss / total_loss_count


def evaluate(model, x_eval, x_alt_eval, y_eval, relation_static=None, rnn_length=None):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_alt_eval[i - rnn_length + 1: i + 1],
                       relation_static=relation_static)
        output = output.detach().cpu()
        preds.append(output.numpy())
        trues.append(y_eval[i].cpu().numpy())
    acc, auc = metrics(trues, preds)
    return acc, auc


def research(max_epoch, hidn_rnn, heads_att, hidn_att, lr, rnn_length, weight_constraint, dropout, clip,
             model_name="AD_GAT", relation="None", random_seed=2021):
    gc.collect()
    torch.cuda.empty_cache()
    task_name = model_name + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    set_seed(random_seed)
    record = dict(max_epoch=max_epoch, hidn_rnn=hidn_rnn, heads_att=heads_att,
                  hidn_att=hidn_att, lr=lr, rnn_length=rnn_length, weight_constraint=weight_constraint,
                  dropout=dropout, clip=clip,
                  model_name=model_name, relation=relation, random_seed=random_seed)
    if relation != "None":
        static = 1
        pass
    else:
        static = 0
        relation_static = None
        
    # load dataset
    print("loading dataset")
    x, y, x_alternative, relation_static = load_dataset(DEVICE, relation)
    # hyper-parameters
    T = x.size(0)
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_ALTER = x_alternative.size(2)
    MAX_EPOCH = max_epoch
    t_mix = 0

    # train-test split
    t_train = int(T * 0.9)
    t_eval = int(T * 0.1)

    x_train = x[: t_train]
    x_eval = x[t_train - rnn_length: t_train + t_eval]
    x_test = x[t_train + t_eval - rnn_length:]
    
    y_train = y[: t_train]
    y_eval = y[t_train - rnn_length: t_train + t_eval]
    y_test = y[t_train + t_eval - rnn_length:]
    
    x_alternative_train = x_alternative[: t_train]
    x_alternative_eval = x_alternative[t_train - rnn_length: t_train + t_eval]
    x_alternative_test = x_alternative[t_train + t_eval - rnn_length:]

    # initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    if model_name == "AD_GAT":
        model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_alter=D_ALTER,
                       d_hidden=D_MARKET, hidn_rnn=hidn_rnn, heads_att=heads_att,
                       hidn_att=hidn_att, dropout=dropout, t_mix=t_mix)
    elif model_name == "LSTM":
        model = LSTM(num_stock=NUM_STOCK, d_market=D_MARKET, d_alter=D_ALTER,
                       d_hidden=D_MARKET, hidn_rnn=hidn_rnn, heads_att=heads_att,
                       hidn_att=hidn_att, dropout=dropout, t_mix=t_mix)
    elif model_name == "DeepQuant":
        pass
    else:
        pass

    if "cuda" in DEVICE:
        model.cuda(device=DEVICE)
    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_constraint)

    # train
    save_file_name = f"../SavedModels/{task_name}/"
    abs_path = os.path.abspath(save_file_name)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)

    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train, x_alternative_train, y_train, relation_static=relation_static, optimizer=optimizer, rnn_length=rnn_length, clip=clip)
        eval_acc, eval_auc = evaluate(model, x_eval, x_alternative_eval, y_eval, relation_static=relation_static, rnn_length=rnn_length)
        test_acc, test_auc = evaluate(model, x_test, x_alternative_test, y_test, relation_static=relation_static, rnn_length=rnn_length)
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch,
                                                                                                                     train_loss,
                                                                                                                     eval_auc,
                                                                                                                     eval_acc,
                                                                                                                     test_auc,
                                                                                                                     test_acc)
        record.update(dict(epoch=epoch, eval_auc=eval_auc, eval_da=eval_acc, test_auc=test_auc, test_da=test_acc))
        print(eval_str)

        with open(fr"../Records/{task_name}.txt", "a") as f:
            f.write(str(record))
            f.write("\n")

        if eval_auc > eval_epoch_best:
            eval_epoch_best = eval_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if save:
                if best_model_file:
                    os.remove(best_model_file)

                best_model_file = save_file_name + "epoch{}_evalauc{:.4f}_da{:.4f}_testauc{:.4f}_da{:.4f}".format(epoch, eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > wait_epoch_threshold:
            print("saved_model_result:", eval_best_str)
            break
        epoch += 1

if __name__ == "__main__":
    # research(max_epoch=100, hidn_rnn=128, heads_att=4, hidn_att=40, lr=5e-4, rnn_length=20, weight_constraint=1e-5, dropout=0.2, clip=0.0001,
    #          model_name="AD_GAT", relation="supply", random_seed=2021)
    research(max_epoch=100, hidn_rnn=10, heads_att=3, hidn_att=3, lr=5e-4, rnn_length=2, weight_constraint=1e-5, dropout=0.2, clip=0.0001,
             model_name="LSTM", relation="supply", random_seed=13)