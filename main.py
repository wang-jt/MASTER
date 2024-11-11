from master import MASTERModel
import pickle

universe = 'csi300' # or 'csi800'

# Please install qlib first before load the data.
with open(f'/usr/wjt/master/data/{universe}/{universe}_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)
with open(f'/usr/wjt/master/data/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'/usr/wjt/master/data/{universe}/{universe}_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)
print("Data Loaded.")

d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index=158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 10
elif universe == 'csi800':
    beta = 5

n_epoch = 40
lr = 8e-6
GPU = 0
seed = 0
train_stop_loss_thred = 0.95
Load_And_Test = False

model = MASTERModel(
    d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
    beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
    n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
    save_path='/usr/wjt/master/model/', save_prefix=universe
)

if not Load_And_Test:
    model.fit(dl_train, dl_valid)
    print("Model Trained.")
    predictions, metrics = model.predict(dl_test)
    print(metrics)
else:
    # Load and Test
    param_path = f'/usr/wjt/master/model/{universe}master_0.pkl'
    print(f'Model Loaded from {param_path}')
    model.load_param(param_path)
    predictions, metrics, targets = model.predict(dl_test)
    print(metrics)
    from plot import plot_predictions_vs_target as ppt, plot_multiple_stocks as pms
    for s_code in ['SH601318', 'SH601939', 'SH601995', 'SH601857']:
        ppt(predictions, targets, s_code, '58', universe)
    #pms(predictions, targets, ['SH601318', 'SH601939', 'SH601995', 'SH601857'])


