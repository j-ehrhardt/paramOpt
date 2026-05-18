import gc
import math
import random
from tqdm import tqdm
import shutil

from sklearn.metrics import mean_squared_error

from data import *
from model.net import *
from tabpfn import TabPFNRegressor
from tabpfn_extensions import AutoTabPFNRegressor


class TrainModule():
    def __init__(self, hparam, modus, scaling=False):
        """
        tabpfn --> TabPFN Regressor
        autotabpfn --> AutoTabPFN Regressor
        mdn --> MixtureDensity Model
        hres --> Heteroscedatic Residual Network
        res --> Residual Network
        ff --> Feed Forward Network
        """
        self.hparam = hparam                            # all data
        self.modus = modus                              # data modus, from which part of the table is the test set selected?: start, middle, end, random
        self.scaling = scaling                          # use scaler for data
        self.model_type = self.hparam['MODEL_TYPE']     # which model_tpye:tabpfn, autotabpfn, mdn, hres, ff, res

        self.loss_l1 = nn.SmoothL1Loss()
        self.loss_mse = nn.MSELoss()
        self.loss_gnll = nn.GaussianNLLLoss()

        self.model = None                               # model placeholder

        model_devices = {'tabpfn': 'cpu', 'autotabpfn': 'cuda'}
        self.device = model_devices.get(hparam['MODEL_TYPE'], 'cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def gaussian_nll(self, y_hat, y):
        mu = y_hat[:, 0]
        logvar = y_hat[:, 1]
        var = torch.exp(logvar).clamp(min=1e-6)
        return self.loss_gnll(mu, y.squeeze(), var)

    def mdn_loss(self, y_hat, y):
        mu, log_var, pi = torch.split(y_hat, 8, dim=1)
        var = torch.exp(log_var).clamp(min=1e-6)
        y = y.expand_as(mu)
        comp = -0.5 * ((y - mu) ** 2 / var + log_var + math.log(2 * math.pi))
        log_prob = torch.logsumexp(torch.log(pi + 1e-8) + comp, dim=1)
        return -log_prob.mean()

    @staticmethod
    def get_tabpfn_data(dl):
        # get dataloader and transform in tabular data for tabPFN
        xs, ys = [], []
        for batch in dl:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch['x'], batch['y']
            else:
                x, y = batch, None
            xs.append(x)
            if y is not None: ys.append(y)

        xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)

        X = xs.detach().cpu().numpy()
        Y = ys.detach().cpu().numpy()
        return X, Y

    def train_tabpfn(self, dl_train, dl_val):
        x_train, y_train = self.get_tabpfn_data(dl_train)
        x_val, y_val = self.get_tabpfn_data(dl_val)

        if self.model_type == 'tabpfn':
            self.model = TabPFNRegressor()
        elif self.model_type == 'autotabpfn':
            self.model = AutoTabPFNRegressor(max_time=300, device=self.device)

        self.model.fit(x_train, y_train)
        y_val_hat = self.model.predict(x_val)
        loss_val = float(mean_squared_error(y_val, y_val_hat))

        gc.collect()
        torch.cuda.empty_cache()
        return loss_val

    def test_tabpfn(self, dl_test):
        x_test, y_test = self.get_tabpfn_data(dl_test)
        y_test_hat = self.model.predict(x_test)

        test_groundtruths = y_test.reshape(-1, 1)
        test_predictions = y_test_hat.reshape(-1, 1)

        per_sample_abs_err = np.abs(test_groundtruths - test_predictions)
        test_losses = per_sample_abs_err.ravel().tolist()  # list of Python floats

        test_groundtruths_out = test_groundtruths.ravel().tolist()
        test_predictions_out = test_predictions.ravel().tolist()

        avg_test_loss = float(mean_squared_error(test_groundtruths, test_predictions))

        return avg_test_loss, test_groundtruths_out, test_predictions_out, test_losses

    def train_step(self, dl_train, model, optimizer, device):
        model.train()
        total_loss = 0.0
        for x, y in dl_train:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            if self.model_type == 'mdn': loss = self.mdn_loss(y_hat, y.unsqueeze(1))
            elif self.model_type == 'hres': loss = self.gaussian_nll(y_hat, y.unsqueeze(1))
            else: loss = torch.nn.functional.mse_loss(y_hat, y.unsqueeze(1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dl_train)

    def val_step(self, dl_val, model, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)

                y_hat = model(x)

                if self.model_type == 'mdn': loss = self.mdn_loss(y_hat, y.unsqueeze(1))
                elif self.model_type == 'hres': loss = self.gaussian_nll(y_hat, y.unsqueeze(1))
                else: loss = torch.nn.functional.mse_loss(y_hat, y.unsqueeze(1))

                total_loss += loss.item()
        return total_loss / len(dl_val)

    def test_step(self, dl_test, model, device):
        model.eval()

        test_ground_truths = []
        test_predictions = []
        test_losses = []

        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)

                y_hat = model(x)

                if self.model_type == 'mdn': loss = self.mdn_loss(y_hat, y.unsqueeze(1))
                elif self.model_type == 'hres': loss = self.gaussian_nll(y_hat, y.unsqueeze(1))
                else: loss = torch.nn.functional.mse_loss(y_hat, y.unsqueeze(1))

                test_ground_truths.append(y)
                test_predictions.append(y_hat)
                test_losses.append(loss)

        avg_test_loss = sum(test_losses) / len(dl_test)

        test_ground_truths_out = [t.squeeze().detach().cpu().tolist() for t in test_ground_truths]
        test_predictions_out   = [t.squeeze().detach().cpu().tolist() for t in test_predictions]
        test_losses_out        = [t.detach().cpu().tolist() for t in test_losses]

        avg_test_loss      = avg_test_loss.detach().cpu().tolist()

        return avg_test_loss, test_ground_truths_out, test_predictions_out, test_losses_out

    def training(self):
        hparam = self.hparam
        random.seed(hparam['SEED'])
        np.random.seed(hparam['SEED'])
        torch.manual_seed(hparam['SEED'])

        data_module = DataModuleUsw(hparam, modus=self.modus, scaling=self.scaling)
        dl_train, dl_val = data_module.get_train_dataloaders()
        dl_test = data_module.get_test_dataloader()

        device = self.device

        test_losses = []
        if self.model_type == 'mdn':
            self.model = MixtureDensityNet(input_dim=hparam['INPUT_DIM'], hidden_dim=hparam['HIDDEN_DIM'], output_dim=hparam['OUTPUT_DIM'], n_layers=hparam['N_LAYERS'], dropout=hparam['DROPOUT'])
        elif self.model_type == 'hres':
            self.model = HeteroscedaticResidualNet(input_dim=hparam['INPUT_DIM'], hidden_dim=hparam['HIDDEN_DIM'], output_dim=hparam['OUTPUT_DIM'], n_layers=hparam['N_LAYERS'], dropout=hparam['DROPOUT'])
        elif self.model_type == 'ff':
            self.model = Net(input_dim=hparam['INPUT_DIM'], hidden_dim=hparam['HIDDEN_DIM'], output_dim=hparam['OUTPUT_DIM'], n_layers=hparam['N_LAYERS'], dropout=hparam['DROPOUT'])
        elif self.model_type == 'res':
            self.model = ResidualNet(input_dim=hparam['INPUT_DIM'], hidden_dim=hparam['HIDDEN_DIM'], output_dim=hparam['OUTPUT_DIM'], n_layers=hparam['N_LAYERS'], dropout=hparam['DROPOUT'])
        elif 'tab' in self.model_type:
            validation_loss_avg = self.train_tabpfn(dl_train, dl_val)
            avg_test_loss, test_ground_truths, test_predictions, test_losses = self.test_tabpfn(dl_test)

            results_dict = {
                "ID": "exp1",
                "HPARAMS": hparam,
                "TRAIN_LOSSES": None,
                "VAL_LOSSES": validation_loss_avg,
                "TEST_LOSSES": test_losses,
                "TEST_LOSSES_AVG": avg_test_loss,
                "TEST_PREDICTIONS": test_predictions,
                "TEST_GROUND_TRUTHS": test_ground_truths,
                "LEN_TESTSET": len(dl_test),
            }

            return results_dict

        self.model.apply(self.init_weights)
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=hparam['LR'], weight_decay=hparam['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparam['MAX_EPOCHS'], eta_min=1e-6)

        training_losses = []
        validation_losses = []

        loss_val_step = 1
        with tqdm(range(hparam['MAX_EPOCHS']), unit='epoch') as tepoch:
            # train and val
            for e in tepoch:
                loss_train_step = self.train_step(dl_train, self.model, optimizer, device)
                training_losses.append(loss_train_step)

                scheduler.step()

                if math.isnan(loss_train_step):
                    break

                if e % 5 == 0 and e != 0:
                    loss_val_step = self.val_step(dl_val, self.model, device)
                    validation_losses.append(loss_val_step)

                    tepoch.set_postfix(loss_train=loss_train_step, loss_val=loss_val_step)

                else:
                    tepoch.set_postfix(loss_train=loss_train_step, loss_val=loss_val_step)


            avg_test_loss, test_ground_truths, test_predictions, test_losses = self.test_step(dl_test, self.model, device)

        results_dict = {
            "ID": "exp1",
            "HPARAMS": hparam,
            "TRAIN_LOSSES": training_losses,
            "VAL_LOSSES": validation_losses,
            "TEST_LOSSES": test_losses,
            "TEST_LOSSES_AVG": avg_test_loss,
            "TEST_PREDICTIONS": test_predictions,
            "TEST_GROUND_TRUTHS": test_ground_truths,
            "LEN_TESTSET": len(dl_test),
        }

        gc.collect()
        torch.cuda.empty_cache()
        return results_dict



if __name__ == '__main__':
    hparam = {
        "ID": "ds1_",
        "DS_ID": "ds1",
        "SEED": 42,
        "DEVICE":1,
        "DATA_DIR": "../data/ds4/usw.csv",
        "STUDY_DIR": "../exp/exp2_redodo",
        "LOG_DIR": "../exp/exp2_redodo/ds2/ds2_exp_xx",
        'MODEL_TYPE': 'autotabpfn',
        "N_AUG_SAMPLES": 0,
        "BATCH_SIZE": 1,
        "INPUT_DIM": 3,
        "OUTPUT_DIM": 1,
        "HIDDEN_DIM": 64,
        "N_LAYERS": 4,
        "DROPOUT": 0.2,
        "MAX_EPOCHS": 1000,
        "LR": 0.0005,
        "WEIGHT_DECAY": 0.0001,
        "METHOD": "is",
        "FOR_OPT_PARAMS": [False,False,True],
        "OPT": "SGD",
        "OPT_LR": 0.05,
        "OPT_MOMENTUM": 0.1,
        "OPT_THRESHOLD": 0.1
    }

    TrainingClass = TrainModule(hparam=hparam, modus='random')
    TrainingClass.training()
    TrainingClass.testing()
