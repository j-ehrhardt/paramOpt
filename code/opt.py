from model import *
from utils import *
from tqdm import tqdm
from data import *

import random
import copy


class GeneticAlgorithm:
    def __init__(self, hparam, model, model_type):
        self.hparam = hparam
        self.model = load_model(self.hparam['MODEL_DIR'], model_type=model_type) if model is None else model

        self.population = None
        self.population_size = 30
        self.parent_gen = None
        self.n_parents = self.population_size // 2

        self.opt_vars = None

        self.random_mutation_rate = 0.2
        self.swap_mutation_rate = 0.2

        self.loss = nn.MSELoss()

    def init_individual(self, x_in, opt_vars, lower_bound=-1, upper_bound=1, method='random'):
        individual = x_in[:]

        if method == 'random':
            for i, opt in enumerate(opt_vars):
                if opt:
                    individual[i] = random.uniform(lower_bound, upper_bound)
        elif method == 'plusminus':
            for i, opt in enumerate(opt_vars):
                if opt:
                    val = random.uniform(lower_bound, upper_bound)
                    individual[i] = individual[i] + val
        elif method == 'grid':
            step = (upper_bound - lower_bound) / self.population_size
            values = [lower_bound + step * i for i in range(self.population_size - 1)] # TODO maybe remove the -1
            index = 0
            for i, opt in enumerate(opt_vars):
                if opt:
                    individual[i] = values[index % len(values)]
                    index += 1
        return individual

    def init_population(self, x_in, opt_vars):
        self.opt_vars = opt_vars
        self.population = []
        for i in range(0, self.population_size):
            individual = self.init_individual(x_in, opt_vars, method='plusminus')
            self.population.append(individual)
        return self.population

    def init_parent_gen(self):
        parent_gen = random.sample(self.population, self.n_parents)
        return parent_gen

    def select_parents(self, next_gen, next_gen_y, next_gen_loss_x, next_gen_loss_y, n_parents):
        sorted_individuums = sorted(zip(next_gen, next_gen_y, next_gen_loss_x, next_gen_loss_y), key=lambda x: x[3])

        top_n = sorted_individuums[:n_parents]
        parent_gen, parent_gen_y, top_losses_x, top_losses_y = zip(*top_n)

        parent_gen = list(parent_gen)
        parent_gen_y = list(parent_gen_y)
        top_losses_x = list(top_losses_x)
        top_losses_y = list(top_losses_y)
        return parent_gen, top_losses_x[0], top_losses_y[0], parent_gen[0], parent_gen_y[0]

    def pairing(self, parent1, parent2, opt_vars):
        child1, child2 = parent1[:], parent2[:]

        for i, opt in enumerate(opt_vars):
            if opt:
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def random_mutation(self, individual, opt_vars, mutation_rate):
        for i, opt in enumerate(opt_vars):
            if opt and random.uniform(0, 1) < mutation_rate:
                individual[i] = individual[i] + random.uniform(-5.0, 5.0)
        return individual

    def step(self, x_gt, y_gt, i):
        if i == 0:
            self.parent_gen = self.init_parent_gen()

        next_gen, next_gen_y, next_gen_y_loss, next_gen_x_loss = [], [], [], []

        # derive device from x_gt or y_gt
        device = x_gt.device

        while len(next_gen) < self.population_size:
            # parent sampling and pairing
            parent1, parent2 = random.sample(self.parent_gen, 2)
            child1, child2 = self.pairing(parent1=parent1, parent2=parent2, opt_vars=self.opt_vars)

            # random mutation
            child1 = self.random_mutation(child1, opt_vars=self.opt_vars, mutation_rate=self.random_mutation_rate)
            child2 = self.random_mutation(child2, opt_vars=self.opt_vars, mutation_rate=self.random_mutation_rate)

            # make tensors on the right device
            t_child1 = torch.tensor(child1, dtype=torch.float32, device=device).unsqueeze(0)
            t_child2 = torch.tensor(child2, dtype=torch.float32, device=device).unsqueeze(0)

            self.model.eval()
            y_hat1 = self.model(t_child1)
            y_hat2 = self.model(t_child2)

            # losses stay on device, tolist() will move them to CPU implicitly
            y_loss1 = self.loss(y_hat1, y_gt).detach().tolist()
            y_loss2 = self.loss(y_hat2, y_gt).detach().tolist()

            x_loss1 = self.loss(t_child1, x_gt).detach().tolist()
            x_loss2 = self.loss(t_child2, x_gt).detach().tolist()

            # logging
            next_gen.append(child1)
            next_gen.append(child2)

            next_gen_y.append(y_hat1.detach().item())
            next_gen_y.append(y_hat2.detach().item())

            next_gen_y_loss.append(y_loss1)
            next_gen_y_loss.append(y_loss2)

            next_gen_x_loss.append(x_loss1)
            next_gen_x_loss.append(x_loss2)

        self.population = next_gen

        self.parent_gen, loss_x, loss_y, x_hat, y_hat = self.select_parents(
            next_gen=self.population,
            next_gen_y=next_gen_y,
            next_gen_loss_x=next_gen_x_loss,
            next_gen_loss_y=next_gen_y_loss,
            n_parents=self.n_parents
        )

        return loss_x, loss_y, x_hat, y_hat


class BeamSearch:
    def __init__(self, hparam, model, model_type):
        self.hparam = hparam
        self.model = load_model(self.hparam['MODEL_DIR'], model_type) if model is None else model
        if 'tabpfn' not in model_type: self.model.eval()
        self.loss = nn.MSELoss()

    def init_beam(self, x_in):
        # set beam parameters
        self.beam_width = 5
        self.candidates_per_step = 25
        self.perturbation = 1.0

        # initialize beam
        intial_vector = copy.deepcopy(x_in)
        for idx in range(len(intial_vector[0])):
            if self.hparam["FOR_OPT_PARAMS"][idx] == True:
                intial_vector[0][idx] = 0
        initial_beam = [(intial_vector, 1.0)]
        return initial_beam

    def step(self, beam, x_gt, y_gt):
        new_beam = copy.deepcopy(beam)
        for candidate, score in beam:
            # Generate candidates by sampling values for all missing indices
            for _ in range(self.candidates_per_step):
                new_candidate = copy.deepcopy(candidate)
                for idx in range(len(candidate[0])):
                    if self.hparam["FOR_OPT_PARAMS"][idx] == True:
                        # defines the exploration around the beam
                        new_candidate[0][idx] = new_candidate[0][idx] + (np.random.uniform(*(-1,1)) * self.perturbation)

                # Evaluate the candidate using the model
                y_hat = self.model(new_candidate)
                new_score = self.loss(y_gt, y_hat).item()

                # Add the new candidate and its score to the new beam
                new_beam.append((new_candidate, new_score))

        # Sort the new beam by score and keep the top beam_width candidates
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=False)
        beam = new_beam[:self.beam_width]

        # Calculate the losses for the best result
        x_hat = beam[0][0]
        x_loss = self.loss(x_hat, x_gt)#.item()
        y_hat = self.model(x_hat)
        y_loss = self.loss(y_hat[0], y_gt[0])#.item()

        return beam, x_loss, y_loss, x_hat, y_hat


class ParamOpt:
    def __init__(self, hparam, model, model_type):
        self.hparam = hparam
        self.model_type = model_type
        self.model = load_model(self.hparam['MODEL_DIR'], self.model_type) if model is None else model
        if 'tabpfn' not in self.model_type: self.model.eval()
        self.loss = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.min_bound, self.max_bound = self.init_observer(hparam)
        #print(f'min bound {self.min_bound}, max bound {self.max_bound}')

    def init_observer(self, hparam):
        data_module = DataModuleUsw(hparam=hparam, modus='random', scaling=False)
        train_loader, _ = data_module.get_train_dataloaders()

        # Extract all data from the DataLoader
        all_x = []
        for batch_x, batch_y in train_loader:
            all_x.append(batch_x)

        # Concatenate all batches
        training_data = torch.cat(all_x, dim=0)

        # Compute bounds across all samples
        min_bounds = training_data.min(dim=0)[0].tolist()
        max_bounds = training_data.max(dim=0)[0].tolist()

        return min_bounds, max_bounds

    def init_optimizer(self, x_guess, lr=None):
        lr = lr if lr is not None else self.hparam['OPT_LR']

        params = x_guess if isinstance(x_guess, list) else [x_guess]

        # select optimizer
        if self.hparam['OPT'] == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr)
        elif self.hparam['OPT'] == 'SGD-nesterov':
            optimizer = torch.optim.SGD(params, lr=lr, nesterov=True, momentum=self.hparam['OPT_MOMENTUM'])
        elif self.hparam['OPT'] == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        elif self.hparam['OPT'] == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr=lr)
        elif self.hparam['OPT'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr=lr)

        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def predict_tabpfn(self, x_t: torch.Tensor):
        x_np = x_t.detach().cpu().numpy()
        y_np = self.model.predict(x_np)
        y_t = torch.from_numpy(y_np).to(torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)  # (N,1)
        return y_t

    def finite_diff_approx(self, x_guess, y_gt, epsilon):
        grads = []

        for i, t in enumerate(x_guess):
            if not t.requires_grad:
                grads.append(0.0)
                continue

            # +epsilon
            plus = [ti.clone().detach() for ti in x_guess]
            plus[i] = plus[i] + epsilon
            x_plus = torch.cat(plus, dim=0).unsqueeze(0)
            y_plus = self.predict_tabpfn(x_plus)
            loss_plus = self.loss(y_plus.squeeze(0), y_gt).item()

            # -epsilon
            minus = [ti.clone().detach() for ti in x_guess]
            minus[i] = minus[i] - epsilon
            x_minus = torch.cat(minus, dim=0).unsqueeze(0)
            y_minus = self.predict_tabpfn(x_minus)
            loss_minus = self.loss(y_minus.squeeze(0), y_gt).item()

            grads.append((loss_plus - loss_minus) / (2.0 * epsilon))
        return grads

    def grad_update(self, x_guess, grads, lr):
        with torch.no_grad():
            for idx, (x_var, grad) in enumerate(zip(x_guess, grads)):
                if x_var.requires_grad:
                    x_var.sub_(lr * torch.tensor(grad, dtype=x_var.dtype))
                    #x_var.add_(lr * torch.tensor(grad, dtype=x_var.dtype))

                    # when using observer
                    x_var.clamp_(min=self.min_bound[idx], max=self.max_bound[idx])
        return x_guess

    def step(self, x_guess, x_gt, y_gt, opt_vars):
        if 'tabpfn' in self.model_type:
            x_guess, x_gt, y_gt, y_gt = x_guess.to('cpu'), x_gt.to('cpu'), y_gt.to('cpu'), y_gt.to('cpu')

            grads = self.finite_diff_approx(x_guess, y_gt, epsilon=1e0)
            x_guess = self.grad_update(x_guess, grads, lr=1e-3)
            y_hat = self.predict_tabpfn(x_guess)

            loss_y = self.loss(y_hat.squeeze(0), y_gt)
            loss_x = self.loss(x_guess, x_gt)

            x_guess, x_gt, y_gt, y_hat = x_guess.to(self.device), x_gt.to(self.device), y_gt.to(self.device), y_hat.to(self.device)
        else:
            x_guess = [param.clone().detach().requires_grad_(True) for param in x_guess.squeeze(0)]

            for i, param in enumerate(x_guess):
                param.requires_grad_(opt_vars[i])

            optimizer = self.init_optimizer(x_guess=x_guess, lr=self.hparam['OPT_LR'])

            x_tensor = torch.stack(x_guess).unsqueeze(0)
            y_hat = self.model(x_tensor)
            if self.model_type == 'hres' or self.model_type == 'mdn':
                y_hat = y_hat[:, 0]

            loss_elem = self.loss(y_hat, y_gt.unsqueeze(0))
            loss_elem.backward()

            optimizer.step()

            # projected gradient descent
            with torch.no_grad():
                for i, param in enumerate(x_guess):
                    if param.requires_grad:
                        param.clamp_(min=self.min_bound[i], max=self.max_bound[i])

            x_guess = torch.stack(x_guess).unsqueeze(0)

            loss_y = self.loss(y_hat.squeeze(dim=0), y_gt)  # squeeze gt because batch size 1
            loss_x = self.loss(x_guess.squeeze(), x_gt.squeeze())  # squeeze gt because batch size 1

        return loss_x.detach(), loss_y.detach(), x_guess.detach(), y_hat.detach()


class OptModule():
    def __init__(self, hparam, model):
        self.hparam = hparam
        self.model = model
        self.model_type = hparam['MODEL_TYPE']
        self.loss = nn.MSELoss()
        self.threshold = hparam['OPT_THRESHOLD']
        self.method = hparam['METHOD']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.GA = GeneticAlgorithm(self.hparam, self.model, self.model_type)
        self.BS = BeamSearch(self.hparam, self.model, self.model_type)
        self.GS = ParamOpt(self.hparam, self.model, self.model_type)

    def find_params(self, x_guess, y_guess, x_gt, y_gt, opt_vars, lr=None):
        """
        This function holds the core functionalities for reconstructing paramters.
        As guesses only tensors of dim1 are allowed ... no batches.

        :param x_guess, y_guess: guesses for reconstruction standard at 0, other initializations possible.
        :param x_gt, y_gt: ground truths for plotting.
        :param opt_vars: Is a list of booleans that describes which input parameters to optimize.
        :param method: Describes whether uninformed search (us), semi-informed search (sis) or gradient-based search (is) should be used.
        """

        if x_guess.dim() == 1:
            x_guess = x_guess.unsqueeze(0)
            x_gt = x_gt.unsqueeze(0)
            y_guess = y_guess.unsqueeze(0)
            y_gt = y_gt.unsqueeze(0)

        #x_guess = x_guess.to(self.device).clone().detach().requires_grad_(True) # detach and reattach

        x_guess = x_guess.to(self.device)

        if self.method == "is":
            x_guess = x_guess.clone().detach().requires_grad_(True)
        else:
            x_guess = x_guess.clone().detach()

        y_guess = y_guess.to(self.device)
        x_gt = x_gt.to(self.device)
        y_gt = y_gt.to(self.device)

        # init methods
        if self.method == "us": # uninformed search
            print(f'genetic algorithm')
            x_in = x_guess.squeeze().tolist()
            self.GA.init_population(x_in=x_in, opt_vars=opt_vars)
        if self.method == "sis": # semi-informed search
            print(f'beamsearch')
            beam = self.BS.init_beam(x_in=x_guess)
        if self.method == "is": # informed search
            print(f'paramopt | optimizer {self.hparam["OPT"]}')

        prediction_variance_runs = 8 #16 TODO change back to 16
        modus = 'input_variance' # 'dropout_variance', 'input_variance'
        # add variance through noise and dropout then caclulating mean and variance of multple runs
        x_guess_history = []

        # meta logging lists
        logging_x_ground_truth = []
        logging_y_ground_truth = []
        logging_x_guess = []
        logging_y_prediction = []

        with tqdm(total=prediction_variance_runs, desc="progress") as pbar:
            for var_seed in range(0, prediction_variance_runs):
                # per variance logging lists
                logging_x_ground_truth_ind = []
                logging_y_ground_truth_ind = []
                logging_x_guess_ind = []
                logging_y_prediction_ind = []

                # warmup and early stop
                max_cycles = self.hparam['OPT_MAX_CYCLES']
                max_consecutive_cycles = self.hparam['OPT_MAX_CON_CYCLES']

                warmup_window = 20
                consecutive_cycles = 0
                n_cycles = 0
                B = x_gt.size(0)

                best_losses = torch.full((B,), float('inf'), device=self.device)
                stalls = torch.zeros(B, dtype=torch.long, device=self.device)

                loss_history = []

                # applying variance either in form of dropout
                if modus == 'dropout_variance':
                    random.seed(var_seed)
                    torch.manual_seed(var_seed)
                elif modus == 'input_variance':
                    random.seed(var_seed)
                    epsilon = random.random() * 10
                    #x_guess = x_guess + epsilon
                    mask = torch.tensor([1.0 if v else 0.0 for v in opt_vars], device=self.device, dtype=x_guess.dtype)
                    x_guess = x_guess + epsilon * mask.unsqueeze(0)


                while consecutive_cycles < max_consecutive_cycles and n_cycles < max_cycles:
                    if self.method == "us":  # uninformed search
                        x_loss, y_loss, x_guess, y_hat = self.GA.step(x_gt=x_gt, y_gt=y_gt, i=n_cycles)
                        x_guess = torch.tensor(x_guess, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if self.method == "sis":  # semi-informed search
                        beam, x_loss, y_loss, x_guess, y_hat = self.BS.step(beam=beam, x_gt=x_gt, y_gt=y_gt)

                    if self.method == "is": #  informed search
                        x_loss, y_loss, x_guess, y_hat = self.GS.step(x_guess=x_guess, x_gt=x_gt, y_gt=y_gt, opt_vars=opt_vars)
                        #x_loss, y_loss, x_guess, y_hat = x_loss.detach().tolist(), y_loss.detach.tolist(), x_guess.detach().tolist(), y_hat.detach().tolist()
                        x_loss, y_loss, x_guess, y_hat = x_loss.detach(), y_loss.detach(), x_guess.detach(), y_hat.detach()

                    # bring y_loss to float
                    if isinstance(y_loss, torch.Tensor):
                        y_loss_tensor = y_loss.to(self.device)
                        y_loss_float = float(y_loss_tensor.item())
                    else:
                        y_loss_float = float(y_loss)
                        y_loss_tensor = torch.tensor(y_loss_float, device=self.device)

                    loss_history.append(y_loss_float)

                    # per-sample stall tracking
                    if n_cycles >= warmup_window:
                        if n_cycles == warmup_window:
                            baseline = sum(loss_history[:warmup_window]) / max(1, len(loss_history[:warmup_window]))
                            min_delta = max(1e-6, 0.01 * baseline)

                        improved = (y_loss_tensor < (best_losses - min_delta))
                        best_losses = torch.minimum(best_losses, y_loss_tensor)
                        stalls = torch.where(improved, torch.zeros_like(stalls), stalls + 1)
                    else:
                        best_losses = torch.minimum(best_losses, y_loss_tensor)

                    # global “no improvement” stopping
                    if len(loss_history) > 1:
                        delta = abs(loss_history[-1] - loss_history[-2])  # both floats
                        threshold = self.hparam['OPT_THRESHOLD']
                        if delta <= threshold:
                            consecutive_cycles += 1
                        else:
                            consecutive_cycles = 0

                    logging_x_ground_truth_ind.append(x_gt)
                    logging_y_ground_truth_ind.append(y_gt)
                    logging_x_guess_ind.append(x_guess)
                    logging_y_prediction_ind.append(y_hat)

                    n_cycles += 1


                x_guess_history.append(x_guess)

                logging_x_ground_truth.append(logging_x_ground_truth_ind)
                logging_y_ground_truth.append(logging_y_ground_truth_ind)
                logging_x_guess.append(logging_x_guess_ind)
                logging_y_prediction.append(logging_y_prediction_ind)

                pbar.update(1)

        return logging_x_ground_truth, logging_y_ground_truth, logging_x_guess, logging_y_prediction
