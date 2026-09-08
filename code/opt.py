from tqdm import tqdm
import random
import copy

import numpy as np
import torch
import torch.nn as nn

from data.usw_data_loader import DataModuleUsw
from model.net import load_model


OBJECTIVE_TARGET_MATCH = 'target_match'
OBJECTIVE_MAXIMIZE = 'maximize'
VALID_OBJECTIVES = {OBJECTIVE_TARGET_MATCH, OBJECTIVE_MAXIMIZE}
METHOD_ALIASES = {'is': 'paramopt', 'sis': 'beam_search', 'us': 'genetic_algorithm'}


def normalize_method(method):
    """Return public method names while accepting legacy result configurations."""
    return METHOD_ALIASES.get(method, method)


def objective_loss(loss, prediction, target, objective):
    """Return the scalar minimisation objective shared by all search methods."""
    if objective == OBJECTIVE_MAXIMIZE:
        return -prediction.mean()
    if objective == OBJECTIVE_TARGET_MATCH:
        if target is None:
            raise ValueError('target_match optimization requires a target quality.')
        return loss(prediction.reshape(-1), target.to(prediction.device).reshape(-1))
    raise ValueError(f'Unknown optimization objective: {objective}')


class GeneticAlgorithm:
    def __init__(self, hparam, model, model_type, min_bound=None, max_bound=None):
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
        self.objective = hparam.get('OPT_OBJECTIVE', OBJECTIVE_TARGET_MATCH)
        self.min_bound = min_bound
        self.max_bound = max_bound

    def project_individual(self, individual, opt_vars):
        """Keep every candidate in the same feasible region as paramOpt."""
        if self.min_bound is None or self.max_bound is None:
            return individual
        for i, optimise in enumerate(opt_vars):
            if optimise:
                individual[i] = min(max(individual[i], self.min_bound[i]), self.max_bound[i])
        return individual

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
        return self.project_individual(individual, opt_vars)

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
        return self.project_individual(individual, opt_vars)

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
            y_loss1 = objective_loss(self.loss, y_hat1, y_gt, self.objective).detach().tolist()
            y_loss2 = objective_loss(self.loss, y_hat2, y_gt, self.objective).detach().tolist()

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
    def __init__(self, hparam, model, model_type, min_bound=None, max_bound=None):
        self.hparam = hparam
        self.model = load_model(self.hparam['MODEL_DIR'], model_type) if model is None else model
        if 'tabpfn' not in model_type: self.model.eval()
        self.loss = nn.MSELoss()
        self.objective = hparam.get('OPT_OBJECTIVE', OBJECTIVE_TARGET_MATCH)
        self.min_bound = min_bound
        self.max_bound = max_bound

    def project_candidate(self, candidate):
        if self.min_bound is None or self.max_bound is None:
            return candidate
        for idx, optimise in enumerate(self.hparam['FOR_OPT_PARAMS']):
            if optimise:
                candidate[0][idx] = candidate[0][idx].clamp(
                    min=self.min_bound[idx], max=self.max_bound[idx]
                )
        return candidate

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
        intial_vector = self.project_candidate(intial_vector)
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
                new_candidate = self.project_candidate(new_candidate)

                # Evaluate the candidate using the model
                y_hat = self.model(new_candidate)
                new_score = objective_loss(self.loss, y_hat, y_gt, self.objective).item()

                # Add the new candidate and its score to the new beam
                new_beam.append((new_candidate, new_score))

        # Sort the new beam by score and keep the top beam_width candidates
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=False)
        beam = new_beam[:self.beam_width]

        # Calculate the losses for the best result
        x_hat = beam[0][0]
        x_loss = self.loss(x_hat, x_gt)#.item()
        y_hat = self.model(x_hat)
        y_loss = objective_loss(self.loss, y_hat, y_gt, self.objective)

        return beam, x_loss, y_loss, x_hat, y_hat


class ParamOpt:
    def __init__(self, hparam, model, model_type):
        self.hparam = hparam
        self.model_type = model_type
        self.model = load_model(self.hparam['MODEL_DIR'], self.model_type) if model is None else model
        if 'tabpfn' not in self.model_type:
            self.model.eval()
            # Optimisation is over the input vector only.  Freezing explicitly
            # avoids accumulating unused surrogate-weight gradients.
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.loss = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.objective = hparam.get('OPT_OBJECTIVE', OBJECTIVE_TARGET_MATCH)
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"OPT_OBJECTIVE must be one of {sorted(VALID_OBJECTIVES)}, got {self.objective!r}."
            )

        self.min_bound, self.max_bound = self.init_observer(hparam)
        #print(f'min bound {self.min_bound}, max bound {self.max_bound}')

    def init_observer(self, hparam):
        """Return deterministic, documented optimisation bounds.

        ``OPT_BOUNDS`` may supply physical limits as ``(lower, upper)``.  If it
        is omitted, use the complete cleaned DoE rather than a stochastic,
        drop-last training loader.  This makes every method operate on the same
        feasible region and avoids a different bound for each test sample.
        """
        configured_bounds = hparam.get('OPT_BOUNDS')
        if configured_bounds is not None:
            lower, upper = configured_bounds
            if len(lower) != hparam['INPUT_DIM'] or len(upper) != hparam['INPUT_DIM']:
                raise ValueError('OPT_BOUNDS must contain one lower and upper value per input feature.')
            return list(lower), list(upper)

        data_module = DataModuleUsw(hparam=hparam, modus='end', scaling=False)
        cleaned_data = data_module.clean(data_module.load_data())
        machine_parameters = torch.tensor(
            cleaned_data.iloc[:, :-1].to_numpy(), dtype=torch.float32
        )
        return (
            machine_parameters.min(dim=0)[0].tolist(),
            machine_parameters.max(dim=0)[0].tolist(),
        )

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

    def init_whitebox_state(self, x_guess):
        """Create one optimised input tensor and one persistent optimizer."""
        x_parameter = nn.Parameter(x_guess.detach().clone())
        return x_parameter, self.init_optimizer(x_parameter, lr=self.hparam['OPT_LR'])

    def predict_tabpfn(self, x_t: torch.Tensor):
        x_np = x_t.detach().cpu().numpy()
        y_np = self.model.predict(x_np)
        y_t = torch.from_numpy(y_np).to(torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)  # (N,1)
        return y_t

    def finite_diff_approx(self, x_guess, y_target, opt_vars, epsilon):
        """Estimate one gradient component per optimised input feature."""
        grads = []

        for i, optimise in enumerate(opt_vars):
            if not optimise:
                grads.append(0.0)
                continue

            # +epsilon
            x_plus = x_guess.clone().detach()
            x_plus[0, i] += epsilon
            y_plus = self.predict_tabpfn(x_plus)
            loss_plus = objective_loss(self.loss, y_plus.squeeze(0), y_target, self.objective).item()

            # -epsilon
            x_minus = x_guess.clone().detach()
            x_minus[0, i] -= epsilon
            y_minus = self.predict_tabpfn(x_minus)
            loss_minus = objective_loss(self.loss, y_minus.squeeze(0), y_target, self.objective).item()

            grads.append((loss_plus - loss_minus) / (2.0 * epsilon))
        return grads

    def grad_update(self, x_guess, grads, opt_vars, lr):
        with torch.no_grad():
            for idx, (optimise, grad) in enumerate(zip(opt_vars, grads)):
                if optimise:
                    x_guess[0, idx].sub_(lr * torch.tensor(grad, dtype=x_guess.dtype))
                    x_guess[0, idx].clamp_(min=self.min_bound[idx], max=self.max_bound[idx])
        return x_guess

    def step(self, x_guess, x_gt, y_target, opt_vars, optimizer=None, fixed_values=None):
        if 'tabpfn' in self.model_type:
            x_guess, x_gt = x_guess.to('cpu'), x_gt.to('cpu')
            if y_target is not None:
                y_target = y_target.to('cpu')

            grads = self.finite_diff_approx(x_guess, y_target, opt_vars, epsilon=1e0)
            x_guess = self.grad_update(x_guess, grads, opt_vars, lr=self.hparam['OPT_LR'])
            y_hat = self.predict_tabpfn(x_guess)

            loss_y = objective_loss(self.loss, y_hat.squeeze(0), y_target, self.objective)
            loss_x = self.loss(x_guess, x_gt)

            x_guess, x_gt, y_hat = x_guess.to(self.device), x_gt.to(self.device), y_hat.to(self.device)
        else:
            if not isinstance(x_guess, nn.Parameter):
                x_guess, optimizer = self.init_whitebox_state(x_guess)
            elif optimizer is None:
                optimizer = self.init_optimizer(x_guess=x_guess, lr=self.hparam['OPT_LR'])

            if fixed_values is None:
                fixed_values = x_gt.detach().clone()
            optimise_mask = torch.tensor(opt_vars, dtype=torch.bool, device=x_guess.device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = self.model(x_guess)
            if self.model_type == 'hres' or self.model_type == 'mdn':
                y_hat = y_hat[:, 0]

            target = y_target.unsqueeze(0) if y_target is not None else None
            loss_elem = objective_loss(self.loss, y_hat, target, self.objective)
            loss_elem.backward()

            # The surrogate weights are never passed to the optimiser.  Mask
            # gradients so the unobserved machine parameters remain fixed.
            if x_guess.grad is not None:
                x_guess.grad[:, ~optimise_mask] = 0

            optimizer.step()

            # projected gradient descent
            with torch.no_grad():
                x_guess[:, ~optimise_mask] = fixed_values[:, ~optimise_mask]
                for i, optimise in enumerate(opt_vars):
                    if optimise:
                        x_guess[:, i].clamp_(min=self.min_bound[i], max=self.max_bound[i])

            with torch.no_grad():
                y_hat = self.model(x_guess)
                if self.model_type == 'hres' or self.model_type == 'mdn':
                    y_hat = y_hat[:, 0]
                loss_y = objective_loss(self.loss, y_hat, target, self.objective)
            loss_x = self.loss(x_guess.squeeze(), x_gt.squeeze())  # squeeze gt because batch size 1

        return loss_x.detach(), loss_y.detach(), x_guess.detach(), y_hat.detach()


class OptModule():
    def __init__(self, hparam, model):
        self.hparam = hparam
        self.model = model
        self.model_type = hparam['MODEL_TYPE']
        self.loss = nn.MSELoss()
        self.threshold = hparam['OPT_THRESHOLD']
        self.method = normalize_method(hparam['METHOD'])
        self.objective = hparam.get('OPT_OBJECTIVE', OBJECTIVE_TARGET_MATCH)
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(
                f"OPT_OBJECTIVE must be one of {sorted(VALID_OBJECTIVES)}, got {self.objective!r}."
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.GS = ParamOpt(self.hparam, self.model, self.model_type)
        shared_bounds = (self.GS.min_bound, self.GS.max_bound)
        self.GA = GeneticAlgorithm(self.hparam, self.model, self.model_type, *shared_bounds)
        self.BS = BeamSearch(self.hparam, self.model, self.model_type, *shared_bounds)

    def find_params(self, x_guess, x_gt, y_gt=None, opt_vars=None, lr=None):
        """
        This function holds the core functionalities for reconstructing paramters.
        As guesses only tensors of dim1 are allowed ... no batches.

        :param x_guess: starting machine-parameter vector.
        :param x_gt: reference vector used only for reconstruction metrics.
        :param y_gt: required target quality for ``target_match``; omit for ``maximize``.
        :param opt_vars: Is a list of booleans that describes which input parameters to optimize.
        :param method: ``paramopt``, ``beam_search``, or ``genetic_algorithm``.
        """

        if x_guess.dim() == 1:
            x_guess = x_guess.unsqueeze(0)
            x_gt = x_gt.unsqueeze(0)
            if y_gt is not None:
                y_gt = y_gt.unsqueeze(0)

        #x_guess = x_guess.to(self.device).clone().detach().requires_grad_(True) # detach and reattach

        x_guess = x_guess.to(self.device)

        if self.method == "paramopt":
            x_guess = x_guess.clone().detach().requires_grad_(True)
        else:
            x_guess = x_guess.clone().detach()

        x_gt = x_gt.to(self.device)
        if y_gt is not None:
            y_gt = y_gt.to(self.device)

        # Repetitions are independent restarts.  Paper-level seed variation is
        # driven by the experiment runner; keeping this at one avoids treating
        # sequential perturbations as independent seeds.
        n_restarts = int(self.hparam.get('OPT_RESTARTS', 1))
        if n_restarts < 1:
            raise ValueError('OPT_RESTARTS must be at least one.')
        initial_guess = x_guess.detach().clone()

        x_guess_history = []

        # meta logging lists
        logging_x_ground_truth = []
        logging_y_ground_truth = []
        logging_x_guess = []
        logging_y_prediction = []

        with tqdm(total=n_restarts, desc="restarts") as pbar:
            for restart in range(n_restarts):
                # Reset every mutable optimisation state.  A restart must not
                # continue from the previous restart's terminal point.
                x_guess = initial_guess.detach().clone()
                random.seed(self.hparam.get('SEED', 0) + restart)
                np.random.seed(self.hparam.get('SEED', 0) + restart)
                torch.manual_seed(self.hparam.get('SEED', 0) + restart)

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
                loss_history = []

                if self.method == "genetic_algorithm":
                    x_in = x_guess.squeeze().tolist()
                    self.GA.init_population(x_in=x_in, opt_vars=opt_vars)
                elif self.method == "beam_search":
                    beam = self.BS.init_beam(x_in=x_guess)
                elif self.method == "paramopt" and 'tabpfn' not in self.model_type:
                    whitebox_guess, whitebox_optimizer = self.GS.init_whitebox_state(x_guess)
                    fixed_values = x_guess.detach().clone()

                while n_cycles < max_cycles:
                    if self.method == "genetic_algorithm":
                        x_loss, y_loss, x_guess, y_hat = self.GA.step(x_gt=x_gt, y_gt=y_gt, i=n_cycles)
                        x_guess = torch.tensor(x_guess, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if self.method == "beam_search":
                        beam, x_loss, y_loss, x_guess, y_hat = self.BS.step(beam=beam, x_gt=x_gt, y_gt=y_gt)

                    if self.method == "paramopt":
                        step_guess = whitebox_guess if 'tabpfn' not in self.model_type else x_guess
                        x_loss, y_loss, x_guess, y_hat = self.GS.step(
                            x_guess=step_guess,
                            x_gt=x_gt,
                            y_target=y_gt,
                            opt_vars=opt_vars,
                            optimizer=whitebox_optimizer if 'tabpfn' not in self.model_type else None,
                            fixed_values=fixed_values if 'tabpfn' not in self.model_type else None,
                        )
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

                    # The publication benchmark uses a fixed 1,000-step
                    # budget.  Early stopping remains available for ordinary
                    # use, but must be explicitly enabled in its configuration.
                    if self.hparam.get('OPT_EARLY_STOP', True) and len(loss_history) > 1:
                        delta = abs(loss_history[-1] - loss_history[-2])  # both floats
                        threshold = self.hparam['OPT_THRESHOLD']
                        if delta <= threshold:
                            consecutive_cycles += 1
                        else:
                            consecutive_cycles = 0

                    logging_x_ground_truth_ind.append(x_gt.detach())
                    logging_y_ground_truth_ind.append(None if y_gt is None else y_gt.detach())
                    logging_x_guess_ind.append(x_guess)
                    logging_y_prediction_ind.append(y_hat)

                    n_cycles += 1

                    if (
                        self.hparam.get('OPT_EARLY_STOP', True)
                        and n_cycles >= warmup_window
                        and consecutive_cycles >= max_consecutive_cycles
                    ):
                        break


                x_guess_history.append(x_guess)

                logging_x_ground_truth.append(logging_x_ground_truth_ind)
                logging_y_ground_truth.append(logging_y_ground_truth_ind)
                logging_x_guess.append(logging_x_guess_ind)
                logging_y_prediction.append(logging_y_prediction_ind)

                pbar.update(1)

        return logging_x_ground_truth, logging_y_ground_truth, logging_x_guess, logging_y_prediction
