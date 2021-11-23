import logging

import advertorch
import advertorch.bpda
import click
import numpy as np

import attacks

logger = logging.getLogger(__name__)

domains = ['cifar10', 'mnist']
architectures = ['a', 'b', 'c', 'wong_small', 'wong_large', 'b2', 'b3', 'b4', 'b5']
attack_types = ['defense', 'evasion', 'standard', 'training']
supported_attacks = ['bim', 'brendel', 'carlini',
                     'deepfool', 'fast_gradient', 'mip', 'pgd', 'uniform']
epsilon_attacks = ['bim', 'fast_gradient', 'pgd', 'uniform']
attacks_with_binary_search = ['bim', 'fast_gradient', 'pgd', 'uniform']
targeted_attacks = ['bim', 'carlini', 'brendel', 'fast_gradient', 'mip', 'pgd']
er_attacks = ['bim', 'carlini', 'pgd', 'uniform']
fb_binary_search_attacks = ['brendel', 'deepfool']
distances = ['l2', 'linf']
misclassification_policies = ['ignore', 'remove', 'use_predicted']
log_levels = ['debug', 'info', 'warning', 'error', 'critical']
_log_level_to_number = {'debug': logging.DEBUG,
                        'info': logging.INFO,
                        'warning': logging.WARNING,
                        'error': logging.ERROR,
                        'critical': logging.CRITICAL}
_distance_to_p = {'l2': 2, 'linf': np.inf}

training_options = [
    click.option('--optimiser', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True,
                 help='The optimiser that will be used for training.'),
    click.option('--learning-rate', type=float, default=1e-3, show_default=True,
                 help='The learning rate for the optimiser.'),
    click.option('--weight-decay', type=float, default=0, show_default=True,
                 help='The weight decay for the optimiser.'),
    click.option('--adam-betas', nargs=2, type=click.Tuple([float, float]), default=(0.9, 0.999), show_default=True,
                 help='The two beta values. Ignored if the optimiser is not "adam".'),
    click.option('--adam-epsilon', type=float, default=1e-8, show_default=True,
                 help='The value of epsilon. Ignored if the optimiser is not "adam".'),
    click.option('--adam-amsgrad', is_flag=True,
                 help='Enables AMSGrad. Ignored if the optimiser is not "adam".'),
    click.option('--sgd-momentum', type=float, default=0, show_default=True,
                 help='The intensity of momentum. Ignored if the optimiser is not "sgd".'),
    click.option('--sgd-dampening', type=float, default=0, show_default=True,
                 help='The intensity of dampening. Ignored if the optimiser is not "sgd".'),
    click.option('--sgd-nesterov', is_flag=True,
                 help='Enables Nesterov Accelerated Gradient. Ignored if the optimiser is not "adam".'),
    click.option('--l1-regularization', type=float, default=0, show_default=True,
                 help='The weight of L1 regularization. 0 disables L1 regularization.'),
    click.option('--validation-dataset', default=None,
                 help='Validation dataset. Mutually exclusive with --validation-split.'),
    click.option('--validation-split', type=float, default=0,
                 help='Uses a portion (0-1) of the train dataset as validation dataset. 0 disables the split. '
                 'Mutually exclusive with --validation-dataset.'),
    click.option('--early-stopping', type=click.IntRange(0, None), default=0, show_default=True,
                 help='The patience of early stopping. 0 disables early stopping. Requires either '
                 '--validation-dataset or --validation-split.'),
    click.option('--early-stopping-delta', type=float, default=0, show_default=True,
                 help='The minimum improvement required to reset early stopping\'s patience.'),
    click.option('--shuffle', type=bool, default=True, show_default=True),
    click.option('--checkpoint-every', type=click.IntRange(1, None), default=None,
                help='How often the program saves a checkpoint.'),
    click.option('--load-checkpoint', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
                help='If passed, the program will load an existing checkpoint.'),
    click.option('--choose-best', is_flag=True,
                help='If passed, the program will save the state_dict with the best validation loss, otherwise '
                'the state_dict of the last epoch will be saved. Requires either --validation-dataset or '
                '--validation-split.')
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def set_log_level(log_level):
    logging.getLogger().setLevel(_log_level_to_number[log_level])



# Targeted FGSM is introduced in
# http://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf

def parse_attack(attack_name, domain, p, attack_type, model, attack_config, device, defended_model=None, seed=None, parameter_overrides=None, suppress_blended_warning=False):
    logger.debug('Parsing %s attack %s-%s (using defended: %s).', attack_type,
                 attack_name, p, defended_model is not None)

    # Convert the float value to its standard name
    if p == 2:
        metric = 'l2'
    elif np.isposinf(p):
        metric = 'linf'
    else:
        raise NotImplementedError(f'Unsupported metric "l{p}"')

    if attack_type not in attack_types:
        raise NotImplementedError(f'Unsupported attack type {attack_type}.')

    kwargs = attack_config.get_arguments(
        attack_name, domain, metric, attack_type)

    logger.debug('Loaded attack kwargs: %s.', kwargs)

    if parameter_overrides is not None:
        logger.debug('Applying parameter overrides: %s.', parameter_overrides)
        for key, value in parameter_overrides.items():
            kwargs[key] = value
        logger.debug('New kwargs: %s.', kwargs)

    if attack_name == 'uniform' and metric != 'linf':
        logger.warning('UniformAttack is designed for the LInf metric. Are you sure that you '
                       'want to use %s?', metric)

    if attack_type == 'evasion' and defended_model is None:
        raise ValueError('Evasion attacks require a defended model.')

    binary_search = kwargs.pop('enable_binary_search', False)
    return_best = kwargs.pop('return_best', False)

    if return_best and attack_name == 'carlini' and np.isposinf(p):
        raise NotImplementedError('Carlini Linf does not support return_best.')

    if attack_type != 'evasion' and defended_model is not None:
        raise ValueError(
            'Passed a defended_model for a non-evasion attack.')

    if domain in ['cifar10', 'mnist']:
        num_classes = 10
    else:
        raise NotImplementedError(f'Unsupported domain "{domain}".')

    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        num_classes += 1

    if binary_search:
        logger.debug('Enabling binary search for %s.', attack_name)
        # Remove standard arguments
        kwargs.pop('eps', None)
        if attack_name not in attacks_with_binary_search:
            raise NotImplementedError(
                f'Attack {attack_name} does not support binary search.')
    elif attack_name in attacks_with_binary_search:
        logger.warning(
            'Binary search for attack %s is disabled in the configuration file, despite being supported.', attack_name)

    # Pop binary search arguments
    min_eps = kwargs.pop('min_eps', None)
    max_eps = kwargs.pop('max_eps', None)
    eps_initial_search_steps = kwargs.pop('eps_initial_search_steps', None)
    eps_initial_search_factor = kwargs.pop('eps_initial_search_factor', None)
    eps_binary_search_steps = kwargs.pop('eps_binary_search_steps', None)

    # Pop epsilon attack arguments
    force_eps = kwargs.pop('force_eps', None)

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    # TODO: Check compatibility between evasion and return_best

    if return_best:
        logger.debug('Wrapping in BestSampleWrapper.')
        target_model = attacks.BestSampleWrapper(target_model)

    if attack_name == 'bim':
        if metric == 'l2':
            attack = advertorch.attacks.L2BasicIterativeAttack(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfBasicIterativeAttack(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'blended_noise':
        if not suppress_blended_warning:
            logger.warning('blended_noise is supposed to be used only as a starting point for brendel. To suppress this warning, '
            'use suppress_blended_warning=True.')

        attack = attacks.LinearSearchBlendedUniformNoiseAttack(target_model, **kwargs)
    elif attack_name == 'brendel':
        # Brendel supports passing an init_attack, which we need to parse
        if return_best:
            logger.warning('Brendel&Bethge already has a form of return_best behaviour.')

        if 'init_attack' in kwargs:
            init_attack_name = kwargs.pop('init_attack')
            init_attack_parameter_overrides = kwargs.pop('init_attack_parameter_overrides', None)
            init_attack = parse_attack(init_attack_name, domain, p, attack_type, target_model, attack_config, device,
                                       defended_model=defended_model, seed=seed, parameter_overrides=init_attack_parameter_overrides,
                                       suppress_blended_warning=True)
        else:
            init_attack = None

        attack = attacks.BrendelBethgeAttack(target_model, p, init_attack=init_attack, **kwargs)
    elif attack_name == 'carlini':
        if metric == 'l2':
            attack = advertorch.attacks.CarliniWagnerL2Attack(
                target_model, num_classes, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            cuda_optimized = device == 'cuda'
            attack = attacks.get_carlini_linf_attack(target_model, num_classes,
                targeted=evade_detector, return_best=return_best, cuda_optimized=cuda_optimized, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'deepfool':
        attack = attacks.DeepFoolAttack(target_model, p, **kwargs)
    elif attack_name == 'fast_gradient':
        # FGM is the L2 variant, FGSM is the Linf variant
        if metric == 'l2':
            attack = advertorch.attacks.FGM(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.FGSM(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'mip':
        if attack_type == 'evasion':
            raise NotImplementedError('MIP does not support evasion.')
        attack = attacks.MIPAttack(
            target_model, p, targeted=evade_detector, seed=seed, **kwargs)
    elif attack_name == 'pgd':
        if metric == 'l2':
            attack = advertorch.attacks.L2PGDAttack(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfPGDAttack(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'uniform':
        attack = attacks.UniformNoiseAttack(
            target_model, p, targeted=evade_detector, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported attack "{attack_name}".')

    # Add support for epsilon
    if attack_name in epsilon_attacks:
        if attack_name in ['bim', 'pgd', 'fast_gradient']:
            unsqueeze = False
        elif attack_name in ['uniform']:
            unsqueeze = True
        else:
            raise ValueError(
                f'Unsupported epsilon attack "{attack_name}"')

        epsilon_attack_kwargs = {}

        if force_eps is not None:
            epsilon_attack_kwargs['force_eps'] = force_eps

        attack = attacks.EpsilonAttack(attack, unsqueeze, **epsilon_attack_kwargs)

    # If necessary, wrap the attack in a binary search wrapper
    if binary_search:
        logger.debug('Adding binary search.')
        binary_search_kwargs = dict()
        if min_eps is not None:
            binary_search_kwargs['min_eps'] = min_eps
        if max_eps is not None:
            binary_search_kwargs['max_eps'] = max_eps
        if eps_initial_search_steps is not None:
            binary_search_kwargs['eps_initial_search_steps'] = eps_initial_search_steps
        if eps_initial_search_factor is not None:
            binary_search_kwargs['eps_initial_search_factor'] = eps_initial_search_factor
        if eps_binary_search_steps is not None:
            binary_search_kwargs['eps_binary_search_steps'] = eps_binary_search_steps

        attack = attacks.EpsilonBinarySearchAttack(
            attack, p, targeted=evade_detector, **binary_search_kwargs)

    # Complete the best sample wrapping
    if return_best:
        logger.debug('Finalizing best sample wrapping.')
        suppress_warning = attack_name in fb_binary_search_attacks
        attack = attacks.BestSampleAttack(
            target_model, attack, p, targeted=evade_detector, suppress_warning=suppress_warning)

    # Convert targeted evasion attacks into untargeted ones
    if evade_detector and (attack_name in targeted_attacks):
        logger.debug('Converting targeted to untargeted attack.')
        attack = attacks.KBestTargetEvasionAttack(target_model, attack)

    return attack


def parse_attack_pool(attack_names, domain, p, attack_type, model, attack_config, device, defended_model=None, seed=None):
    logger.debug('Parsing %s attack pool %s for %s (using defended: %s).', attack_type,
                 attack_names, p, defended_model is not None)
    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    attack_pool = []
    for attack_name in attack_names:
        attack_pool.append(parse_attack(attack_name, domain, p, attack_type,
                                      model, attack_config, device, defended_model=defended_model, seed=seed))

    if len(attack_pool) == 1:
        return attack_pool[0]
    else:
        return attacks.AttackPool(target_model, evade_detector, attack_pool, p)

def validate_lp_distance(ctx, param, value):
    if value == 'l2':
        return 2
    elif value == 'linf':
        return np.inf
    elif value is None:
        return None
    else:
        raise NotImplementedError(f'Unsupported distance metric "{value}".')