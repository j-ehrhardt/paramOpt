"""Build the paper's exact-versus-approximated-gradient table (Table 3)."""

try:
    from .eval_exp3 import build_mae_table_per_setup_with_sd
except ImportError:  # Allow ``python eval_exp4.py`` from this directory.
    from eval_exp3 import build_mae_table_per_setup_with_sd


PAPER_SETUPS = [
    '[True, False, False]',
    '[False, True, False]',
    '[False, True, True]',
    '[True, True, True]',
]


if __name__ == '__main__':
    for model_type in ['ff', 'res', 'hres', 'tabpfn']:
        table = build_mae_table_per_setup_with_sd(
            model_type=model_type,
            dataset_ids=['ds1'],
            testset_selection='end',
            setups=PAPER_SETUPS,
            paradigms=['paramopt'],
        )
        print(f'======== {model_type} ========')
        print(table)
