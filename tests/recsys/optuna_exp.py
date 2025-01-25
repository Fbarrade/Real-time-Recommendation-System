from optuna.integration.wandb import WeightsAndBiasesCallback
import optuna
from optuna.trial import TrialState

from recsys.models import train_step_optuna
import torch 


def objective(trial): 
    trainset, testset, valset = torch.load("./data/gnn_datasets.pt")

    loss = train_step_optuna(trial, trainset=trainset)

    return loss 


if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))