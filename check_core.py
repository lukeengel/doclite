from doclite.configs.core import ENV, WEIGHTS, DUModel, DistillStage

print("ROOT:", ENV.ROOT)
print("Dirs:", ENV.DATA, ENV.PROCESSED, ENV.CHECKPOINTS, ENV.LOGS)
print("Weights:", WEIGHTS)
print("Models:", list(DUModel))
print("Stages:", list(DistillStage))