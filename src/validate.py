from run_model import validate 


for i in progressbar(range(train_args['epochs'], 0, -2)):
    curr_path = f"../model_pkl/DUDE/{validate_args['frname_prefix']}{i}.pkl"
    validate_args['model'] = DrugVQA(modelArgs, block = ResidualBlock)
    validate_args['model'].load_state_dict(torch.load(curr_path))
    validate(validate_args, i)

