from run_model import validate


for i in progressbar(range(train_args['epochs'], 0, -2)):
    curr_path = f"../model_pkl/DUDE/{validate_args['fname_prefix']}{i}.pkl"
    validate_args['model'] = DrugVQA(modelArgs, block=ResidualBlock)
    validate_args['model'].load_state_dict(torch.load(curr_path),
                                            map_location=device)
    validate_args['model'] = validate_args['model'].to(device)
    validate(validate_args, i)

