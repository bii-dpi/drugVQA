from utils import  *
#from data_pre import  *
from pickle import dump
from sklearn import metrics
from progressbar import progressbar


def train(train_args):
    """Train the model."""
    device = train_args["device"]
    train_loader = train_args["train_loader"]
    optimizer = train_args["optimizer"]
    criterion = train_args["criterion"]
    attention_model = train_args["model"]
    attention_model.train()

    for i in progressbar(range(train_args["train_from"], train_args["epochs"])):
        torch.manual_seed(train_args["seed"])

        total_loss = 0
        correct = 0
        all_pred = np.array([])
        all_target = np.array([])

        for lines, contactmap, properties in train_loader:
            input, seq_lengths, y = make_variables(lines, properties, smiles_letters, device)
            attention_model.hidden_state = attention_model.init_hidden()
            contactmap = contactmap.to(device)
            y_pred, att = attention_model(input, contactmap)
            y_pred = y_pred.clamp(0, 1)

            #penalization AAT - I
            if train_args["use_regularizer"]:
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1))
                identity = identity.unsqueeze(0).expand(train_loader.batch_size,
                                                        att.size(1), att.size(1)).to(device)
                penal = attention_model.l2_matrix_norm(att@attT - identity)

            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                            y.type(torch.DoubleTensor)).data.sum()
            all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()),
                                        axis=0)
            all_target = np.concatenate((all_target, y.data.cpu().numpy()),
                                        axis=0)

            if train_args["use_regularizer"]:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                    y.type(torch.DoubleTensor))+(train_args["penal_coeff"] * \
                                            penal.cpu() / train_loader.batch_size)
            else:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                    y.type(torch.DoubleTensor))

            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if train_args["clip"]:
                torch.nn.utils.clip_grad_norm_(attention_model.parameters(), 0.5)

            optimizer.step()

        # Across all batches
        accuracy = correct.numpy() / (len(train_loader.dataset))
        recall = metrics.recall_score(all_target, np.round(all_pred))
        precision = metrics.precision_score(all_target, np.round(all_pred))
        AUC = metrics.roc_auc_score(all_target, all_pred)
        AUPR = metrics.average_precision_score(all_target, all_pred)
        avg_loss = total_loss.item() / len(train_loader.dataset)

        roce_1 = get_ROCE(all_pred, all_target, 0.5)
        roce_2 = get_ROCE(all_pred, all_target, 1)
        roce_3 = get_ROCE(all_pred, all_target, 2)
        roce_4 = get_ROCE(all_pred, all_target, 5)

        torch.save(attention_model.state_dict(),
                    f"../model_pkl/DUDE/{train_args['fname_prefix']}{i + 1}.pkl")

        with open(f"../results/{train_args['fname_prefix']}train_results.csv", "a") as f:
            f.write((f"{i + 1}, {accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {avg_loss}, "
                        f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n"))


def validate(validate_args, epoch):
    """Validate the model."""
    device = validate_args["device"]

    validate_loader = validate_args["validate_loader"]
    criterion = validate_args["criterion"]
    attention_model = validate_args["model"]
    attention_model.eval()

    total_loss = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])

    with torch.no_grad():
        for lines, contactmap, properties in validate_loader:
            input, seq_lengths, y = make_variables(lines, properties, validate_args['smiles_letters'], device)
            attention_model.hidden_state = attention_model.init_hidden()
            contactmap = contactmap.to(device)
            y_pred, att = attention_model(input, contactmap)
            y_pred = y_pred.clamp(0, 1)

            #pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
            """
            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                            y.type(torch.DoubleTensor)).data.sum()
            """
            correct += torch.eq(torch.round(y_pred.squeeze(1)), y).data.sum()
            all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis = 0)
            all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis = 0)

            if validate_args["use_regularizer"]:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                y.type(torch.DoubleTensor)) + \
                                        (C * penal.cpu() / validate_loader.batch_size)
            else:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                    y.type(torch.DoubleTensor))

            total_loss += loss.data

    accuracy = correct.cpu().numpy() / (len(validate_loader.dataset))
    recall = metrics.recall_score(all_target, np.round(all_pred))
    precision = metrics.precision_score(all_target, np.round(all_pred))
    AUC = metrics.roc_auc_score(all_target, all_pred)
    AUPR = metrics.average_precision_score(all_target, all_pred)
    loss = total_loss.item() / len(validate_loader)

    roce_1 = get_ROCE(all_pred, all_target, 0.5)
    roce_2 = get_ROCE(all_pred, all_target, 1)
    roce_3 = get_ROCE(all_pred, all_target, 2)
    roce_4 = get_ROCE(all_pred, all_target, 5)

    np.save(f"../model_pred/{validate_args['fname_prefix']}_{epoch}_pred",
            all_pred)
    np.save(f"../model_pred/{validate_args['fname_prefix']}_{epoch}_target",
            all_target)

    with open(f"../results/{validate_args['fname_prefix']}validate_results.csv", "a") as f:
        f.write((f"{epoch}, {accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {loss}, "
                    f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n"))

