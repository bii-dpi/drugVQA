from utils import  * 
from dataPre import  * 
from sklearn import metrics
from progressbar import progressbar


def getROCE(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key = lambda x:x[1], reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate * n)/100)):
                break
    roce = (tp1 * n)/(p*fp1)
    return roce


def train(trainArgs):
    """
    args:
        model           : {object} model
        lr              : {float} learning rate
        train_loader    : {DataLoader} training data loaded into a dataloader
        doTest          : {bool} do test or not
        test_proteins   : {list} proteins list for test
        testDataDict    : {dict} test data dict
        seqContactDict  : {dict} seq-contact dict
        optimizer       : optimizer
        criterion       : loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
        epochs          : {int} number of epochs
        use_regularizer : {bool} use penalization or not
        penal_coeff     : {int} penalization coeff
        clip            : {bool} use gradient clipping or not
    """
    print("Training...")

    train_loader = trainArgs['train_loader']
    optimizer = trainArgs['optimizer']
    criterion = trainArgs["criterion"]
    attention_model = trainArgs['model']
    attention_model.train()

    csv = ""
    for i in progressbar(range(trainArgs['epochs'])):

        total_loss = 0
        correct = 0
        all_pred = np.array([])
        all_target = np.array([])

        for lines, contactmap, properties in train_loader:  
            input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            attention_model.hidden_state = attention_model.init_hidden()
            contactmap = create_variable(contactmap)
            y_pred, att = attention_model(input, contactmap)

            #penalization AAT - I
            if trainArgs['use_regularizer']:
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size, att.size(1), att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)

            #binary classification
            #Adding a very small value to prevent BCELoss from outputting NaN's
            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                            y.type(torch.DoubleTensor)).data.sum()
            all_pred=np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis = 0)
            all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis = 0)

            if trainArgs['use_regularizer']:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                    y.type(torch.DoubleTensor))+(trainArgs['penal_coeff'] * \
                                            penal.cpu()/train_loader.batch_size)
            else:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                    y.type(torch.DoubleTensor))

            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward() #retain_graph=True
       
            #gradient clipping
            if trainArgs['clip']:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(), 0.5)

            optimizer.step()
        
        # Across all batches
        accuracy = correct.numpy()/(len(train_loader.dataset))
        recall = metrics.recall_score(all_target, np.round(all_pred))
        precision = metrics.precision_score(all_target, np.round(all_pred))
        AUC = metrics.roc_auc_score(all_target, all_pred)
        AUPR = metrics.average_precision_score(all_target, all_pred)
        avg_loss = total_loss.item()/len(test_loader)

        roce_1 = getROCE(all_pred, all_target, 0.5)
        roce_2 = getROCE(all_pred, all_target, 1)
        roce_3 = getROCE(all_pred, all_target, 2)
        roce_4 = getROCE(all_pred, all_target, 5)

        if (trainArgs['doSave']):
            torch.save(attention_model.state_dict(),
                        f"../model_pkl/DUDE/{trainArgs['saveNamePre']}{i + 1}.pkl")

        csv += \
            (f"{accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {avg_loss}, "
                f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n")

    # done
    with open(f"../results/{trainArgs['saveNamePre']}train_results.csv", "w") as f:
        f.write(csv)
    
    print("Finished training.")
   

def test(testArgs, epoch):
    print("Validating...")

    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    attention_model.eval()

    total_loss = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])

    with torch.no_grad():
        for lines, contactmap, properties in test_loader:
            input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            attention_model.hidden_state = attention_model.init_hidden()
            contactmap = contactmap.cuda()
            y_pred, att = attention_model(input, contactmap)

            #binary classification
            #Adding a very small value to prevent BCELoss from outputting NaN's
            #pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                            y.type(torch.DoubleTensor)).data.sum()
            all_pred=np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis = 0)
            all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis = 0)
            if trainArgs['use_regularizer']:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),
                                y.type(torch.DoubleTensor)) + \
                                        (C * penal.cpu() / train_loader.batch_size)
            else:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))

            total_loss +=loss.data

    accuracy = correct.numpy()/(len(test_loader.dataset))
    recall = metrics.recall_score(all_target, np.round(all_pred))
    precision = metrics.precision_score(all_target, np.round(all_pred))
    AUC = metrics.roc_auc_score(all_target, all_pred)
    AUPR = metrics.average_precision_score(all_target, all_pred)
    loss = total_loss.item()/len(test_loader)

    roce_1 = getROCE(all_pred, all_target, 0.5)
    roce_2 = getROCE(all_pred, all_target, 1)
    roce_3 = getROCE(all_pred, all_target, 2)
    roce_4 = getROCE(all_pred, all_target, 5)

    csv = \
        (f"{epoch}, {accuracy}, {recall}, {precision}, {AUC}, {AUPR}, {loss}, "
            f"{roce_1}, {roce_2}, {roce_3}, {roce_4}\n")

    with open(f"../results/{testArgs['saveNamePre']}test_results.csv", "a") as f:
        f.write(csv)

