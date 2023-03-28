import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import numpy as np
import os



def compute_acc(out: torch.Tensor, truth:torch.Tensor):
    # pred = torch.argmax(
    #     torch.softmax(out, dim = 1), 
    #     dim = 1,
    # )
    return (out.round().flatten() == truth).sum().item() / len(out)


def train_loop(epochs, model, device, train_loader, val_loader, optimizer, criterion, log_freq, name = ''):
    save_dir_name = 'checkpoint_models'

    train_losses = []
    val_losses = []
    train_accs_tot = []
    val_accs_tot = []

    model = model.to(device)

    for epoch in range(epochs):
        print(f'-- Epoch {epoch + 1:02d} --')

        start = time.time()

        # for checkpoints
        min_val_loss = np.inf
        max_acc = 0
        best_epoch_v = 0        # validation
        best_epoch_a = 0        # accuracy
        best_model_v = np.nan   # validation
        best_model_a = np.nan   # accuracy

        # training
        model.train()
        run_loss = .0
        tmp_run_loss = .0
        train_accs = .0

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = inputs.to(device)
            for ele in inputs:
                inputs[ele] = inputs[ele].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                inputs['input_ids'],
                inputs['attention_mask'],
            ).reshape(len(labels))      # same as flatten but flutten doesn't work on mps

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # stats
            run_loss += loss.item()
            tmp_run_loss += loss.item()

            train_accs += compute_acc(outputs, labels)

            # logging
            if i % log_freq == log_freq - 1:        # print every 2000 mini-batches
                print(f'   {epoch + 1:02d}) [train, {i + 1:5d}] \tloss: {tmp_run_loss / log_freq:.3f}')
                tmp_run_loss = .0
            
        # evaluation
        model.eval()
        eval_loss = .0
        tmp_eval_loss = .0
        eval_accs = .0
        for i, (inputs, labels) in enumerate(val_loader):
            # inputs = inputs.to(device)
            for ele in inputs:
                inputs[ele] = inputs[ele].to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                ).reshape(len(labels))

            loss = criterion(outputs, labels)

            # stats
            eval_loss += loss.item()
            tmp_eval_loss += loss.item()
            eval_accs += compute_acc(outputs, labels)

            # logging
            if i % log_freq == log_freq - 1:        # print every 2000 mini-batches
                print(f'   {epoch + 1:02d}) [eval, {i + 1:5d}] \tloss: {tmp_run_loss / log_freq:.3f}')
                tmp_run_loss = .0
        
        loss_t = run_loss / len(train_loader)
        loss_e = eval_loss / len(val_loader)
        acc_t = train_accs / len(train_loader)
        acc_e = eval_accs / len(val_loader)

        train_losses.append(loss_t)
        val_losses.append(loss_e)
        train_accs_tot.append(acc_t)
        val_accs_tot.append(acc_e)

        print(f'   [Recap {epoch + 1:02d} epoch] - train_loss: {loss_t:.3f}, train_acc: {acc_t:.4f} | eval_loss: {loss_e:.3f}, eval_acc: {acc_e:.4f} | elapsed time: {time.time() - start:.1f}s', end = '')
        
        if min_val_loss > loss_e:
            best_epoch_v = epoch
            min_val_loss = loss_e
            best_model_v = model
            print('\t <-- Best epoch so far, val', end = '')
            if not os.path.exists(f'./{save_dir_name}'):
                os.makedirs(f'./{save_dir_name}')
            torch.save(model.state_dict(), f'./{save_dir_name}/model_min_val_loss_{name}.pth')

        if max_acc < acc_e:
            best_epoch_a = epoch
            max_acc = acc_e
            best_model_a = model
            print('\t <-- Best epoch so far, acc', end = '')
            if not os.path.exists(f'./{save_dir_name}'):
                os.makedirs(f'./{save_dir_name}')
            torch.save(model.state_dict(), f'./{save_dir_name}/model_max_acc_{name}.pth')
        

        print('\n')

    print('Done')

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs_tot,
        'val_accs': val_accs_tot,
        'best_epoch_v': best_epoch_v,
        'best_epoch_a': best_epoch_a,
        'best_model_v': best_model_v,
        'best_model_a': best_model_a,
        'final_model': model,
    }

    if not os.path.exists(f'./{save_dir_name}'):
                os.makedirs(f'./{save_dir_name}')
    torch.save(model.state_dict(), f'./{save_dir_name}/model_final_{name}.pth')

    return history