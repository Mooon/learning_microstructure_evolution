#!/bin/env python

"""
Main function to train the model based on the MS dataset
Author: Monica Rotulo
"""

# system modules
from pathlib import Path
import os, platform, argparse
from numpy import column_stack

# connection to wandb for experiment tracking
import wandb

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# custom dataset
from ms_dataloader import MicroS_Dataset, ImglistToTensor
# the model
from convlstmnet import ConvLSTMNet


# default config/hyperparameter values
LR_DECAY_EPOCH, LR_DECAY_RATE, LR_DECAY_MODE = 5, 0.98, False
SSR_DECAY_MODE, SSR_DECAY_EPOCH, SSR_DECAY_RATIO, SCHEDULED_SAMPLING_RATIO  = False, 1, 4e-3, 1


def check_device():
    """
    This function to check whether to use GPU (or CPU)
    and whether to use multi-GPU (or single-GPU)
    """
    use_cuda  = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    multi_gpu = True
    multi_gpu = use_cuda and multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0

    return use_cuda, device, multi_gpu, num_gpus


def build_network(config):
    """
    This function define the Conv-LSTM model
    Args:
        config: model settings, "model_order", "model_steps", "model_rank", "kernel"
    """
    model = ConvLSTMNet(
        input_channels = 3, 
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels = (32, 48, 48, 32), 
        skip_stride = 2,
        cell = 'convlstm', cell_params = {"order": config.model_order,
        "steps": config.model_steps, "rank": config.model_rank},
        kernel_size = config.kernel, bias = True,
        output_sigmoid = False)
    
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    ### set path 
    if platform.system() == "Windows":
        videos_root = os.path.join(os.getcwd(), 'code\data\dataset_I')  
        train_path = os.path.join(videos_root, 'config_file_train') 
        valid_path = os.path.join(videos_root, 'config_file_test') 
        dir_checkpoint = os.path.join(os.getcwd(), 'code\checkpoints') 
    else:        
        videos_root = os.path.join(os.getcwd(), 'data/dataset_I')  
        train_path = os.path.join(os.getcwd(), 'data/dataset_I/config_file_train')
        valid_path = os.path.join(os.getcwd(), 'data/dataset_I/config_file_test') 
        dir_checkpoint = os.path.join(os.getcwd(), 'code/checkpoints')   

    save_checkpoint = True

    ### whether to use GPU
    use_cuda, device, multi_gpu, num_gpus = check_device()
    print("Use of cuda, num gpus ", num_gpus)

    if use_cuda:
        tot_frames, batch_size, num_epochs, learning_rate = 20, 8, 250, 1e-3
    else:
        tot_frames, batch_size, num_epochs, learning_rate = 8, 1, 5, 1e-3

    ### initialise a wandb.ai run
    wandb.init(
        project="esa0",
        entity="username",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_order": 3, 
            "model_steps": 3, 
            "model_rank": 8, 
            "kernel": 3,
            "total_frames": tot_frames
            })
    config = wandb.config

    ### Construct the MODEL
    model = build_network(config)
    model.to(device)
    if multi_gpu: 
        model = nn.DataParallel(model)

    print("model with parameters num:", count_parameters(model))

    # loss function for training
    loss_func = lambda pred, origin: (
        F.l1_loss( pred, origin, reduction = "mean") + 
        F.mse_loss(pred, origin, reduction = "mean"))

    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    ### DATA loading
    n_input, n_output = tot_frames//2, tot_frames//2

    transform = transforms.Compose([
        ImglistToTensor()  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    ])

    ### TRAIN dataset
    train_data = MicroS_Dataset(
        root=videos_root,
        config_path=train_path,
        n_frames_input=n_input,
        n_frames_output=n_output,
        imagefile_template='RenderView1_{:06d}.jpg',
        transform=transform,
        is_train=True
    )
    
    # dataset is the object, dataset[0] returns 1 sample (input, label), dataset[0][0] is input, dataset[0][1] is label
    sample = train_data[0]     # for example dataset[0] is the first video
    total_frames = train_data.n_frames_input + train_data.n_frames_output
    input_frames = sample[0]  
    label = sample[1]

    print("train dataset created, total frames ", total_frames)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 * max(num_gpus, 1),
        #pin_memory=True
    )

    train_size = len(train_dataloader) * batch_size
    
    print("train dataset LOADED, train size ", train_size)

    ### VALIDATION dataset
    valid_data = MicroS_Dataset(
        root=videos_root,
        config_path=valid_path,
        n_frames_input=n_input,
        n_frames_output=n_output,
        imagefile_template='RenderView1_{:06d}.jpg',
        transform=transform,
        is_train=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 * max(num_gpus, 1),
        #pin_memory=True
    ) 
    valid_size = len(valid_dataloader) * batch_size

    print("valid dataset created and loaded, valid size ", valid_size)



    ## Main script for training and validation
    # scheduling sampling
    global SSR_DECAY_MODE, SSR_DECAY_EPOCH, SSR_DECAY_RATIO, SCHEDULED_SAMPLING_RATIO
    # optimizer and learning rate
    global LR_DECAY_EPOCH, LR_DECAY_RATE, LR_DECAY_MODE

    # best model in validation loss
    min_epoch, min_loss = 0, float("inf")
    gradient_clipping, clipping_threshold = True, 3

    wandb.watch(model, loss_func, log="all", log_freq=100)

    tot_batches = len(train_dataloader)*num_epochs
    example_ct, batch_ct = 0, 0

    print("start Training...")
    for epoch in range(0, num_epochs):
        ## Phase 1: Learning on the training set
        model.train()

        samples, running_loss = 0, 0   #samples is just a counter variable
        
        for _, (input_batch, output_batch) in enumerate(train_dataloader):
            samples += train_dataloader.batch_size
            batch_ct += 1

            all_frames = torch.cat((input_batch,output_batch),1)
            all_frames = all_frames.to(device)

            # check this later
            example_ct +=  len(all_frames)
            print('tot_batches: {}, samples: {}, batch_ct: {}, example_ct: {}'.format(
                        tot_batches, samples, batch_ct, example_ct ))

            inputs = all_frames[:, :-1] 
            gtruth = all_frames[:, -train_data.n_frames_output:]

            pred = model(inputs, 
                input_frames  = train_data.n_frames_input, 
                future_frames = train_data.n_frames_output, 
                output_frames = train_data.n_frames_output,        # output of the model
                teacher_forcing = True, 
                scheduled_sampling_ratio = SCHEDULED_SAMPLING_RATIO)
            
            #print("prediction done: ", pred.shape)

            # calculate loss + backward
            loss = loss_func(pred, gtruth)

            running_loss += loss.item()
            loss_aver = loss.item() / train_dataloader.batch_size

            loss.backward()
            
            if gradient_clipping: 
                nn.utils.clip_grad_norm_(
                    model.parameters(), clipping_threshold)

            # update the optimizer parameters
            optimizer.step()
            optimizer.zero_grad()


            # statistics
            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
            #if index % 10 == 0:
                wandb.log({
                "Epoch": epoch,
                "Loss": loss.item(),
                "Avg loss": loss_aver,
                }, step = epoch)

                print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(
                        epoch, num_epochs, samples, len(train_dataloader), loss.item()))


        avg_loss = running_loss / len(train_dataloader)
        wandb.log({
            "Avg loss": avg_loss
            })

        # adjust the learning rate of the optimizer
        if LR_DECAY_MODE and (epoch + 1) % LR_DECAY_EPOCH == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= LR_DECAY_RATE

        # adjust the scheduled sampling ratio
        if SSR_DECAY_MODE and (epoch + 1) % SSR_DECAY_EPOCH == 0:
            SCHEDULED_SAMPLING_RATIO = max(SCHEDULED_SAMPLING_RATIO - SSR_DECAY_RATIO, 0)
    

        ## Phase 2: Evaluation on the validation set
        model.eval()
        print('Now validation phase...epoch: {}/{}', epoch, num_epochs)
        record_data = []
        with torch.no_grad():
           
            samples, LOSS = 0, 0.0
            for _, (input_batch, output_batch) in enumerate(valid_dataloader):
                samples += valid_dataloader.batch_size

                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                pred = model(input_batch, 
                    input_frames  = valid_data.n_frames_input, 
                    future_frames = valid_data.n_frames_output, 
                    output_frames = valid_data.n_frames_output, 
                    teacher_forcing = False)

                valid_loss = loss_func(pred, output_batch)
                LOSS += valid_loss

                for pred_image, true_image in zip(pred[0], output_batch[0]):
                    record_data.append(
                        [samples, 
                        wandb.Image(pred_image),
                        wandb.Image(true_image)] 
                    )

                wandb.log({
                    "Valid Sum Loss": LOSS,
                    "valid loss": valid_loss
                })
                

        table = wandb.Table(data= record_data, columns=['sample num.', 'predicted', 'true'])


        LOSS /= valid_size
        wandb.log({
            "Valid Loss avg": LOSS, 
            "Valid images": table
        })
        if LOSS < min_loss:
            min_epoch, min_loss = epoch + 1, LOSS

        ## Phase 3: learning rate and scheduling sampling ratio adjustment
        decay_log_epochs = 20
        if not SSR_DECAY_MODE and epoch > min_epoch + decay_log_epochs:
            min_epoch = epoch
            SSR_DECAY_MODE = True

        if not  LR_DECAY_MODE and epoch > min_epoch + decay_log_epochs:
           LR_DECAY_MODE  = True
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': LOSS,
            }, "checkpoint_epoch{}.pth".format(epoch))


    # WandB – Save the model checkpoint. 
    # This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), "overall_checkpoint.pt")



if __name__ == "__main__":

    main()
