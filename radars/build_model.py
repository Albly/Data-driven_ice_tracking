import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
from copy import deepcopy
from PIL import Image, ImageDraw
#from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class RadarDataset(Dataset):

    def __init__(self, data, device, days_in=1, days_out=1, gfs_data=None):
        self.data = torch.from_numpy(data).to(device)
        self.days_in = days_in
        self.days_out = days_out
        self.device = device
        if gfs_data is None:
            self.gfs_data = None
        else:
            self.gfs_data = torch.from_numpy(gfs_data).to(device)
        #self.device = device

    def __getitem__(self, day_idx):
        item = {}
        item["past"] = self.data[day_idx:day_idx+self.days_in]#.to(self.device)
        item["past_last_layer"] = self.data[day_idx+self.days_in-1:day_idx+self.days_in].cpu()
        #item["future"] = self.data[day_idx+self.days_in]#.to(self.device)
        item["future"] = self.data[day_idx+self.days_in:
                                   day_idx+self.days_in+self.days_out]
        if day_idx == 0: # FIXME: unsafe? check gfs indexes
            day_idx = 1
        item["past_dif"] = self.data[day_idx:day_idx+self.days_in] - \
            self.data[day_idx-1:day_idx-1+self.days_in]
        item["future_dif"] = self.data[day_idx+self.days_in] - \
            self.data[day_idx-1+self.days_in]

        if self.gfs_data is not None:
            item["gfs"] = self.gfs_data[day_idx]
        else:
            gfs_shape = list(item["past"].shape)
            gfs_shape[0] = 0
            item["gfs"] = torch.zeros((gfs_shape)).to(self.device)
        return item

    def __len__(self):
        return self.data.shape[0]-self.days_in-self.days_out+1



class Persistence(nn.Module):

    def __init__(self, days_out=1):
        super(Persistence, self).__init__()
        self.days_out = days_out

    def forward(self, x):
        out = x[:,-1,:,:].unsqueeze(1).repeat(1, self.days_out, 1, 1)
        return out



def concat_preds_to_future(reals, preds, device):
    reals, preds = reals.to(device), preds.to(device)
    reals[reals<=0.2] = preds[reals<=0.2]
    return reals



class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()

    def forward(self, preds, targets, mask):

        days = preds.shape[1]
        pixels_in_day = int(len(mask)/days)

        preds = preds.view(-1)
        targets = targets.view(-1)

        loss_dict = {}
        for day in range(days):
            preds_i = preds[day*pixels_in_day:(day+1)*pixels_in_day].detach().cpu()
            targets_i = targets[day*pixels_in_day:(day+1)*pixels_in_day].detach().cpu()
            mask_i = mask[day*pixels_in_day:(day+1)*pixels_in_day].detach().cpu()

            preds_i = preds_i[mask_i.bool()]
            targets_i = targets_i[mask_i.bool()]
            loss_i = (preds_i - targets_i) ** 2
            loss_dict[day] = torch.mean(loss_i)

        preds = preds[mask.bool()]
        targets = targets[mask.bool()]

        loss = (preds - targets) ** 2
        loss_len = len(loss)
        loss = torch.mean(loss)

        return loss, loss_len, loss_dict



def eval_model(model, past, future, loss_function, device):
    past, future = past.to(device), future.to(device)
    preds = model(past.detach()) # past[0][0].detach ????? - убирать детэч у всех кроме последнего
    shape = future[0][0].shape
    #mask = (preds*future).bool().view(-1)
    #mask = future.bool().view(-1)
    #mask = torch.round(future).bool().view(-1)
    mask = deepcopy(future)
    mask[mask<=0.1] = 0.
    mask = mask.bool().view(-1)
    #loss, loss_len, loss_dict = loss_function(torch.nan_to_num(preds), torch.nan_to_num(future), mask)
    loss, loss_len, loss_dict = loss_function(preds, future, mask)
    mask = mask.reshape(shape)
    return loss, loss_len, preds, loss_dict, mask



def plot_metrics(y, x=None, persistence_loss=None, title="Metrics", color="red"):

    plt.figure(figsize=(7,5))
    if x:
        plt.plot(x, y, color=color, label="Model metrics")
    else:
        plt.plot(y, color=color, label="Model metrics")

    if persistence_loss is not None:
        if x:
            plt.plot(x, [persistence_loss]*len(y), color="yellow", label="Persistence metrics")
        else:
            plt.plot([persistence_loss]*len(y), color="yellow", label="Persistence metrics")

    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.show()



def add_image_to_base_layer(arr, path_to_base_layer, alpha=0.5):

    overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    overlay[..., :3] = 255
    overlay[..., 3] = arr*alpha*256.
    overlay = Image.fromarray(overlay, 'RGBA')

    img = Image.open(path_to_base_layer).convert('RGBA')
    img = img.resize(overlay.size)

    result = Image.alpha_composite(img, overlay)
    result = np.array(result)
    result = np.delete(result, 3, axis=2)

    return result


def add_base_layer_to_timeseries(timeseries, path_to_base_layer, alpha=0.5):
    results = []
    for arr in timeseries:
        new_arr = add_image_to_base_layer(arr, path_to_base_layer, alpha=alpha)
        results.append(new_arr)
    results = np.array(results)
    return results


def no_transform(x, device=None):
    return x



def create_animation(data, names, filename, i=0):
    # создаем пустой список для кадров
    frames = []

    # проходимся по каждому моменту времени
    for t in range(data.shape[0]):
        # создаем пустой список для изображений
        images = []

        # проходимся по каждому параметру
        for idx, x in enumerate(data[t]):
            # создаем изображение с подписью параметра
            '''img = np.random.rand(data.shape[2], data.shape[3])
            img = np.uint8(img * 255)
            img = np.stack([img] * 3, axis=-1)
            #img[10:20, 10:20, :] = [255, 0, 0]'''
            img = Image.fromarray(x)
            draw = ImageDraw.Draw(img)
            #font = ImageFont.load("arial.pil")
            draw.text((10, 5), names[idx], 1)

            # добавляем изображение в список
            images.append(np.array(img))

        # добавляем список изображений в список кадров
        frames.append(np.concatenate(images, axis=1))

    # сохраняем анимацию в видео формата mp4
    frames = np.array(frames)
    print(frames.shape)
    imageio.mimsave(f'some_files/gfs_{filename}_day{i+1}.mp4', frames, fps=1)



def make_video(arr, model, filename, device,
               train_frac=0.8, days_in=7, concat_pers=False, path_to_base_layer=None, days_out=1,
               arr_gfs=None, gfs_names=None, preds_transform=no_transform):

    val_from = int(len(arr)*train_frac)
    if arr_gfs is not None:
        arr_gfs = arr_gfs[val_from:]
    dataset_val = RadarDataset(arr[val_from:], # Как мы понимаем что в данных нет повторов?
                               days_in=days_in,
                               days_out=days_out,
                               device=device,
                               gfs_data=arr_gfs)

    dataloader_val = DataLoader(
                                dataset=dataset_val,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=False,
                                )

    loss_function = MaskedLoss()
    persistence = Persistence(days_out=days_out)
    preds = None
    past = None
    pers = []
    val = []

    all_future, all_preds, all_pers, all_gfs = [], [], [], []
    history_bugs = []

    #all_preds = []

    for (idx, batch) in tqdm(enumerate(dataloader_val)): # two cycles: for predictions creation & for loss estimation?
                                                         # requires rewrite batch generation & eval func

        future = deepcopy(batch["future"]).float()
        future = preds_transform(future, device=device)
        past_orig = deepcopy(batch["past"].float())

        loss, _, pers_preds, loss_dict, mask = eval_model(persistence, past_orig, future, loss_function, device=device)
        pers.append(loss.cpu().detach().numpy())
        pers_preds = pers_preds.cpu().detach().numpy()[0]
        if len(all_pers) > 0: # check that it's don't use in eval
            pers_preds[pers_preds<=0.2] = all_pers[-1][pers_preds<=0.2]
        #plt.imshow(pers_preds[-1])
        #plt.show()
        all_pers.append(pers_preds)

        future = batch["future"].float()
        future = preds_transform(future, device=device)
        if past is None:
            past = batch["past"].float()

        past = past[:, :days_in, :, :] # is it ok?

        '''plt.imshow(batch["past_last_layer"][-1][-1].float())
        plt.show()'''

        '''print("PAST BEFORE CONCAT")
        plt.imshow(past[-1][-1].cpu())
        plt.show()'''

        if concat_pers:
            past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
            past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                        past[:, -2:-1, :, :],
                                                        device=device)

        elif preds is not None:
            past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
            past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                        preds.detach(),
                                                        device=device)

        #HHHHHHHHHHHH
        #print(past.shape)

        '''print("LAST PAST LAYER")
        plt.imshow(batch["past_last_layer"][-1][-1].float())
        plt.show()'''

        #print("PAST")
        #plt.imshow(past[-1][-1].cpu())
        #plt.show()
        past = torch.cat((past, batch["gfs"].float()), axis=1)


        # re-create past data right here to strictly push it into model
        loss, val_loss_len, preds, loss_dict, mask = eval_model(model, past, future, loss_function, device=device)
        # are there any ways to make black areas after this code string?

        #print("PREDS")
        # plt.imshow(preds[-1][-1].cpu().detach().numpy())
        # plt.show()

        val.append(loss.cpu().detach().numpy())
        future = preds_transform(future, device=device)
        all_future.append(future.cpu().detach().numpy()[0])

        new_gfs = batch["gfs"].float().cpu().detach().numpy()
        for g in new_gfs:
            new_gfs[:]
        all_gfs.append(new_gfs)
        #all_preds.append(preds.cpu().detach().numpy()[0])

        preds_shape = preds.cpu().detach().numpy()[0][0].shape
        #print(preds_shape)
        if len(preds[preds<0.2]) >= 0.05 * preds_shape[0] * preds_shape[1]:
            print("PREDS WITH BLAK AREA IDX: ", idx)
            his = list(deepcopy(past.cpu().detach().numpy())[0])
            print(mask.shape)
            his.append(mask.cpu().detach().numpy())
            his.append(preds.cpu().detach().numpy()[0][0])
            his.append(future.cpu().detach().numpy()[0][0])
            history_bugs.append(np.concatenate(his, axis=1))


        all_preds.append(preds_transform(preds, device="cpu").cpu().detach().numpy()[0])

    if len(history_bugs)>0:
        imageio.mimsave(f'some_files/bugs.mp4',
                        np.array(history_bugs), fps=1)
    else:
        print("NO BLACK AREA BUGS")

    all_future = np.array(all_future)
    all_preds = np.array(all_preds)
    all_pers = np.array(all_pers)
    all_gfs = np.array(all_gfs)

    val, pers = np.array(val), np.array(pers)
    val = val[val==val]
    pers = pers[pers==pers]
    print("VAL LOSS: ", np.mean(val))
    print("PERS LOSS: ", np.mean(pers))

    for i in range(days_out):
        all_preds_i = all_preds[:, i, :, :]
        all_future_i = all_future[:, i, :, :]
        all_pers_i = all_pers[:, i, :, :]
        if path_to_base_layer is None:
            arr_0 = np.zeros((all_future_i.shape[0], all_future_i.shape[1], 3))
            arr_1 = np.zeros((all_future_i.shape[0], all_future_i.shape[1], 3))
            arr_1[:, :, :] = 1.
            concated_arr =  np.concatenate([all_preds_i,
                                            arr_0, arr_1, arr_0,
                                            all_future_i,
                                            arr_0, arr_1, arr_0,
                                            all_pers_i
                                            ],
                                            axis=2)

            imageio.mimsave(f'some_files/{filename}_day{i+1}.mp4',
                            np.nan_to_num(concated_arr), fps=1)

            if arr_gfs is not None:
                all_gfs_i = all_gfs[:, i, :, :, :]
                print('GFS I SHAPE: ', all_gfs_i.shape)
                print('PREDS I SHAPE: ', all_preds_i.shape)
                print('PERS I SHAPE: ', all_pers_i.shape)
                print('FUTURE I SHAPE: ', all_future_i.shape)
                all_gfs_i = np.concatenate([all_gfs_i,
                                            np.expand_dims(all_pers_i, 1),
                                            np.expand_dims(all_future_i, 1),
                                            np.expand_dims(all_preds_i, 1)], axis=1)

                gfs_names = gfs_names + ["Persistense", "Real Future", "Prediction"]
                create_animation(all_gfs_i, names=gfs_names, filename=filename, i=i)



        else:
            all_preds_i = add_base_layer_to_timeseries(all_preds_i, path_to_base_layer, alpha=0.9)
            all_future_i = add_base_layer_to_timeseries(all_future_i, path_to_base_layer, alpha=0.9)
            all_pers_i = add_base_layer_to_timeseries(all_pers_i, path_to_base_layer, alpha=0.9)

            arr_0 = np.zeros((all_future_i.shape[0], all_future_i.shape[1], 3, all_future_i.shape[3], ))
            arr_1 = np.zeros((all_future_i.shape[0], all_future_i.shape[1], 3, all_future_i.shape[3], ))
            arr_1[:, :, :, :] = 1.

            concated_arr = np.concatenate([all_preds_i,
                                           arr_0, arr_1, arr_0,
                                           all_future_i,
                                           arr_0, arr_1, arr_0,
                                           all_pers_i
                                           ],
                                          axis=2)
            video_writer = imageio.get_writer(f'some_files/{filename}_day{i+1}.mp4', fps=1)
            for frame in concated_arr:
                video_writer.append_data(frame)
            video_writer.close()


def plot_several_days_loss(loss_dict, x=None, title="Metrics"):

    plt.figure(figsize=(7, 5))
    for day in loss_dict.keys():
        y = loss_dict[day]
        if x:
            plt.plot(x, y, label=f"Day {day}")
        else:
            plt.plot(y, label=f"Day {day}")

    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.show()






def train_model(model, dataset_arr, device, video_name, epochs=10,
                train_size=0.8, val_size=0.1,
                validate_every=2, concat_preds=False, lr=0.01,
                concat_val=False, days_in=1, days_out=1,
                loss_function=None, concat_pers=False, path_to_base_layer=None, arr_gfs=None,
                preds_transform=no_transform, gfs_names=None):

    model = model.to(device)



    train_size = int(train_size*len(dataset_arr))
    if arr_gfs is not None:
        gfs_train = arr_gfs[:train_size]
        gfs_val = arr_gfs[train_size:]
    else:
        gfs_train, gfs_val = None, None

    dataset_train = RadarDataset(dataset_arr[:train_size],
                                 days_in=days_in,
                                 days_out=days_out,
                                 device=device,
                                 gfs_data=gfs_train)
    dataset_val = RadarDataset(dataset_arr[train_size:],#[train_size:train_size+val_size],
                               days_in=days_in,
                               days_out=days_out,
                               device=device,
                               gfs_data=gfs_val)

    dataloader_train = DataLoader(
                                  dataset=dataset_train,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=False,
                                  )

    dataloader_val = DataLoader(
                                  dataset=dataset_val,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=False,
                                  )

    if loss_function is None:
        loss_function = MaskedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_epochs = []

    train_losses_dict = {}
    for day in range(days_out):
        train_losses_dict[day] = []

    val_losses_dict = {}
    for day in range(days_out):
        val_losses_dict[day] = []

    for epoch in range(epochs):

        past = None
        preds = None
        loss_epoch = []

        losses_dict = {}
        for day in range(days_out):
            losses_dict[day] = []

        for (idx, batch) in tqdm(enumerate(dataloader_train)):

            model.zero_grad()
            future = batch["future"].float()
            future = preds_transform(future, device=device)
            if past is None:
                past = batch["past"].float()

            past = past[:, :days_in, :, :]

            if concat_preds and preds is not None:
                past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                            preds.detach(),
                                                            device=device)

            elif concat_pers:
                past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                            past[:, -2:-1, :, :],
                                                            device=device)
            else:
                past = batch["past"].float()

            past = torch.cat((past, batch["gfs"].float()), axis=1)
            loss, train_loss_len, preds, loss_dict, mask = eval_model(model, past, future, loss_function, device=device)
            #preds = preds_transform(preds, device=device)
            loss_epoch.append(loss.cpu().detach().numpy())
            for day in range(days_out):
                losses_dict[day].append(loss_dict[day].cpu().detach().numpy())



            if train_loss_len>0:
                loss.backward(retain_graph=True)
                optimizer.step()

        loss_epoch = np.array(loss_epoch)
        loss_epoch = loss_epoch[~np.isnan(loss_epoch)]
        loss_epoch = np.mean(loss_epoch)
        train_losses.append(loss_epoch)
        for day in range(days_out):
            d = np.array(losses_dict[day])
            d = d[~np.isnan(d)]
            train_losses_dict[day].append(np.mean(d))

        #print(train_losses_dict)

        if epoch % validate_every == 0:
            preds = None
            val_epochs.append(epoch)
            val_loss = []
            model.eval()     # Optional when not using Model Specific layer

            losses_dict = {}
            for day in range(days_out):
                losses_dict[day] = []

            for (idx, batch) in enumerate(dataloader_val):

                future = batch["future"].float()
                future = preds_transform(future, device=device)
                if past is None:
                    past = batch["past"].float()

                past = past[:, :days_in, :, :]

                if concat_preds and preds is not None:
                    past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                    past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                                preds.detach(),
                                                                device=device)

                elif concat_pers:
                    past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                    past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                                past[:, -2:-1, :, :],
                                                                device=device)
                else:
                    past = batch["past"].float()

                past = torch.cat((past, batch["gfs"].float()), axis=1)

                loss, val_loss_len, preds, loss_dict, mask = eval_model(model, past, future, loss_function, device=device)
                #preds = preds_transform(preds, device=device)
                val_loss.append(loss.cpu().detach().numpy())
                for day in range(days_out):
                    losses_dict[day].append(loss_dict[day])

            val_loss = np.array(val_loss)
            val_loss = np.mean(val_loss[~np.isnan(val_loss)])
            val_losses.append(val_loss)
            for day in range(days_out):
                d = np.array(losses_dict[day])
                d = d[~np.isnan(d)]
                val_losses_dict[day].append(np.mean(d))
            print(f"epoch {epoch},    train loss: {loss_epoch:0.04f},  train loss len: {train_loss_len}, "
                  f"  val loss: {val_loss:0.04f}")

    persistence = Persistence(days_out=days_out)
    persistence_train_losses = []
    for batch in dataloader_train:
        past, future = batch["past"].float(), batch["future"].float()
        future = preds_transform(future, device=device)
        loss, _, _, loss_dict, mask = eval_model(persistence, past, future, loss_function, device=device)
        persistence_train_losses.append(loss.cpu())
    persistence_train_losses = np.array(persistence_train_losses)
    persistence_train_losses = persistence_train_losses[persistence_train_losses==persistence_train_losses]

    persistence_val_losses = []
    for batch in dataloader_val:
        past, future = batch["past"].float(), batch["future"].float()
        future = preds_transform(future, device=device)
        loss, _, _, loss_dict, mask = eval_model(persistence, past, future, loss_function, device=device)
        persistence_val_losses.append(loss.cpu())
    persistence_val_losses = np.array(persistence_val_losses)
    persistence_val_losses = persistence_val_losses[persistence_val_losses==persistence_val_losses]

    print(f"TRAIN PERSISTENCE: {np.mean(persistence_train_losses):0.04f}")
    print("TRAIN LOSSES: ", train_losses_dict)
    plot_several_days_loss(train_losses_dict, x=None, title="Train loss")

    print(f"VAL PERSISTENCE: {np.mean(persistence_val_losses):0.04f}")
    plot_several_days_loss(val_losses_dict, x=val_epochs, title="Val loss")

    make_video(dataset_arr, model, video_name, device=device, days_in=days_in, days_out=days_out,
               path_to_base_layer=path_to_base_layer, concat_pers=concat_pers, arr_gfs=arr_gfs,
               gfs_names=gfs_names,
               preds_transform=no_transform)

    return model, train_losses, val_losses, val_epochs




# EXP MODEL


class ExpMaskedLoss(nn.Module):
    def __init__(self):
        super(ExpMaskedLoss, self).__init__()

    def forward(self, preds, targets, exp_mask, mask, exp_rate, device):

        preds = preds.view(-1)
        targets = targets.view(-1)
        exp_mask = torch.tensor(exp_mask).view(-1)
        #print("EXP SUM: ", torch.sum(exp_mask))
        #mask = exp_mask==exp_mask
        exp_mask = np.exp(-exp_mask*exp_rate)

        preds = preds[mask]
        targets = targets[mask]
        exp_mask = exp_mask[mask]

        loss = (preds - targets) ** 2 * exp_mask.to(device)
        loss_len = len(loss)
        loss = torch.mean(loss)

        return loss, loss_len



def eval_model_exp(model, past, future, exp_mask, device, exp_rate=0):

    loss_function = ExpMaskedLoss()
    preds = model(past.detach())
    bool_mask = future.cpu().bool().detach()   # те которые есть
    mask = (preds*future).bool().view(-1)

    exp_mask[exp_mask is not np.nan and ~bool_mask.bool()] += 1
    #exp_mask[exp_mask is not np.nan and ~bool_mask.bool()] += 1   # давность данных в днях
    exp_mask[bool_mask] = 0                                       # если данные обновились сегодня - 0
                                                                  # остальные поля - nan
    loss, loss_len = loss_function(torch.nan_to_num(preds), torch.nan_to_num(future),
                                   exp_mask, mask, exp_rate=exp_rate, device=device)
    return loss, loss_len, preds, exp_mask



def train_exp_model(model, dataset_arr, video_name, device,
                    epochs=10, train_size=0.8, val_size=0.1,
                    validate_every=2, concat_preds=False, concat_pers=False, lr=0.01,
                    concat_val=False, exp_rate=0,
                    days_in=1):

    model = model.to(device)

    train_size = int(train_size*len(dataset_arr))
    val_size = int(val_size*len(dataset_arr))

    dataset_train = RadarDataset(dataset_arr[:train_size],
                                 days_in=days_in,
                                 device=device)
    dataset_val = RadarDataset(dataset_arr[train_size:],
                               days_in=days_in,
                               device=device)

    dataloader_train = DataLoader(
                                  dataset=dataset_train,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=False,
                                  )

    dataloader_val = DataLoader(
                                  dataset=dataset_val,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=False,
                                  )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_epochs = []


    for epoch in range(epochs):
        exp_mask = None
        preds = None
        past = None
        #past_new_layer = None

        loss_epoch = []
        for (idx, batch) in tqdm(enumerate(dataloader_train)):

            past, future = batch["past"].float(), batch["future"].float()
            model.zero_grad()

            if exp_mask is None:
                exp_mask = np.empty(future.shape)
                exp_mask[:] = np.nan

            if concat_preds and preds is not None:
                past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                            preds,
                                                            device=device)
            elif concat_pers:
                past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                            past[:, -2:-1, :, :],
                                                            device=device)

            loss, train_loss_len, preds, exp_mask = eval_model_exp(model, past, future, exp_mask, device, exp_rate)
            loss_epoch.append(loss.cpu().detach().numpy())

            if train_loss_len>0:
                loss.backward(retain_graph=True)
                optimizer.step()

        loss_epoch = np.array(loss_epoch)
        loss_epoch = np.mean(loss_epoch[~np.isnan(loss_epoch)])
        train_losses.append(loss_epoch)

        if epoch % validate_every == 0:
            preds = None
            exp_mask = None
            val_epochs.append(epoch)
            val_loss = []
            model.eval()     # Optional when not using Model Specific layer

            for (idx, batch) in enumerate(dataloader_val):

                past, future = batch["past"].float(), batch["future"].float()

                if exp_mask is None:
                    exp_mask = np.empty(future.shape)
                    exp_mask[:] = np.nan

                if concat_preds and preds is not None:
                    past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                    past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                                preds,
                                                                device=device)

                elif concat_pers:
                    past[:, :-1, :, :] = past.clone()[:, 1:, :, :]
                    past[:, -1:, :, :] = concat_preds_to_future(batch["past_last_layer"].float(),
                                                                past[:, -2:-1, :, :],
                                                                device=device)

                loss, loss_len, preds, exp_mask = eval_model_exp(model, past, future, exp_mask, device,  exp_rate)
                val_loss.append(loss.cpu().detach().numpy())

            val_loss = np.array(val_loss)
            val_loss = np.mean(val_loss[~np.isnan(val_loss)])
            val_losses.append(val_loss)
            print(f"epoch {epoch},    train loss: {loss_epoch:0.04f},  train loss len: {train_loss_len}, "
                  f"  val loss: {val_loss:0.04f}")

    loss_function_pers = MaskedLoss()
    persistence = Persistence()
    persistence_train_losses = []
    for batch in dataloader_train:
        past, future = batch["past"].float(), batch["future"].float()
        loss, _, _ = eval_model(persistence, past, future, loss_function_pers, device=device)
        persistence_train_losses.append(loss.cpu())
    persistence_train_losses = np.array(persistence_train_losses)
    persistence_train_losses = persistence_train_losses[persistence_train_losses==persistence_train_losses]

    persistence_val_losses = []
    for batch in dataloader_val:
        past, future = batch["past"].float(), batch["future"].float()
        loss, _, _ = eval_model(persistence, past, future, loss_function_pers, device=device)
        persistence_val_losses.append(loss.cpu())
    persistence_val_losses = np.array(persistence_val_losses)
    persistence_val_losses = persistence_val_losses[persistence_val_losses==persistence_val_losses]

    print(f"TRAIN PERSISTENCE: {np.mean(persistence_train_losses):0.04f}")
    plot_metrics(train_losses, title="Train loss", color="red",)

    print(f"VAL PERSISTENCE: {np.mean(persistence_val_losses):0.04f}")
    plot_metrics(val_losses, val_epochs, title="Val loss", color="green",)

    make_video(dataset_arr, model, video_name, device=device, days_in=days_in, concat_pers=concat_pers)

    return model, train_losses, val_losses, val_epochs