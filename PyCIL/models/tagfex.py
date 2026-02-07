# Please note that only "cifar100_aa" and "cifar10_aa" are supported for TagFex in PyCIL.
# For large datasets like ImageNet, please refer to the offical code repo https://github.com/bwnzheng/TagFex_CVPR2025.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import TagFexNet
from utils.toolkit import count_parameters, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005
momentum = 0.9

epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class TagFex(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TagFexNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        self.last_ta_net = self._network.get_freezed_copy_ta()
        self.last_projector = self._network.get_freezed_copy_projector()
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        torch.backends.cudnn.benchmark = True
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        
        prog_bar = tqdm(range(init_epoch))
       
        
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                out = self._network(inputs)
                logits = out["logits"]
                embedding = out["embedding"]

                ce_loss = F.cross_entropy(logits, targets)
                infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])
                loss = ce_loss + infonce_loss * self.args['contrast_factor']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1, inputs2, targets = inputs1.to(self._device), inputs2.to(self._device), targets.to(self._device)

                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
                
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                embedding = outputs['embedding']

                infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                predicted_feature = outputs['predicted_feature']
                old_ta_feature = self.last_ta_net(inputs.contiguous())['features']
                kd_loss = infoNCE_distill_loss(self.last_projector(predicted_feature), self.last_projector(old_ta_feature), self.args['infonce_kd_temp'])
                trans_logits = outputs["trans_logits"]
                cur_task_mask = (targets >= self._known_classes)
                trans_cls_loss = F.cross_entropy(trans_logits[cur_task_mask], targets[cur_task_mask] - self._known_classes)

                if trans_cls_loss < loss_clf:
                    T = self.args['kd_temp']
                    transfer_loss = F.kl_div((logits[cur_task_mask][:, self._known_classes:] / T).log_softmax(dim=1), (trans_logits.detach()[cur_task_mask] / T).softmax(dim=1), reduction='batchmean')
                else:
                    transfer_loss = torch.tensor(0., device=self._device)

                auto_kd_factor = self._known_classes / self._total_classes
                loss = loss_clf + \
                self.args['aux_factor'] * loss_aux + \
                self.args['contrast_factor'] * (infonce_loss * (1 - auto_kd_factor) + self.args['contrast_kd_factor'] * kd_loss * auto_kd_factor) + \
                self.args['trans_cls_factor'] * trans_cls_loss + \
                self.args['transfer_factor'] * transfer_loss         

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

def infoNCE_loss(feats, t):
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def infoNCE_distill_loss(p_feats, z_feats, t):
    # print(p_feats.shape, z_feats.shape)
    cos_sim = F.cosine_similarity(p_feats[:,None,:], z_feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll
