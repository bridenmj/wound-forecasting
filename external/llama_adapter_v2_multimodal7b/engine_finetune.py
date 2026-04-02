import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    clip_grad=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask, prompt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        #if data_iter_step == 0:  # first iter
        #print(data_iter_step == 0)
        # ignore_idx = -100
        # with torch.no_grad():
        #     print("shapes", examples.shape, labels.shape)
        #     sup = (labels != ignore_idx).sum().item()
        #     tot = labels.numel()
        #     print("supervised tokens:", sup, "/", tot, "=", sup/tot)
        #     print("labels min/max:", labels[labels!=ignore_idx].min().item(), labels[labels!=ignore_idx].max().item())
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        examples = examples.to(device)
        labels = labels.to(device)
        #print(examples.dtype, labels.dtype)
        prompt = prompt.to(device, non_blocking=True)
        
        # print("examples.sum()", examples.sum().item())
        # print("labels.sum()", labels.sum().item())
        # print("prompt.sum()", prompt.sum().item())
        # assert examples.sum() > 0
        # assert prompt.sum() > 0
        # assert labels.sum() > 0
        #torch.compiler.cudagraph_mark_step_begin()
        with torch.cuda.amp.autocast():
             c_loss, m_loss = model(examples, labels, prompt)
        loss = c_loss  #+ m_loss * 0
        loss_value = loss.item()
        c_loss_value = float(c_loss.detach())
        m_loss_value = float(m_loss.detach())

        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad norm: {param.grad.norm().item()}")
                    
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),clip_grad = clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                
        if data_iter_step == 0:  # only check first batch
                    total_grad = 0.0
                    nonzero_count = 0
                    max_g = 0.0
                    worst_param = None
                
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            grad_norm = p.grad.detach().float().norm().item()
                            if grad_norm > 0:
                                nonzero_count += 1
                
                            total_grad += grad_norm
                            if grad_norm > max_g:
                                max_g = grad_norm
                                worst_param = name
                
                    print(f"Grad summary [step {data_iter_step}]:")
                    print(f"  nonzero grads: {nonzero_count}")
                    print(f"  sum-grad-norm: {total_grad:.4e}")
                    print(f"  max grad norm: {max_g:.4e} (param: {worst_param})")
                
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
 
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if name in ["text_encoder.norm.weight", "adapter_query.weight"]:
        #             print(f"{name} grad norm: {param.grad.norm().item()}") 
        #print("data_iter_step", data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


        #if data_iter_step % 100 == 0 and data_iter_step != 1:
            # #print(f"Grad norm: {grad_norm:.6f} | Loss scale: {loss_scaler._scaler.get_scale()}")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad norm: {param.grad.norm().item()}")
            #print("len(model.named_parameters())", len(model.named_parameters()))
            # for name, param in model.named_parameters():
            #     print("param.grad is None",name, param.grad is not None)
            #     if param.grad is not None:
            #         print(f"{name} grad norm: {param.grad.norm().item()}")
