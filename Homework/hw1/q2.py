###Q2: Parameter Server###
###please implement PS method, using  parameter update&optimizor states all be held in rank0###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def server(params, opt, world):
    # ---- aggregate grads from workers ----
    print("params", len(params))
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous() #usage: tranfer a list tensor to one 1-D tensor
    print("flat_grad", flat_grad.shape)
    ##here, you should generate one big 1-D tensor containing all parameters to make the transfer process easy
    agg = flat_grad.clone() #agg as a aggregated counter to record sum gradients
    print("agg", agg.shape)

    #                                                                   #
    #                                                                   #
    # your code here: receive gradients form worker, and add them to agg#
    #                                                                   #
    #                                                                   #
    grad_1 = torch.empty_like(flat_grad)
    r_1 = dist.irecv(grad_1, src=1)
    r_1.wait()

    grad_2 = torch.empty_like(flat_grad)
    r_2 = dist.irecv(grad_2, src=2)

    print("grad_1", grad_1.shape)
    print("grad_2", grad_2.shape)

    # TODO: May cause issues with autograd
    # Finding average
    agg += grad_1
    agg += grad_2
    agg /= 3

    synced_grads = _unflatten_dense_tensors(agg, [p.grad for p in params])
    # ---- set averaged grads locally & step ----
    for g, s in zip([p.grad for p in params], synced_grads):
        g.copy_(s)
    opt.step()

    # ---- broadcast updated params for this subset ----
    flat_param = _flatten_dense_tensors([p.data for p in params]).contiguous()
    print("flat param", flat_param.shape)
    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D parameter tensor to all workers   #
    #                                                                   #
    #                                                                   #
    s_1 = dist.isend(flat_param, dst=1)
    s_1.wait()

    s_2 = dist.isend(flat_param, dst=2)
    s_2.wait()

def worker(params):
    print("params", len(params))
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    print("flat_grads", flat_grad.shape)
    # ---- push grads to server ----

    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D gradient to server
    #                                                                   #
    #                                                                   #
    s = dist.isend(flat_grad, dst=0)
    s.wait()

    # ---- receive updated params, write into local model ----
    
    #                                                                   #
    #                                                                   #
    # your code here: please get correct 1-D packed parameter from server
    #           And then unpacked it and store in synced_params
    #                                                                   #
    new_params = torch.empty_like(flat_grad)
    r = dist.irecv(new_params, src=0)
    r.wait()

    synced_params = _unflatten_dense_tensors(new_params, [p.data for p in params])


    # ---- syncronize the parameters ----
    for p, s in zip(params, synced_params):
        p.data.copy_(s)

def PS_grads_(model,world_size=None, rankid=None, opt=None):
    """
    Synchronous PS step:
      - Rank 0: receive grads, sum/avg, set grads, opt.step(), broadcast updated params.
      - Rank >0: send grads, receive updated params, write into local model.
    Only processes the subset of parameters with non-None grads.
    """
    world = world_size
    rank  = rankid
    
    # Fast path: single process
    if world == 1:
        opt.step()
        return

    #not necessary, as in most case, params won't be empty list...
    # Collect params that participated in this backward pass
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        # No grads this step; only server might still want to advance schedulers, etc.
        if rank == 0:
            opt.step()
        return

    if rank == 0:
        server(params, opt, world)

    else:
        worker(params)

    # Optional hard step boundary
    dist.barrier()