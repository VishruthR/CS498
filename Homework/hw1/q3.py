###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist
import torch.nn.functional as F

def reduce_scatter(chunks, count, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    # TODO: Remove waits to optimize comm-compute overlap
    chunk_to_send = (rank - count) % world
    s = dist.isend(chunks[chunk_to_send], dst=right)
    s.wait()

    chunk_to_recv = (rank - count - 1) % world
    new_chunk = torch.empty_like(chunks[0])
    r = dist.irecv(new_chunk, src=left)
    r.wait()

    # update chunk
    chunk[chunk_to_recv] += new_chunk
    
        
def all_gather(chunks, count, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    # first iter, chunk rank + 1 % world is done
    # send that off to right
    # recv chunk rank % world from left
    # when you receive all gather, update the chunk and divide by world size
    # chunk rank + 1 - count % world gets sent to right
    # recv chunk rank - count % world from left
    # TODO: Do we need to do opt.step to get new params? I don't even know if we're working with grads
    chunk_to_send = (rank + 1 - count) % world
    if count == 0:
        # on first iter of all gather, average grads
        chunks[chunk_to_send] /= world

    s = dist.isend(chunks[chunk_to_send], dst=right)
    s.wait()

    chunk_to_recv = (rank - count) % world
    new_chunk = torch.empty_like(chunks[0])
    r = dist.irecv(new_chunk, src=left)
    r.wait()

    chunk[chunk_to_recv] = new_chunk # update since new_chunk should be averaged already

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padding_needed = world_size - (n % world_size)
    padded_flat = F.pad(flat, pad=(0, padding_needed), value=0)
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    for i in range(world - 1):
        print(f"reduce_scatter {i}")
        reduce_scatter(chunks, i, world, rank, left, right)

    for i in range(world - 1):
        print(f"all_gather {i}")
        all_gather(chunks, i, world, rank, left, right)


    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return