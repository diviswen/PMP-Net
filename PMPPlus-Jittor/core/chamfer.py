import jittor as jt

def select_vertices(verts, idxs):
    batch_size = verts.shape[0]
    assert idxs.shape[0] == batch_size

    verts = verts.reindex([batch_size, idxs.shape[1], 3], [
        'i0',
        '@e0(i0, i1)',
        'i2'
    ], extras=[idxs])
    return verts


cpu_src = '''
    for (int bs = 0; bs < in0_shape0; ++bs)
        for (int i = 0; i < in0_shape1; ++i) {
            float min_dis = (@in0(bs, i, 0) - @in1(bs, 0, 0)) * (@in0(bs, i, 0) - @in1(bs, 0, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, 0, 1)) * (@in0(bs, i, 1) - @in1(bs, 0, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, 0, 2)) * (@in0(bs, i, 2) - @in1(bs, 0, 2));
            @out(bs, i) = 0;
            for (int j = 1; j < in1_shape1; ++j) {
                float dis = (@in0(bs, i, 0) - @in1(bs, j, 0)) * (@in0(bs, i, 0) - @in1(bs, j, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, j, 1)) * (@in0(bs, i, 1) - @in1(bs, j, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, j, 2)) * (@in0(bs, i, 2) - @in1(bs, j, 2));
                if (dis < min_dis) {
                    min_dis = dis;
                    @out(bs, i) = j;
                }
            }
        }
'''

cuda_src = '''
    __global__ void chamfer_loss_min_idx_kernel(@ARGS_DEF) {
        @PRECALC
        int bs = blockIdx.x;
        int n = in0_shape1;
        int m = in1_shape1;

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float min_dis = (@in0(bs, i, 0) - @in1(bs, 0, 0)) * (@in0(bs, i, 0) - @in1(bs, 0, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, 0, 1)) * (@in0(bs, i, 1) - @in1(bs, 0, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, 0, 2)) * (@in0(bs, i, 2) - @in1(bs, 0, 2));
            @out(bs, i) = 0;
            for (int j = 1; j < m; ++j) {
                float dis = (@in0(bs, i, 0) - @in1(bs, j, 0)) * (@in0(bs, i, 0) - @in1(bs, j, 0)) +
                            (@in0(bs, i, 1) - @in1(bs, j, 1)) * (@in0(bs, i, 1) - @in1(bs, j, 1)) +
                            (@in0(bs, i, 2) - @in1(bs, j, 2)) * (@in0(bs, i, 2) - @in1(bs, j, 2));
                if (dis < min_dis) {
                    min_dis = dis;
                    @out(bs, i) = j;
                }
            }
        }
    }

    chamfer_loss_min_idx_kernel<<<in0_shape0, 512>>>(@ARGS);
'''


def chamfer_loss(pc1, pc2, reduction='mean', sqrt=True):
    '''
    return the chamfer loss from pc1 to pc2.

    Parameters:
    ===========
        pc1:  [B, N, xyz]
        pc2:  [B, N, xyz]
        reduction: 'mean', 'sum', or None
    '''
    batch_size_1, n_samples_pc1, _ = pc1.shape
    batch_size_2, n_samples_pc2, _ = pc2.shape

    assert batch_size_1 == batch_size_2
    batch_size = batch_size_1

    idx = jt.code([batch_size, n_samples_pc1], 'int32', [pc1, pc2],
                        cpu_src=cpu_src,
                        cuda_src=cuda_src)

    nearest_pts = select_vertices(pc2, idx)
    if sqrt:
        chamfer_distance = (((pc1 - nearest_pts) ** 2).sum(dim=-1)).sqrt()
    else:
        chamfer_distance = (((pc1 - nearest_pts) ** 2).sum(dim=-1))

    if reduction is None:
        return chamfer_distance
    elif reduction == 'sum':
        return jt.sum(chamfer_distance)
    elif reduction == 'mean':
        return jt.mean(chamfer_distance)


def chamfer_loss_bidirectional_sqrt(pc1, pc2):
    '''
    return the chamfer loss between two point clouds.
    '''
    return (chamfer_loss(pc1, pc2, sqrt=True) + chamfer_loss(pc2, pc1, sqrt=True)) / 2


def chamfer_loss_bidirectional(pc1, pc2):
    '''
    return the chamfer loss between two point clouds.
    '''
    l = chamfer_loss(pc1, pc2, sqrt=False)
    r = chamfer_loss(pc2, pc1, sqrt=False)
    return l + r