import unittest
import torch

from fairseq.modules.temporal_multihead_attention import TemporalMultiheadAttention


class TestTemporalMultiheadAttention(unittest.TestCase):

    def test_random_computation(self):
        print(f"{self.test_random_computation.__name__} is testing......")
        # fix random seed since attention contains randomly initialized parameters
        torch.manual_seed(0)
        head = 1
        time = 1
        batch = 1

        attn = TemporalMultiheadAttention(8, head)
        # (tgt) Time x (src + tgt, output) Time x Batch x Channel
        # 3 x 5 x 2 x 8
        query = torch.randn(3, 5, 2, 8)
        # Batch x (src + tgt, input) Time
        # 2 x 5
        key_padding_mask = torch.Tensor(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0]]
        ).byte()
        # (tgt) Time x (src + tgt, input) Time
        # 3 x 5
        attn_mask = torch.Tensor(
            [[0, 0, 0, float('-inf'), float('-inf')],
             [0, 0, 0, 0, float('-inf')],
             [0, 0, 0, 0, 0]]
        )

        # correct answer, verified in 2D cases (num_heads=1)
        part = query[time, :, batch, :]
        q, k, v = attn.in_proj_qkv(part)
        q *= attn.scaling
        weight = torch.matmul(q, k.transpose(0, 1))
        weight += attn_mask[time, :].unsqueeze(0)
        weight = weight.float().masked_fill(
            key_padding_mask[batch, :].unsqueeze(0),
            float('-inf'),
        ).type_as(weight)
        weight = torch.nn.functional.softmax(weight, dim=-1).type_as(weight)
        ans = torch.matmul(weight, v)
        ans = attn.out_proj(ans)

        res, weights = attn(query, query, query, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # check input output dims
        for i, (in_dim, out_dim) in enumerate(zip(res.size(), query.size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")

        # check output data
        self.assertTrue(torch.all(torch.eq(ans, res[time, :, batch, :])),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")

    def test_fix_computation(self):
        print(f"{self.test_fix_computation.__name__} is testing......")
        # fix random seed since attention contains randomly initialized parameters
        torch.manual_seed(0)
        head = 1
        time = 1
        batch = 1

        attn = TemporalMultiheadAttention(8, head)
        # (tgt) Time x (src + tgt, output) Time x Batch x Channel
        # 3 x 5 x 2 x 8
        query = torch.Tensor(
            [[[[1., 1., -2., 1., 1., 1., 1., 1.],
               [-1., 1., -0., -0., 1., -1., -0., -2.]],

              [[-0., -0., 1., -0., 1., -0., -3., -1.],
               [-0., -1., -2., -1., -0., -1., -0., 1.]],

              [[1., 2., 1., -0., -2., -1., -1., 2.],
               [1., -0., -0., 1., -0., 1., -2., -0.]],

              [[1., 3., -0., 2., -0., 2., -0., 2.],
               [-0., 2., 1., 1., -0., -1., 1., -2.]],

              [[-0., -0., 1., -0., -1., 1., 2., -0.],
               [-0., 1., 1., 1., -0., -2., 1., 4.]]],

             [[[-1., 1., -0., -1., -1., -0., 1., -0.],
               [1., -0., -0., -0., 1., -1., -1., -1.]],

              [[-1., -0., 1., -0., -0., -0., 1., -0.],
               [2., 1., 1., -1., -1., 1., -1., 1.]],

              [[1., -0., 1., -1., -0., 1., 3., 1.],
               [-0., 1., -0., 1., -0., -0., 1., -2.]],

              [[2., 3., 1., 1., 1., -2., 1., 1.],
               [-1., -0., -0., 1., -2., 1., -0., 2.]],

              [[-1., 1., 1., 1., 1., -0., -0., 2.],
               [1., -1., 1., 1., 1., -2., 1., -0.]]],

             [[[-1., 1., -0., -1., -0., -1., -0., 2.],
               [-0., 2., 2., -1., 1., 1., -0., 3.]],

              [[-0., -1., 1., 2., 2., 2., -0., 2.],
               [-0., -0., -0., 1., -0., -1., -0., -3.]],

              [[-0., -2., -0., 1., 1., -0., -0., -0.],
               [2., 2., -0., 1., 1., 1., -0., -0.]],

              [[-0., -1., 1., 1., 2., 2., 1., -0.],
               [-2., -0., -1., 2., 1., -0., 3., 1.]],

              [[1., 2., -0., 1., -0., 1., 1., -0.],
               [2., -1., 1., -0., 1., 1., 2., -0.]]]]
        )
        # Batch x (src + tgt, input) Time
        # 2 x 5
        key_padding_mask = torch.Tensor(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0]]
        ).byte()
        # (tgt) Time x (src + tgt, input) Time
        # 3 x 5
        attn_mask = torch.Tensor(
            [[0, 0, 0, float('-inf'), float('-inf')],
             [0, 0, 0, 0, float('-inf')],
             [0, 0, 0, 0, 0]]
        )

        # correct answer, verified in 2D cases (num_heads=1)
        ans = torch.Tensor(
            [[0.1022, 0.2863, 0.2398, 0.3441, -0.3025, -0.3131, -0.3484, 0.5686],
             [0.0658, 0.1594, 0.1905, 0.2390, -0.1565, -0.1553, -0.1953, 0.3024],
             [-0.0014, 0.0510, 0.2906, 0.1588, -0.0130, -0.1198, -0.1511, 0.2360],
             [-0.1450, -0.4273, 0.1309, -0.2354, 0.5407, 0.4568, 0.4101, -0.7381],
             [0.0184, 0.1435, 0.3526, 0.2372, -0.1161, -0.2531, -0.2787, 0.4596]]
        )

        res, weights = attn(query, query, query, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # check input output dims
        for i, (in_dim, out_dim) in enumerate(zip(res.size(), query.size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")

        # check output data
        self.assertTrue(torch.all(torch.le(torch.abs(res[time, :, batch, :] - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")


if __name__ == "__main__":
    unittest.main()
