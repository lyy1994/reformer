import unittest
import torch

from fairseq.modules.separable_attention import SeparableAttention


class TestSeparableAttention(unittest.TestCase):

    def test_dec_self_attn(self):
        print(f"{self.test_dec_self_attn.__name__} is testing......")
        # fix random seed since attention contains randomly initialized parameters
        torch.manual_seed(0)
        head = 1
        time = 1
        batch = 1

        attn = SeparableAttention(8, head)
        # Time x Source x Batch x Channel
        # 3 x 5 x 2 x 8
        query = torch.randn(3, 5, 2, 8)
        # Batch x Source
        # 2 x 5
        key_padding_mask = torch.Tensor(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1]]
        ).byte()
        # (output) Time x (input) Time
        # 3 x 3
        attn_mask = torch.Tensor(
            [[0, float('-inf'), float('-inf')],
             [0, 0, float('-inf')],
             [0, 0, 0]]
        )

        # in parallel training
        res, _ = attn(query, query, query,
                      key_padding_mask=None, attn_mask=attn_mask)
        res = res.masked_fill(
            key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1),
            0
        ).type_as(res)
        # in one step decoding (without cache)
        res2, _ = attn(query[time:time + 1, :, batch:batch + 1, :].clone(),
                       query[:time + 1, :, batch:batch + 1, :],
                       query[:time + 1, :, batch:batch + 1, :],
                       key_padding_mask=None, attn_mask=None)
        res2 = res2.masked_fill(
            key_padding_mask[batch:batch + 1, :].transpose(0, 1).unsqueeze(0).unsqueeze(-1),
            0
        ).type_as(res2)
        res2 = res2.squeeze(2)
        # in one step decoding (with cache)
        incremental_state = {}
        for i in range(time + 1):
            res3, _ = attn(query[i:i + 1, :, batch:batch + 1, :],
                           query[i:i + 1, :, batch:batch + 1, :],
                           query[i:i + 1, :, batch:batch + 1, :],
                           key_padding_mask=None, attn_mask=None,
                           incremental_state=incremental_state)
        res3 = res3.masked_fill(
            key_padding_mask[batch:batch + 1, :].transpose(0, 1).unsqueeze(0).unsqueeze(-1),
            0
        ).type_as(res3)
        res3 = res3.squeeze(2)

        # correct answer, verified in 2D cases (num_heads=1)
        part = query[:time + 1, :, batch, :]
        # Source x Time x Channel, 5 x 3 x 8
        part = part.transpose(0, 1)
        # 5 x 1 x 8
        q = attn.in_proj_q(part[:, time:time + 1, :])
        k, v = attn.in_proj_kv(part)
        q *= attn.scaling
        weight = torch.bmm(q, k.transpose(1, 2))
        weight = torch.nn.functional.softmax(weight, dim=-1).type_as(weight)
        # 5 x 1 x 8
        ans = torch.bmm(weight, v)
        ans = attn.out_proj(ans)
        ans = ans.transpose(0, 1)
        ans = ans.masked_fill(
            key_padding_mask[batch, :].unsqueeze(0).unsqueeze(-1),
            0
        ).type_as(ans)

        # check input output dims
        for i, (in_dim, out_dim) in enumerate(zip(res.size(), query.size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")
        # check output data
        self.assertTrue(torch.all(torch.le(torch.abs(res[time:time + 1, :, batch, :] - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")
        print("| parallel training passed")

        for i, (in_dim, out_dim) in enumerate(zip(res2.size(), query[time:time + 1, :, batch, :].size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")
        self.assertTrue(torch.all(torch.le(torch.abs(res2 - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")
        print("| one step decoding (without cache) passed")

        for i, (in_dim, out_dim) in enumerate(zip(res3.size(), query[time:time + 1, :, batch, :].size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")
        self.assertTrue(torch.all(torch.le(torch.abs(res3 - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")
        print("| one step decoding (with cache) passed")

    def test_enc_self_attn(self):
        print(f"{self.test_enc_self_attn.__name__} is testing......")
        # fix random seed since attention contains randomly initialized parameters
        torch.manual_seed(0)
        head = 1
        time = 1
        batch = 1

        attn = SeparableAttention(8, head, tgt_attn=False)
        # Time x Source x Batch x Channel
        # 3 x 5 x 2 x 8
        query = torch.randn(3, 5, 2, 8)
        # Batch x Source
        # 2 x 5
        key_padding_mask = torch.Tensor(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1]]
        ).byte()
        # (output) Time x (input) Time
        # 3 x 3
        attn_mask = torch.Tensor(
            [[0, float('-inf'), float('-inf')],
             [0, 0, float('-inf')],
             [0, 0, 0]]
        )

        # in parallel training
        res, _ = attn(query, query, query,
                      key_padding_mask=key_padding_mask, attn_mask=None)
        # in one step training / decoding (without cache)
        res2, _ = attn(query[time:time + 1, :, batch:batch + 1, :],
                       query[time:time + 1, :, batch:batch + 1, :],
                       query[time:time + 1, :, batch:batch + 1, :],
                       key_padding_mask=key_padding_mask[batch:batch+1, :], attn_mask=None)
        res2 = res2.squeeze(2)

        # correct answer, verified in 2D cases (num_heads=1)
        # Time x Source x Channel, 3 x 5 x 8
        part = query[time:time + 1, :, batch, :]
        # 3 x 5 x 8
        q = attn.in_proj_q(part)
        k, v = attn.in_proj_kv(part)
        q *= attn.scaling
        # 3 x 5 x 5
        weight = torch.bmm(q, k.transpose(1, 2))
        weight = weight.float().masked_fill(
            key_padding_mask[batch, :].unsqueeze(0).unsqueeze(0),
            float('-inf'),
        ).type_as(weight)
        weight = torch.nn.functional.softmax(weight, dim=-1).type_as(weight)
        # 3 x 5 x 8
        ans = torch.bmm(weight, v)
        ans = attn.out_proj(ans)

        # check input output dims
        for i, (in_dim, out_dim) in enumerate(zip(res.size(), query.size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")
        # check output data
        self.assertTrue(torch.all(torch.le(torch.abs(res[time:time + 1, :, batch, :] - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")
        print("| parallel training passed")

        for i, (in_dim, out_dim) in enumerate(zip(res2.size(), query[time:time + 1, :, batch, :].size())):
            self.assertEqual(in_dim, out_dim,
                             msg=f"{i}th dim of result ({in_dim}) and q ({out_dim}) is not compatible")
        self.assertTrue(torch.all(torch.le(torch.abs(res2 - ans), 1e-4)),
                        msg=f"result and q are not equal in time={time}, batch={batch} when num_heads={head}")
        print("| one step training / decoding passed")


if __name__ == "__main__":
    unittest.main()
