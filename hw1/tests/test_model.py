import unittest

import torch
from torch.nn import functional as F

from hw1_imitation.model import FlowMatchingPolicy, MSEPolicy


class TestMSEPolicy(unittest.TestCase):
    def test_predicts_action_chunks_and_uses_mse_loss(self) -> None:
        policy = MSEPolicy(3, 2, 4, hidden_dims=(5,))
        state = torch.randn(2, 3)
        target = torch.randn(2, 4, 2)

        try:
            prediction = policy.sample_actions(state)
        except NotImplementedError:
            self.fail("MSEPolicy.sample_actions is not implemented")

        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(
            torch.allclose(
                policy.compute_loss(state, target), F.mse_loss(prediction, target)
            )
        )


class TestFlowMatchingPolicy(unittest.TestCase):
    def test_sampling_rejects_nonpositive_step_count(self) -> None:
        policy = FlowMatchingPolicy(3, 2, 4)

        try:
            with self.assertRaisesRegex(ValueError, "positive"):
                policy.sample_actions(torch.randn(2, 3), num_steps=0)
        except ZeroDivisionError:
            self.fail("sample_actions should reject nonpositive num_steps")

    def test_loss_uses_the_straight_line_velocity_target(self) -> None:
        policy = FlowMatchingPolicy(3, 2, 4, hidden_dims=())
        linear = next(
            module for module in policy.modules() if isinstance(module, torch.nn.Linear)
        )
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.zero_()

        state = torch.randn(2, 3)
        expert_chunk = torch.randn(2, 4, 2)
        torch.manual_seed(0)
        noise = torch.randn_like(expert_chunk)
        expected_loss = F.mse_loss(torch.zeros_like(expert_chunk), expert_chunk - noise)

        torch.manual_seed(0)
        try:
            loss = policy.compute_loss(state, expert_chunk)
        except NotImplementedError:
            self.fail("FlowMatchingPolicy.compute_loss is not implemented")

        torch.testing.assert_close(loss, expected_loss)

    def test_constructed_policy_samples_action_chunks(self) -> None:
        policy = FlowMatchingPolicy(3, 2, 4, hidden_dims=(5,))
        state = torch.randn(2, 3)

        try:
            action_chunk = policy.sample_actions(state, num_steps=2)
        except AttributeError:
            self.fail("FlowMatchingPolicy does not build its velocity network")

        self.assertEqual(action_chunk.shape, (2, 4, 2))

    def test_sampling_integrates_the_learned_velocity(self) -> None:
        policy = FlowMatchingPolicy(3, 2, 4, hidden_dims=())
        linear = torch.nn.Linear(3 + 4 * 2 + 1, 4 * 2)
        policy.net = linear
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.fill_(2.0)

        state = torch.randn(2, 3)
        torch.manual_seed(0)
        initial_noise = torch.randn(2, 4, 2)
        torch.manual_seed(0)
        try:
            action_chunk = policy.sample_actions(state, num_steps=4)
        except NotImplementedError:
            self.fail("FlowMatchingPolicy.sample_actions is not implemented")

        torch.testing.assert_close(action_chunk, initial_noise + 2.0)


if __name__ == "__main__":
    unittest.main()
