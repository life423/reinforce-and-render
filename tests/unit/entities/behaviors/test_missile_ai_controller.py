"""
Unit tests for the missile AI controller.

Tests the missile guidance logic including velocity adjustments,
target tracking, and blend factors between model predictions and direct targeting.
"""
import math
import pytest
import torch
from unittest.mock import Mock

from ai_platform_trainer.entities.components.missile import Missile
from ai_platform_trainer.entities.behaviors.missile_ai_controller import update_missile_ai


@pytest.fixture


def missile():
    """Return a standard missile instance for testing."""
    return Missile(
        x=100,
        y=200,
        speed=5.0,
        vx=5.0,
        vy=0.0,
        birth_time=1000,
        lifespan=2000
    )


@pytest.fixture


def player_pos():
    """Return standard player position for testing."""
    return {"x": 50, "y": 50}


@pytest.fixture


def enemy_pos():
    """Return standard enemy position for testing."""
    return {"x": 300, "y": 200}


@pytest.fixture


def mock_model():
    """Create a mock missile model for testing."""
    model = Mock()
    # Mock the model to return a turn rate value
    model.return_value = torch.tensor([2.0])  # 2 degree turn
    return model


class TestMissileAIController:
    """Tests for the missile AI controller."""

    def test_update_missile_ai_direct_targeting(self, missile, player_pos, enemy_pos, mock_model):
        """Test that missiles can track targets directly."""
        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Test with 0 model influence (pure direct targeting)
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=mock_model,
            model_blend_factor=0.0,  # No model influence
            max_turn_rate=5.0
        )

        # Get the angle from missile to enemy
        target_angle = math.atan2(
            enemy_pos["y"] - missile.pos["y"],
            enemy_pos["x"] - missile.pos["x"]
        )

        # Get the new missile direction
        new_angle = math.atan2(missile.vy, missile.vx)

        # The new angle should be closer to the target angle
        # We expect some turn, but limited by max_turn_rate
        # Calculate original angle difference
        original_angle = math.atan2(0.0, 5.0)  # From initial vx=5.0, vy=0.0

        initial_diff = abs(target_angle - original_angle)
        current_diff = abs(target_angle - new_angle)

        # The difference should be smaller (missile turning toward target)
        assert current_diff < initial_diff

        # The turn should respect max_turn_rate
        max_angle_change = math.radians(5.0)  # 5 degrees max turn
        angle_change = abs(new_angle - original_angle)
        assert angle_change <= max_angle_change + 1e-6  # Add small tolerance for floating point

    def test_update_missile_ai_model_influence(self, missile, player_pos, enemy_pos, mock_model):
        """Test that model predictions influence missile trajectory."""
        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Store original velocity
        original_vx, original_vy = missile.vx, missile.vy
        original_angle = math.atan2(original_vy, original_vx)

        # Update with 100% model influence
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=mock_model,
            model_blend_factor=1.0,  # Full model influence
            max_turn_rate=5.0
        )

        # Get the new missile direction
        new_angle = math.atan2(missile.vy, missile.vx)

        # The model returned 2.0 degrees turn
        expected_angle_change = math.radians(2.0)
        angle_change = new_angle - original_angle

        # Normalize angle difference to [-pi, pi]
        while angle_change > math.pi:
            angle_change -= 2 * math.pi
        while angle_change < -math.pi:
            angle_change += 2 * math.pi

        # Check that the angle changed by approximately the model's predicted turn rate
        assert math.isclose(angle_change, expected_angle_change, abs_tol=1e-4)

        # Make sure the model was called with the expected input
        mock_model.assert_called_once()
        # Verify input tensor was filled correctly (at least check shape)
        args, _ = mock_model.call_args
        assert args[0].shape == (1, 9)  # 9 features in input

    def test_update_missile_ai_blend_factors(self, missile, player_pos, enemy_pos):
        """Test that blending between model and direct targeting works."""
        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Create two identical missiles
        missile1 = Missile(x=100, y=200, vx=5.0, vy=0.0)
        missile2 = Missile(x=100, y=200, vx=5.0, vy=0.0)

        # Create two mock models with different turn predictions
        model1 = Mock()
        model1.return_value = torch.tensor([10.0])  # 10 degree turn

        model2 = Mock()
        model2.return_value = torch.tensor([10.0])  # Same 10 degree turn

        # Update with different blend factors
        update_missile_ai(
            missiles=[missile1],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=model1,
            model_blend_factor=0.2,  # 20% model influence
            max_turn_rate=15.0
        )

        update_missile_ai(
            missiles=[missile2],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=model2,
            model_blend_factor=0.8,  # 80% model influence
            max_turn_rate=15.0
        )

        # Get new angles
        angle1 = math.atan2(missile1.vy, missile1.vx)
        angle2 = math.atan2(missile2.vy, missile2.vx)

        # Missile2 should rely more on the model (10 degree turn)
        # Missile1 should rely more on direct targeting

        # If 0.8 blend gives closer to 10 degrees and 0.2 gives closer to direct angle,
        # then |angle2 - model_angle| should be smaller than |angle1 - model_angle|
        model_angle = math.atan2(0.0, 5.0) + math.radians(10.0)  # Original + 10 degree turn

        # Calculate differences
        diff1 = abs(model_angle - angle1)
        diff2 = abs(model_angle - angle2)

        # Missile2 should be closer to the model prediction
        assert diff2 < diff1

    def test_update_missile_ai_multiple_missiles(self, player_pos, enemy_pos, mock_model):
        """Test updating multiple missiles simultaneously."""
        # Create multiple missiles with different positions/velocities
        missiles = [
            Missile(x=100, y=100, vx=5.0, vy=0.0),
            Missile(x=200, y=200, vx=0.0, vy=5.0),
            Missile(x=300, y=100, vx=-5.0, vy=0.0)
        ]

        # Store original velocities
        original_velocities = [(m.vx, m.vy) for m in missiles]

        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Update all missiles
        update_missile_ai(
            missiles=missiles,
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=mock_model,
            model_blend_factor=0.5,  # Equal blending
            max_turn_rate=5.0
        )

        # Verify all missiles were updated
        for i, missile in enumerate(missiles):
            orig_vx, orig_vy = original_velocities[i]
            assert (missile.vx, missile.vy) != (orig_vx, orig_vy)

            # Speed should remain the same
            orig_speed = math.sqrt(orig_vx**2 + orig_vy**2)
            new_speed = math.sqrt(missile.vx**2 + missile.vy**2)
            assert math.isclose(orig_speed, new_speed, abs_tol=1e-6)

    def test_update_missile_ai_no_enemy(self, missile, player_pos, mock_model):
        """Test behavior when no enemy is present."""
        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Save original values
        orig_vx, orig_vy = missile.vx, missile.vy

        # Update with no enemy
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=None,  # No enemy
            shared_input_tensor=shared_input,
            missile_model=mock_model,
            model_blend_factor=0.5,
            max_turn_rate=5.0
        )

        # Missile should still move but in a way determined by the model
        # Ensure velocities changed in some way
        assert (missile.vx, missile.vy) != (orig_vx, orig_vy)

        # Speed should still be preserved
        orig_speed = math.sqrt(orig_vx**2 + orig_vy**2)
        new_speed = math.sqrt(missile.vx**2 + missile.vy**2)
        assert math.isclose(orig_speed, new_speed, abs_tol=1e-6)

        # Check model was still called
        mock_model.assert_called_once()

    def test_update_missile_ai_max_turn_rate(self, missile, player_pos, enemy_pos):
        """Test that max turn rate is respected."""
        # Setup a shared input tensor
        shared_input = torch.zeros((1, 9), dtype=torch.float32)

        # Create a mock model that wants to turn hard
        model = Mock()
        model.return_value = torch.tensor([30.0])  # 30 degree turn (exceeds max)

        # Original direction
        orig_angle = math.atan2(missile.vy, missile.vx)

        # Update with limited turn rate
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input,
            missile_model=model,
            model_blend_factor=1.0,  # Full model influence
            max_turn_rate=5.0  # Only allow 5 degrees
        )

        # New direction
        new_angle = math.atan2(missile.vy, missile.vx)

        # Calculate angle change in degrees
        angle_change_degrees = math.degrees(abs(new_angle - orig_angle))

        # Should respect the max turn rate
        assert angle_change_degrees <= 5.0 + 1e-6  # Allow small floating point error
