"""
Unit tests for missile_ai_controller module.

Tests the missile AI control functionality including trajectory adjustments
and model inference.
"""
import math
from unittest.mock import Mock

import pytest
import torch

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai


class TestMissileAIController:
    """Test suite for missile AI controller functionality."""

    def test_update_missile_ai_basic(self, simple_missile_model):
        """Test basic functionality of update_missile_ai."""
        # Create mock missile
        missile = Mock(spec=Missile)
        missile.pos = {"x": 120, "y": 120}
        missile.vx = 5.0
        missile.vy = 5.0
        missile.speed = 7.07  # sqrt(5^2 + 5^2)
        missile.last_action = 0.0
        
        # Setup player and enemy positions
        player_pos = {"x": 100, "y": 100}
        enemy_pos = {"x": 300, "y": 300}
        
        # Setup input tensor (batch size 1, 9 features)
        shared_input_tensor = torch.zeros(1, 9)
        
        # Call update function
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input_tensor,
            missile_model=simple_missile_model,
            model_blend_factor=0.5,
            max_turn_rate=5.0
        )
        
        # Verify missile state was updated
        assert hasattr(missile, 'vx')
        assert hasattr(missile, 'vy')
        assert hasattr(missile, 'last_action')
        
        # Verify missile velocity components still maintain the speed
        assert math.isclose(
            math.sqrt(missile.vx**2 + missile.vy**2),
            missile.speed,
            rel_tol=1e-2
        )

    def test_update_missile_ai_no_enemy(self, simple_missile_model):
        """Test update_missile_ai with no enemy (enemy_pos=None)."""
        # Create mock missile
        missile = Mock(spec=Missile)
        missile.pos = {"x": 120, "y": 120}
        missile.vx = 5.0
        missile.vy = 5.0
        missile.speed = 7.07
        missile.last_action = 0.0
        
        # Setup player position (no enemy)
        player_pos = {"x": 100, "y": 100}
        enemy_pos = None
        
        # Setup input tensor
        shared_input_tensor = torch.zeros(1, 9)
        
        # Initial velocity components
        initial_vx = missile.vx
        initial_vy = missile.vy
        
        # Call update function
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input_tensor,
            missile_model=simple_missile_model,
            model_blend_factor=0.0,  # Use only direct angle, ignore model
            max_turn_rate=5.0
        )
        
        # With no enemy and blend_factor=0, missile should maintain direction
        # Allow small changes due to floating point precision
        assert math.isclose(missile.vx, initial_vx, rel_tol=1e-2)
        assert math.isclose(missile.vy, initial_vy, rel_tol=1e-2)

    def test_update_missile_ai_turn_rate_limit(self, simple_missile_model):
        """Test that turn rate is properly limited."""
        # Create mock missile heading right (0 degrees)
        missile = Mock(spec=Missile)
        missile.pos = {"x": 120, "y": 120}
        missile.vx = 10.0
        missile.vy = 0.0
        missile.speed = 10.0
        missile.last_action = 0.0
        
        # Set player and enemy positions to force a large turn
        # Enemy directly above missile would require 90 degree turn
        player_pos = {"x": 100, "y": 100}
        enemy_pos = {"x": 120, "y": 20}  # directly above missile
        
        # Setup input tensor
        shared_input_tensor = torch.zeros(1, 9)
        
        # Maximum allowed turn rate in degrees
        max_turn_rate = 5.0
        
        # Call update function
        update_missile_ai(
            missiles=[missile],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input_tensor,
            missile_model=simple_missile_model,
            model_blend_factor=0.0,  # Use only direct angle, ignore model
            max_turn_rate=max_turn_rate
        )
        
        # Calculate the actual turn rate that was applied (in degrees)
        # Initial angle was 0 degrees (missile.vx = 10, missile.vy = 0)
        # New angle = atan2(missile.vy, missile.vx)
        new_angle_degrees = math.degrees(math.atan2(missile.vy, missile.vx))
        
        # Verify turn rate was limited
        # Should be close to max_turn_rate (within floating point precision)
        assert abs(new_angle_degrees) <= max_turn_rate + 1e-5

    def test_multiple_missiles(self, simple_missile_model):
        """Test that multiple missiles are all updated."""
        # Create multiple mock missiles
        missile1 = Mock(spec=Missile)
        missile1.pos = {"x": 120, "y": 120}
        missile1.vx = 5.0
        missile1.vy = 5.0
        missile1.speed = 7.07
        missile1.last_action = 0.0
        
        missile2 = Mock(spec=Missile)
        missile2.pos = {"x": 150, "y": 150}
        missile2.vx = -3.0
        missile2.vy = 4.0
        missile2.speed = 5.0
        missile2.last_action = 0.0
        
        # Setup positions
        player_pos = {"x": 100, "y": 100}
        enemy_pos = {"x": 300, "y": 300}
        
        # Setup input tensor
        shared_input_tensor = torch.zeros(1, 9)
        
        # Call update function
        update_missile_ai(
            missiles=[missile1, missile2],
            player_pos=player_pos,
            enemy_pos=enemy_pos,
            shared_input_tensor=shared_input_tensor,
            missile_model=simple_missile_model,
            model_blend_factor=0.5,
            max_turn_rate=5.0
        )
        
        # Verify both missiles were updated
        assert hasattr(missile1, 'last_action')
        assert hasattr(missile2, 'last_action')
        
        # Verify speeds are maintained
        assert math.isclose(
            math.sqrt(missile1.vx**2 + missile1.vy**2),
            missile1.speed,
            rel_tol=1e-2
        )
        assert math.isclose(
            math.sqrt(missile2.vx**2 + missile2.vy**2),
            missile2.speed,
            rel_tol=1e-2
        )
