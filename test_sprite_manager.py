#!/usr/bin/env python
"""
Script to test sprite manager functionality and validate sprite loading.

Run this script to ensure all sprites in the assets directory are properly 
loaded or have appropriate placeholders.
"""
import logging

from ai_platform_trainer.utils.sprite_manager import SpriteManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Run the sprite validation test."""
    # Initialize pygame for sprite testing
    import pygame
    pygame.init()
    pygame.display.set_mode((800, 600))

    print("Testing SpriteManager functionality...")
    sprite_manager = SpriteManager()
    
    # Validate all critical sprites
    failures = sprite_manager.validate_sprites()
    
    if failures:
        print(f"\n❌ Found {len(failures)} missing or placeholder sprites:")
        for name in failures:
            print(f"  - {name}")
        print("\nFix these by ensuring the sprite files exist in the expected locations:")
        print("  - Check that file names match what the code expects")
        print("  - Verify the sprites are in the correct subdirectories")
        print("  - Fix any permissions issues that might prevent loading")
    else:
        print("\n✅ All sprites loaded successfully!")
    
    # Test specific obstacle sprites (wall, rock)
    print("\nTesting obstacle sprite variants...")
    test_size = (50, 50)
    
    for name in ["wall", "rock", "obstacle"]:
        sprite = sprite_manager.load_sprite(name, test_size)
        is_ph = sprite_manager.is_placeholder(sprite)
        result = "✅ OK" if not is_ph else "❌ PLACEHOLDER"
        print(f"  - {name}: {result}")
    
    # Test orientation variants (wall_h, wall_v)
    print("\nTesting orientation variants...")
    for variant in ["wall_h", "wall_v", "rock_h", "rock_v"]:
        sprite = sprite_manager.load_sprite(variant, test_size)
        is_ph = sprite_manager.is_placeholder(sprite)
        
        # Consider success if either:
        # 1. A actual sprite was loaded (not a placeholder)
        # 2. We got a placeholder but base sprite exists (expected fallback)
        if not is_ph:
            result = "✅ OK - Direct variant sprite loaded"
        else:
            # Check if base exists (e.g., wall.png for wall_h)
            base_name = variant.split("_")[0]
            base_sprite = sprite_manager.load_sprite(base_name, test_size)
            base_is_ph = sprite_manager.is_placeholder(base_sprite)
            
            if not base_is_ph:
                result = "✅ OK - Fallback to base sprite"
            else:
                result = "❌ PLACEHOLDER - No variant or base sprite found"
        
        print(f"  - {variant}: {result}")
    
    # Test animation loading
    print("\nTesting animation loading...")
    explosion_frames = sprite_manager.load_animation("explosion", (50, 50), 4)
    print(f"  - Loaded {len(explosion_frames)} explosion animation frames")
    
    print("\nSprite validation complete!")


if __name__ == "__main__":
    main()
