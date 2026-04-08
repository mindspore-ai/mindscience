# Obstacle Handling

Obstacles define boundary conditions inside the simulation domain.

## Stationary Obstacles

```python
obstacle = Obstacle(Sphere(center=(16, 20), radius=5))
velocity, pressure = fluid.make_incompressible(velocity, obstacle)
```

## Moving Obstacles

```python
obstacle = Obstacle(
    Sphere(center=(16, 20), radius=5),
    velocity=(0.5, 0)  # Linear velocity
)
```

## Rotating Obstacles

```python
obstacle = Obstacle(
    Sphere(center=(16, 20), radius=5),
    angular_velocity=1.0  # Rotation speed
)
```

## Complex Geometries

```python
# Box obstacle
box_obstacle = Obstacle(Box(x=(10, 20), y=(10, 20)))

# Multiple obstacles
obstacles = [
    Obstacle(Sphere(center=(10, 10), radius=3)),
    Obstacle(Box(x=(20, 30), y=(15, 25)))
]
velocity, pressure = fluid.make_incompressible(velocity, obstacles)
```

## Obstacle Properties

```python
obstacle.is_stationary  # True if not moving or rotating
obstacle.is_moving      # True if has linear velocity
obstacle.is_rotating    # True if has angular velocity
```

## Transforming Obstacles

```python
# Shift position
shifted = obstacle.shifted((5, 0))

# Rotate
rotated = obstacle.rotated(math.pi/4)

# Move to specific position
at_pos = obstacle.at((20, 15))
```

## Boundary Conditions

Obstacles automatically enforce no-slip boundary conditions:
- Velocity inside obstacle matches obstacle velocity
- Pressure solver respects obstacle geometry
- Compatible with 2nd order schemes only

## Performance Notes

Obstacles add computational overhead. Consider:
- Using simple geometries (Sphere, Box) when possible
- Merging nearby obstacles
- Using 2nd order schemes (higher-order not supported)
