ENGLISH | [简体中文](README_CN.md)

# 2D Cylinder Flow

## Overview

Flow past cylinder problem is a two-dimensional low velocity steady flow around a cylinder which is only related to the `Re` number. When `Re` is less than or equal to 1, the inertial force in the flow field is secondary to the viscous force, the streamlines in the upstream and downstream directions of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to `Re` . The flow around this `Re` number range is called the Stokes zone; With the increase of `Re` , the streamlines in the upstream and downstream of the cylinder gradually lose symmetry. This special phenomenon reflects the peculiar nature of the interaction between the fluid and the surface of the body. Solving flow past a cylinder is a classical problem in hydromechanics. This case uses PINNs to solve the wake flow field around a cylinder.

![cylinder flow](images/cylinder_flow.gif)

![flow](images/image-flow.png)

![Time Error](images/TimeError_epoch5000.png)

[See More](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/physics_driven/cylinder_flow/navier_stokes2D.ipynb)

## Contributor

liulei277