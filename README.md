# fluid

A small C++17 real-time 3D fluid simulation demo rendered with OpenGL.

The project simulates fluid particles inside a cube using a MAC grid, pressure solve, gravity, and particle advection, then visualizes the result as blue particles inside a rotating wireframe box.

## Features

- 3D particle-based fluid visualization
- MAC grid velocity storage
- Gravity and pressure projection step
- OpenGL rendering with `glad` and `GLFW`
- Simple interactive camera presentation with cube rotation toggle

## Controls

- `R`: toggle cube rotation
- `Esc`: close the application

## Requirements

- CMake 3.20+
- A C++17 compiler
- OpenGL 3.3 compatible system
- `glad`
- `glfw3`
- vcpkg toolchain installed at `C:/vcpkg/scripts/buildsystems/vcpkg.cmake`

## Install Dependencies

This project expects `glad` and `glfw3` to be available through vcpkg.

Example:

```powershell
vcpkg install glad glfw3
```

If your vcpkg installation is not located at `C:/vcpkg`, update the `CMAKE_TOOLCHAIN_FILE` path in [CMakeLists.txt](/d:/Kapil/cod/GitHub/fluid/CMakeLists.txt).

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

From the repo root:

```powershell
.\build\Release\fluid.exe
```

Depending on your generator, the executable may also be placed directly under `build`.

## Project Structure

- [main.cpp](/d:/Kapil/cod/GitHub/fluid/main.cpp): simulation and rendering code
- [CMakeLists.txt](/d:/Kapil/cod/GitHub/fluid/CMakeLists.txt): build configuration

## How It Works

Each frame the app:

1. Classifies grid cells containing fluid particles.
2. Applies gravity to the velocity field.
3. Computes grid divergence.
4. Solves pressure iteratively to reduce compression.
5. Applies the pressure gradient to the velocity field.
6. Advects particles through the grid using RK2 integration.
7. Uploads particle positions to the GPU and renders them.

## Notes

- The simulation grid is currently fixed at `16 x 16 x 16`.
- The fluid domain is a unit cube.
- Rendering uses point sprites for particles and line segments for the container.
- The implementation is intentionally compact and lives in a single source file, making it a good starting point for experimentation.
