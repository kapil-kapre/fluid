#if __has_include(<glad/glad.h>)
#define FLUID_GLAD_V1 1
#include <glad/glad.h>
#elif __has_include(<glad/gl.h>)
#define FLUID_GLAD_V2 1
#include <glad/gl.h>
#else
#error "GLAD header not found. Install GLAD and add include paths for glad/glad.h or glad/gl.h."
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int WINDOW_W = 1280;
static constexpr int WINDOW_H = 720;

static constexpr int NX = 16;
static constexpr int NY = 16;
static constexpr int NZ = 16;
static constexpr float DT = 1.0f / 60.0f;
static constexpr float H = 1.0f / float(NX); // cube is [0,1]^3
static constexpr float GRAVITY = -9.8f;
static constexpr int PRESSURE_ITERS = 40;
static constexpr int PARTICLES_PER_CELL_AXIS = 2;
static constexpr float WALL_DAMPING = -0.4f;

struct Vec3 {
    float x{}, y{}, z{};

    Vec3() = default;
    Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

    Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

static inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

static inline Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return (len > 1e-8f) ? v / len : Vec3{};
}

struct Mat4 {
    float m[16]{};

    static Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
};

static Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c) {
        for (int r0 = 0; r0 < 4; ++r0) {
            r.m[c * 4 + r0] =
                a.m[0 * 4 + r0] * b.m[c * 4 + 0] +
                a.m[1 * 4 + r0] * b.m[c * 4 + 1] +
                a.m[2 * 4 + r0] * b.m[c * 4 + 2] +
                a.m[3 * 4 + r0] * b.m[c * 4 + 3];
        }
    }
    return r;
}

static Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
    Mat4 r{};
    float f = 1.0f / std::tan(fovy * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 r = Mat4::identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 rotateY(float a) {
    Mat4 r = Mat4::identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;   r.m[8] = s;
    r.m[2] = -s;  r.m[10] = c;
    return r;
}

static Mat4 rotateX(float a) {
    Mat4 r = Mat4::identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[5] = c;   r.m[9] = -s;
    r.m[6] = s;   r.m[10] = c;
    return r;
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(size_t(len), '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        throw std::runtime_error(log);
    }
    return s;
}

static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(size_t(len), '\0');
        glGetProgramInfoLog(p, len, nullptr, log.data());
        throw std::runtime_error(log);
    }
    return p;
}

enum class CellType : uint8_t { Air = 0, Fluid = 1 };

struct Particle {
    Vec3 p;
};

struct MACGrid {
    // Face velocities
    std::vector<float> u; // (NX+1)*NY*NZ
    std::vector<float> v; // NX*(NY+1)*NZ
    std::vector<float> w; // NX*NY*(NZ+1)

    std::vector<float> pressure;
    std::vector<float> div;
    std::vector<CellType> cellType;

    MACGrid() {
        u.assign((NX + 1) * NY * NZ, 0.0f);
        v.assign(NX * (NY + 1) * NZ, 0.0f);
        w.assign(NX * NY * (NZ + 1), 0.0f);
        pressure.assign(NX * NY * NZ, 0.0f);
        div.assign(NX * NY * NZ, 0.0f);
        cellType.assign(NX * NY * NZ, CellType::Air);
    }

    static int idxCell(int i, int j, int k) {
        return (k * NY + j) * NX + i;
    }

    static int idxU(int i, int j, int k) {
        return (k * NY + j) * (NX + 1) + i;
    }

    static int idxV(int i, int j, int k) {
        return (k * (NY + 1) + j) * NX + i;
    }

    static int idxW(int i, int j, int k) {
        return (k * NY + j) * NX + i;
    }

    bool inCell(int i, int j, int k) const {
        return i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ;
    }

    float& P(int i, int j, int k) { return pressure[idxCell(i,j,k)]; }
    float& D(int i, int j, int k) { return div[idxCell(i,j,k)]; }
    CellType& T(int i, int j, int k) { return cellType[idxCell(i,j,k)]; }

    float& U(int i, int j, int k) { return u[idxU(i,j,k)]; }
    float& V(int i, int j, int k) { return v[idxV(i,j,k)]; }
    float& W(int i, int j, int k) { return w[idxW(i,j,k)]; }

    float P(int i, int j, int k) const { return pressure[idxCell(i,j,k)]; }
    float D(int i, int j, int k) const { return div[idxCell(i,j,k)]; }
    CellType T(int i, int j, int k) const { return cellType[idxCell(i,j,k)]; }

    float U(int i, int j, int k) const { return u[idxU(i,j,k)]; }
    float V(int i, int j, int k) const { return v[idxV(i,j,k)]; }
    float W(int i, int j, int k) const { return w[idxW(i,j,k)]; }
};

static std::vector<Particle> g_particles;
static MACGrid g_grid;
static std::mt19937 g_rng(1337);

static float clampf(float x, float a, float b) {
    return std::max(a, std::min(b, x));
}

static Vec3 cubeToWorld(const Vec3& p) {
    // center cube around origin, scale to [-0.5,0.5]^3
    return Vec3(p.x - 0.5f, p.y - 0.5f, p.z - 0.5f);
}

static float trilerpScalar(const std::vector<float>& a, int sx, int sy, int sz, float gx, float gy, float gz) {
    gx = clampf(gx, 0.0f, float(sx - 1) - 1e-4f);
    gy = clampf(gy, 0.0f, float(sy - 1) - 1e-4f);
    gz = clampf(gz, 0.0f, float(sz - 1) - 1e-4f);

    int i0 = int(std::floor(gx)), j0 = int(std::floor(gy)), k0 = int(std::floor(gz));
    int i1 = std::min(i0 + 1, sx - 1);
    int j1 = std::min(j0 + 1, sy - 1);
    int k1 = std::min(k0 + 1, sz - 1);

    float fx = gx - float(i0);
    float fy = gy - float(j0);
    float fz = gz - float(k0);

    auto idx = [sx, sy](int i, int j, int k) {
        return (k * sy + j) * sx + i;
    };

    auto lerp = [](float a0, float a1, float t) { return a0 + (a1 - a0) * t; };

    float c000 = a[idx(i0,j0,k0)];
    float c100 = a[idx(i1,j0,k0)];
    float c010 = a[idx(i0,j1,k0)];
    float c110 = a[idx(i1,j1,k0)];
    float c001 = a[idx(i0,j0,k1)];
    float c101 = a[idx(i1,j0,k1)];
    float c011 = a[idx(i0,j1,k1)];
    float c111 = a[idx(i1,j1,k1)];

    float c00 = lerp(c000, c100, fx);
    float c10 = lerp(c010, c110, fx);
    float c01 = lerp(c001, c101, fx);
    float c11 = lerp(c011, c111, fx);

    float c0 = lerp(c00, c10, fy);
    float c1 = lerp(c01, c11, fy);

    return lerp(c0, c1, fz);
}

static Vec3 sampleVelocity(const MACGrid& g, const Vec3& p) {
    // staggered sampling positions:
    // u at x-faces: (i*h, (j+0.5)h, (k+0.5)h)
    // v at y-faces: ((i+0.5)h, j*h, (k+0.5)h)
    // w at z-faces: ((i+0.5)h, (j+0.5)h, k*h)
    float ux = trilerpScalar(g.u, NX + 1, NY, NZ, p.x / H, p.y / H - 0.5f, p.z / H - 0.5f);
    float vy = trilerpScalar(g.v, NX, NY + 1, NZ, p.x / H - 0.5f, p.y / H, p.z / H - 0.5f);
    float wz = trilerpScalar(g.w, NX, NY, NZ + 1, p.x / H - 0.5f, p.y / H - 0.5f, p.z / H);
    return {ux, vy, wz};
}

static void zeroBoundaryVelocities(MACGrid& g) {
    for (int j = 0; j < NY; ++j)
    for (int k = 0; k < NZ; ++k) {
        g.U(0,j,k) = 0.0f;
        g.U(NX,j,k) = 0.0f;
    }

    for (int i = 0; i < NX; ++i)
    for (int k = 0; k < NZ; ++k) {
        g.V(i,0,k) = 0.0f;
        g.V(i,NY,k) = 0.0f;
    }

    for (int i = 0; i < NX; ++i)
    for (int j = 0; j < NY; ++j) {
        g.W(i,j,0) = 0.0f;
        g.W(i,j,NZ) = 0.0f;
    }
}

static void classifyCells(MACGrid& g, const std::vector<Particle>& particles) {
    std::fill(g.cellType.begin(), g.cellType.end(), CellType::Air);
    for (const auto& pt : particles) {
        int i = clampf(int(pt.p.x / H), 0, NX - 1);
        int j = clampf(int(pt.p.y / H), 0, NY - 1);
        int k = clampf(int(pt.p.z / H), 0, NZ - 1);
        g.T(i,j,k) = CellType::Fluid;
    }
}

static void seedInitialFluid(std::vector<Particle>& particles) {
    particles.clear();
    std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);

    for (int k = 2; k < NZ - 2; ++k) {
        for (int j = 2; j < NY / 2; ++j) {
            for (int i = 2; i < NX - 2; ++i) {
                for (int pk = 0; pk < PARTICLES_PER_CELL_AXIS; ++pk)
                for (int pj = 0; pj < PARTICLES_PER_CELL_AXIS; ++pj)
                for (int pi = 0; pi < PARTICLES_PER_CELL_AXIS; ++pi) {
                    Vec3 p{
                        (float(i) + (pi + 0.5f) / PARTICLES_PER_CELL_AXIS + 0.15f * jitter(g_rng)) * H,
                        (float(j) + (pj + 0.5f) / PARTICLES_PER_CELL_AXIS + 0.15f * jitter(g_rng)) * H,
                        (float(k) + (pk + 0.5f) / PARTICLES_PER_CELL_AXIS + 0.15f * jitter(g_rng)) * H
                    };
                    particles.push_back({p});
                }
            }
        }
    }
}

static void initializeVelocityField(MACGrid& g) {
    std::fill(g.u.begin(), g.u.end(), 0.0f);
    std::fill(g.v.begin(), g.v.end(), 0.0f);
    std::fill(g.w.begin(), g.w.end(), 0.0f);

    // small initial swirl
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY + 1; ++j) {
            for (int i = 0; i < NX; ++i) {
                float x = (i + 0.5f) * H - 0.5f;
                float z = (k + 0.5f) * H - 0.5f;
                g.V(i,j,k) = 0.0f;
                if (j > 0 && j < NY) {
                    // nothing here initially
                }
                if (k < NZ && i < NX) {
                    // set a mild rotating motion around vertical axis
                }
            }
        }
    }

    for (int k = 0; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX + 1; ++i) {
        float z = (k + 0.5f) * H - 0.5f;
        g.U(i,j,k) = -1.5f * z;
    }

    for (int k = 0; k < NZ + 1; ++k)
    for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
        float x = (i + 0.5f) * H - 0.5f;
        g.W(i,j,k) = 1.5f * x;
    }

    zeroBoundaryVelocities(g);
}

static void applyGravity(MACGrid& g, float dt) {
    for (int k = 0; k < NZ; ++k)
    for (int j = 1; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
        g.V(i,j,k) += GRAVITY * dt;
    }
}

static void computeDivergence(MACGrid& g) {
    for (int k = 0; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
        if (g.T(i,j,k) != CellType::Fluid) {
            g.D(i,j,k) = 0.0f;
            continue;
        }

        float div =
            (g.U(i+1,j,k) - g.U(i,j,k)) +
            (g.V(i,j+1,k) - g.V(i,j,k)) +
            (g.W(i,j,k+1) - g.W(i,j,k));
        g.D(i,j,k) = div / H;
    }
}

static void solvePressure(MACGrid& g, float dt) {
    std::fill(g.pressure.begin(), g.pressure.end(), 0.0f);
    std::vector<float> nextP = g.pressure;

    for (int iter = 0; iter < PRESSURE_ITERS; ++iter) {
        for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            if (g.T(i,j,k) != CellType::Fluid) {
                nextP[MACGrid::idxCell(i,j,k)] = 0.0f;
                continue;
            }

            float sum = 0.0f;
            int count = 0;

            auto accum = [&](int ni, int nj, int nk) {
                if (!g.inCell(ni,nj,nk)) return;
                sum += g.P(ni,nj,nk);
                count++;
            };

            accum(i-1,j,k);
            accum(i+1,j,k);
            accum(i,j-1,k);
            accum(i,j+1,k);
            accum(i,j,k-1);
            accum(i,j,k+1);

            if (count > 0) {
                nextP[MACGrid::idxCell(i,j,k)] = (sum - (H * H / dt) * g.D(i,j,k)) / float(count);
            } else {
                nextP[MACGrid::idxCell(i,j,k)] = 0.0f;
            }
        }
        g.pressure.swap(nextP);
    }
}

static void applyPressure(MACGrid& g, float dt) {
    // u faces
    for (int k = 0; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
    for (int i = 1; i < NX; ++i) {
        float pR = g.P(i,j,k);
        float pL = g.P(i-1,j,k);
        bool fluidR = (g.T(i,j,k) == CellType::Fluid);
        bool fluidL = (g.T(i-1,j,k) == CellType::Fluid);
        if (fluidR || fluidL) {
            g.U(i,j,k) -= dt * (pR - pL) / H;
        }
    }

    // v faces
    for (int k = 0; k < NZ; ++k)
    for (int j = 1; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
        float pT = g.P(i,j,k);
        float pB = g.P(i,j-1,k);
        bool fluidT = (g.T(i,j,k) == CellType::Fluid);
        bool fluidB = (g.T(i,j-1,k) == CellType::Fluid);
        if (fluidT || fluidB) {
            g.V(i,j,k) -= dt * (pT - pB) / H;
        }
    }

    // w faces
    for (int k = 1; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
        float pF = g.P(i,j,k);
        float pB = g.P(i,j,k-1);
        bool fluidF = (g.T(i,j,k) == CellType::Fluid);
        bool fluidB = (g.T(i,j,k-1) == CellType::Fluid);
        if (fluidF || fluidB) {
            g.W(i,j,k) -= dt * (pF - pB) / H;
        }
    }

    zeroBoundaryVelocities(g);
}

static void advectParticlesRK2(const MACGrid& g, std::vector<Particle>& particles, float dt) {
    for (auto& pt : particles) {
        Vec3 v1 = sampleVelocity(g, pt.p);
        Vec3 mid = pt.p + v1 * (0.5f * dt);
        mid.x = clampf(mid.x, 0.001f, 0.999f);
        mid.y = clampf(mid.y, 0.001f, 0.999f);
        mid.z = clampf(mid.z, 0.001f, 0.999f);

        Vec3 v2 = sampleVelocity(g, mid);
        pt.p += v2 * dt;

        // simple box collision
        if (pt.p.x < 0.001f) pt.p.x = 0.001f;
        if (pt.p.y < 0.001f) pt.p.y = 0.001f;
        if (pt.p.z < 0.001f) pt.p.z = 0.001f;

        if (pt.p.x > 0.999f) pt.p.x = 0.999f;
        if (pt.p.y > 0.999f) pt.p.y = 0.999f;
        if (pt.p.z > 0.999f) pt.p.z = 0.999f;
    }
}

static void simulationStep(float dt) {
    classifyCells(g_grid, g_particles);
    applyGravity(g_grid, dt);
    computeDivergence(g_grid);
    solvePressure(g_grid, dt);
    applyPressure(g_grid, dt);
    advectParticlesRK2(g_grid, g_particles, dt);
    classifyCells(g_grid, g_particles);
}

static GLuint cubeVAO = 0, cubeVBO = 0;
static GLuint particleVAO = 0, particleVBO = 0;
static GLuint lineProgram = 0, particleProgram = 0;

static void initCubeGeometry() {
    // 12 lines
    const float cubeLines[] = {
        -0.5f,-0.5f,-0.5f,  0.5f,-0.5f,-0.5f,
         0.5f,-0.5f,-0.5f,  0.5f, 0.5f,-0.5f,
         0.5f, 0.5f,-0.5f, -0.5f, 0.5f,-0.5f,
        -0.5f, 0.5f,-0.5f, -0.5f,-0.5f,-0.5f,

        -0.5f,-0.5f, 0.5f,  0.5f,-0.5f, 0.5f,
         0.5f,-0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
         0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
        -0.5f, 0.5f, 0.5f, -0.5f,-0.5f, 0.5f,

        -0.5f,-0.5f,-0.5f, -0.5f,-0.5f, 0.5f,
         0.5f,-0.5f,-0.5f,  0.5f,-0.5f, 0.5f,
         0.5f, 0.5f,-0.5f,  0.5f, 0.5f, 0.5f,
        -0.5f, 0.5f,-0.5f, -0.5f, 0.5f, 0.5f,
    };

    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeLines), cubeLines, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, g_particles.size() * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

static void uploadParticles() {
    std::vector<float> pts;
    pts.reserve(g_particles.size() * 3);
    for (const auto& p : g_particles) {
        Vec3 q = cubeToWorld(p.p);
        pts.push_back(q.x);
        pts.push_back(q.y);
        pts.push_back(q.z);
    }
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float), pts.data(), GL_DYNAMIC_DRAW);
}

static void framebufferSizeCallback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static bool loadOpenGLFunctions() {
#if defined(FLUID_GLAD_V1)
    return gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0;
#elif defined(FLUID_GLAD_V2)
    return gladLoadGL((GLADloadfunc)glfwGetProcAddress) != 0;
#endif
}

int main() {
    if (!glfwInit()) return EXIT_FAILURE;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(WINDOW_W, WINDOW_H, "3D MAC Fluid Cube", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSwapInterval(1);

    if (!loadOpenGLFunctions()) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    const char* lineVS = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
    )";

    const char* lineFS = R"(
        #version 330 core
        out vec4 FragColor;
        void main() { FragColor = vec4(0.85, 0.88, 0.95, 1.0); }
    )";

    const char* particleVS = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            gl_PointSize = 5.0;
        }
    )";

    const char* particleFS = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            vec2 d = gl_PointCoord - vec2(0.5);
            if (dot(d,d) > 0.25) discard;
            FragColor = vec4(0.15, 0.55, 0.95, 0.9);
        }
    )";

    try {
        lineProgram = makeProgram(lineVS, lineFS);
        particleProgram = makeProgram(particleVS, particleFS);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    seedInitialFluid(g_particles);
    initializeVelocityField(g_grid);
    classifyCells(g_grid, g_particles);
    initCubeGeometry();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const GLint lineMVP = glGetUniformLocation(lineProgram, "uMVP");
    const GLint partMVP = glGetUniformLocation(particleProgram, "uMVP");

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        simulationStep(DT);
        uploadParticles();

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        float aspect = (h == 0) ? 1.0f : float(w) / float(h);

        float t = float(glfwGetTime());
        Mat4 proj = perspective(45.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);
        Mat4 view = translate(0.0f, 0.0f, -2.4f);
        Mat4 model = multiply(rotateY(0.35f * t), rotateX(-0.45f));
        Mat4 mvp = multiply(proj, multiply(view, model));

        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(lineProgram);
        glUniformMatrix4fv(lineMVP, 1, GL_FALSE, mvp.m);
        glBindVertexArray(cubeVAO);
        glDrawArrays(GL_LINES, 0, 24);

        glUseProgram(particleProgram);
        glUniformMatrix4fv(partMVP, 1, GL_FALSE, mvp.m);
        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)g_particles.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(lineProgram);
    glDeleteProgram(particleProgram);
    glDeleteBuffers(1, &cubeVBO);
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &particleVBO);
    glDeleteVertexArrays(1, &particleVAO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}