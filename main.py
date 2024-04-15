from os import times
from random import random
from re import L
from turtle import pos
from xmlrpc.client import Boolean
import taichi as ti
import taichi.math as tm

ti.init(arch = ti.cuda, device_memory_GB = 5, cpu_max_num_threads = 1)

# for rigid body simulation (this shall be also represented by particles)
# this is actually a cube
radius = 0.8
length = 8
rigidV = ti.field(dtype = tm.vec3, shape = ()) # rigid velocity
rigidP = ti.field(dtype = tm.vec3, shape = ()) # rigid angular motion
rigidF = ti.field(dtype = tm.vec3, shape = ()) # rigid force
rigidT = ti.field(dtype = tm.vec3, shape = ()) # rigid torque (moment?)
deltaV = ti.field(dtype = tm.vec3, shape = ())
deltaP = ti.field(dtype = tm.vec3, shape = ())
rigidPosition = ti.field(dtype = tm.vec3, shape = ())
rigidParticles = ti.field(dtype = tm.vec3, shape = (length, length, length))
rigidGroupedParticles = ti.field(dtype = tm.vec3, shape = (length ** 3, ))
rigidGroupedColor = ti.field(dtype = tm.vec3, shape = (length ** 3, ))
rigidMass = 15
rigidDensity = 0.7
rigidTotalMass = rigidMass * length ** 3
rigidJ = rigidTotalMass * ((length - 1) * radius) ** 2 / 6

# for fluid simulation
a, b, c = 50, 50, 155
na, nb, nc = 27, 27, 27

m = 1
density0 = 0.21
bounce = 0.8
rigidBounce = 0.5

viscosity = 1
h = radius * 4
h6 = h ** 6
h9 = h ** 9

timeStep = 0.000015
stepCnt = 120
shift = 1e-2
eps = 1e-7

B = 60.0
gamma = 7.0
K_poly6 = m * 315 / (64 * tm.pi * h9)
K_spiky = m * 45 / (tm.pi * h6)

gravity = tm.vec3(0, 0, -100)
velocity = ti.field(dtype = tm.vec3, shape = (na, nb, nc))
tmpVelocity = ti.field(dtype = tm.vec3, shape = (na, nb, nc))
position = ti.field(dtype = tm.vec3, shape = (na, nb, nc))
groupedPosition = ti.field(dtype = tm.vec3, shape = (na * nb * nc, ))
groupedColor = ti.field(dtype = tm.vec3, shape = (na * nb * nc, ))
density = ti.field(dtype = ti.f32, shape = (na, nb, nc))

vca, vcb, vcc = int((a - radius) // h + 1), int((b - radius) // h + 1), int((c - radius) // h + 1)
vcd = 400
voxel = ti.field(dtype = tm.ivec3, shape = (vca, vcb, vcc, vcd))
voxelCnt = ti.field(dtype = ti.i32, shape = (vca, vcb, vcc))

normalList = ti.field(dtype = tm.vec3, shape = (6, ))
coordinateList = ti.field(dtype = ti.i32, shape = (6, ))
boundaryList = ti.field(dtype = ti.f32, shape = (6, ))

# for coupling
rigidVoxel = ti.field(dtype = tm.ivec3, shape = (vca, vcb, vcc, vcd))
rigidVoxelCnt = ti.field(dtype = ti.i32, shape = (vca, vcb, vcc))
rigidVol = ti.field(dtype = ti.f32, shape = (length, length, length))
couplingViscosity = 0.0001

currentTime = ti.field(dtype = ti.f32, shape = ())

@ti.func
def getVoxel(pos: tm.vec3) -> tm.ivec3:
    return tm.ivec3(pos.x // h, pos.y // h, pos.z // h)

@ti.func
def voxelAdd(idx: tm.ivec3, pos: tm.ivec3, type: ti.i32):
    if type == 0: # fluid
        voxel[pos.x, pos.y, pos.z, voxelCnt[pos.x, pos.y, pos.z]] = idx
        voxelCnt[pos.x, pos.y, pos.z] += 1
    elif type == 1:
        rigidVoxel[pos.x, pos.y, pos.z, rigidVoxelCnt[pos.x, pos.y, pos.z]] = idx
        rigidVoxelCnt[pos.x, pos.y, pos.z] += 1

@ti.func
def setVoxel():
    for i, j, k in ti.ndrange(vca, vcb, vcc):
        voxelCnt[i, j, k] = 0
        rigidVoxelCnt[i, j, k] = 0
    ti.loop_config(serialize = True)
    for i, j, k in ti.ndrange(na, nb, nc):
        voxelPos = getVoxel(position[i, j, k])
        voxelAdd(tm.ivec3(i, j, k), voxelPos, 0)
    ti.loop_config(serialize = True)
    for i, j, k in ti.ndrange(length, length, length):
        if i == 0 or i == length - 1 or j == 0 or j == length - 1 or k == 0 or k == length - 1:
            voxelPos = getVoxel(rigidParticles[i, j, k])
            voxelAdd(tm.ivec3(i, j, k), voxelPos, 1)

@ti.func
def densityKernel(w: ti.f32) -> ti.f32:
    return (h * h - w * w) ** 3

@ti.func
def volumeCorrection(idx: tm.ivec3, rho0: ti.f32) -> ti.f32:
    return rigidVol[idx.x, idx.y, idx.z] * rho0

@ti.func
def pressureKernel(p: tm.ivec3) -> ti.f32:
    return pow(max(density[p.x, p.y, p.z] / density0, 1), gamma) - 1

@ti.func
def pressureGradientKernel(p: tm.ivec3, q: tm.ivec3) -> tm.vec3:
    pos1, pos2 = position[p.x, p.y, p.z], position[q.x, q.y, q.z]
    den1, den2 = density[p.x, p.y, p.z], density[q.x, q.y, q.z]
    p1, p2 = pressureKernel(p), pressureKernel(q)
    r = pos2 - pos1
    rn = r.norm()
    return ((p1 + p2) / (2 * den1 * den2)) * ((h - rn) ** 2) * r / rn

@ti.func
def couplingPressureKernel(p: tm.ivec3, q: tm.ivec3) -> tm.vec3:
    posf, posr = position[p.x, p.y, p.z], rigidParticles[q.x, q.y, q.z]
    den = density[p.x, p.y, p.z]
    p1 = pressureKernel(p)
    r = posr - posf
    rn = r.norm()
    res = volumeCorrection(q, density0) * (p1 / (den * den)) * ((h - rn) ** 2) * r / rn
    return volumeCorrection(q, density0) * (p1 / (den * den)) * ((h - rn) ** 2) * r / rn

@ti.func
def viscosityKernel(p: tm.ivec3, q: tm.ivec3) -> tm.vec3:
    pos1, pos2 = position[p.x, p.y, p.z], position[q.x, q.y, q.z]
    den1, den2 = density[p.x, p.y, p.z], density[q.x, q.y, q.z]
    v1, v2 = velocity[p.x, p.y, p.z], velocity[q.x, q.y, q.z]
    r = pos2 - pos1
    rn = r.norm()
    return ((h - rn) / (den1 * den2)) * (v2 - v1)

@ti.func
def couplingViscosityKernel(p: tm.ivec3, q: tm.ivec3) -> tm.vec3:
    omega = rigidP[None] / rigidJ
    posf, posr = position[p.x, p.y, p.z], rigidParticles[q.x, q.y, q.z]
    v1 = velocity[p.x, p.y, p.z]
    v2 = tm.cross(omega, rigidParticles[q.x, q.y, q.z] - rigidPosition[None]) + rigidV[None]
    den = density[p.x, p.y, p.z]
    cs = 20
    r = posr - posf
    rn = r.norm()
    nu = (couplingViscosity * h * cs) / (2 * den)
    pi = -nu * min(tm.dot(v2 - v1, r), 0) / (rn * rn + shift * h * h)
    return -m * volumeCorrection(q, density0) * pi * \
           (-1.5 * rn * rn / (h ** 3) + 2 * r / (h * h) - h / (2 * rn * rn)) * r / rn * \
           15 / (2 * tm.pi * h ** 3)

@ti.func
def fluidParticleCollisionHandler(idx: tm.ivec3):
    v = tmpVelocity[idx.x, idx.y, idx.z]
    pos = position[idx.x, idx.y, idx.z] + v * timeStep
    for p in range(6):
        flag = (-1) ** p
        if pos[coordinateList[p]] * flag < boundaryList[p] * flag:
            pos[coordinateList[p]] = boundaryList[p] + flag * eps
            v[coordinateList[p]] = -bounce * v[coordinateList[p]]
    if v.norm() > 40:
        v = v.normalized() * 40
    velocity[idx.x, idx.y, idx.z] = v
    position[idx.x, idx.y, idx.z] = pos

@ti.func
def rigidPosUpdate(idx: tm.ivec3):
    omega = rigidP[None] / rigidJ
    nextV = tm.cross(omega, rigidParticles[idx.x, idx.y, idx.z] - rigidPosition[None])
    rigidParticles[idx.x, idx.y, idx.z] += (nextV + rigidV[None]) * timeStep

@ti.func
def rigidParticleCollisionHandler(pos: tm.vec3, normal: tm.vec3, nowV: tm.vec3, cnt: ti.i32):
    ratio = rigidBounce
    rn = tm.vec3(0, 0, 0)
    normalV = -tm.dot(nowV, normal)
    relPos = pos - rigidPosition[None]
    rn = tm.cross(relPos, normal)
    pulse = (1 + ratio) * normalV / (1 / rigidTotalMass + tm.dot(rn, rn) / rigidJ) / cnt
    rigidV[None] += pulse * normal / rigidTotalMass
    rigidP[None] += pulse * rn

@ti.func
def rigidCollisionHandler():
    omega = rigidP[None] / rigidJ
    for p in range(6):
        flag = (-1) ** p
        cnt = 0
        avgPos = tm.vec3(0, 0, 0)
        avgV = tm.vec3(0, 0, 0)
        for i, j, k in ti.ndrange(length, length, length):
            if 0 < i < length - 1 and 0 < j < length - 1 and 0 < k < length - 1:
                continue
            nowV = tm.cross(omega, rigidParticles[i, j, k] - rigidPosition[None]) + rigidV[None]
            pos = rigidParticles[i, j, k] + nowV * timeStep
            if pos[coordinateList[p]] * flag < boundaryList[p] * flag:
                cnt += 1
                avgPos += pos
                avgV += nowV
        if cnt != 0:
            rigidParticleCollisionHandler(avgPos / cnt, normalList[p], avgV / cnt, 1)

@ti.func
def updateRigidKinematics(idx: tm.ivec3, force: tm.vec3):
    pos = rigidParticles[idx.x, idx.y, idx.z]
    rigidF[None] += force
    rigidT[None] += tm.cross(pos - rigidPosition[None], force)

@ti.kernel
def update():
    currentTime[None] += timeStep
    rigidF[None] = gravity * rigidTotalMass
    rigidT[None] = tm.vec3(0, 0, 0)
    # update fluid density first
    for i, j, k in ti.ndrange(na, nb, nc):
        tmpDensity = 0.0
        cnt = 0
        voxelPos = getVoxel(position[i, j, k])
        for p, q, r in ti.ndrange((max(0, voxelPos.x - 1), min(vca, voxelPos.x + 2)), 
                                  (max(0, voxelPos.y - 1), min(vcb, voxelPos.y + 2)), 
                                  (max(0, voxelPos.z - 1), min(vcc, voxelPos.z + 2))):
            # contribution from fluid
            for z in range(voxelCnt[p, q, r]):
                idx = voxel[p, q, r, z]
                w = (position[i, j, k] - position[idx.x, idx.y, idx.z]).norm()
                if w <= h:
                    tmpDensity += densityKernel(w)
            # contribution from rigid
            for z in range(rigidVoxelCnt[p, q, r]):
                idx = rigidVoxel[p, q, r, z]
                w = (position[i, j, k] - rigidParticles[idx.x, idx.y, idx.z]).norm()
                if w <= h:
                    tmpDensity += volumeCorrection(idx, density0) * densityKernel(w)
        density[i, j, k] = tmpDensity * K_poly6 + shift
    # calculate acceleration & update velocity
    for i, j, k in ti.ndrange(na, nb, nc):
        tmpPressure = tm.vec3(0, 0, 0)
        tmpViscosity = tm.vec3(0, 0, 0)
        voxelPos = getVoxel(position[i, j, k])
        acc = gravity
        ttv = tm.vec3(0, 0, 0)
        for p, q, r in ti.ndrange((max(0, voxelPos.x - 1), min(vca, voxelPos.x + 2)), 
                                  (max(0, voxelPos.y - 1), min(vcb, voxelPos.y + 2)), 
                                  (max(0, voxelPos.z - 1), min(vcc, voxelPos.z + 2))):
            # contribution from fluid
            for z in range(voxelCnt[p, q, r]):
                idx = voxel[p, q, r, z]
                w = (position[i, j, k] - position[idx.x, idx.y, idx.z]).norm()
                if w <= h and w > shift:
                    tmpPressure += pressureGradientKernel(tm.ivec3(i, j, k), voxel[p, q, r, z])
                    tmpViscosity += viscosityKernel(tm.ivec3(i, j, k), voxel[p, q, r, z]) * viscosity * K_spiky
            # contribution from rigid
            for z in range(rigidVoxelCnt[p, q, r]):
                idx = rigidVoxel[p, q, r, z]
                w = (position[i, j, k] - rigidParticles[idx.x, idx.y, idx.z]).norm()
                if w <= h:
                    tp = couplingPressureKernel(tm.ivec3(i, j, k), tm.ivec3(idx.x, idx.y, idx.z))
                    tmpPressure += tp / m
                    updateRigidKinematics(idx, B * K_spiky * tp)
                    tv = couplingViscosityKernel(tm.ivec3(i, j, k), tm.ivec3(idx.x, idx.y, idx.z))
                    tmpViscosity += tv / m
                    ttv += tv
                    updateRigidKinematics(idx, -tv)
        acc += -B * K_spiky * tmpPressure + tmpViscosity
        tmpVelocity[i, j, k] = velocity[i, j, k] + acc * timeStep
    # handle collision
    for i, j, k in ti.ndrange(na, nb, nc):
        fluidParticleCollisionHandler(tm.ivec3(i, j, k))
        groupedPosition[i * (nb * nc) + j * nb + k] = position[i, j, k].xzy
    rigidV[None] += rigidF[None] / rigidTotalMass * timeStep
    rigidP[None] += rigidT[None]
    rigidCollisionHandler()
    rigidPosition[None] += rigidV[None] * timeStep
    for i, j, k in ti.ndrange(length, length, length):
        rigidPosUpdate(tm.ivec3(i, j, k))
    corRatio = tm.vec3(-length / 2 + 0.5, -length / 2 + 0.5 , -length / 2 + 0.5).norm() * radius * 2 / \
               (rigidParticles[0, 0, 0] - rigidPosition[None]).norm()
    for i, j, k in ti.ndrange(length, length, length):
        rigidParticles[i, j, k] = rigidPosition[None] + (rigidParticles[i, j, k] - rigidPosition[None]) * corRatio
        rigidGroupedParticles[i * length * length + j * length + k] = rigidParticles[i, j, k].xzy
    setVoxel()
        
@ti.func
def calcRigidVol(idx: tm.ivec3):
    i, j, k = idx.x, idx.y, idx.z
    voxelPos = getVoxel(rigidParticles[i, j, k])
    rho = 0.0
    for p, q, r in ti.ndrange((max(0, voxelPos.x - 1), min(vca, voxelPos.x + 2)), 
                              (max(0, voxelPos.y - 1), min(vcb, voxelPos.y + 2)), 
                              (max(0, voxelPos.z - 1), min(vcc, voxelPos.z + 2))):
        for z in range(rigidVoxelCnt[p, q, r]):
            idx = rigidVoxel[p, q, r, z]
            w = (rigidParticles[i, j, k] - rigidParticles[idx.x, idx.y, idx.z]).norm()
            if w <= h:
                rho += densityKernel(w)
    rigidVol[i, j, k] = 1 / rho * 120000

@ti.kernel
def initRigid():
    rigidPosition[None] = tm.vec3(a / 2, b / 2, (length - 1) / 2 * radius + radius + 50)
    for i, j, k in ti.ndrange(length, length, length):
        rigidParticles[i, j, k] = rigidPosition[None] + tm.vec3(i - length / 2 + 0.5, j - length / 2 + 0.5 , k - length / 2 + 0.5) * radius * 2
        rat = (i + j + k) / (3 * length - 3)
        rigidGroupedColor[i * length * length + j * length + k] = 1.5 * rat * tm.vec3(1.95, 0.56, 0.2) + (1 - rat) * tm.vec3(1.75, 1.9, 0.2)
    rigidP[None] = rigidJ * tm.vec3(1, 1, 1) * 1
    rigidV[None] = (0, 0, 0)
    rigidT[None] = (0, 0, 0)
    rigidF[None] = gravity    
    K_poly6 = m * 315 / (64 * tm.pi * h9)

@ti.kernel
def initFluid():
    ratio = 0.98 * 2
    xBase = a / 2 - (na - 1) / 2 * ratio * radius
    yBase = b / 2 - (nb - 1) / 2 * ratio * radius
    zBase = radius
    initVelocity = tm.vec3(ti.random(float), ti.random(float), 0)
    for i, j, k in ti.ndrange(na, nb, nc):
        position[i, j, k] = tm.vec3(xBase + ratio * i * radius, yBase + ratio * j * radius, zBase + ratio * k * radius)
        velocity[i, j, k] = initVelocity
        rat = (i + j) / (na + nb - 2)
        groupedColor[i * (nb * nc) + j * nc + k] = 2.5 * (rat * tm.vec3(0.2, 0.2, 0.9) + (1 - rat) * tm.vec3(0.9, 0.2, 0.2))

@ti.kernel
def initList():
    normalList[0] = tm.vec3(1, 0, 0)
    normalList[1] = tm.vec3(-1, 0, 0)
    normalList[2] = tm.vec3(0, 1, 0)
    normalList[3] = tm.vec3(0, -1, 0)
    normalList[4] = tm.vec3(0, 0, 1)
    normalList[5] = tm.vec3(0, 0, -1)
    coordinateList[0] = 0
    coordinateList[1] = 0
    coordinateList[2] = 1
    coordinateList[3] = 1
    coordinateList[4] = 2
    coordinateList[5] = 2
    boundaryList[0] = radius - eps
    boundaryList[1] = a - radius + eps
    boundaryList[2] = radius - eps
    boundaryList[3] = b - radius + eps
    boundaryList[4] = radius - eps
    boundaryList[5] = c - radius + eps

@ti.kernel
def initVolume():
    setVoxel()
    # init rigid particle volume
    for i, j, k in ti.ndrange(length, length, length):
        if i == 0 or i == length - 1 or j == 0 or j == length - 1 or k == 0 or k == length - 1:
            calcRigidVol(tm.ivec3(i, j, k))

# main simulation procedure
window = ti.ui.Window("SPH Fluid Simulation on GGUI", (1024, 1024),
                      fps_limit = 60, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

initFluid()
initRigid()
initList()
initVolume()

# exit(0)

result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir = result_dir, framerate = 60, automatic_build = False)
frameCnt = 0

while frameCnt < 1500:
    # update particle position
    for i in range(stepCnt):
        update()
    # rendering configuration
    camera.position(-125.0, 80.0, -75.0)
    camera.lookat(35.0, 45.0, 35.0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    # render particles
    scene.particles(groupedPosition, radius = radius, per_vertex_color = groupedColor)
    scene.particles(rigidGroupedParticles, radius = radius, per_vertex_color = rigidGroupedColor)
    canvas.scene(scene)
    img = window.get_image_buffer_as_numpy()
    video_manager.write_frame(img)
    print(f'\rFrame {frameCnt + 1}/1500 is recorded', end='')
    frameCnt += 1
    window.show()
    
print()
print('Exporting .mp4 and .gif videos...')
video_manager.make_video(gif = True, mp4 = True)
print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')