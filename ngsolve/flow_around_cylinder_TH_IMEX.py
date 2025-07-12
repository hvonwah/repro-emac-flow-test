# ------------------------------ LOAD LIBRARIES -------------------------------
import os
from ngsolve import *
from meshes import mesh1, mesh2
from math import isnan
from time import time
import pandas as pd
import argparse

SetNumThreads(4)
ngsglobals.msg_level = 2
SetHeapSize(100 * 1000 * 1000)

# -------------------------------- PARAMETERS ---------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=8, help='Mesh size')
parser.add_argument('-m', '--mesh', type=int, default=1, choices=[1, 2])
parser.add_argument('-k', '--order', type=int, default=4, help='Polynomial order')
parser.add_argument('-dt', '--timestep', type=float, default=0.01, help='Time step')
parser.add_argument('-Re', '--reynolds', type=float, default=500, help='Reynolds number')
parser.add_argument('-conv', '--convect', type=str, default='EMAC', choices=['EMAC', 'conv', 'skew'], help='Convective')
options = vars(parser.parse_args())
print(options)

order = options['order']
hmax = options['mesh_size']
mesh_choice = options['mesh']
dt = options['timestep']
conv_form = options['convect']

Re = options['reynolds']
nu = 2 / Re
tend = 500
uin = CoefficientFunction((1, 0))

comp = {"realcompile": True, "wait": True}
condense = True
inverse = 'pardiso'

output_dir = 'results'
filename = f'cylinder_flow_Re{Re}_TH{order}{conv_form}_h{hmax}'
filename += f'mesh{mesh_choice}IMEXSBDF2dt{dt}'

vtk_flag = False
vtk_dir = 'vtk'
vtk_freq = int(0.1 / dt)
vtk_subdiv = 0

check_nan_freq = int(0.1 / dt)

# ------------------------------ GEOMETRY & MESH ------------------------------
if mesh_choice == 1:
    mesh = mesh1(hmax)
elif mesh_choice == 2:
    mesh = mesh2(hmax)
mesh.Curve(order)

# ------------------------------- SET-UP SOLVER -------------------------------
V = VectorH1(mesh, order=order, dirichlet='inlet|wall|cyl|outlet')
Q = H1(mesh, order=order - 1)
N = NumberSpace(mesh)
X = V * Q * N
gfu = GridFunction(X)
vel, pre, lagr = gfu.components
if conv_form == 'EMAC':
    pre = pre + 0.5 * vel * vel

print(f'Space has {X.ndof}({sum(X.FreeDofs(condense))}) unknowns')
(u, p, lam), (v, q, mu) = X.TnT()
t = Parameter(0)

m0 = (u | v)

stokes = (nu * (Grad(u) + Grad(u).trans) | Grad(v))
stokes += -div(v) * p - div(u) * q + p * mu + q * lam

if conv_form == 'EMAC':
    c0 = ((Grad(u) + Grad(u).trans) * u + div(u) * u | v)
else:
    c0 = (Grad(u) * u | v)
    if conv_form == 'skew':
        c0 += 0.5 * (div(u) * u | v)


# ------------------------------- BILINEAR FORMS ------------------------------
# Stokes
a = BilinearForm(X, condense=condense, store_inner=condense)
a += (stokes).Compile() * dx

m = BilinearForm(X)
m += m0.Compile() * dx

# SBDF1
mstar1 = BilinearForm(X, condense=condense)
mstar1 += (m0 + dt * stokes).Compile() * dx

# SBDF2
mstar2 = BilinearForm(X, condense=condense)
mstar2 += (3 / 2 * m0 + dt * stokes).Compile(**comp) * dx

# Explicit Convection
c = BilinearForm(X, nonassemble=True)
c += c0.Compile(**comp) * dx


# ------------------------------- FUNCTIONALS ---------------------------------
drag_x_test, drag_y_test = GridFunction(X), GridFunction(X)
drag_x_test.components[0].Set(CoefficientFunction((1.0, 0)),
                              definedon=mesh.Boundaries('cyl'))
drag_y_test.components[0].Set(CoefficientFunction((0, 1.0)),
                              definedon=mesh.Boundaries('cyl'))
n = specialcf.normal(mesh.dim)
drag_bnd_form = (nu * (Grad(vel) + Grad(vel).trans) * n - pre * n | v).Compile()

f_test = LinearForm(X)
f_test += drag_bnd_form * dx(definedon=mesh.Boundaries('cyl'), skeleton=True)

cf_div = (div(vel)**2).Compile()
cf_ke = (vel | vel).Compile()
cf_vel0 = vel[0].Compile()
cf_vel1 = vel[1].Compile()
cf_am = (vel[0] * y - vel[1] * x).Compile()

time_vals, drag_x_vals, drag_y_vals, drag_x_vals_bnd, drag_y_vals_bnd, \
    div_vel, kin_energy, mom0, mom1, angmom = [[] for _ in range(10)]


def compute_functionals(res: BaseVector):
    drag_x_vals.append(InnerProduct(res, drag_x_test.vec))
    drag_y_vals.append(InnerProduct(res, drag_y_test.vec))
    time_vals.append(t.Get())

    f_test.Assemble()
    drag_x_vals_bnd.append(- 1 * InnerProduct(f_test.vec, drag_x_test.vec))
    drag_y_vals_bnd.append(- 1 * InnerProduct(f_test.vec, drag_y_test.vec))

    div_vel.append(sqrt(Integrate(cf_div * dx, mesh)))
    kin_energy.append(1 / 2 * Integrate(cf_ke * dx, mesh))
    mom0.append(Integrate(cf_vel0 * dx, mesh))
    mom1.append(Integrate(cf_vel1 * dx, mesh))
    angmom.append(Integrate(cf_am * dx, mesh))
    return None


# ------------------------------- VISUALISATION -------------------------------
Draw(vel, mesh, 'velocity')
Draw(div(vel), mesh, 'divergece')
Draw(pre, mesh, 'pressure')

# ---------------------------------- OUTPUT -----------------------------------
try:
    data_dir = os.environ['DATA'][:-1]
except KeyError:
    print('DATA environment variable does not exist')
    data_dir = '..'

comp_dir_name = os.getcwd().split('/')[-1]
output_dir_abs = data_dir + '/' + comp_dir_name + '/' + output_dir
if not os.path.isdir(output_dir_abs):
    os.makedirs(output_dir_abs)

if vtk_flag:
    vtk_dir_abs = data_dir + '/' + comp_dir_name + '/' + vtk_dir
    if not os.path.isdir(vtk_dir_abs):
        os.makedirs(vtk_dir_abs)
    coefs = [vel.Compile(), pre.Compile(),
             (Grad(vel)[1, 0] - Grad(vel)[0, 1]).Compile()]
    vtk = VTKOutput(ma=mesh, coefs=coefs, names=['vel', 'pre', 'vort'],
                    filename=f'{vtk_dir_abs}/{filename}',
                    subdivision=vtk_subdiv, order=2, floatsize='single')

print(f'data will be saved in the directory {data_dir}/{comp_dir_name}')

# ------------------------------- INITIALISATION ------------------------------
res, mlast, clast = [gfu.vec.CreateVector() for _ in range(3)]

time_start = time()
with TaskManager():
    vel.Set(uin, definedon=mesh.Boundaries('inlet|wall|outlet'))

    a.Assemble()
    invS = a.mat.Inverse(freedofs=X.FreeDofs(condense), inverse=inverse)
    if condense:
        IdMS = IdentityMatrix()
        AShex = a.harmonic_extension
        AShext = a.harmonic_extension_trans
        inv = (IdMS + AShex) @ invS @ (IdMS + AShext) + a.inner_solve
        AA = (IdMS - AShext) @ (a.mat + a.inner_matrix) @ (IdMS - AShex)
    else:
        inv = invS
        AA = a.mat

    res.data = - AA * gfu.vec
    gfu.vec.data += inv * res
    Redraw(blocking=True)
    if vtk_flag:
        vtk.Do(0)

try:
    del inv, invS
except NameError:
    pass

# ------------------------------- TIME STEPPING -------------------------------
with TaskManager():
    m.Assemble()
    mstar1.Assemble()
    invmstar1 = mstar1.mat.Inverse(X.FreeDofs(condense), inverse=inverse)
    if condense:
        M1Shex = mstar1.harmonic_extension
        M1Shext = mstar1.harmonic_extension_trans
        M1Siii = mstar1.inner_solve
        inv = (IdMS + M1Shex) @ invmstar1 @ (IdMS + M1Shext) + M1Siii
    else:
        inv = invmstar1

    t.Set(dt)
    mlast.data = m.mat * gfu.vec
    c.Apply(gfu.vec, clast)
    res.data = - clast - AA * gfu.vec
    gfu.vec.data += dt * inv * res
    compute_functionals(res)
    Redraw(blocking=True)
    if vtk_flag and vtk_freq == 1:
        vtk.Do(time=t.Get())
    print(f't = {t.Get():.8f}', end='\r')

    try:
        del inv, invmstar1, M1Shex, M1Shext, M1Siii
    except NameError:
        pass

    mstar2.Assemble()
    invmstar2 = mstar2.mat.Inverse(X.FreeDofs(condense), inverse=inverse)
    if condense:
        M2Shex = mstar2.harmonic_extension
        M2Shext = mstar2.harmonic_extension_trans
        M2Siii = mstar2.inner_solve
        inv = (IdMS + M2Shex) @ invmstar2 @ (IdMS + M2Shext) + M2Siii
    else:
        inv = invmstar2

    for i in range(1, int(tend / dt)):
        t.Set(dt * (i + 1))
        res.data = clast - 0.5 / dt * mlast
        mlast.data = m.mat * gfu.vec
        c.Apply(gfu.vec, clast)
        res.data += -2 * clast + 1 / 2 / dt * mlast
        res.data += -AA * gfu.vec
        gfu.vec.data += dt * inv * res
        compute_functionals(res)
        Redraw(blocking=True)

        if vtk_flag and (i + 1) % vtk_freq == 0:
            vtk.Do(time=t.Get())
        print(f't = {t.Get():.8f}', end='\r')
        if (i + 1) % check_nan_freq == 0:
            if isnan(Norm(gfu.vec)):
                print(f'Aborting as NaNs detected!, t={i * dt}')
                break


# ------------------------------ POST-PROCESSING ------------------------------
df = pd.DataFrame({'time': time_vals, 'dragvol': drag_x_vals,
                   'liftvol': drag_y_vals, 'dragbnd': drag_x_vals_bnd,
                   'liftbnd': drag_y_vals_bnd, 'div': div_vel,
                   'kin_energy': kin_energy, 'mom0': mom0, 'mom1': mom1,
                   'angmom': angmom})
df.to_csv(f'{output_dir_abs}/{filename}.txt', sep=' ', index=False)

time_total = time() - time_start

print('\n----- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f} -----\n'.format(
      time_total // (24 * 60**2), time_total % (24 * 60**2) // 60**2,
      time_total % 3600 // 60, time_total % 60))
