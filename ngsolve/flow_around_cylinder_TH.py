# ------------------------------ LOAD LIBRARIES -------------------------------
import os
from ngsolve import *
from newton_solver import quasi_newton_solve, del_jacobean
from meshes import mesh1, mesh2
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
parser.add_argument('-time', '--timediscr', type=str, default='BDF3', help='Time discretisation', choices=['BDF2', 'BDF3', 'CN'])
parser.add_argument('-Re', '--reynolds', type=float, default=500, help='Reynolds number')
parser.add_argument('-conv', '--convect', type=str, default='EMAC', choices=['EMAC', 'conv', 'skew'], help='Convective')
options = vars(parser.parse_args())
print(options)

order = options['order']
hmax = options['mesh_size']
mesh_choice = options['mesh']
dt = options['timestep']
timestepping = options['timediscr']
conv_form = options['convect']

Re = options['reynolds']
nu = 2 / Re
tend = 500
uin = CoefficientFunction((1, 0))

comp = {"realcompile": True, "wait": True}
condense = True
inverse = 'pardiso'
j_tol = 0.1                 # Jacobi update tolerance for quasi-Newton solver

output_dir = 'results'
filename = f'cylinder_flow_Re{Re}_TH{order}{conv_form}_h{hmax}'
filename += f'mesh{mesh_choice}{timestepping}dt{dt}'

vtk_flag = False
vtk_dir = 'vtk'
vtk_freq = int(0.1 / dt)
vtk_subdiv = 0


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
gfu, gfu1, gfu2, gfu3 = [GridFunction(X) for _ in range(4)]
vel, pre, lagr = gfu.components
vel1, vel2, vel3 = gfu1.components[0], gfu2.components[0], gfu3.components[0]
if conv_form == 'EMAC':
    pre = pre + 0.5 * vel * vel

print(f'Space has {X.ndof}({sum(X.FreeDofs(condense))}) unknowns')
(u, p, lam), (v, q, mu) = X.TnT()
t = Parameter(0)

m0 = (u | v)
m1, m2, m3 = (vel1 | v), (vel2 | v), (vel3 | v)

d0 = (nu * (Grad(u) + Grad(u).trans) | Grad(v))
d1 = (nu * (Grad(vel1) + Grad(vel1).trans) | Grad(v))
pre_coupl = -div(v) * p - div(u) * q + p * mu + q * lam

if conv_form == 'EMAC':
    c0 = ((Grad(u) + Grad(u).trans) * u + div(u) * u | v)
    c1 = ((Grad(vel1) + Grad(vel1).trans) * vel1 + div(vel1) * vel1 | v)
else:
    c0 = (Grad(u) * u | v)
    c1 = (Grad(vel1) * vel1 | v)
    if conv_form == 'skew':
        c0 += 0.5 * (div(u) * u | v)
        c1 += 0.5 * (div(vel1) * vel1 | v)


# ------------------------------- BILINEAR FORMS ------------------------------
# Stokes
a0 = BilinearForm(X, condense=condense)
a0 += (d0 + pre_coupl).Compile() * dx

# Crank-Nicolson
a1 = BilinearForm(X, condense=condense)
a1 += (m0 - m1 + dt * (0.5 * (d0 + c0) + 0.5 * (d1 + c1) + pre_coupl)).Compile(**comp) * dx

# BDF2
a2 = BilinearForm(X, condense=condense)
a2 += ((3 * m0 - 4 * m1 + m2) / 2 + dt * (d0 + c0 + pre_coupl)).Compile(**comp) * dx

# BDF3
a3 = BilinearForm(X, condense=condense)
a3 += ((11 * m0 - 18 * m1 + 9 * m2 - 2 * m3) / 6 + dt * (d0 + c0 + pre_coupl)).Compile(**comp) * dx

res = gfu.vec.CreateVector()

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


def compute_functionals(a):
    a.Apply(gfu.vec, res)
    drag_x_vals.append(- 1 / dt * InnerProduct(res, drag_x_test.vec))
    drag_y_vals.append(- 1 / dt * InnerProduct(res, drag_y_test.vec))
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
time_start = time()
with TaskManager():
    vel.Set(uin, definedon=mesh.Boundaries('inlet|wall|outlet'))

    a0.Assemble()
    invS = a0.mat.Inverse(freedofs=X.FreeDofs(condense), inverse=inverse)
    if condense:
        ext = IdentityMatrix() + a0.harmonic_extension
        extT = IdentityMatrix() + a0.harmonic_extension_trans
        inv = ext @ invS @ extT + a0.inner_solve
        gfu.vec.data += a0.harmonic_extension * gfu.vec
    else:
        inv = invS

    res.data = - a0.mat * gfu.vec
    gfu.vec.data += inv * res
    Redraw(blocking=True)
    if vtk_flag:
        vtk.Do(0)

# ------------------------------- TIME STEPPING -------------------------------
with TaskManager():

    for i in range(1, 3):
        gfu2.vec.data = gfu1.vec
        gfu1.vec.data = gfu.vec
        t.Set(i * dt)

        quasi_newton_solve(a1, gfu, reuse=True, inverse=inverse, maxerr=1e-11,
                           printing=True, jacobi_update_tol=j_tol)
        compute_functionals(a1)
        Redraw(blocking=True)
        if vtk_flag and i % vtk_freq == 0:
            vtk.Do(time=t.Get())
        print(f't = {t.Get()}')
        del_jacobean()

    if timestepping == 'BDF2':
        a3 = a2
    elif timestepping == 'CN':
        a3 = a1

    for i in range(2, int(tend / dt)):
        gfu3.vec.data = gfu2.vec
        gfu2.vec.data = gfu1.vec
        gfu1.vec.data = gfu.vec
        t.Set((i + 1) * dt)
        res.data = gfu.vec
        out = quasi_newton_solve(a3, gfu, reuse=True, inverse=inverse,
                                 maxerr=1e-11, printing=True, maxit=20,
                                 jacobi_update_tol=j_tol)
        if out[0] < 0:
            print('Newton solver failed, retrying without previous Jacobian')
            del_jacobean()
            gfu.vec.data = res
            out = quasi_newton_solve(a3, gfu, reuse=True, inverse=inverse,
                                     maxerr=1e-11, printing=True, maxit=20,
                                     jacobi_update_tol=j_tol)
        compute_functionals(a3)
        Redraw(blocking=True)
        if vtk_flag and (i + 1) % vtk_freq == 0:
            vtk.Do(time=t.Get())
        print(f't = {t.Get()}')
        if out[0] < 0:
            print('Newton solver failed')
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
