from netgen.occ import Rectangle, MoveTo, Circle, Glue, X, Y, OCCGeometry, gp_Pnt2d
import ngsolve

__all__ = ['mesh1', 'mesh2']


def mesh1(hmax: float) -> ngsolve.Mesh:
    base1 = Rectangle(330, 60).Face()
    base1.maxh = hmax
    base1.edges.Min(Y).name = 'wall'
    base1.edges.Max(Y).name = 'wall'
    base1.edges.Min(X).name = 'inlet'
    base1.edges.Max(X).name = 'outlet'
    base2 = MoveTo(25, 25).Rectangle(10, 10).Face()
    base2.maxh = hmax / 2
    base1 -= base2
    base = Glue([base1, base2])
    cyl = Circle(gp_Pnt2d(30, 30), 1).Face()
    cyl.edges.name = 'cyl'
    cyl.edges.maxh = hmax / 100
    base -= cyl
    geo = OCCGeometry(base, dim=2)
    return ngsolve.Mesh(geo.GenerateMesh())


def mesh2(hmax: float) -> ngsolve.Mesh:
    base = Rectangle(330, 60).Face()
    cyl = Circle(gp_Pnt2d(30, 30), 1).Face()
    cyl.edges.name = 'cyl'
    cyl.edges.maxh = hmax / 250

    base.edges.Min(X).name = 'inlet'
    base.edges.Max(X).name = 'outlet'
    base.edges.Min(Y).name = 'wall'
    base.edges.Max(Y).name = 'wall'
    base -= cyl
    geo = OCCGeometry(base, dim=2)
    return ngsolve.Mesh(geo.GenerateMesh(maxh=hmax, grading=0.25))
