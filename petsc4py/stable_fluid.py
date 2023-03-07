import sys
import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt
import petsc4py

petsc4py.init(sys.argv)

# Optional
import cmasher as cmr
from rich.progress import track
from petsc4py import PETSc

DOMAIN_SIZE = 1.0
N_POINTS = 41
N_TIME_STEPS = 100
TIME_STEP_LENGTH = 0.1

KINEMATIC_VISCOSITY = 0.0001

rtol = 1e-3

def forcing_function(time, point):
    time_decay = np.maximum(
        2.0 - 0.5 * time,
        0.0,
    )

    forced_value = (
        time_decay
        *
        np.where(
            (
                (point[0] > 0.4)
                &
                (point[0] < 0.6)
                &
                (point[1] > 0.1)
                &
                (point[1] < 0.3)
            ),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        )
    )

    return forced_value


def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    scalar_dof = N_POINTS**2
    vector_shape = (N_POINTS, N_POINTS, 2)

    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    # Using "ij" indexing makes the differential operators more logical. Take
    # care when plotting.
    X, Y = np.meshgrid(x, y, indexing="ij")

    coordinates = np.concatenate(
        (
            X[..., np.newaxis],
            Y[..., np.newaxis],
        ),
        axis=-1,
    )

    forcing_function_vectorized = np.vectorize(
        pyfunc=forcing_function,
        signature="(),(d)->(d)",
    )

    def partial_derivative_x(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[2:  , 1:-1]
                -
                field[0:-2, 1:-1]
            ) / (
                2 * element_length
            )
        )

        return diff

    def partial_derivative_y(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[1:-1, 2:  ]
                -
                field[1:-1, 0:-2]
            ) / (
                2 * element_length
            )
        )

        return diff
    
    def divergence(vector_field):
        divergence_applied = (
            partial_derivative_x(vector_field[..., 0])
            +
            partial_derivative_y(vector_field[..., 1])
        )

        return divergence_applied
    
    def gradient(field):
        gradient_applied = np.concatenate(
            (
                partial_derivative_x(field)[..., np.newaxis],
                partial_derivative_y(field)[..., np.newaxis],
            ),
            axis=-1,
        )

        return gradient_applied
    
    def curl_2d(vector_field):
        curl_applied = (
            partial_derivative_x(vector_field[..., 1])
            -
            partial_derivative_y(vector_field[..., 0])
        )

        return curl_applied

    def advect(field, vector_field):
        backtraced_positions = np.clip(
            (
                coordinates
                -
                TIME_STEP_LENGTH
                *
                vector_field
            ),
            0.0,
            DOMAIN_SIZE,
        )

        advected_field = interpolate.interpn(
            points=(x, y),
            values=field,
            xi=backtraced_positions,
        )

        return advected_field

    def vector_boundary(vector_field):
        vector_field[0, ...] = 0.0
        vector_field[N_POINTS-1, ...] = 0.0
        vector_field[:, 0, :] = 0.0
        vector_field[:, N_POINTS-1, :] = 0.0

    plt.style.use("dark_background")
    plt.figure(figsize=(5, 5), dpi=160)

    # Allocate space
    velocities_prev = np.zeros(vector_shape)
    velocities_diffused = np.zeros(vector_shape)
    velocities_forces_applied = np.zeros(vector_shape)
    velocities_advected = np.zeros(vector_shape)

    time_current = 0.0

    # Assemble sparse Poission matrix
    A = PETSc.Mat().createAIJ([scalar_dof, scalar_dof])
    A.setUp()

    diagonal_entry = -4.0/element_length**2
    off_diagonal_entry = 1.0/element_length**2

    for i in range(0, scalar_dof):
        A.setValue(i, i, diagonal_entry)

    for i in range(0, scalar_dof-N_POINTS):
        A.setValue(i, i+1, off_diagonal_entry)
        A.setValue(i, i+N_POINTS, off_diagonal_entry)

    for i in range(N_POINTS, scalar_dof):
        A.setValue(i, i-1, off_diagonal_entry)
        A.setValue(i, i-N_POINTS, off_diagonal_entry)

    A.assemble()

    # Assemble sparse Diffusive matrix
    B = PETSc.Mat().createAIJ([scalar_dof, scalar_dof])
    B.setUp()

    diagonal_entry = 1.0+4.0*KINEMATIC_VISCOSITY*TIME_STEP_LENGTH/element_length**2
    off_diagonal_entry = -KINEMATIC_VISCOSITY*TIME_STEP_LENGTH/element_length**2

    for i in range(0, scalar_dof):
        B.setValue(i, i, diagonal_entry)

    for i in range(0, scalar_dof-N_POINTS):
        B.setValue(i, i+1, off_diagonal_entry)
        B.setValue(i, i+N_POINTS, off_diagonal_entry)

    for i in range(N_POINTS, scalar_dof):
        B.setValue(i, i-1, off_diagonal_entry)
        B.setValue(i, i-N_POINTS, off_diagonal_entry)

    B.assemble()

    # Assemble the initial rhs to the linear system
    b = PETSc.Vec().createSeq(scalar_dof)

    # Allocate a PETSc vector storing the solution to the linear system
    res = PETSc.Vec().createSeq(scalar_dof)

    # Instantiate a linear solver: Krylov subspace linear iterative solver
    ksp_p = PETSc.KSP().create()
    ksp_p.setOperators(A)

    # Set tolerence : rtol, abstol
    ksp_p.setTolerances(rtol, None)

    # print(f"Solving Poisson with solver {ksp_p.getType():}")
    # print(f"Solving Poisson with PC {ksp_p.getPC().getType():}")

    ksp_d = PETSc.KSP().create()
    ksp_d.setOperators(B)

    # Set tolerence : rtol, abstol
    ksp_d.setTolerances(rtol, None)

    # print(f"Solving Poisson with solver {ksp_d.getType():}")
    # print(f"Solving Poisson with PC {ksp_d.getPC().getType():}")

    for i in track(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH

        forces = forcing_function_vectorized(
            time_current,
            coordinates,
        )

        # (1) Apply Forces
        velocities_forces_applied = (
            velocities_prev
            +
            TIME_STEP_LENGTH
            *
            forces
        )

        # (2) Nonlinear convection (=self-advection)
        velocities_advected = advect(
            field=velocities_forces_applied,
            vector_field=velocities_forces_applied,
        )

        # (3) Diffuse using PETSc
        vel_x = velocities_advected[...,0].flatten()
        for i in range(scalar_dof):
            b.setValue(i, vel_x[i])

        ksp_d.solve(b, res)
        velocities_diffused[...,0] = res.getArray().reshape(N_POINTS, N_POINTS)

        vel_y = velocities_advected[...,1].flatten()
        for i in range(scalar_dof):
            b.setValue(i, vel_y[i])

        ksp_d.solve(b, res)
        # print('Diffusion iterations : ', ksp_d.getIterationNumber())

        velocities_diffused[...,1] = res.getArray().reshape(N_POINTS, N_POINTS)

        vector_boundary(velocities_diffused)
        # (4.1) Poisson using PETSc
        div = divergence(velocities_diffused).flatten()

        for i in range(scalar_dof):
            b.setValue(i, div[i])

        ksp_p.solve(b, res)
        # print('Poisson iterations : ', ksp_p.getIterationNumber())

        pressure = res.getArray()
        pressure = pressure.reshape(N_POINTS, N_POINTS)

        # (4.2) Correct the velocities to be incompressible
        velocities_projected = (
            velocities_diffused
            -
            gradient(pressure)
        )

        # Advance to next time step
        velocities_prev = velocities_projected

        # Plot
        curl = curl_2d(velocities_projected)
        plt.contourf(
            X,
            Y,
            curl,
            cmap=cmr.redshift,
            levels=100,
        )
        plt.quiver(
            X,
            Y,
            velocities_projected[..., 0],
            velocities_projected[..., 1],
            color="dimgray",
        )
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    plt.show()

if __name__ == "__main__":
    main()

