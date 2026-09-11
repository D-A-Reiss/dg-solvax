# Mathematical background

The core functionality of this package is to numerically solve (systems of) partial differential equations (PDEs) of the following form:
$$ \frac{\partial y}{\partial t} = \sum_{\gamma=1}^{N_x} A_\gamma \frac{\partial y}{\partial x_\gamma} + B y, $$

where $y = y(x, t)$ is the $N_y$-dimensional vector field to be determined by solving the PDEs, $t$ usually denotes the time, $N_x$ the number of spatial dimensions, $x_\gamma$ the components of $x$, and $A$ as well as $B$ are $N_y \times N_y$ matrices, independent of $t$ and $x$.

In order to numerically solve the (system of) PDEs above, a _discontinuous Galerkin scheme_ is applied, which can be derived for example in 1+1D in the following way (for the sake of brevity, the index $\gamma$ will be dropped):
1. Divide the spatial domain of interest into $N$ _cells_ with boundaries at $x_i$, $i = 0, \dots, N$ (in the context of finite-element-method solvers, these cells are called finite elements).
2. For each cell $i$ (with boundaries at $x_i$ and $x_{i + 1}$) choose a set of $N_\phi$ normalized functions $\phi_{i \alpha} (x)$ which are zero outside the cell and orthogonal to each other, i.e.,
$$ \int\limits_{x_0}^{x_N} dx \, \phi_{i \alpha} (x) \phi_{j \beta} (x) = \delta_{ij} \delta_{\alpha \beta},$$
where $\delta_{ij}$ denotes the Kronecker symbol.
3. Approximate the vector field $y$ as a linear combination of these orthonormal functions with so-far unknown coefficients $y_{i \alpha}$, i.e.,
$$ y(x, t) = \sum_{j, \beta} y_{j \beta}(t) \phi_{j \beta} (x).$$
4. Insert this approximation into the PDE(s), multiply with $\phi_{i \alpha}$, integrate over $x$ from $x_0$ to $x_N$, and use the orthonormality defined above in order to arrive at
$$ \frac{\partial y_{i \alpha}}{\partial t} = A \int\limits_{x_i}^{x_{i + 1}} dx \, \phi_{i \alpha} (x) \frac{\partial y}{\partial x}  + B y_{i \alpha}(t), $$
5. Apply integration by parts (or in higher dimensions its generalizations, the generalized Stokes' theorem), resulting in
$$ \frac{\partial y_{i \alpha}}{\partial t} = A \Bigg[  \phi_{i \alpha} (x) y (x, t) \Big|_{x_i}^{x_{i + 1}} - \int\limits_{x_i}^{x_{i + 1}} dx \, \frac{\partial \phi_{i \alpha}}{\partial x} \phi_{i \beta} (x) y_{i \beta} (t) \Bigg]  + B y_{i \alpha}(t), $$
where the first term in square brackets is commonly interpreted as fluxes through the cell boundaries.
6. In derivations of discontinuous Galerkin schemes, at this point it's common to choose a numerical approximation for the boundary fluxes, like upwind fluxes, etc., which often seems like an ad-hoc choice, for which it’s unclear how to generalize it from one single PDE to a system of PDEs. In contrast, here the fluxes are computed by solving the related Riemann problem: Assume piecewise constant initial conditions for the same governing PDE(s) (except $B = 0$, which is an approximation justified for small enough time-steps; considering the case of two linear advection equations, with one mode propagating to the left and one to the right it’s straightforward to check that the following computation yields the well-known and well-established upwind fluxes), i.e., $y(x, 0) = y_{L/R}$ for $x < x_i $/$ x > x_i$, where $y_{L/R}$ and $x_i$ are arbitrary. Then the solution can be constructed as a linear superposition of the _characteristics_ of the PDE(s), by computing:
   - the eigenvalues $\lambda_\gamma$ and eigenvectors $v_\gamma$ of the matrix $A$ (note that any characteristic $c_\gamma v_\gamma e^{\lambda_\gamma t + x}$ is a solution of the PDE(s) _without_ initial and boundary conditions, and characteristics with $\lambda_\gamma < 0 $ / $\lambda_\gamma > 0$ can be interpreted as solutions propagating to the right/left);
   - the coefficients $c_{\gamma L/R}$ in the decomposition $y_{L/R} = \sum_\gamma c_{\gamma L/R} v_\gamma$ via solving these two systems of linear equations $y_{L/R} = V c_{L/R}$;
   - and the solution for $t > 0$ at $x = x_i$ as
$$y(x_i, t) = \sum\limits_{\gamma: \lambda_\gamma < 0} c_{\gamma L} v_\gamma + \sum\limits_{\gamma: \lambda_\gamma > 0} c_{\gamma R} v_\gamma, $$
where formally the coefficients c_{\gamma L/R} can be expressed as $c_{L/R} = V^{-1} y_{L/R}$ (although numerically it’s more efficient to solve the linear system of equations above than computing the matrix inverse $V^{-1}$).

7. Finally, inserting the last equation above into the eq. of step 5 and analogous to step 3 approximating $y_{L/R}$ as a linear combination of the chosen functions $\phi_{i \alpha} (x)$, one arrives at a closed system of ODEs which determine the yet unknown coefficients $y_{i \alpha}(t)$. To this system, the JAX solver of [diffrax](https://github.com/patrick-kidger/diffrax) is applied and from its solution for $y_{i \alpha}(t)$ the full solution $y(x, t)$ is reconstructed.