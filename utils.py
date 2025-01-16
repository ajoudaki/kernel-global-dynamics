from functools import lru_cache
from numba import njit 
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from scipy import integrate
from scipy.special import hermitenorm
from functools import wraps 
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.integrate import odeint

def handle_multidim(func):
    """
    Decorator to handle multi-dimensional input for Numba functions.
    The decorated function should expect 1D input.
    """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        original_shape = x.shape
        x_flat = x.ravel()
        result_flat = func(x_flat, *args, **kwargs)
        return result_flat.reshape(original_shape)
    
    return wrapper

def gaussian_pdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

@lru_cache(maxsize=1000)
def compute_hermite_coefs_integration(f, order=25, a=-15, b=15, norm=True):
    """
    Compute Hermite coefficients using numerical integration.
    
    Parameters:
    f (callable): The function to approximate
    order (int): The number of coefficients to compute
    a, b (float): The integration limits
    norm (bool): Whether to use normalized Hermite polynomials
    
    Returns:
    np.array: The computed Hermite coefficients
    """
    def integrand(x, k):
        C = np.sqrt(np.float64(math.factorial(k))) 
        he_k = lambda x: hermitenorm(k)(x)/C
        return gaussian_pdf(x) * f(x) * he_k(x)
    
    hermite_coefs = np.zeros(order)
    
    for k in range(order):
        result, _ = integrate.quad(integrand, a, b, args=(k,))
        if norm:
            hermite_coefs[k] = result 
        else:
            hermite_coefs[k] = result
    
    return hermite_coefs




@lru_cache(maxsize=1000)
def get_activation_function(name='relu', **config):
    """
    Returns the activation function based on the provided name and optional configuration parameters.

    Parameters:
    name (str): The name of the activation function. 
                Options are 'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 
                'leaky_relu', 'elu', 'selu', 'celu', 'gelu', 'swish'.
    config (dict): A dictionary of parameters for the activation function. 
                   If None, default values are used.

    Returns:
    function: A function that computes the specified activation using the provided parameters.

    Example use:
    
    f = get_activation_function('relu', {'alpha': 0.1})
    coefs = compute_hermite_coefs(f, coefs_len=20)
    f2 = hermite_expansion(coefs)
    print(coefs)
    x = np.linspace(-3, 3, 100)
    """

    @njit
    def relu(x):
        return np.maximum(0, x)
    
    @njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @njit
    def tanh(x):
        return np.tanh(x)
    
    @njit
    def exp(x):
        return np.exp(x)
    
    @njit
    def softmax(x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    @njit
    def linear(x):
        return x
    
    @njit
    def leaky_relu(x, alpha=config.get('alpha', 0.01)):
        return np.where(x > 0, x, alpha * x)
    
    @njit
    def elu(x, alpha=config.get('alpha', 1.0)):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    @njit
    def selu(x, alpha=config.get('alpha', 1.67326), scale=config.get('scale', 1.0507)):
        return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    @njit
    def celu(x, alpha=config.get('alpha', 1.0)):
        return np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1))
    
    @njit
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    @njit
    def swish(x, beta=config.get('beta', 1.0)):
        return x / (1 + np.exp(-beta * x))

    # Dictionary of activation functions
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'exp': exp,
        'softmax': softmax,
        'linear': linear,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'selu': selu,
        'celu': celu,
        'gelu': gelu,
        'swish': swish
    }
    
    return activations.get(name.lower(), None)

@lru_cache(maxsize=1000)
def factorial(k):
    assert k >= 0 and k == int(k)
    if k<=1:
        return 1
    return k * factorial(k-1)

def coefs_derivative(coefs, r):
    if r == 0:
        return coefs
    coefs = coefs.copy()
    # print(f'derivative 0 = ', ', '.join(f"c{i}={c}" for i,c in enumerate(coefs)))
    for i in range(r):
        for k in range(1,len(coefs)):
            coefs[k-1] = k * coefs[k]
        coefs[len(coefs)-1-i] = 0
        # print(f'derivative {i+1} = ', ', '.join(f"c{i}={c}" for i,c in enumerate(coefs)))
    return coefs

def coefs_to_poly(coefs):
    @njit
    def f(x):
        K = len(coefs)
        x_powers = np.zeros((K,len(x)),dtype=np.float64)
        for i in range(K):
            x_powers[i] = x**i
        return coefs @ x_powers
    return f

def hermite_poly_coefs(k):
    if k == 0:
        return np.array([1,])
    elif k == 1:
        return np.array([0,1])
    else:
        H_k_minus_1 = hermite_poly_coefs(k-1)
        H_k_minus_2 = hermite_poly_coefs(k-2)
        return np.concatenate([[0], H_k_minus_1]) - (k-1) * np.concatenate([H_k_minus_2, [0, 0]])

def hermite_poly_coefs_norm(k):
    coefs = hermite_poly_coefs(k)
    c = factorial(k)**0.5
    return coefs / c

def hermite_expansion(coefs, norm=True, return_coefs=False):
    K = len(coefs)
    cs = np.zeros(len(coefs),dtype=np.float64)
    for k,c in enumerate(coefs):
        if norm:
            cs += c * np.concatenate([hermite_poly_coefs_norm(k), np.zeros(K-1-k)])
        else:
            cs += c * np.concatenate([hermite_poly_coefs(k), np.zeros(K-1-k)])
    if return_coefs:
        return coefs_to_poly(cs), cs
    return coefs_to_poly(cs)

def hermite_poly(k,norm=True):
    coefs = np.zeros(k+1,dtype=np.float64)
    coefs[k] = 1
    return hermite_expansion(coefs, norm=norm)

@lru_cache(maxsize=1000)
def compute_hermite_coefs(f, order, norm=True, num_samples=10**7):
    X = np.random.randn(num_samples)
    hermite_coefs = np.zeros(order)
    for k in tqdm.trange(len(hermite_coefs)):
        hermite_coefs[k] = np.mean(f(X) * hermite_poly(k,norm=norm)(X))
        if not norm:
            hermite_coefs[k] /= factorial(k)
    return hermite_coefs

def kernel_map_emp(f, num_bins=100, num_samples=10**6,atol=1e-2,rtol=1e-2):
    rhos = np.linspace(-1,1,num_bins)
    vals = np.zeros(len(rhos))
    (x,y,z) = np.random.randn(3,num_samples)
    for i,rho in enumerate(rhos):
        ryz = np.sqrt(abs(rho))
        rxz = np.sign(rho) * ryz
        r = np.sqrt(1-abs(rho))
        X = rxz * z + r * y
        Y = ryz * z + r * x
        # test if rho = E[X Y], and variances are 1
        assert(np.allclose(np.mean(X * Y),rho,atol=atol,rtol=rtol))
        assert(np.allclose(np.var(X),1,atol=atol,rtol=rtol))
        assert(np.allclose(np.var(Y),1,atol=atol,rtol=rtol))
        vals[i] = np.mean(f(X) * f(Y))
    @njit
    def kernel(x):
        closest_indices = np.abs(rhos[:, np.newaxis] - x).argmin(axis=0)
        return vals[closest_indices]
    
    return kernel


# compute kernel map from Hermite coefficients
def kernel_map(coefs,r=0, norm=True):
    # cross terms dissapear, since E[He_k He_l] = 0 for k != l, leaving squared terms 
    coefs = coefs ** 2
    # if not normalized, E[He_k^2] = k!, if normalized E[He_k^2] = 1
    if not norm:
        c = 1
        for k in range(1,len(coefs)):
            c *= k
            coefs[k] = coefs[k] * c
    coefs = coefs_derivative(coefs, r)
    def kappa(x):
        return np.sum([(coefs[k]) * x**k for k in range(len(coefs))], axis=0)
    return kappa



def fixed_point_iteration(func, rho0, tol=1e-5, max_iterations=1000):
    x_values = [rho0]
    for _ in range(max_iterations):
        x_values.append(func(x_values[-1]))
        if len(x_values)>10 and abs(x_values[-1] - x_values[-2]) < tol:
            break
    return x_values


def coefs2name(poly_coefs):
    def sgn(c):
        if c==0:
            return ''
        if c>0:
            return '+'
        else:
            return '-'
    def t(i,c):
        return f'{sgn(c)}{abs(c):.1f}' + (f"x^{i}" if i>0 else "")
    act_name = (f"\\phi(x)={''.join(t(i,c) for i,c in enumerate(poly_coefs))}")
    return act_name 

def coefs2kernel_name(coefs):
    def term(c,i):
        if c==0:
            return ''
        elif i==0:
            return f'+{c:.1f}'
        else:
            return f'+{c:.1f}\\rho^{i}'
    kernel_name = (f"\\kappa(\\rho) ={''.join(term(c**2,i) for i,c in enumerate(coefs))}")
    return kernel_name


def test_hermite_coefs_integration(act_name = 'relu', order = 10):
    # input a test function name (e.g., ReLU)
    f = get_activation_function(act_name)
    
    # Compute coefficients with integration 
    coefs = compute_hermite_coefs_integration(f, order)
    
    # Compute sampling-based coefficinets
    coefs_sampling = compute_hermite_coefs(f, order)
    
    print("\nComparison with sampling-based method:")
    for i, (coef_int, coef_samp) in enumerate(zip(coefs, coefs_sampling)):
        print(f"c_{i}: Integration = {coef_int:10.7f}, Sampling = {coef_samp:10.7f}, error = {coef_int-coef_samp:10.7f}")

def test_kernel_map_from_coefs(coefs, norm, atol=1e-2, rtol=1e-2, plot=True):
    f = hermite_expansion(coefs, norm=norm)
    kernel_emp = kernel_map_emp(f)
    kernel_theory = kernel_map(coefs, norm=norm)
    x = np.linspace(-1,1,30)
    emp = kernel_emp(x)
    theory = kernel_theory(x)
    if plot:
        plt.figure()
        plt.plot(x, kernel_emp(x), label='empirical kappa(x)',marker='o')
        plt.plot(x, kernel_theory(x), label='theoretical kappa(x)')
        plt.xlabel('$\\rho$')
        plt.ylabel('$\\kappa(\\rho)$')
        plt.title(f'Kernel map in {"normalized (he)" if norm else "unnormalized (He)"} Hermite basis')
        plt.legend()
    if not np.allclose(emp, theory, atol=atol, rtol=rtol):
        print("Success: Kernel map theory and empirical values are close")
    else:
        print("Failed: Kernel map theory and empirical values are not close")


def test_kernel_map_properties_from_coefs(coefs,norm, num_samples=10**7,atol=1e-2, rtol=1e-2):
    f = hermite_expansion(coefs, norm=norm)
    kappa = kernel_map(coefs, norm=norm)
    kappa_prime = kernel_map(coefs, r=1, norm=norm)
    X = np.random.randn(num_samples)
    c0 = coefs[0]
    c1 = coefs[1]
    if norm:
        c2_sum = np.sum(coefs**2)
    else:
        c2_sum = np.sum([factorial(k) * c**2 for k,c in enumerate(coefs)])
    Ef = np.mean(f(X))
    Efx = np.mean(X * f(X))
    Ef2 = np.mean(f(X)**2)
    k0 = kappa(0)
    kprime_0 = kappa_prime(0)
    k1 = kappa(1)
    np.testing.assert_allclose(c0, Ef, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c0**2, k0, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c1, Efx, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c1**2, kprime_0, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c2_sum, Ef2, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c2_sum, k1, atol=atol,rtol=rtol)
    print(f"Success: Kernel map properties in {'normalized (he)' if norm else 'unnormalized (He)'} basis are satisfied")

# test orthogonality of polynomials 
def test_orthogonality(K=4, tol = 1e-2):
    X = np.random.randn(10**7)
    for norm in [True, False]:
        poly_name = "he" if norm else "He"
        print(f"Testing orthogonality of {'normalized' if norm else ''} Hermite polynomials ({poly_name}(x))")
        for k in range(K):
            for l in range(k,K):
                f = hermite_poly(k,norm)
                g = hermite_poly(l,norm)
                theory = float(k==l)
                if not norm:
                    theory *= factorial(k)
                emp = np.mean(f(X) * g(X))
                error = np.abs(theory - emp)
                if error > tol:
                    message = "WARNING: "
                else:
                    message = ""
                print(f"{message} E [{poly_name}_{k}(X) {poly_name}_{l}(X)], theory = {theory:5.4f}, emp =  {emp:5.4f}, error = {error:5.4f}")


        # test if we can recover the coefficients of the expansion
    
def test_recovery(coefs, tol = 5e-2):
    for norm in [True, False]:
        poly_name = "he" if norm else "He"
        print(f"Testing recovery of {'normalized' if norm else ''} Hermite coefficients ({poly_name}(x))")
        # test with He_k (not normalized)
        f = hermite_expansion(coefs,norm=norm)
        hermite_coefs = compute_hermite_coefs(f, len(coefs)+3, norm=norm)

        for k,c in enumerate(hermite_coefs):
            c_org = coefs[k] if k < len(coefs) else 0
            err = np.abs(c_org - c) 
            if err > tol:
                message = "WARNING: "
            else:
                message = ""
            print(f"{message} c_k: original = {c_org:5.4f}, recovered = {c:5.4f}, error = {err:5.4f}")

def run_tests(tol=0.02):
    test_orthogonality(K=4)
    
    # tests these properties for some random activation
    for _ in range(2):
        coefs = np.random.randn(4)
        test_recovery(coefs)
        coefs = np.random.randn(5)
        test_kernel_map_from_coefs(coefs, norm=False)
        test_kernel_map_properties_from_coefs(coefs, norm=False)
        test_kernel_map_from_coefs(coefs, norm=True)
        test_kernel_map_properties_from_coefs(coefs, norm=True)

    max_order = 25
    for act_name in ['tanh', 'relu', 'leaky_relu', 'exp', 'gelu', 'selu', 'celu', 'elu', 'sigmoid']:
        f = get_activation_function(act_name)
        coefs = compute_hermite_coefs_integration(f, order=max_order, a=-20,b=20)
        f2 = hermite_expansion(coefs)
        x = np.arange(-2,2,0.01)
        mae = np.mean(np.abs(f(x)-f2(x)))
        s = "WARNING" if mae > tol  else ""
        s += f"Hermite approx for {act_name:10} of order {max_order}, MAE = {mae:.3f}"
        print(s)
    
    
## you can uncomment lines bellow to test if the basic properties hold 
# run_tests()