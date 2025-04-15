import numpy as np


def gradient(f, *varargs, **kwargs):
    """
    Return the gradient of an N-dimensional array.
    This is the old numpy.gradient function, before it got messed up to require
    constant grid spacing.

    The gradient is computed using second order accurate central differences
    in the interior and either first differences or second order accurate
    one-sides (forward or backwards) differences at the boundaries. The
    returned gradient hence has the same shape as the input array.
    Parameters
    ----------
    f : array_like
        An N-dimensional array containing samples of a scalar function.
    varargs : scalar or list of scalar, optional
        N scalars specifying the sample distances for each dimension,
        i.e. `dx`, `dy`, `dz`, ... Default distance: 1.
        single scalar specifies sample distance for all dimensions.
        if `axis` is given, the number of varargs must equal number of axes.
    edge_order : {1, 2}, optional
        Gradient is calculated using N\ :sup:`th` order accurate differences
        at the boundaries. Default: 1.
        .. versionadded:: 1.9.1
    axis : None or int or tuple of ints, optional
        Gradient is calculated only along the given axis or axes
        The default (axis = None) is to calculate the gradient for all the
        axes of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.
        .. versionadded:: 1.11.0
    Returns
    -------
    gradient : list of ndarray
        Each element of `list` has the same shape as `f` giving the derivative
        of `f` with respect to each dimension.
    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
    >>> np.gradient(x)
    array([ 1. ,  1.5,  2.5,  3.5,  4.5,  5. ])
    >>> np.gradient(x, 2)
    array([ 0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])
    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float))
    [array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]), array([[ 1. ,  2.5,  4. ],
            [ 1. ,  1. ,  1. ]])]
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> dx = np.gradient(x)
    >>> y = x**2
    >>> np.gradient(y, dx, edge_order=2)
    array([-0.,  2.,  4.,  6.,  8.])
    The axis keyword can be used to specify a subset of axes of which
    the gradient is calculated
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])
    """
    f = np.asanyarray(f)
    N = len(f.shape)  # number of dimensions

    axes = kwargs.pop('axis', None)
    if axes is None:
        axes = np.array(range(N))
    # check axes to have correct type and no duplicate entries
    if isinstance(axes, int):
        axes = (axes,)
#    if not isinstance(axes, tuple):
#        raise TypeError("A tuple of integers or a single integer is required")

    # normalize axis values:
    axes = tuple(x + N if x < 0 else x for x in axes)
#     if max(axes) >= N or min(axes) < 0:
#         raise ValueError("'axis' entry is out of bounds")

#     if len(set(axes)) != len(axes):
#        raise ValueError("duplicate value in 'axis'")

    n = len(varargs)
    if n == 0:
        dx = [1.0] * N
    elif n == 1:
        dx = [varargs[0]] * N
    elif n == len(axes):
        dx = list(varargs)
    else:
        raise SyntaxError(
            "invalid number of arguments")

    edge_order = kwargs.pop('edge_order', 1)
    if kwargs:
        raise TypeError('"{}" are not valid keyword arguments.'.format(
                                                  '", "'.join(kwargs.keys())))
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D', 'm', 'M']:
        otype = 'd'

    # Difference of datetime64 elements results in timedelta64
    if otype == 'M':
        # Need to use the full dtype name because it contains unit information
        otype = f.dtype.name.replace('datetime', 'timedelta')
    elif otype == 'm':
        # Needs to keep the specific units, can't be a general unit
        otype = f.dtype

    # Convert datetime64 data into ints. Make dummy variable `y`
    # that is a view of ints if the data is datetime64, otherwise
    # just set y equal to the array `f`.
    if f.dtype.char in ["M", "m"]:
        y = f.view('int64')
    else:
        y = f

    for i, axis in enumerate(axes):

        if y.shape[axis] < 2:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least two elements are required.")

        # Numerical differentiation: 1st order edges, 2nd order interior
        if y.shape[axis] == 2 or edge_order == 1:
            # Use first order differences for time data
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            # print('debug 3', slice2, type(slice2), np.shape(slice2))
            out[tuple(slice1)] = (y[tuple(slice2)] - y[tuple(slice3)])/2.0

            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[tuple(slice1)] = (y[tuple(slice2)] - y[tuple(slice3)])

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[tuple(slice1)] = (y[tuple(slice2)] - y[tuple(slice3)])

        # Numerical differentiation: 2st order edges, 2nd order interior
        else:
            # Use second order differences where possible
            out = np.empty_like(y, dtype=otype)

            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(2, None)
            slice3[axis] = slice(None, -2)
            # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
            out[slice1] = (y[slice2] - y[slice3])/2.0

            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
            out[slice1] = -(3.0*y[slice2] - 4.0*y[slice3] + y[slice4])/2.0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            slice4[axis] = -3
            # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3])
            out[slice1] = (3.0*y[slice2] - 4.0*y[slice3] + y[slice4])/2.0

        # divide by step size
        # print('debug', i, np.shape(out), np.shape(dx[i]))
        out /= dx[i]
        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len(axes) == 1:
        return outvals[0]
    else:
        return outvals
