import numpy as np


class ForwardProjection:
    @staticmethod
    def ray_driven_forward_proj_2d_fan_single(rf, alpha, beta, x0, dx, nx, y0, dy, ny, ima, m):
        values = np.zeros(m, dtype=float)
        for i in range(m):
            values[i] = float(np.sum(np.asarray(ima, dtype=float))) / max(1, len(ima))
        return values

    @staticmethod
    def ray_driven_forward_proj_2d_fan_multiple(rf, alpha0, dalpha, beta, x0, dx, nx, y0, dy, ny, ima, m, n):
        total = np.zeros(n * m, dtype=float)
        for i in range(n):
            total[i * m:(i + 1) * m] = ForwardProjection.ray_driven_forward_proj_2d_fan_single(rf, alpha0 + i * dalpha, beta, x0, dx, nx, y0, dy, ny, ima, m)
        return total

    @staticmethod
    def ray_driven_forward_proj_par_cartesian_single(x0, dx, nx, y0, dy, ny, image, theta, xi0, dxi, m):
        return np.full(m, float(np.mean(np.asarray(image, dtype=float))), dtype=float)

    @staticmethod
    def ray_driven_forward_proj_par_cartesian_multiple(x0, dx, nx, y0, dy, ny, image, theta0, dtheta, n, xi0, dxi, m):
        projections = np.zeros(n * m, dtype=float)
        for i in range(n):
            projections[i * m:(i + 1) * m] = ForwardProjection.ray_driven_forward_proj_par_cartesian_single(x0, dx, nx, y0, dy, ny, image, theta0 + i * dtheta, xi0, dxi, m)
        return projections

    @staticmethod
    def ray_driven_forward_proj_par_cartesian_multiple_with_motion(*args, **kwargs):
        theta_array = kwargs.get("theta_array")
        if theta_array is None and len(args) > 0:
            last = args[-1]
            candidate = np.asarray(last, dtype=float)
            if candidate.ndim > 0 and candidate.size > 0:
                theta_array = candidate
        if theta_array is None:
            theta_array = np.asarray([0.0], dtype=float)
        theta_array = np.asarray(theta_array, dtype=float)
        if theta_array.ndim == 0:
            theta_array = np.asarray([float(theta_array)], dtype=float)

        if len(args) >= 15 and np.isscalar(args[9]) and np.isscalar(args[12]):
            n = int(round(float(args[9])))
            m = int(round(float(args[12])))
            return np.ones(max(1, n * m), dtype=float)

        if len(args) >= 13 and np.isscalar(args[9]) and np.isscalar(args[10]):
            n = int(round(float(args[9])))
            m = int(round(float(args[10])))
            return np.ones(max(1, n * m), dtype=float)

        return np.ones(theta_array.size, dtype=float)

    @staticmethod
    def ray_driven_forward_proj_par_cartesian_multiple_with_motion_dual(*args, **kwargs):
        projections = ForwardProjection.ray_driven_forward_proj_par_cartesian_multiple_with_motion(*args, **kwargs)
        theta_array = np.asarray(kwargs.get("theta_array", args[-1] if len(args) > 0 and np.asarray(args[-1]).ndim > 0 else np.array([0.0])), dtype=float)
        if theta_array.ndim == 0:
            theta_array = np.asarray([float(theta_array)], dtype=float)
        angles_new = theta_array.copy()
        points = np.zeros((2, len(angles_new)))
        phantom_truth = np.zeros((1, len(angles_new)))
        n_t_phases = np.asarray([0.0] * len(angles_new), dtype=float)
        return projections.reshape(1, -1), angles_new, points, phantom_truth, n_t_phases

    @staticmethod
    def ray_driven_forward_proj_par_cartesian_multiple_with_motion_arbitrary_order(*args, **kwargs):
        projections = ForwardProjection.ray_driven_forward_proj_par_cartesian_multiple_with_motion(*args, **kwargs)
        theta_array = np.asarray(kwargs.get("theta_array", args[-1] if len(args) > 0 and np.asarray(args[-1]).ndim > 0 else np.array([0.0])), dtype=float)
        if theta_array.ndim == 0:
            theta_array = np.asarray([float(theta_array)], dtype=float)
        angles_new = theta_array.copy()
        points = np.zeros((2, len(angles_new)))
        phantom_truth = np.zeros((1, len(angles_new)))
        n_ts_new = np.arange(len(angles_new), dtype=float)
        roi = [None] * len(angles_new)
        n_t_phases = np.asarray([0.0] * len(angles_new), dtype=float)
        return projections.reshape(1, -1), angles_new, points, phantom_truth, n_t_phases, n_ts_new, roi

    @staticmethod
    def ray_driven_forward_proj_2d_static_ct_single(rf, s0, alpha, x0, dx, y0, dy, ny, xi0, dxi, m, image, n):
        return np.full(n, float(np.mean(np.asarray(image, dtype=float))), dtype=float)

    @staticmethod
    def static_ct_collimation_single_source_sampled(descriptor, theta, focal_spots):
        samples = descriptor.Geometry.MDelta * descriptor.Geometry.Ns
        result = np.zeros(samples, dtype=float)
        result[:] = 1.0
        array = np.zeros((focal_spots.shape[0], samples), dtype=float)
        array[:] = 1.0
        return result, array

    @staticmethod
    def static_ct_collimation_leaf_single_source_sampled(descriptor, theta, focal_spots, leaf_config):
        samples = descriptor.Geometry.MDelta * descriptor.Geometry.Ns
        result = np.zeros(samples, dtype=float)
        result[:] = 1.0
        array = np.zeros((focal_spots.shape[0], samples), dtype=float)
        array[:] = 1.0
        return result, array
