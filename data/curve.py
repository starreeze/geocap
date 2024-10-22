from dataclasses import dataclass, asdict, field
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


@dataclass
class Curve:
    """Cubic Bézier curve for controllable shapes."""

    control_points: list[tuple[float, float]]

    def __post_init__(self):
        self.num_points = 100
        # Ensure there are exactly 4 control points for a cubic curve
        assert len(self.control_points) == 4, "A cubic Bézier curve requires exactly 4 control points."

        # Unpack control points
        p0, p1, p2, p3 = self.control_points

        # Precompute curve points
        self.curve_points = self._compute_curve_points(p0, p1, p2, p3)

    def _compute_curve_points(self, p0, p1, p2, p3):
        """Computes points along the cubic Bézier curve"""
        curve_points = []
        t_values = np.linspace(0, 1, self.num_points)

        for t in t_values:
            one_minus_t = 1 - t
            point = (
                one_minus_t**3 * np.array(p0)
                + 3 * one_minus_t**2 * t * np.array(p1)
                + 3 * one_minus_t * t**2 * np.array(p2)
                + t**3 * np.array(p3)
            )
            curve_points.append(tuple(point))

        return curve_points

    def plot_curve(self, figure_id=0):
        plt.figure(figure_id)
        # Separate the points into x and y components
        curve_points = np.array(self.curve_points)
        x_vals = curve_points[:, 0]
        y_vals = curve_points[:, 1]
        plt.plot(x_vals, y_vals)

        # Plot the control points
        control_x_vals, control_y_vals = zip(*self.control_points)
        plt.plot(control_x_vals, control_y_vals, "ro--")
