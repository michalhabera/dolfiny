import logging

logger = logging.Logger("dolfiny")

from dolfiny import (expression, function, interpolation, io, la, localsolver, mesh, odeint,
                     projection, restriction, slepcblockproblem,
                     snesblockproblem)

__all__ = ["expression", "function", "io", "interpolation", "la", "localsolver", "mesh",
           "odeint", "projection", "restriction", "slepcblockproblem", "snesblockproblem"]
